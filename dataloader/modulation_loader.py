#!/usr/bin/env python3

"""Lazy loader and normalization utilities for native COD latent tensors."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader.conditioning import build_conditioning_sources


def compute_latent_statistics(records):
    count = 0
    total = None
    total_square = None
    for record in records:
        with np.load(record["latent_path"]) as data:
            latent = np.asarray(data["posterior_mean"], dtype=np.float64)
        flattened = latent.reshape(-1, latent.shape[-1])
        count += flattened.shape[0]
        value_sum = flattened.sum(axis=0)
        square_sum = np.square(flattened).sum(axis=0)
        total = value_sum if total is None else total + value_sum
        total_square = square_sum if total_square is None else total_square + square_sum
    if count == 0:
        raise ValueError("cannot compute latent statistics from an empty dataset")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-12)
    return mean.astype(np.float32).reshape(1, 1, -1), np.sqrt(variance).astype(np.float32).reshape(1, 1, -1)


def save_latent_statistics(path, mean, std):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        np.savez(output, mean=mean, std=std)


class ModulationLoader(Dataset):
    def __init__(
        self,
        data_path,
        split_file=None,
        conditioning=None,
        conditioning_sources=None,
        records=None,
        latent_stats_path=None,
        normalize=True,
    ):
        super().__init__()
        self.conditioning_sources = (
            conditioning_sources
            if conditioning_sources is not None
            else build_conditioning_sources(conditioning)
        )
        self.records = records or self.build_records(
            data_path, split_file, self.conditioning_sources
        )
        self.validate_required_conditioning_cache()
        self.normalize = bool(normalize)

        stats_path = Path(latent_stats_path or Path(data_path) / "latent_stats.npz")
        if stats_path.is_file():
            with np.load(stats_path) as data:
                mean, std = data["mean"], data["std"]
        else:
            mean, std = compute_latent_statistics(self.records)
            save_latent_statistics(stats_path, mean, std)
        self.mean = torch.from_numpy(np.asarray(mean, dtype=np.float32))
        self.std = torch.from_numpy(np.asarray(std, dtype=np.float32)).clamp_min(1e-6)

        if self.records:
            with np.load(self.records[0]["latent_path"]) as data:
                shape = data["posterior_mean"].shape
            print("COD modulation shape, dataset len:", shape, len(self.records))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(record["latent_path"]) as data:
            latent = torch.from_numpy(
                np.asarray(data["posterior_mean"], dtype=np.float32)
            )
            logvar = (
                torch.from_numpy(np.asarray(data["posterior_logvar"], dtype=np.float32))
                if "posterior_logvar" in data
                else None
            )
        if self.normalize:
            latent = (latent - self.mean.squeeze(0)) / self.std.squeeze(0)
        conditioning = {
            source.name: source.load(record) for source in self.conditioning_sources
        }
        conditioning_paths = {
            source.name: source.resolve(record) for source in self.conditioning_sources
        }
        item = {
            "latent": latent,
            "dataset": record["dataset"],
            "class_name": record["class_name"],
            "object_id": record["instance_name"],
            "conditioning": conditioning,
            "conditioning_paths": conditioning_paths,
        }
        if logvar is not None:
            item["posterior_logvar"] = logvar
        return item

    @staticmethod
    def build_records(data_source, split, conditioning_sources=None, f_name="modulation.npz"):
        conditioning_sources = conditioning_sources or []
        records = []
        for dataset, classes in split.items():
            for class_name, instance_names in classes.items():
                for instance_name in instance_names:
                    path = Path(data_source) / class_name / instance_name / f_name
                    if not path.is_file():
                        continue
                    record = {
                        "dataset": dataset,
                        "class_name": class_name,
                        "instance_name": instance_name,
                        "latent_path": str(path),
                    }
                    if all(source.exists(record) for source in conditioning_sources):
                        records.append(record)
        return records

    def validate_required_conditioning_cache(self):
        missing_paths = []
        for source in self.conditioning_sources:
            if not getattr(source, "require_cached", False):
                continue
            for record in self.records:
                path = source.resolve_cache_path(record)
                if not Path(path).is_file():
                    missing_paths.append(path)
        if missing_paths:
            raise FileNotFoundError(
                "Missing cached conditioning features; run conditioning preparation. First paths: "
                + ", ".join(map(str, missing_paths[:5]))
            )
