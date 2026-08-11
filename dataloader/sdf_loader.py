#!/usr/bin/env python3

"""Dataset for COD surface encoding and independently sampled SDF supervision."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader.conditioning import build_conditioning_sources


class SdfLoader(Dataset):
    def __init__(
        self,
        data_source,
        split_file,
        samples_per_mesh=16000,
        surface_point_count=2048,
        near_surface_ratio=0.7,
        modulation_path=None,
        condition_surface=False,
        conditioning=None,
        conditioning_sources=None,
        deterministic_sampling=False,
        deterministic_surface_sampling=False,
        paired_surface_sampling=False,
        encoder_surface_jitter_std=0.0,
        sampling_seed=0,
        **_,
    ):
        self.samples_per_mesh = int(samples_per_mesh)
        self.surface_point_count = int(surface_point_count)
        self.near_surface_ratio = float(near_surface_ratio)
        self.condition_surface = bool(condition_surface)
        self.conditioning_sources = (
            conditioning_sources
            if conditioning_sources is not None
            else build_conditioning_sources(conditioning)
        )
        self.deterministic_sampling = bool(deterministic_sampling)
        self.deterministic_surface_sampling = bool(
            deterministic_surface_sampling
        )
        self.paired_surface_sampling = bool(paired_surface_sampling)
        self.encoder_surface_jitter_std = float(encoder_surface_jitter_std)
        self.sampling_seed = int(sampling_seed)
        if not 0 <= self.near_surface_ratio <= 1:
            raise ValueError("near_surface_ratio must be in [0, 1]")
        if self.surface_point_count <= 0:
            raise ValueError("surface_point_count must be positive")
        if self.encoder_surface_jitter_std < 0:
            raise ValueError("encoder_surface_jitter_std must be non-negative")

        self.records = []
        data_source = Path(data_source)
        modulation_path = Path(modulation_path) if modulation_path else None
        for dataset, classes in split_file.items():
            for class_name, object_ids in classes.items():
                for object_id in object_ids:
                    path = data_source / dataset / class_name / object_id / "cod_sdf.npz"
                    if modulation_path is not None:
                        modulation = modulation_path / class_name / object_id / "modulation.npz"
                        if not modulation.is_file():
                            continue
                    record = {
                        "data_path": str(path),
                        "dataset": dataset,
                        "class_name": class_name,
                        "instance_name": object_id,
                    }
                    if not path.is_file():
                        logging.warning("Requested non-existent file '%s'", path)
                        continue
                    if not all(
                        source.exists(record)
                        for source in self.conditioning_sources
                    ):
                        logging.warning(
                            "Missing conditioning data for '%s/%s/%s'",
                            dataset, class_name, object_id,
                        )
                        continue
                    self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        query_rng = (
            np.random.RandomState(self.sampling_seed + index)
            if self.deterministic_sampling
            else np.random
        )
        surface_rng = (
            np.random.RandomState(self.sampling_seed + index)
            if self.deterministic_sampling
            or self.deterministic_surface_sampling
            else np.random
        )
        record = self.records[index]
        path = Path(record["data_path"])
        dataset = record["dataset"]
        class_name = record["class_name"]
        object_id = record["instance_name"]
        with np.load(path) as data:
            surface_indices = surface_rng.choice(
                len(data["surface_points"]),
                self.surface_point_count,
                replace=len(data["surface_points"]) < self.surface_point_count,
            )
            surface = data["surface_points"][surface_indices]
            paired_surface = None
            if self.paired_surface_sampling:
                paired_surface_indices = surface_rng.choice(
                    len(data["surface_points"]),
                    self.surface_point_count,
                    replace=len(data["surface_points"]) < self.surface_point_count,
                )
                paired_surface = data["surface_points"][paired_surface_indices]
            surface_normals = (
                data["surface_normals"][surface_indices]
                if "surface_normals" in data else None
            )
            near_count = round(self.samples_per_mesh * self.near_surface_ratio)
            uniform_count = self.samples_per_mesh - near_count
            near_indices = query_rng.choice(
                len(data["near_surface_query_points"]),
                near_count,
                replace=len(data["near_surface_query_points"]) < near_count,
            )
            uniform_indices = query_rng.choice(
                len(data["uniform_query_points"]),
                uniform_count,
                replace=len(data["uniform_query_points"]) < uniform_count,
            )
            query_points = np.concatenate(
                (
                    data["near_surface_query_points"][near_indices],
                    data["uniform_query_points"][uniform_indices],
                ),
                axis=0,
            )
            query_sdf = np.concatenate(
                (
                    data["near_surface_sdf"][near_indices],
                    data["uniform_sdf"][uniform_indices],
                ),
                axis=0,
            )

        permutation = query_rng.permutation(len(query_points))
        query_is_near = np.concatenate(
            (np.ones(near_count, dtype=np.bool_), np.zeros(uniform_count, dtype=np.bool_))
        )[permutation]
        surface = torch.from_numpy(np.asarray(surface, dtype=np.float32))
        encoder_surface = surface.clone()
        if self.encoder_surface_jitter_std > 0:
            jitter = surface_rng.normal(
                0.0, self.encoder_surface_jitter_std, tuple(surface.shape)
            ).astype(np.float32)
            encoder_surface = (encoder_surface + torch.from_numpy(jitter)).clamp(
                -1.0, 1.0
            )
        item = {
            "surface_points": surface,
            "encoder_surface_points": encoder_surface,
            "query_points": torch.from_numpy(
                np.asarray(query_points[permutation], dtype=np.float32)
            ),
            "query_sdf": torch.from_numpy(
                np.asarray(query_sdf[permutation], dtype=np.float32)
            ).reshape(-1),
            "query_is_near": torch.from_numpy(query_is_near),
            "object_id": object_id,
            "dataset": dataset,
            "class_name": class_name,
        }
        if paired_surface is not None:
            paired_surface = torch.from_numpy(
                np.asarray(paired_surface, dtype=np.float32)
            )
            paired_encoder_surface = paired_surface.clone()
            if self.encoder_surface_jitter_std > 0:
                paired_jitter = surface_rng.normal(
                    0.0,
                    self.encoder_surface_jitter_std,
                    tuple(paired_surface.shape),
                ).astype(np.float32)
                paired_encoder_surface = (
                    paired_encoder_surface + torch.from_numpy(paired_jitter)
                ).clamp(-1.0, 1.0)
            item["paired_surface_points"] = paired_surface
            item["paired_encoder_surface_points"] = paired_encoder_surface
        if self.conditioning_sources:
            item["conditioning"] = {
                source.name: source.load(record)
                for source in self.conditioning_sources
            }
            item["conditioning_paths"] = {
                source.name: source.resolve(record)
                for source in self.conditioning_sources
            }
        elif self.condition_surface:
            item["conditioning"] = {"point_cloud": surface}
        if surface_normals is not None:
            item["surface_normals"] = torch.from_numpy(
                np.asarray(surface_normals, dtype=np.float32)
            )
        return item
