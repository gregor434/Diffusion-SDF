#!/usr/bin/env python3

"""Dataset for COD surface encoding and independently sampled SDF supervision."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

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
        deterministic_sampling=False,
        sampling_seed=0,
        **_,
    ):
        self.samples_per_mesh = int(samples_per_mesh)
        self.surface_point_count = int(surface_point_count)
        self.near_surface_ratio = float(near_surface_ratio)
        self.condition_surface = bool(condition_surface)
        self.deterministic_sampling = bool(deterministic_sampling)
        self.sampling_seed = int(sampling_seed)
        if not 0 <= self.near_surface_ratio <= 1:
            raise ValueError("near_surface_ratio must be in [0, 1]")
        if self.surface_point_count <= 0:
            raise ValueError("surface_point_count must be positive")

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
                    if not path.is_file():
                        logging.warning("Requested non-existent file '%s'", path)
                        continue
                    self.records.append((path, dataset, class_name, object_id))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rng = (
            np.random.RandomState(self.sampling_seed + index)
            if self.deterministic_sampling
            else np.random
        )
        path, dataset, class_name, object_id = self.records[index]
        with np.load(path) as data:
            surface_indices = rng.choice(
                len(data["surface_points"]),
                self.surface_point_count,
                replace=len(data["surface_points"]) < self.surface_point_count,
            )
            surface = data["surface_points"][surface_indices]
            surface_normals = (
                data["surface_normals"][surface_indices]
                if "surface_normals" in data else None
            )
            near_count = round(self.samples_per_mesh * self.near_surface_ratio)
            uniform_count = self.samples_per_mesh - near_count
            near_indices = rng.choice(
                len(data["near_surface_query_points"]),
                near_count,
                replace=len(data["near_surface_query_points"]) < near_count,
            )
            uniform_indices = rng.choice(
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

        permutation = rng.permutation(len(query_points))
        query_is_near = np.concatenate(
            (np.ones(near_count, dtype=np.bool_), np.zeros(uniform_count, dtype=np.bool_))
        )[permutation]
        surface = torch.from_numpy(np.asarray(surface, dtype=np.float32))
        item = {
            "surface_points": surface,
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
        if self.condition_surface:
            item["conditioning"] = {"point_cloud": surface}
        if surface_normals is not None:
            item["surface_normals"] = torch.from_numpy(
                np.asarray(surface_normals, dtype=np.float32)
            )
        return item
