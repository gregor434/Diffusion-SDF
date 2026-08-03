#!/usr/bin/env python3

"""Creation, caching, and loading utilities for native COD latent tensors."""

import copy
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader.conditioning import build_conditioning_sources


SPLIT_KEYS = ("TrainSplit", "ValSplit", "TestSplit", "ModulationSplit")


def _load_split(value):
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text())


def merge_splits(splits):
    """Return the stable union of dataset/class/object entries in ``splits``."""
    merged = {}
    seen = set()
    for split in splits:
        for dataset, classes in split.items():
            for class_name, object_ids in classes.items():
                output = merged.setdefault(dataset, {}).setdefault(class_name, [])
                for object_id in object_ids:
                    key = (dataset, class_name, object_id)
                    if key not in seen:
                        seen.add(key)
                        output.append(object_id)
    return merged


def configured_splits(specs):
    """Load every split supplied by a stage-two configuration."""
    return {
        key: _load_split(specs[key])
        for key in SPLIT_KEYS
        if specs.get(key) is not None
    }


def extraction_split_name(specs, test_split_only=False):
    """Select the stage-one extraction manifest."""
    if test_split_only:
        return "TestSplit"
    return "ModulationSplit" if specs.get("ModulationSplit") else "TestSplit"


def _modulation_filename(variant_index, base_name="modulation.npz"):
    if variant_index == 0:
        return base_name
    base = Path(base_name)
    return f"{base.stem}_{variant_index:03d}{base.suffix}"


def _split_records(split, cache_path=None, modulation_variants=1):
    records = []
    for dataset, classes in split.items():
        for class_name, object_ids in classes.items():
            for object_id in object_ids:
                for variant_index in range(max(1, int(modulation_variants))):
                    record = {
                        "dataset": dataset,
                        "class_name": class_name,
                        "instance_name": object_id,
                        "variant_index": variant_index,
                    }
                    if cache_path is not None:
                        record["latent_path"] = str(
                            Path(cache_path)
                            / class_name
                            / object_id
                            / _modulation_filename(variant_index)
                        )
                    records.append(record)
    return records


class SurfacePointLoader(Dataset):
    """Read only the surface samples required by the COD latent encoder."""

    def __init__(
        self,
        data_source,
        records,
        surface_point_count=2048,
        deterministic_sampling=False,
        sampling_seed=0,
    ):
        self.surface_point_count = int(surface_point_count)
        self.deterministic_sampling = bool(deterministic_sampling)
        self.sampling_seed = int(sampling_seed)
        self.records = []
        missing = []
        root = Path(data_source)
        for record in records:
            path = (
                root
                / record["dataset"]
                / record["class_name"]
                / record["instance_name"]
                / "cod_sdf.npz"
            )
            if path.is_file():
                self.records.append({**record, "surface_path": str(path)})
            else:
                missing.append(path)
        if missing:
            preview = ", ".join(map(str, missing[:5]))
            raise FileNotFoundError(
                f"missing {len(missing)} COD surface files required for modulation "
                f"caching; first paths: {preview}"
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        rng = (
            np.random.RandomState(self.sampling_seed + index)
            if self.deterministic_sampling
            else np.random
        )
        with np.load(record["surface_path"]) as data:
            surface = data["surface_points"]
            indices = rng.choice(
                len(surface),
                self.surface_point_count,
                replace=len(surface) < self.surface_point_count,
            )
            surface = np.asarray(surface[indices], dtype=np.float32)
        return {
            "surface_points": torch.from_numpy(surface),
            "dataset": record["dataset"],
            "class_name": record["class_name"],
            "object_id": record["instance_name"],
            "latent_path": record["latent_path"],
        }


def _save_modulation(path, object_id, mean, logvar):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            object_id=np.asarray(object_id),
            posterior_mean=np.asarray(mean, dtype=np.float32),
            posterior_logvar=np.asarray(logvar, dtype=np.float32),
        )
    temporary.replace(path)


def _load_stage1_encoder(checkpoint_path, fallback_specs):
    """Load a stage-one model and discard all decoder-only components."""
    from models.sdf_model import SdfModel

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            "modulation_ckpt_path must reference a Lightning checkpoint with a state_dict"
        )
    stage1_specs = copy.deepcopy(
        checkpoint.get("hyper_parameters", {}).get("specs", fallback_specs)
    )
    # The Lightning checkpoint supersedes the upstream COD checkpoint. Avoid
    # loading the latter just to overwrite it immediately.
    stage1_specs.setdefault("CODVaeSpecs", {})["checkpoint_path"] = None
    model = SdfModel(stage1_specs)
    state = {
        key[len("sdf_model."):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("sdf_model.")
    }
    if not state:
        raise RuntimeError(f"no sdf_model parameters found in {checkpoint_path}")
    model.load_state_dict(state, strict=True)

    # Surface encoding uses only the point encoder and variational projection.
    # Remove the much larger triplane/SDF and latent decoder modules before the
    # model is transferred to the accelerator.
    del model.feature_adapter
    del model.sdf_decoder
    del model.cod_vae.latent_proj_out
    del model.cod_vae.latent_decoder
    del model.cod_vae.autoencoder.decoder
    del model.cod_vae.autoencoder.head
    del state
    del checkpoint
    gc.collect()
    return model, stage1_specs


@torch.no_grad()
def ensure_modulation_cache(
    specs,
    exp_dir,
    batch_size=8,
    workers=0,
    device=None,
):
    """Create missing modulations for the union of all configured splits.

    By default the cache is local to the stage-two experiment. Continuation
    experiments can set ``modulation_cache_path`` to reuse the exact cached
    latents and normalization statistics of the experiment they initialize
    from. The stage-one model is loaded only when files are missing and is
    explicitly released before this function returns.
    """
    splits = configured_splits(specs)
    if "TrainSplit" not in splits:
        raise ValueError("diffusion training requires TrainSplit")
    all_split = merge_splits(splits.values())
    cache_path = Path(
        specs.get("modulation_cache_path", Path(exp_dir) / "modulations")
    )
    cache_path.mkdir(parents=True, exist_ok=True)
    modulation_variants = max(1, int(specs.get("modulation_variants", 1)))
    # Validation/test objects need only the canonical variant. Training objects
    # receive additional independently surface-sampled encodings.
    expected_by_path = {
        record["latent_path"]: record
        for record in _split_records(all_split, cache_path)
    }
    for record in _split_records(
        splits["TrainSplit"], cache_path, modulation_variants
    ):
        expected_by_path.setdefault(record["latent_path"], record)
    expected = list(expected_by_path.values())
    missing = [record for record in expected if not Path(record["latent_path"]).is_file()]

    if missing:
        checkpoint_path = specs.get("modulation_ckpt_path")
        if not checkpoint_path:
            raise ValueError(
                "modulation_ckpt_path is required to create stage-two modulations"
            )
        encoder = None
        surface_batch = None
        latent = None
        posterior = None
        encoded = None
        try:
            encoder, stage1_specs = _load_stage1_encoder(checkpoint_path, specs)
            data_source = specs.get("DataSource", stage1_specs.get("DataSource"))
            if not data_source:
                raise ValueError(
                    "DataSource must be set in stage two or embedded in the stage-one checkpoint"
                )
            surface_count = int(
                specs.get(
                    "ModulationSurfacePointCount",
                    stage1_specs.get("SurfacePointCount", 2048),
                )
            )
            dataset = SurfacePointLoader(
                data_source,
                missing,
                surface_count,
                deterministic_sampling=bool(
                    specs.get("DeterministicModulationSurfaceSampling", False)
                ),
                sampling_seed=int(specs.get("ModulationSurfaceSamplingSeed", 0)),
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=max(1, int(batch_size)),
                num_workers=max(0, int(workers)),
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )
            device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            encoder = encoder.to(device).eval()
            for batch in loader:
                surface_batch = batch["surface_points"].to(
                    device, non_blocking=True
                )
                latent, posterior, encoded = encoder.encode_surface(
                    surface_batch,
                    sample_posterior=False,
                )
                if posterior is None:
                    raise RuntimeError("the stage-one encoder did not return a posterior")
                means = posterior.mean.detach().cpu().numpy()
                logvars = posterior.logvar.detach().cpu().numpy()
                for index, object_id in enumerate(batch["object_id"]):
                    path = batch["latent_path"][index]
                    _save_modulation(path, object_id, means[index], logvars[index])
                del surface_batch, latent, posterior, encoded
                surface_batch = latent = posterior = encoded = None
        finally:
            del surface_batch, latent, posterior, encoded
            if encoder is not None:
                del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    incomplete = [
        record["latent_path"]
        for record in expected
        if not Path(record["latent_path"]).is_file()
    ]
    if incomplete:
        raise RuntimeError(
            f"modulation caching left {len(incomplete)} files missing; first paths: "
            + ", ".join(incomplete[:5])
        )

    train_records = ModulationLoader.build_records(
        cache_path,
        splits["TrainSplit"],
        modulation_variants=modulation_variants,
    )
    expected_train = _split_records(
        splits["TrainSplit"], modulation_variants=modulation_variants
    )
    if len(train_records) != len(expected_train):
        raise RuntimeError(
            f"expected {len(expected_train)} training modulations, found {len(train_records)}"
        )
    mean, std = compute_latent_statistics(
        train_records,
        include_posterior_variance=bool(
            specs.get("sample_posterior_latents", False)
        ),
    )
    stats_path = cache_path / "latent_stats.npz"
    save_latent_statistics(stats_path, mean, std)
    return str(cache_path), str(stats_path)


def compute_latent_statistics(records, include_posterior_variance=False):
    count = 0
    total = None
    total_square = None
    for record in records:
        with np.load(record["latent_path"]) as data:
            latent = np.asarray(data["posterior_mean"], dtype=np.float64)
            posterior_variance = (
                np.exp(np.asarray(data["posterior_logvar"], dtype=np.float64))
                if include_posterior_variance
                else None
            )
        flattened = latent.reshape(-1, latent.shape[-1])
        count += flattened.shape[0]
        value_sum = flattened.sum(axis=0)
        square_values = np.square(flattened)
        if posterior_variance is not None:
            square_values += posterior_variance.reshape(
                -1, posterior_variance.shape[-1]
            )
        square_sum = square_values.sum(axis=0)
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
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as output:
        np.savez(output, mean=mean, std=std)
    temporary.replace(path)


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
        sample_posterior=False,
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
        self.sample_posterior = bool(sample_posterior)

        stats_path = Path(latent_stats_path or Path(data_path) / "latent_stats.npz")
        if stats_path.is_file():
            with np.load(stats_path) as data:
                mean, std = data["mean"], data["std"]
        else:
            mean, std = compute_latent_statistics(
                self.records,
                include_posterior_variance=self.sample_posterior,
            )
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
        if self.sample_posterior:
            if logvar is None:
                raise ValueError(
                    f"posterior sampling requested but logvar is missing: "
                    f"{record['latent_path']}"
                )
            latent = latent + torch.exp(0.5 * logvar) * torch.randn_like(latent)
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
    def build_records(
        data_source,
        split,
        conditioning_sources=None,
        f_name="modulation.npz",
        modulation_variants=1,
    ):
        conditioning_sources = conditioning_sources or []
        records = []
        for dataset, classes in split.items():
            for class_name, instance_names in classes.items():
                for instance_name in instance_names:
                    for variant_index in range(max(1, int(modulation_variants))):
                        path = (
                            Path(data_source)
                            / class_name
                            / instance_name
                            / _modulation_filename(variant_index, f_name)
                        )
                        if not path.is_file():
                            continue
                        record = {
                            "dataset": dataset,
                            "class_name": class_name,
                            "instance_name": instance_name,
                            "variant_index": variant_index,
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
