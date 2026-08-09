#!/usr/bin/env python3
"""Compare two stage-one checkpoints under identical surface resampling."""

import argparse
import gc
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.sdf_model import SdfModel


STABILITY_METRICS = (
    "latent_aligned_mse",
    "latent_set_chamfer",
    "sdf_pairwise_mae",
    "sdf_sign_flip_fraction",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Measure how two COD encoders respond to identical resamplings of "
            "the same validation surfaces. Lower values are better."
        )
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-source", type=Path)
    parser.add_argument("--surface-samples", type=int, default=8)
    parser.add_argument("--surface-points", type=int, default=2048)
    parser.add_argument("--query-points", type=int, default=4096)
    parser.add_argument("--near-surface-ratio", type=float)
    parser.add_argument("--variant-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def validate_arguments(args):
    if args.surface_samples < 2:
        raise ValueError("surface-samples must be at least 2")
    if args.surface_points <= 0:
        raise ValueError("surface-points must be positive")
    if args.query_points <= 0:
        raise ValueError("query-points must be positive")
    if args.variant_batch_size <= 0:
        raise ValueError("variant-batch-size must be positive")
    if args.near_surface_ratio is not None and not (
        0.0 <= args.near_surface_ratio <= 1.0
    ):
        raise ValueError("near-surface-ratio must be in [0, 1]")


def flatten_split(split, data_source):
    records = []
    for dataset, classes in split.items():
        for class_name, object_ids in classes.items():
            for object_id in object_ids:
                path = (
                    data_source
                    / dataset
                    / class_name
                    / object_id
                    / "cod_sdf.npz"
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                records.append(
                    {
                        "dataset": dataset,
                        "class_name": class_name,
                        "object_id": object_id,
                        "path": path,
                    }
                )
    if not records:
        raise ValueError("the evaluation split contains no objects")
    return records


def sample_object(path, object_index, args, near_surface_ratio):
    """Return paired surface variants and one fixed SDF query set."""
    base_seed = int(args.seed) + object_index * 100_003
    with np.load(path) as data:
        source_surface = data["surface_points"]
        surfaces = []
        for variant_index in range(args.surface_samples):
            rng = np.random.RandomState(base_seed + variant_index * 997)
            indices = rng.choice(
                len(source_surface),
                args.surface_points,
                replace=len(source_surface) < args.surface_points,
            )
            surfaces.append(
                np.asarray(source_surface[indices], dtype=np.float32)
            )

        query_rng = np.random.RandomState(base_seed + 97_000_003)
        near_count = round(args.query_points * near_surface_ratio)
        uniform_count = args.query_points - near_count
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
        ).astype(np.float32, copy=False)
        query_sdf = np.concatenate(
            (
                data["near_surface_sdf"][near_indices],
                data["uniform_sdf"][uniform_indices],
            ),
            axis=0,
        ).astype(np.float32, copy=False)
        permutation = query_rng.permutation(args.query_points)

    return (
        np.stack(surfaces),
        np.asarray(query_points[permutation], dtype=np.float32),
        np.asarray(query_sdf[permutation], dtype=np.float32),
    )


def checkpoint_metadata(checkpoint, path):
    callbacks = checkpoint.get("callbacks", {})
    scores = []
    for state in callbacks.values() if isinstance(callbacks, dict) else ():
        if isinstance(state, dict) and state.get("best_model_score") is not None:
            value = state["best_model_score"]
            scores.append(float(value.item() if torch.is_tensor(value) else value))
    return {
        "path": str(path),
        "epoch_index": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "best_model_score": min(scores) if scores else None,
    }


def load_stage1_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"not a Lightning checkpoint with a state_dict: {path}")
    specs = checkpoint.get("hyper_parameters", {}).get("specs")
    if not isinstance(specs, dict):
        raise ValueError(f"checkpoint has no embedded experiment specs: {path}")
    specs = json.loads(json.dumps(specs))
    specs.setdefault("CODVaeSpecs", {})["checkpoint_path"] = None
    model = SdfModel(specs)
    state = {
        key[len("sdf_model."):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("sdf_model.")
    }
    if not state:
        raise RuntimeError(f"checkpoint contains no sdf_model parameters: {path}")
    model.load_state_dict(state, strict=True)
    metadata = checkpoint_metadata(checkpoint, path)
    del state, checkpoint
    gc.collect()
    return model.to(device).eval(), specs, metadata


@torch.inference_mode()
def evaluate_checkpoint(path, records, args, device, near_surface_ratio):
    model, specs, metadata = load_stage1_model(path, device)
    all_latents = []
    all_predictions = []
    targets = []
    try:
        for object_index, record in enumerate(
            tqdm(records, desc=f"evaluating {path.name}")
        ):
            surfaces, queries, target = sample_object(
                record["path"], object_index, args, near_surface_ratio
            )
            object_latents = []
            object_predictions = []
            for start in range(0, args.surface_samples, args.variant_batch_size):
                surface_batch = torch.from_numpy(
                    surfaces[start : start + args.variant_batch_size]
                ).to(device)
                query_batch = torch.from_numpy(queries).to(device).unsqueeze(0)
                query_batch = query_batch.expand(len(surface_batch), -1, -1)
                latent, posterior, _ = model.encode_surface(
                    surface_batch, sample_posterior=False
                )
                if posterior is None:
                    raise RuntimeError("resampling comparison requires VAE posteriors")
                latent = posterior.mode()
                decoded = model.decode_latent(latent)
                prediction = model.query_sdf(decoded["planes"], query_batch)
                object_latents.append(posterior.mean.detach().cpu().numpy())
                object_predictions.append(prediction.detach().cpu().numpy())
            all_latents.append(np.concatenate(object_latents, axis=0))
            all_predictions.append(np.concatenate(object_predictions, axis=0))
            targets.append(target)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "latents": np.stack(all_latents),
        "predictions": np.stack(all_predictions),
        "targets": np.stack(targets),
        "specs": specs,
        "metadata": metadata,
    }


def distribution_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize an empty metric")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def checkpoint_metrics(result):
    latents = np.asarray(result["latents"], dtype=np.float32)
    predictions = np.asarray(result["predictions"], dtype=np.float32)
    targets = np.asarray(result["targets"], dtype=np.float32)
    channel_mean = latents.mean(axis=(0, 1, 2), keepdims=True)
    channel_std = latents.std(axis=(0, 1, 2), keepdims=True)
    channel_std = np.maximum(channel_std, 1e-6)
    normalized = (latents - channel_mean) / channel_std

    pair_indices = list(combinations(range(latents.shape[1]), 2))
    aligned_mse = []
    set_chamfer = []
    sdf_mae = []
    sign_flip = []
    for object_index in range(latents.shape[0]):
        for first_index, second_index in pair_indices:
            first = normalized[object_index, first_index]
            second = normalized[object_index, second_index]
            aligned_mse.append(float(np.square(first - second).mean()))
            distances = np.square(first[:, None, :] - second[None, :, :]).mean(
                axis=-1
            )
            set_chamfer.append(
                float(distances.min(axis=1).mean() + distances.min(axis=0).mean())
            )
            first_sdf = predictions[object_index, first_index]
            second_sdf = predictions[object_index, second_index]
            sdf_mae.append(float(np.abs(first_sdf - second_sdf).mean()))
            sign_flip.append(
                float(np.not_equal(first_sdf >= 0.0, second_sdf >= 0.0).mean())
            )

    reconstruction = np.abs(predictions - targets[:, None, :]).mean(axis=-1)
    return {
        "latent_aligned_mse": distribution_summary(aligned_mse),
        "latent_set_chamfer": distribution_summary(set_chamfer),
        "sdf_pairwise_mae": distribution_summary(sdf_mae),
        "sdf_sign_flip_fraction": distribution_summary(sign_flip),
        "sdf_ground_truth_mae": distribution_summary(reconstruction.reshape(-1)),
        "normalization": {
            "channel_mean": channel_mean.reshape(-1).astype(float).tolist(),
            "channel_std": channel_std.reshape(-1).astype(float).tolist(),
        },
    }


def compare_metrics(baseline, candidate):
    comparison = {}
    for name in (*STABILITY_METRICS, "sdf_ground_truth_mae"):
        baseline_median = float(baseline[name]["median"])
        candidate_median = float(candidate[name]["median"])
        comparison[name] = {
            "lower_is_better": True,
            "baseline_median": baseline_median,
            "candidate_median": candidate_median,
            "absolute_delta": candidate_median - baseline_median,
            "relative_delta": (
                candidate_median / baseline_median - 1.0
                if baseline_median != 0.0
                else None
            ),
            "relative_reduction": (
                1.0 - candidate_median / baseline_median
                if baseline_median != 0.0
                else None
            ),
            "candidate_improves_median": candidate_median < baseline_median,
        }
    comparison["candidate_improves_all_stability_medians"] = all(
        comparison[name]["candidate_improves_median"]
        for name in STABILITY_METRICS
    )
    return comparison


def main():
    args = build_parser().parse_args()
    validate_arguments(args)
    device = torch.device(args.device)
    split = json.loads(args.split.read_text())

    baseline_checkpoint = torch.load(args.baseline_checkpoint, map_location="cpu")
    baseline_specs = baseline_checkpoint.get("hyper_parameters", {}).get("specs", {})
    del baseline_checkpoint
    gc.collect()
    data_source = args.data_source or Path(baseline_specs.get("DataSource", "datasets"))
    near_surface_ratio = (
        float(args.near_surface_ratio)
        if args.near_surface_ratio is not None
        else float(baseline_specs.get("NearSurfaceRatio", 0.7))
    )
    records = flatten_split(split, Path(data_source))

    baseline_result = evaluate_checkpoint(
        args.baseline_checkpoint, records, args, device, near_surface_ratio
    )
    candidate_result = evaluate_checkpoint(
        args.candidate_checkpoint, records, args, device, near_surface_ratio
    )
    if baseline_result["latents"].shape != candidate_result["latents"].shape:
        raise ValueError(
            "checkpoint latent shapes differ: "
            f"{baseline_result['latents'].shape} versus "
            f"{candidate_result['latents'].shape}"
        )
    np.testing.assert_array_equal(
        baseline_result["targets"], candidate_result["targets"]
    )

    baseline_metrics = checkpoint_metrics(baseline_result)
    candidate_metrics = checkpoint_metrics(candidate_result)
    report = {
        "schema_version": 1,
        "settings": {
            "split": str(args.split),
            "data_source": str(data_source),
            "objects": len(records),
            "surface_samples_per_object": args.surface_samples,
            "surface_points": args.surface_points,
            "query_points": args.query_points,
            "near_surface_ratio": near_surface_ratio,
            "seed": args.seed,
            "posterior_sampling": False,
            "pairing": "all surface-sample pairs per object",
        },
        "baseline": {
            "checkpoint": baseline_result["metadata"],
            "latent_shape": list(baseline_result["latents"].shape[2:]),
            "metrics": baseline_metrics,
        },
        "candidate": {
            "checkpoint": candidate_result["metadata"],
            "latent_shape": list(candidate_result["latents"].shape[2:]),
            "metrics": candidate_metrics,
        },
        "comparison": compare_metrics(baseline_metrics, candidate_metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["comparison"], indent=2))
    print(f"Saved paired resampling report to {args.output}")


if __name__ == "__main__":
    main()
