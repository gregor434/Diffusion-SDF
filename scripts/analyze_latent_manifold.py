#!/usr/bin/env python3
"""Assess COD latent-set manifold structure and diffusion sample alignment."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.combined_model import CombinedModel


def load_split(path):
    return json.loads(Path(path).read_text())


def split_ids(split):
    return [
        (dataset, class_name, object_id)
        for dataset, classes in split.items()
        for class_name, object_ids in classes.items()
        for object_id in object_ids
    ]


def load_statistics(path):
    with np.load(path) as data:
        mean = np.asarray(data["mean"], dtype=np.float32).reshape(1, -1)
        std = np.asarray(data["std"], dtype=np.float32).reshape(1, -1)
    return mean, std


def load_latents(cache, split, mean, std, include_variants=False):
    latents = []
    keys = []
    logvars = []
    for _, class_name, object_id in split_ids(split):
        directory = cache / class_name / object_id
        paths = (
            sorted(directory.glob("modulation*.npz"))
            if include_variants
            else [directory / "modulation.npz"]
        )
        for path in paths:
            with np.load(path) as data:
                latent = np.asarray(data["posterior_mean"], dtype=np.float32)
                logvar = np.asarray(data["posterior_logvar"], dtype=np.float32)
            latents.append((latent - mean) / std)
            logvars.append(logvar)
            keys.append(f"{class_name}/{object_id}/{path.name}")
    return np.stack(latents), keys, np.stack(logvars)


def invariant_features(latents):
    """Permutation-invariant summary retaining channel covariance and quantiles."""
    quantiles = np.quantile(latents, (0.1, 0.25, 0.5, 0.75, 0.9), axis=1)
    quantiles = quantiles.transpose(1, 0, 2).reshape(len(latents), -1)
    centered = latents - latents.mean(axis=1, keepdims=True)
    covariance = np.einsum("ntd,nte->nde", centered, centered) / latents.shape[1]
    upper = np.triu_indices(latents.shape[-1])
    return np.concatenate(
        (
            latents.mean(axis=1),
            latents.std(axis=1),
            quantiles,
            covariance[:, upper[0], upper[1]],
        ),
        axis=1,
    )


def fit_pca(train_features, *feature_sets):
    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_std = train_features.std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0
    standardized = [
        (features - feature_mean) / feature_std
        for features in (train_features, *feature_sets)
    ]
    u, singular, vh = np.linalg.svd(standardized[0], full_matrices=False)
    del u
    eigenvalues = singular**2 / max(1, len(train_features) - 1)
    explained = eigenvalues / eigenvalues.sum()
    projections = [features @ vh[:2].T for features in standardized]
    return projections, explained


def set_chamfer_matrix(queries, references, query_chunk=8, reference_chunk=32):
    """Squared symmetric token-set Chamfer distances."""
    queries = torch.from_numpy(np.asarray(queries, dtype=np.float32))
    references = torch.from_numpy(np.asarray(references, dtype=np.float32))
    result = np.empty((len(queries), len(references)), dtype=np.float32)
    with torch.no_grad():
        for query_start in range(0, len(queries), query_chunk):
            query = queries[query_start : query_start + query_chunk]
            for reference_start in range(0, len(references), reference_chunk):
                reference = references[
                    reference_start : reference_start + reference_chunk
                ]
                difference = (
                    query[:, None, :, None, :]
                    - reference[None, :, None, :, :]
                )
                distances = difference.square().mean(dim=-1)
                chamfer = (
                    distances.min(dim=-1).values.mean(dim=-1)
                    + distances.min(dim=-2).values.mean(dim=-1)
                )
                result[
                    query_start : query_start + len(query),
                    reference_start : reference_start + len(reference),
                ] = chamfer.cpu().numpy()
    return result


def nearest_distances(queries, train, self_comparison=False):
    distances = set_chamfer_matrix(queries, train)
    if self_comparison:
        np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def generate_latents(specs, checkpoint, count, seed, device):
    model = CombinedModel.load_from_checkpoint(
        checkpoint, specs=specs, map_location="cpu"
    ).to(device).eval()
    if model.diffusion_model.model.conditional:
        raise ValueError(
            "automatic generation currently supports unconditional checkpoints; "
            "omit --checkpoint to analyze cached encoder latents only"
        )
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        count,
        model.diffusion_model.latent_tokens,
        model.diffusion_model.latent_dimension,
        generator=generator,
        device=device,
    )
    with torch.no_grad():
        generated = model.diffusion_model.sample(count, noise=noise)
    return generated.cpu().numpy()


def distribution_summary(values):
    values = np.asarray(values)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def relative_stability(current, baseline):
    """Compare median resampling distances without imposing a pass/fail gate."""
    comparison = {}
    for name, current_summary in current.items():
        baseline_summary = baseline.get(name)
        if not isinstance(current_summary, dict) or not isinstance(
            baseline_summary, dict
        ):
            continue
        baseline_median = float(baseline_summary["median"])
        current_median = float(current_summary["median"])
        comparison[name] = {
            "baseline_median": baseline_median,
            "current_median": current_median,
            "relative_delta": (
                current_median / baseline_median - 1.0
                if baseline_median != 0.0
                else None
            ),
            "relative_reduction": (
                1.0 - current_median / baseline_median
                if baseline_median != 0.0
                else None
            ),
        }
    return comparison


def save_embedding_plot(path, projections, labels):
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    styles = {
        "train": ("#3366cc", 18, 0.55),
        "validation": ("#109618", 26, 0.8),
        "generated": ("#dc3912", 28, 0.8),
        "gaussian": ("#990099", 22, 0.55),
    }
    for points, label in zip(projections, labels):
        color, size, alpha = styles[label]
        axis.scatter(
            points[:, 0], points[:, 1], s=size, alpha=alpha,
            c=color, label=label, edgecolors="none",
        )
    axis.set_title("Permutation-invariant COD latent-set PCA")
    axis.set_xlabel("PC 1")
    axis.set_ylabel("PC 2")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_neighborhood_plot(path, groups):
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    names = list(groups)
    values = [groups[name] for name in names]
    parts = axis.violinplot(values, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.55)
    axis.set_xticks(range(1, len(names) + 1), names, rotation=15)
    axis.set_ylabel("Nearest training-set token Chamfer (squared)")
    axis.set_title("Latent manifold neighborhood distance")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_structure_plot(path, explained, posterior_std):
    cumulative = np.cumsum(explained)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(np.arange(1, len(cumulative) + 1), cumulative, color="#3366cc")
    axes[0].axhline(0.9, color="black", linestyle="--", linewidth=1)
    axes[0].axhline(0.95, color="black", linestyle=":", linewidth=1)
    axes[0].set_xlim(1, min(100, len(cumulative)))
    axes[0].set_ylim(0, 1.01)
    axes[0].set_xlabel("PCA components")
    axes[0].set_ylabel("Cumulative explained variance")
    axes[0].set_title("Invariant-feature intrinsic dimension")
    axes[0].grid(alpha=0.2)

    image = axes[1].imshow(
        posterior_std, aspect="auto", interpolation="nearest", cmap="magma"
    )
    axes[1].set_xlabel("latent channel")
    axes[1].set_ylabel("latent token index")
    axes[1].set_title("Mean encoder posterior standard deviation")
    fig.colorbar(image, ax=axes[1], shrink=0.85)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("config/cod/stage2_transformer_diffusion_multiray21_medium"),
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint", type=Path)
    checkpoint_group.add_argument(
        "--cache-only",
        action="store_true",
        help=(
            "analyze cached encoder latents without loading an existing "
            "diffusion checkpoint or generating diffusion samples"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help=(
            "Optional report.json from the FPS-initialized baseline; adds "
            "source-relative resampling deltas without applying a threshold"
        ),
    )
    parser.add_argument("--samples", type=int, default=76)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    specs = json.loads((args.exp_dir / "specs.json").read_text())
    cache = Path(specs["data_path"])
    mean, std = load_statistics(specs["latent_stats_path"])
    train_split = load_split(specs["TrainSplit"])
    validation_split = load_split(specs["ValSplit"])
    train, train_keys, train_logvar = load_latents(
        cache, train_split, mean, std, include_variants=False
    )
    train_variants, variant_keys, _ = load_latents(
        cache, train_split, mean, std, include_variants=True
    )
    validation, validation_keys, validation_logvar = load_latents(
        cache, validation_split, mean, std, include_variants=False
    )

    generated = None
    checkpoint = args.checkpoint
    if checkpoint is None and not args.cache_only:
        candidate = args.exp_dir / "best.ckpt"
        checkpoint = candidate if candidate.is_file() else None
    if checkpoint is not None:
        generated = generate_latents(
            specs, checkpoint, args.samples, args.seed, torch.device(args.device)
        )
    rng = np.random.default_rng(args.seed)
    gaussian = rng.standard_normal(
        (args.samples, train.shape[1], train.shape[2])
    ).astype(np.float32)

    feature_sets = {
        "train": invariant_features(train),
        "validation": invariant_features(validation),
    }
    if generated is not None:
        feature_sets["generated"] = invariant_features(generated)
    feature_sets["gaussian"] = invariant_features(gaussian)
    names = list(feature_sets)
    projections, explained = fit_pca(
        feature_sets["train"],
        *(feature_sets[name] for name in names[1:]),
    )

    train_nearest = nearest_distances(train, train, self_comparison=True)
    validation_nearest = nearest_distances(validation, train)
    gaussian_nearest = nearest_distances(gaussian, train)
    neighborhoods = {
        "train LOO": train_nearest,
        "validation": validation_nearest,
        "Gaussian": gaussian_nearest,
    }
    generated_nearest = None
    if generated is not None:
        generated_nearest = nearest_distances(generated, train)
        neighborhoods["generated"] = generated_nearest

    canonical_by_object = {
        key.rsplit("/", 1)[0]: latent for key, latent in zip(train_keys, train)
    }
    variant_pairs = [
        (latent, canonical_by_object[key.rsplit("/", 1)[0]])
        for key, latent in zip(variant_keys, train_variants)
        if not key.endswith("/modulation.npz")
    ]
    variant_distance = np.asarray(
        [
            set_chamfer_matrix(first[None], second[None])[0, 0]
            for first, second in variant_pairs
        ],
        dtype=np.float32,
    )
    variant_aligned_mse = np.asarray(
        [np.square(first - second).mean() for first, second in variant_pairs],
        dtype=np.float32,
    )

    all_logvar = np.concatenate((train_logvar, validation_logvar), axis=0)
    posterior_std = np.exp(0.5 * all_logvar).mean(axis=0)
    participation_ratio = float(
        explained.sum() ** 2 / np.square(explained).sum()
    )
    components_90 = int(np.searchsorted(np.cumsum(explained), 0.9) + 1)
    components_95 = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)
    train_median = float(np.median(train_nearest))

    report = {
        "experiment": str(args.exp_dir),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "latent_shape": list(train.shape[1:]),
        "counts": {
            "train_objects": len(train),
            "train_cached_variants": len(train_variants),
            "validation_objects": len(validation),
            "generated": 0 if generated is None else len(generated),
        },
        "normalization": {
            "train_value_mean": float(train.mean()),
            "train_value_std": float(train.std()),
            "validation_value_mean": float(validation.mean()),
            "validation_value_std": float(validation.std()),
        },
        "intrinsic_dimension": {
            "participation_ratio": participation_ratio,
            "components_for_90_percent": components_90,
            "components_for_95_percent": components_95,
            "feature_dimension": int(feature_sets["train"].shape[1]),
        },
        "nearest_training_set_chamfer": {
            "train_leave_one_out": distribution_summary(train_nearest),
            "validation": distribution_summary(validation_nearest),
            "gaussian": distribution_summary(gaussian_nearest),
            "generated": (
                distribution_summary(generated_nearest)
                if generated_nearest is not None else None
            ),
            "validation_to_train_median_ratio": float(
                np.median(validation_nearest) / train_median
            ),
            "generated_to_train_median_ratio": (
                float(np.median(generated_nearest) / train_median)
                if generated_nearest is not None else None
            ),
        },
        "encoder_surface_resampling_stability": {
            "variant_to_canonical_chamfer": distribution_summary(variant_distance),
            "variant_to_canonical_aligned_mse": distribution_summary(
                variant_aligned_mse
            ),
            "variant_to_train_median_ratio": float(
                np.median(variant_distance) / train_median
            ),
        },
        "posterior": {
            "mean_std": float(np.exp(0.5 * all_logvar).mean()),
            "median_std": float(np.median(np.exp(0.5 * all_logvar))),
            "maximum_mean_token_channel_std": float(posterior_std.max()),
        },
    }
    if args.baseline_report is not None:
        baseline_report = json.loads(args.baseline_report.read_text())
        baseline_stability = baseline_report.get(
            "encoder_surface_resampling_stability", {}
        )
        report["encoder_surface_resampling_comparison"] = relative_stability(
            report["encoder_surface_resampling_stability"],
            baseline_stability,
        )

    output_dir = args.output_dir or args.exp_dir / "latent_manifold"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    save_embedding_plot(output_dir / "embedding.png", projections, names)
    save_neighborhood_plot(output_dir / "neighborhoods.png", neighborhoods)
    save_structure_plot(
        output_dir / "structure.png", explained, posterior_std
    )
    print(json.dumps(report, indent=2))
    print(f"Saved latent-manifold analysis to {output_dir}")


if __name__ == "__main__":
    main()
