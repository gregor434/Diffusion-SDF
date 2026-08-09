#!/usr/bin/env python3
from __future__ import annotations

"""Convert ShapeNetPart chair point clouds into stage-two COD records.

The normalized ShapeNetPart archive stores one text file per object. Its first
three columns are XYZ coordinates; the remaining columns (normals and part
labels) are not inputs to the COD surface encoder. This script writes the
surface-only ``cod_sdf.npz`` layout expected by ``SurfacePointLoader`` and
translates the archive's official train/validation/test lists into this
repository's split-manifest format.

These records are suitable for latent extraction and diffusion training. They
do not contain SDF supervision and therefore cannot be used for stage-one SDF
training.
"""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_SOURCE_DIR = Path(
    "datasets/shapenetpart/"
    "shapenetcore_partanno_segmentation_benchmark_v0_normal"
)
DEFAULT_DATASETS_ROOT = Path("datasets")
DEFAULT_DATASET_KEY = "shapenetpart"
DEFAULT_CLASS_NAME = "CHAIR"
DEFAULT_CATEGORY_ID = "03001627"
DEFAULT_SPLIT_PREFIX = "shapenetpart_CHAIR"
SPLIT_FILENAMES = {
    "train": "shuffled_train_file_list.json",
    "val": "shuffled_val_file_list.json",
    "test": "shuffled_test_file_list.json",
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=(
            "Extracted ShapeNetPart root containing the category directory and "
            "train_test_split. A parent containing the standard nested archive "
            "directory is also accepted."
        ),
    )
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--dataset-key", default=DEFAULT_DATASET_KEY)
    parser.add_argument("--class-name", default=DEFAULT_CLASS_NAME)
    parser.add_argument("--category-id", default=DEFAULT_CATEGORY_ID)
    parser.add_argument("--split-prefix", default=DEFAULT_SPLIT_PREFIX)
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=min(8, os.cpu_count() or 1),
        help="Number of point-cloud conversion workers.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Convert only the first N sorted objects (for smoke tests).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep existing cod_sdf.npz records instead of replacing them.",
    )
    return parser.parse_args()


def resolve_source_root(source_dir: Path, category_id: str) -> Path:
    """Resolve either the extracted archive root or its immediate parent."""
    candidates = (
        source_dir,
        source_dir / "shapenetcore_partanno_segmentation_benchmark_v0_normal",
    )
    for candidate in candidates:
        if (candidate / category_id).is_dir() and (
            candidate / "train_test_split"
        ).is_dir():
            return candidate
    raise FileNotFoundError(
        "could not find the ShapeNetPart category and split directories below "
        f"{source_dir}; expected {category_id}/ and train_test_split/"
    )


def normalize_points_abo(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """Apply the ABO/COD bbox-center and isotropic max-abs-0.999 transform."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"expected a non-empty (N, 3) point array, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("point cloud contains non-finite coordinates")

    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    center = ((bounds_min + bounds_max) * np.float32(0.5)).astype(np.float32)
    centered = points - center
    radius = float(np.abs(centered).max())
    if radius <= 0:
        raise ValueError("point cloud has zero extent")
    scale = np.float32(0.999 / radius)
    normalized = (centered * scale).astype(np.float32)
    return normalized, center, scale


def load_xyz(path: Path) -> np.ndarray:
    try:
        points = np.loadtxt(path, dtype=np.float32, usecols=(0, 1, 2), ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(f"failed to read XYZ coordinates from {path}: {error}") from error
    return points


def output_path(
    datasets_root: Path,
    dataset_key: str,
    class_name: str,
    model_id: str,
) -> Path:
    return datasets_root / dataset_key / class_name / model_id / "cod_sdf.npz"


def convert_one(
    source_path: Path,
    destination: Path,
    *,
    skip_existing: bool,
) -> str:
    if skip_existing and destination.is_file():
        return "skipped"

    normalized, center, scale = normalize_points_abo(load_xyz(source_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                surface_points=normalized,
                normalization_center=center,
                normalization_scale=scale,
                source_format=np.asarray("shapenetpart_normal_txt"),
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def split_model_ids(path: Path, category_id: str) -> list[str]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"ShapeNetPart split must contain a JSON list: {path}")

    model_ids = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError(f"non-string entry in ShapeNetPart split: {path}")
        parts = Path(entry).parts
        if category_id not in parts:
            continue
        model_id = Path(entry).name
        if model_id and model_id not in seen:
            seen.add(model_id)
            model_ids.append(model_id)
    return sorted(model_ids)


def write_manifest(
    path: Path,
    dataset_key: str,
    class_name: str,
    model_ids: Iterable[str],
) -> None:
    payload = {dataset_key: {class_name: sorted(model_ids)}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_dataset(args: argparse.Namespace) -> dict[str, object]:
    source_root = resolve_source_root(args.source_dir, args.category_id)
    category_dir = source_root / args.category_id
    sources = sorted(category_dir.glob("*.txt"))
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        raise RuntimeError(f"no point-cloud .txt files found in {category_dir}")

    available_ids = {path.stem for path in sources}
    results = {"written": 0, "skipped": 0}

    def process(source_path: Path) -> str:
        destination = output_path(
            args.datasets_root,
            args.dataset_key,
            args.class_name,
            source_path.stem,
        )
        return convert_one(
            source_path,
            destination,
            skip_existing=args.skip_existing,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, path): path for path in sources}
        for index, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            source_path = futures[future]
            try:
                status = future.result()
            except Exception as error:
                raise RuntimeError(f"failed to convert {source_path}: {error}") from error
            results[status] += 1
            if index % 250 == 0 or index == len(futures):
                print(f"converted {index}/{len(futures)}", flush=True)

    splits: dict[str, list[str]] = {}
    split_dir = source_root / "train_test_split"
    for split_name, filename in SPLIT_FILENAMES.items():
        ids = split_model_ids(split_dir / filename, args.category_id)
        splits[split_name] = [model_id for model_id in ids if model_id in available_ids]
    splits["all"] = sorted(available_ids)

    assigned = set().union(*(splits[name] for name in SPLIT_FILENAMES))
    missing_from_official_splits = sorted(available_ids - assigned)
    if missing_from_official_splits and args.limit is None:
        preview = ", ".join(missing_from_official_splits[:5])
        raise RuntimeError(
            f"{len(missing_from_official_splits)} converted objects are absent from the "
            f"official splits; first IDs: {preview}"
        )

    manifest_paths = {}
    for split_name, ids in splits.items():
        path = args.datasets_root / "splits" / f"{args.split_prefix}_{split_name}.json"
        write_manifest(path, args.dataset_key, args.class_name, ids)
        manifest_paths[split_name] = str(path)

    metadata_path = (
        args.datasets_root / args.dataset_key / "preprocessing_metadata.json"
    )
    metadata = {
        "dataset_key": args.dataset_key,
        "class_name": args.class_name,
        "category_id": args.category_id,
        "source_root": str(source_root),
        "source_format": "ShapeNetPart normalized point clouds (XYZ columns only)",
        "usage": "stage_two_latent_extraction_and_diffusion_only",
        "normalization": "x_prime = scale * (x - bbox_center), isotropic max_abs_0.999",
        "counts": {name: len(ids) for name, ids in splits.items()},
        "records_written": results["written"],
        "records_skipped": results["skipped"],
        "manifests": manifest_paths,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return {**results, "splits": splits, "metadata_path": str(metadata_path)}


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(args)
    print(
        "summary: "
        f"written={summary['written']} skipped={summary['skipped']} "
        f"train={len(summary['splits']['train'])} "
        f"val={len(summary['splits']['val'])} "
        f"test={len(summary['splits']['test'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
