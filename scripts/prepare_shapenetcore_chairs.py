#!/usr/bin/env python3
from __future__ import annotations

"""Convert raw ShapeNetCore-v2 chair meshes into stage-two COD records.

The expected input is the extracted ShapeNetCore-v2 layout::

    <source-dir>/03001627/<model-id>/models/model_normalized.obj

``source-dir`` may also point directly at the ``03001627`` directory.  Each
mesh is normalized from its exact vertex bounds with the same bbox-center and
isotropic max-abs-0.999 convention used by the ABO/COD preprocessing.  Points
are then sampled uniformly by triangle area and saved in the surface-only
``cod_sdf.npz`` layout expected by ``SurfacePointLoader``.

These records are intended for latent extraction and diffusion pretraining.
They contain no SDF supervision.  Consequently, non-watertight meshes are
accepted by default; use ``--require-watertight`` only when preparing data for
a workflow that actually evaluates inside/outside or signed distances.
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import trimesh

try:
    from scripts.prepare_shapenetpart_chairs import (
        normalize_points_abo,
        normalization_extent,
        output_path,
        partition_model_ids,
        positive_int,
        train_ratio,
        write_manifest,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/prepare_....py
    from prepare_shapenetpart_chairs import (  # type: ignore[no-redef]
        normalize_points_abo,
        normalization_extent,
        output_path,
        partition_model_ids,
        positive_int,
        train_ratio,
        write_manifest,
    )


DEFAULT_SOURCE_DIR = Path("datasets/shapenet_raw")
DEFAULT_DATASETS_ROOT = Path("datasets")
DEFAULT_DATASET_KEY = "shapenetcore"
DEFAULT_CLASS_NAME = "CHAIR"
DEFAULT_CATEGORY_ID = "03001627"
DEFAULT_SPLIT_PREFIX = "shapenetcore_CHAIR"
DEFAULT_SURFACE_POINT_COUNT = 10_000
DEFAULT_TRAIN_RATIO = 0.9
DEFAULT_SPLIT_SEED = 0
DEFAULT_SAMPLING_SEED = 0
DEFAULT_NORMALIZATION_EXTENT = 0.999
MESH_RELATIVE_PATH = Path("models/model_normalized.obj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=(
            "Extracted ShapeNetCore-v2 root containing 03001627/, or the "
            "03001627 category directory itself."
        ),
    )
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--dataset-key", default=DEFAULT_DATASET_KEY)
    parser.add_argument("--class-name", default=DEFAULT_CLASS_NAME)
    parser.add_argument("--category-id", default=DEFAULT_CATEGORY_ID)
    parser.add_argument("--split-prefix", default=DEFAULT_SPLIT_PREFIX)
    parser.add_argument(
        "--normalization-extent",
        type=normalization_extent,
        default=DEFAULT_NORMALIZATION_EXTENT,
        help="Maximum absolute normalized mesh coordinate (default: 0.999).",
    )
    parser.add_argument(
        "--surface-point-count",
        type=positive_int,
        default=DEFAULT_SURFACE_POINT_COUNT,
        help="Area-weighted surface samples saved per mesh (default: 10000).",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=DEFAULT_SAMPLING_SEED,
        help="Base seed for deterministic per-object surface sampling.",
    )
    parser.add_argument(
        "--train-ratio",
        type=train_ratio,
        default=DEFAULT_TRAIN_RATIO,
        help="Fraction of all converted chairs assigned to training (default: 0.9).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Seed for the deterministic train/validation partition.",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=min(8, os.cpu_count() or 1),
        help="Number of mesh-conversion workers.",
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
    parser.add_argument(
        "--require-watertight",
        action="store_true",
        help="Reject non-watertight meshes (not needed for latent extraction).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Omit failed meshes from the manifests instead of aborting the run.",
    )
    return parser.parse_args()


def resolve_category_dir(source_dir: Path, category_id: str) -> Path:
    """Resolve either a ShapeNetCore root or the category directory itself."""
    candidates = (source_dir, source_dir / category_id)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob(f"*/{MESH_RELATIVE_PATH}")):
            return candidate
    raise FileNotFoundError(
        "could not find raw ShapeNetCore meshes below "
        f"{source_dir}; expected {category_id}/<model-id>/{MESH_RELATIVE_PATH}"
    )


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False, skip_materials=True)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("mesh scene contains no geometry")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"expected a triangle mesh, got {type(loaded).__name__}")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError("mesh has no vertices or faces")
    return loaded


def sample_surface_area_weighted(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample points uniformly over triangle area without requiring a solid."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"expected a non-empty (F, 3) face array, got {faces.shape}")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("mesh contains out-of-range vertex indices")

    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross.astype(np.float64), axis=1) * 0.5
    area_sum = float(areas.sum())
    if not np.isfinite(areas).all() or area_sum <= 0:
        raise ValueError("mesh has no finite, positive-area triangles")

    chosen = rng.choice(len(faces), size=count, p=areas / area_sum)
    selected = triangles[chosen]
    barycentric = rng.random((count, 2)).astype(np.float32)
    reflected = barycentric.sum(axis=1) > 1.0
    barycentric[reflected] = 1.0 - barycentric[reflected]
    points = (
        selected[:, 0]
        + barycentric[:, :1] * (selected[:, 1] - selected[:, 0])
        + barycentric[:, 1:] * (selected[:, 2] - selected[:, 0])
    )
    return points.astype(np.float32)


def object_seed(base_seed: int, model_id: str) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{model_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def convert_one(
    source_path: Path,
    destination: Path,
    *,
    model_id: str,
    surface_point_count: int,
    sampling_seed: int,
    skip_existing: bool,
    require_watertight: bool,
    normalization_extent: float = DEFAULT_NORMALIZATION_EXTENT,
) -> str:
    if skip_existing and destination.is_file():
        return "skipped"

    mesh = load_mesh(source_path)
    is_watertight = bool(mesh.is_watertight)
    if require_watertight and not is_watertight:
        raise ValueError("mesh is not watertight")

    normalized_vertices, center, scale = normalize_points_abo(
        mesh.vertices, normalization_extent
    )
    rng = np.random.default_rng(object_seed(sampling_seed, model_id))
    surface_points = sample_surface_area_weighted(
        normalized_vertices, mesh.faces, surface_point_count, rng
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                surface_points=surface_points,
                normalization_center=center,
                normalization_scale=scale,
                canonical_extent=np.asarray(normalization_extent, dtype=np.float32),
                source_is_watertight=np.asarray(is_watertight),
                source_format=np.asarray("shapenetcore_v2_model_normalized_obj"),
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def prepare_dataset(args: argparse.Namespace) -> dict[str, object]:
    category_dir = resolve_category_dir(args.source_dir, args.category_id)
    sources = sorted(category_dir.glob(f"*/{MESH_RELATIVE_PATH}"))
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        raise RuntimeError(f"no model_normalized.obj meshes found in {category_dir}")

    results = {"written": 0, "skipped": 0, "failed": 0}
    successful_ids: set[str] = set()
    failures: list[dict[str, str]] = []

    def process(source_path: Path) -> tuple[str, str]:
        model_id = source_path.parent.parent.name
        destination = output_path(
            args.datasets_root, args.dataset_key, args.class_name, model_id
        )
        status = convert_one(
            source_path,
            destination,
            model_id=model_id,
            surface_point_count=args.surface_point_count,
            sampling_seed=args.sampling_seed,
            skip_existing=args.skip_existing,
            require_watertight=args.require_watertight,
            normalization_extent=getattr(
                args, "normalization_extent", DEFAULT_NORMALIZATION_EXTENT
            ),
        )
        return model_id, status

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, path): path for path in sources}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            source_path = futures[future]
            try:
                model_id, status = future.result()
            except Exception as error:
                if not args.continue_on_error:
                    raise RuntimeError(f"failed to convert {source_path}: {error}") from error
                results["failed"] += 1
                failures.append({"path": str(source_path), "error": str(error)})
            else:
                results[status] += 1
                successful_ids.add(model_id)
            if index % 250 == 0 or index == len(futures):
                print(f"converted {index}/{len(futures)}", flush=True)

    splits = partition_model_ids(
        successful_ids, ratio=args.train_ratio, seed=args.split_seed
    )
    manifest_paths: dict[str, str] = {}
    for split_name, ids in splits.items():
        path = args.datasets_root / "splits" / f"{args.split_prefix}_{split_name}.json"
        write_manifest(path, args.dataset_key, args.class_name, ids)
        manifest_paths[split_name] = str(path)

    metadata_path = args.datasets_root / args.dataset_key / "preprocessing_metadata.json"
    metadata = {
        "dataset_key": args.dataset_key,
        "class_name": args.class_name,
        "category_id": args.category_id,
        "source_category_dir": str(category_dir),
        "source_format": "ShapeNetCore-v2 model_normalized.obj",
        "usage": "stage_two_latent_extraction_and_diffusion_only",
        "normalization": (
            "x_prime = scale * (x - exact_mesh_bbox_center), isotropic max_abs_"
            f"{getattr(args, 'normalization_extent', DEFAULT_NORMALIZATION_EXTENT):g}"
        ),
        "canonical_extent": getattr(
            args, "normalization_extent", DEFAULT_NORMALIZATION_EXTENT
        ),
        "surface_sampling": {
            "method": "triangle_area_weighted",
            "points_per_mesh": args.surface_point_count,
            "seed": args.sampling_seed,
        },
        "watertightness": {
            "required": args.require_watertight,
            "note": "not required for surface-only latent extraction",
        },
        "partition": {
            "method": "deterministic_random_train_val_over_successful_chairs",
            "train_ratio": args.train_ratio,
            "seed": args.split_seed,
        },
        "counts": {name: len(ids) for name, ids in splits.items()},
        "records_written": results["written"],
        "records_skipped": results["skipped"],
        "records_failed": results["failed"],
        "failures": failures,
        "manifests": manifest_paths,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_name(f"{metadata_path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(metadata_path)
    finally:
        temporary.unlink(missing_ok=True)

    return {**results, "splits": splits, "metadata_path": str(metadata_path)}


def main() -> None:
    # ShapeNet OBJs reference materials that are irrelevant for geometry-only
    # sampling; suppress one warning per mesh about their textures being skipped.
    logging.getLogger("trimesh").setLevel(logging.ERROR)
    args = parse_args()
    summary = prepare_dataset(args)
    print(
        "summary: "
        f"written={summary['written']} skipped={summary['skipped']} "
        f"failed={summary['failed']} train={len(summary['splits']['train'])} "
        f"val={len(summary['splits']['val'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
