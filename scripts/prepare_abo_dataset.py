#!/usr/bin/env python3
from __future__ import annotations

"""Prepare ABO meshes for COD Diffusion-SDF and emit split-aware metadata.

This generalizes the chair-specific ABO preparation flow:
- consumes downloaded `.glb` files
- stores separate COD surface points and SDF supervision in NPZ records
- writes split manifests for all products and per product type
- writes a metadata JSON keyed directly by `3dmodel_id`
"""

import argparse
import concurrent.futures
import gc
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    import open3d as o3d
except (ImportError, OSError):
    o3d = None
import trimesh


DEFAULT_SOURCE_DIR = Path("datasets/ABO/models_filtered")
DEFAULT_DATASETS_ROOT = Path("datasets")
DEFAULT_DATASET_KEY = "abo"
DEFAULT_CLASS_NAME = "ABO"
DEFAULT_SPLIT_PREFIX = "abo"
DEFAULT_METADATA_IN = Path("datasets/ABO/abo_selected_subset.json")
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_REPAIRED_MESH_DIRNAME = "repaired_meshes_cod_0999"
REPAIR_NONE = "none"
REPAIR_MANIFOLDPLUS = "manifoldplus"
SIGN_RAY_COUNT = 21


def fibonacci_sphere_directions(count: int) -> np.ndarray:
    """Return deterministic, well-separated ray directions on the unit sphere."""
    if count <= 0 or count % 2 == 0:
        raise ValueError("sign ray count must be a positive odd integer")
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    directions = np.empty((count, 3), dtype=np.float32)
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        angle = (index + 0.5) * golden_angle
        directions[index] = (
            radius * np.cos(angle),
            radius * np.sin(angle),
            z,
        )
    return directions


SIGN_RAY_DIRECTIONS = fibonacci_sphere_directions(SIGN_RAY_COUNT)


@dataclass(frozen=True)
class RepairConfig:
    method: str = REPAIR_NONE
    manifoldplus_bin: Path | None = None
    manifoldplus_depth: int = 8
    repaired_mesh_dir: Path | None = None
    force_repair: bool = False


@dataclass(frozen=True)
class RepairResult:
    mesh: trimesh.Trimesh
    used_cache: bool


@dataclass(frozen=True)
class RepairFidelityConfig:
    sample_count: int = 20000
    distance_threshold: float = 0.02
    max_p95_distance: float = 0.02
    max_outlier_fraction: float = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--dataset-key", default=DEFAULT_DATASET_KEY)
    parser.add_argument(
        "--repair-cache-dataset-key",
        default=None,
        help=(
            "Dataset-key subdirectory used only for repaired-mesh cache lookup. "
            "Defaults to --dataset-key; set this when writing records under a new "
            "dataset key while reusing an existing repair cache."
        ),
    )
    parser.add_argument("--class-name", default=DEFAULT_CLASS_NAME)
    parser.add_argument("--split-prefix", default=DEFAULT_SPLIT_PREFIX)
    parser.add_argument("--metadata-in", type=Path, default=DEFAULT_METADATA_IN)
    parser.add_argument("--metadata-out", type=Path, default=None)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Fraction of each split assigned to training; remainder is assigned to validation.",
    )
    parser.add_argument(
        "--surface-point-count",
        type=int,
        default=235000,
        help="Number of area-weighted surface samples stored for COD encoding.",
    )
    parser.add_argument(
        "--near-surface-stds",
        type=float,
        nargs=2,
        default=(0.005, 0.0005),
        metavar=("COARSE_STD", "FINE_STD"),
        help="Standard deviations of the two isotropic Gaussian surface perturbations.",
    )
    parser.add_argument(
        "--uniform-point-count",
        type=int,
        default=262144,
        help="Number of uniformly sampled SDF supervision points in [-1, 1]^3.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Maximum raycast query batch; lower values reduce peak native memory.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--repair-method",
        choices=(REPAIR_NONE, REPAIR_MANIFOLDPLUS),
        default=REPAIR_NONE,
        help="Optional watertight repair method used for SDF signing.",
    )
    parser.add_argument(
        "--manifoldplus-bin",
        type=Path,
        default=None,
        help=(
            "Path to the ManifoldPlus executable. If omitted, MANIFOLDPLUS_BIN "
            "and then PATH are checked when --repair-method manifoldplus is used."
        ),
    )
    parser.add_argument(
        "--manifoldplus-depth",
        type=int,
        default=8,
        help="ManifoldPlus octree depth; higher preserves more detail but costs more time/memory.",
    )
    parser.add_argument(
        "--repaired-mesh-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached watertight proxy meshes. Defaults to "
            "<datasets-root>/repaired_meshes_cod_0999; the COD-normalized cache "
            "must not share proxies with the legacy diagonal-normalized pipeline."
        ),
    )
    parser.add_argument(
        "--force-repair",
        action="store_true",
        help="Regenerate repaired proxy meshes even when cached outputs exist.",
    )
    parser.add_argument(
        "--repair-fidelity-samples",
        type=int,
        default=20000,
        help="Surface samples in each direction used to compare repaired and original meshes.",
    )
    parser.add_argument(
        "--repair-fidelity-distance-threshold",
        type=float,
        default=0.02,
        help="Normalized surface distance counted as a repair-fidelity outlier.",
    )
    parser.add_argument(
        "--repair-fidelity-max-p95",
        type=float,
        default=0.02,
        help="Maximum allowed p95 surface distance in either comparison direction.",
    )
    parser.add_argument(
        "--repair-fidelity-max-outlier-fraction",
        type=float,
        default=0.05,
        help="Maximum fraction above the fidelity distance threshold in either direction.",
    )
    parser.add_argument(
        "--reuse-repair-fidelity",
        action="store_true",
        help=(
            "When regenerating COD/SDF records from cached repaired meshes, reuse a "
            "compatible .obj.fidelity.json sidecar instead of repeating the "
            "bidirectional original/repaired surface comparison."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N models after sorting.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip objects whose cod_sdf.npz record already exists.",
    )
    parser.add_argument(
        "--only-models-in",
        type=Path,
        default=None,
        help="Only regenerate model IDs contained in this split manifest.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only write splits and metadata; do not generate COD/SDF records.",
    )
    parser.add_argument(
        "--per-type-splits-only",
        action="store_true",
        help="Write per-product-type manifests without duplicate aggregate manifests.",
    )
    parser.add_argument(
        "--no-model-isolation",
        action="store_true",
        help=(
            "Process all models in the parent process. By default every model "
            "uses a fresh child process so native memory is returned to the OS."
        ),
    )
    parser.add_argument(
        "--model-workers",
        type=int,
        default=1,
        help=(
            "Number of chair models to preprocess concurrently. Each model still "
            "runs in a fresh isolated process; start with 2-4 because Open3D "
            "raycasting can use substantial memory."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Finish other models, record failures in metadata, then exit nonzero.",
    )
    return parser.parse_args()


def build_repair_config(args: argparse.Namespace) -> RepairConfig:
    repaired_mesh_dir = (
        args.repaired_mesh_dir
        or args.datasets_root / DEFAULT_REPAIRED_MESH_DIRNAME
    )
    manifoldplus_bin = args.manifoldplus_bin
    if manifoldplus_bin is None and args.repair_method == REPAIR_MANIFOLDPLUS:
        env_bin = os.environ.get("MANIFOLDPLUS_BIN")
        if env_bin:
            manifoldplus_bin = Path(env_bin)
        else:
            path_bin = shutil.which("ManifoldPlus")
            if path_bin:
                manifoldplus_bin = Path(path_bin)
    config = RepairConfig(
        method=args.repair_method,
        manifoldplus_bin=manifoldplus_bin,
        manifoldplus_depth=args.manifoldplus_depth,
        repaired_mesh_dir=repaired_mesh_dir,
        force_repair=args.force_repair,
    )
    validate_repair_config(config)
    return config


def validate_repair_config(config: RepairConfig) -> None:
    if config.method == REPAIR_NONE:
        return
    if config.method != REPAIR_MANIFOLDPLUS:
        raise ValueError(f"unsupported repair method: {config.method}")
    if config.manifoldplus_bin is None:
        raise ValueError(
            "ManifoldPlus executable is required when --repair-method manifoldplus is used; "
            "pass --manifoldplus-bin, set MANIFOLDPLUS_BIN, or put ManifoldPlus on PATH"
        )
    if not config.manifoldplus_bin.is_file():
        raise FileNotFoundError(f"ManifoldPlus executable does not exist: {config.manifoldplus_bin}")
    if config.manifoldplus_depth <= 0:
        raise ValueError("--manifoldplus-depth must be positive")
    if config.repaired_mesh_dir is None:
        raise ValueError("repaired mesh directory is required for ManifoldPlus repair")


def build_repair_fidelity_config(args: argparse.Namespace) -> RepairFidelityConfig:
    config = RepairFidelityConfig(
        sample_count=args.repair_fidelity_samples,
        distance_threshold=args.repair_fidelity_distance_threshold,
        max_p95_distance=args.repair_fidelity_max_p95,
        max_outlier_fraction=args.repair_fidelity_max_outlier_fraction,
    )
    if config.sample_count <= 0:
        raise ValueError("--repair-fidelity-samples must be positive")
    if config.distance_threshold <= 0 or config.max_p95_distance <= 0:
        raise ValueError("repair fidelity distance thresholds must be positive")
    if not 0.0 <= config.max_outlier_fraction <= 1.0:
        raise ValueError("--repair-fidelity-max-outlier-fraction must be in [0, 1]")
    return config


def list_model_ids(source_dir: Path) -> list[str]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    return sorted(path.stem for path in source_dir.glob("*.glb"))


def model_ids_from_manifest(manifest_path: Path) -> set[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_ids: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, str):
            model_ids.add(value)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(payload)
    if not model_ids:
        raise ValueError(f"split manifest contains no model IDs: {manifest_path}")
    return model_ids


def load_mesh_asset(mesh_path: Path, process: bool = True) -> trimesh.Trimesh:
    asset = trimesh.load(mesh_path, force="scene", process=process)
    if isinstance(asset, trimesh.Scene):
        if not asset.geometry:
            raise ValueError(f"scene has no geometry: {mesh_path}")
        mesh = asset.dump(concatenate=True)
    elif isinstance(asset, trimesh.Trimesh):
        mesh = asset
    else:
        raise TypeError(f"unsupported mesh type {type(asset)!r} for {mesh_path}")

    return mesh.copy()


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = load_mesh_asset(mesh_path)
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


def normalize_mesh_with_transform(
    mesh: trimesh.Trimesh,
    *,
    copy: bool = True,
) -> tuple[trimesh.Trimesh, np.ndarray, float]:
    if copy:
        mesh = mesh.copy()
    bounds = mesh.bounds.astype(np.float32)
    center = bounds.mean(axis=0)
    radius = float(np.abs(np.asarray(mesh.vertices) - center).max())
    if radius <= 0:
        raise ValueError("mesh has zero extent")
    scale = 0.999 / radius
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    return mesh, center.astype(np.float32), scale


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    return normalize_mesh_with_transform(mesh)[0]


def make_raycast_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    if o3d is None:
        raise RuntimeError(
            "Open3D raycasting is unavailable; install Open3D and its libGL runtime"
        )
    vertices = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
    faces = o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32))
    tmesh = o3d.t.geometry.TriangleMesh(vertices, faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return scene


def load_repaired_mesh(mesh_path: Path) -> trimesh.Trimesh:
    # ManifoldPlus can emit coincident vertices and zero-area faces which are
    # topologically significant. Trimesh's processing merges/removes them and
    # can turn the closed proxy into a non-manifold mesh.
    mesh = load_mesh_asset(mesh_path, process=False)
    if not mesh.is_watertight:
        raise ValueError(f"repaired mesh is not watertight: {mesh_path}")
    return mesh


def repair_mesh_with_manifoldplus(
    mesh: trimesh.Trimesh,
    output_path: Path,
    config: RepairConfig,
) -> RepairResult:
    if config.method != REPAIR_MANIFOLDPLUS:
        raise ValueError(f"cannot repair with method: {config.method}")
    if config.manifoldplus_bin is None:
        raise ValueError("ManifoldPlus executable is required")

    if output_path.is_file() and not config.force_repair:
        try:
            return RepairResult(
                mesh=load_repaired_mesh(output_path), used_cache=True
            )
        except Exception as error:
            warnings.warn(
                f"discarding invalid repaired-mesh cache {output_path}: {error}"
            )
            output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(
        f".{output_path.stem}.tmp.{os.getpid()}{output_path.suffix}"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.obj"
        mesh.export(input_path)
        command = [
            str(config.manifoldplus_bin),
            "--input",
            str(input_path),
            "--output",
            str(temporary_output),
            "--depth",
            str(config.manifoldplus_depth),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            temporary_output.unlink(missing_ok=True)
            raise RuntimeError(
                "ManifoldPlus repair failed for "
                f"{output_path}: {result.stderr.strip() or result.stdout.strip()}"
            )

    try:
        repaired_mesh = load_repaired_mesh(temporary_output)
        temporary_output.replace(output_path)
    finally:
        temporary_output.unlink(missing_ok=True)
    return RepairResult(mesh=repaired_mesh, used_cache=False)


def sample_surface(
    mesh: trimesh.Trimesh,
    count: int,
    rng: np.random.Generator,
    return_normals: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        raise ValueError("surface point count must be positive")

    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("mesh has no sampleable surface area")
    face_indices = rng.choice(len(mesh.faces), size=count, p=areas / areas.sum())
    triangles = np.asarray(
        mesh.vertices[mesh.faces[face_indices]], dtype=np.float32
    )
    barycentric = rng.random((count, 2), dtype=np.float32)
    reflected = barycentric.sum(axis=1) > 1.0
    barycentric[reflected] = 1.0 - barycentric[reflected]
    points = (
        triangles[:, 0]
        + barycentric[:, :1] * (triangles[:, 1] - triangles[:, 0])
        + barycentric[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    )
    if return_normals:
        return points, np.asarray(mesh.face_normals[face_indices], dtype=np.float32)
    return points


def compute_signed_distances(
    scene: o3d.t.geometry.RaycastingScene,
    query_points: np.ndarray,
    batch_size: int,
    sign_method: str = "normal",
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if sign_method not in {"normal", "occupancy"}:
        raise ValueError(f"unsupported sign method: {sign_method}")
    sdf = np.empty((len(query_points), 1), dtype=np.float32)
    for start in range(0, len(query_points), batch_size):
        stop = min(start + batch_size, len(query_points))
        batch = query_points[start:stop]
        tensor = o3d.core.Tensor(batch)
        if sign_method == "occupancy":
            unsigned = scene.compute_distance(tensor).numpy().reshape(-1, 1)
            inside_votes = np.zeros(len(batch), dtype=np.uint8)
            rays = np.empty((len(batch), 6), dtype=np.float32)
            rays[:, :3] = batch
            # Count one direction at a time. This keeps peak ray-buffer memory
            # bounded by batch_size rather than batch_size * SIGN_RAY_COUNT.
            for direction in SIGN_RAY_DIRECTIONS:
                rays[:, 3:] = direction
                intersections = scene.count_intersections(
                    o3d.core.Tensor(rays)
                ).numpy()
                inside_votes += (intersections & 1).astype(np.uint8)
            inside = inside_votes > (SIGN_RAY_COUNT // 2)
            unsigned[inside] *= -1.0
            sdf[start:stop] = unsigned
            del unsigned, inside_votes, rays, intersections, inside
        else:
            # Non-watertight fallback: closest-normal signs are only a pseudo-SDF.
            out = scene.compute_closest_points(tensor)
            closest_points = out["points"].numpy()
            normals = out["primitive_normals"].numpy()
            delta = batch - closest_points
            unsigned = np.linalg.norm(delta, axis=1, keepdims=True)
            sign = np.sign(np.sum(delta * normals, axis=1, keepdims=True)).astype(np.float32)
            sign[sign == 0] = 1.0
            sdf[start:stop] = unsigned * sign
            del out, closest_points, normals, delta, unsigned, sign
        del batch, tensor
    return sdf


def compute_unsigned_distances(
    scene: o3d.t.geometry.RaycastingScene,
    query_points: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    distances = np.empty(len(query_points), dtype=np.float32)
    for start in range(0, len(query_points), batch_size):
        stop = min(start + batch_size, len(query_points))
        tensor = o3d.core.Tensor(query_points[start:stop])
        distances[start:stop] = scene.compute_distance(tensor).numpy()
        del tensor
    return distances


def repaired_mesh_fidelity(
    original_mesh: trimesh.Trimesh,
    repaired_mesh: trimesh.Trimesh,
    original_scene: o3d.t.geometry.RaycastingScene,
    repaired_scene: o3d.t.geometry.RaycastingScene,
    config: RepairFidelityConfig,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    original_points = sample_surface(original_mesh, config.sample_count, rng)
    repaired_points = sample_surface(repaired_mesh, config.sample_count, rng)
    original_to_repaired = compute_unsigned_distances(
        repaired_scene, original_points, batch_size
    )
    repaired_to_original = compute_unsigned_distances(
        original_scene, repaired_points, batch_size
    )

    def summarize(distances: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(distances.mean()),
            "p95": float(np.quantile(distances, 0.95)),
            "p99": float(np.quantile(distances, 0.99)),
            "max": float(distances.max()),
            "outlier_fraction": float(
                np.mean(distances > config.distance_threshold)
            ),
        }

    original_summary = summarize(original_to_repaired)
    repaired_summary = summarize(repaired_to_original)
    accepted = all(
        summary["p95"] <= config.max_p95_distance
        and summary["outlier_fraction"] <= config.max_outlier_fraction
        for summary in (original_summary, repaired_summary)
    )
    return {
        "accepted": accepted,
        "sample_count": config.sample_count,
        "distance_threshold": config.distance_threshold,
        "max_p95_distance": config.max_p95_distance,
        "max_outlier_fraction": config.max_outlier_fraction,
        "original_to_repaired": original_summary,
        "repaired_to_original": repaired_summary,
    }


def repair_fidelity_cache_compatible(
    fidelity: dict[str, Any], config: RepairFidelityConfig
) -> bool:
    try:
        if int(fidelity.get("sample_count", -1)) != config.sample_count:
            return False
        if not np.isclose(
            float(fidelity.get("distance_threshold", np.nan)),
            config.distance_threshold,
        ):
            return False
        for direction in ("original_to_repaired", "repaired_to_original"):
            for statistic in ("mean", "p95", "p99", "max", "outlier_fraction"):
                float(fidelity[direction][statistic])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def apply_repair_fidelity_config(
    fidelity: dict[str, Any], config: RepairFidelityConfig
) -> dict[str, Any]:
    evaluated = dict(fidelity)
    evaluated["sample_count"] = config.sample_count
    evaluated["distance_threshold"] = config.distance_threshold
    evaluated["max_p95_distance"] = config.max_p95_distance
    evaluated["max_outlier_fraction"] = config.max_outlier_fraction
    evaluated["accepted"] = all(
        float(fidelity[direction]["p95"]) <= config.max_p95_distance
        and float(fidelity[direction]["outlier_fraction"])
        <= config.max_outlier_fraction
        for direction in ("original_to_repaired", "repaired_to_original")
    )
    return evaluated


def repair_fidelity_passes(
    fidelity: dict[str, Any], config: RepairFidelityConfig
) -> bool:
    if not repair_fidelity_cache_compatible(fidelity, config):
        return False
    return all(
        float(fidelity[direction]["p95"]) <= config.max_p95_distance
        and float(fidelity[direction]["outlier_fraction"])
        <= config.max_outlier_fraction
        for direction in ("original_to_repaired", "repaired_to_original")
    )


def sample_cod_supervision(
    mesh: trimesh.Trimesh,
    scene: o3d.t.geometry.RaycastingScene,
    surface_point_count: int,
    near_surface_stds: tuple[float, float],
    uniform_point_count: int,
    batch_size: int,
    rng: np.random.Generator,
    sign_method: str = "normal",
) -> dict[str, np.ndarray]:
    if any(std <= 0 for std in near_surface_stds):
        raise ValueError("near-surface standard deviations must be positive")
    if uniform_point_count <= 0:
        raise ValueError("uniform point count must be positive")

    surface_points, surface_normals = sample_surface(
        mesh, surface_point_count, rng, return_normals=True
    )
    near_points = []
    near_sdf = []
    for std in near_surface_stds:
        query_points = surface_points + rng.normal(0.0, std, surface_points.shape).astype(np.float32)
        # COD's tri-plane sampler clamps to this interval. Clamp before
        # computing distances so every stored target corresponds to the exact
        # coordinate later used for feature lookup.
        query_points = np.clip(query_points, -1.0, 0.999).astype(np.float32)
        query_sdf = compute_signed_distances(scene, query_points, batch_size, sign_method=sign_method)
        near_points.append(query_points)
        near_sdf.append(query_sdf)

    uniform_points = np.clip(
        rng.uniform(-1.0, 1.0, size=(uniform_point_count, 3)),
        -1.0,
        0.999,
    ).astype(np.float32)
    uniform_sdf = compute_signed_distances(
        scene, uniform_points, batch_size, sign_method=sign_method
    )
    return {
        "surface_points": surface_points.astype(np.float32),
        "surface_normals": surface_normals.astype(np.float32),
        "near_surface_query_points": np.concatenate(near_points, axis=0),
        "near_surface_sdf": np.concatenate(near_sdf, axis=0).reshape(-1),
        "uniform_query_points": uniform_points,
        "uniform_sdf": uniform_sdf.reshape(-1),
    }


def save_cod_sdf(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary_path.replace(path)


def fidelity_arrays(fidelity: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "repair_fidelity_accepted": np.asarray(
            fidelity["accepted"], dtype=np.bool_
        ),
        "repair_fidelity_sample_count": np.asarray(
            fidelity["sample_count"], dtype=np.int32
        ),
        "repair_fidelity_distance_threshold": np.asarray(
            fidelity["distance_threshold"], dtype=np.float32
        ),
        "repair_fidelity_max_p95_distance": np.asarray(
            fidelity["max_p95_distance"], dtype=np.float32
        ),
        "repair_fidelity_max_outlier_fraction": np.asarray(
            fidelity["max_outlier_fraction"], dtype=np.float32
        ),
    }
    for direction in ("original_to_repaired", "repaired_to_original"):
        for statistic in ("mean", "p95", "p99", "max", "outlier_fraction"):
            arrays[f"repair_{direction}_{statistic}"] = np.asarray(
                fidelity[direction][statistic], dtype=np.float32
            )
    return arrays


def read_repaired_record_fidelity(
    path: Path,
) -> tuple[str | None, dict[str, Any] | None]:
    try:
        with np.load(path) as data:
            if "surface_source" not in data:
                return None, None
            source = str(np.asarray(data["surface_source"]).item())
            fidelity: dict[str, Any] = {}
            scalar_fields = {
                "sample_count": "repair_fidelity_sample_count",
                "distance_threshold": "repair_fidelity_distance_threshold",
                "max_p95_distance": "repair_fidelity_max_p95_distance",
                "max_outlier_fraction": "repair_fidelity_max_outlier_fraction",
            }
            for field, key in scalar_fields.items():
                if key not in data:
                    return source, None
                value = np.asarray(data[key]).item()
                fidelity[field] = int(value) if field == "sample_count" else float(value)
            for direction in ("original_to_repaired", "repaired_to_original"):
                values = {}
                for statistic in (
                    "mean", "p95", "p99", "max", "outlier_fraction"
                ):
                    key = f"repair_{direction}_{statistic}"
                    if key not in data:
                        return source, None
                    values[statistic] = float(np.asarray(data[key]).item())
                fidelity[direction] = values
            fidelity["accepted"] = bool(
                np.asarray(data["repair_fidelity_accepted"]).item()
            )
            return source, fidelity
    except (OSError, ValueError, KeyError):
        return None, None


def object_output_paths(
    datasets_root: Path,
    dataset_key: str,
    class_name: str,
    model_id: str,
) -> Path:
    return datasets_root / dataset_key / class_name / model_id / "cod_sdf.npz"


def repaired_mesh_output_path(
    repaired_mesh_dir: Path,
    dataset_key: str,
    class_name: str,
    model_id: str,
) -> Path:
    return repaired_mesh_dir / dataset_key / class_name / f"{model_id}.obj"


def repair_fidelity_output_path(repaired_mesh_path: Path) -> Path:
    return repaired_mesh_path.with_suffix(repaired_mesh_path.suffix + ".fidelity.json")


def save_repair_fidelity(path: Path, fidelity: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(fidelity, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def load_repair_fidelity(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def mesh_summary(mesh: trimesh.Trimesh) -> dict[str, Any]:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "components": int(mesh.body_count),
    }


def process_model(
    mesh_path: Path,
    datasets_root: Path,
    dataset_key: str,
    class_name: str,
    surface_point_count: int,
    near_surface_stds: tuple[float, float],
    uniform_point_count: int,
    batch_size: int,
    rng: np.random.Generator,
    skip_existing: bool,
    repair_config: RepairConfig | None = None,
    use_repaired_surface: bool = False,
    fidelity_config: RepairFidelityConfig | None = None,
    reuse_repair_fidelity: bool = False,
    repair_cache_dataset_key: str | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    def log_phase(message: str) -> None:
        print(f"  phase: {message}")

    model_id = mesh_path.stem
    output_path = object_output_paths(datasets_root, dataset_key, class_name, model_id)
    proxy_dataset_key = repair_cache_dataset_key or dataset_key

    if skip_existing and output_path.is_file():
        repair_info: dict[str, Any] = {"skipped_existing": True}
        if repair_config is not None and repair_config.method == REPAIR_MANIFOLDPLUS:
            if repair_config.repaired_mesh_dir is None:
                raise ValueError("repaired mesh directory is required")
            proxy_path = repaired_mesh_output_path(
                repair_config.repaired_mesh_dir,
                proxy_dataset_key,
                class_name,
                model_id,
            )
            if not proxy_path.is_file():
                raise FileNotFoundError(
                    "existing COD/SDF record has no matching repaired proxy: "
                    f"{proxy_path}"
                )
            if use_repaired_surface:
                source, fidelity = read_repaired_record_fidelity(output_path)
                cached_fidelity = load_repair_fidelity(
                    repair_fidelity_output_path(proxy_path)
                )
                if fidelity is None:
                    fidelity = cached_fidelity
                if (
                    source != "repaired_mesh"
                    and fidelity is not None
                    and fidelity_config is not None
                    and not repair_fidelity_passes(fidelity, fidelity_config)
                ):
                    repair_info.update(
                        {
                            "method": REPAIR_MANIFOLDPLUS,
                            "surface_source": "repaired_mesh",
                            "sdf_sign_method": "occupancy",
                            "manifoldplus_depth": repair_config.manifoldplus_depth,
                            "repaired_mesh_path": str(proxy_path),
                            "cache_hit": True,
                            "fidelity": fidelity,
                        }
                    )
                    return None, repair_info
                if (
                    source != "repaired_mesh"
                    or fidelity is None
                    or fidelity_config is None
                    or not repair_fidelity_passes(fidelity, fidelity_config)
                ):
                    print(
                        "  existing record uses legacy or rejected surface "
                        "supervision; regenerating"
                    )
                else:
                    load_repaired_mesh(proxy_path)
                    repair_info.update(
                        {
                            "method": REPAIR_MANIFOLDPLUS,
                            "surface_source": source,
                            "sdf_sign_method": "occupancy",
                            "manifoldplus_depth": repair_config.manifoldplus_depth,
                            "repaired_mesh_path": str(proxy_path),
                            "cache_hit": True,
                            "fidelity": fidelity,
                        }
                    )
                    return output_path, repair_info
            else:
                repair_info.update(
                    {
                        "method": REPAIR_MANIFOLDPLUS,
                        "surface_source": "original_mesh",
                        "sdf_sign_method": "occupancy",
                        "manifoldplus_depth": repair_config.manifoldplus_depth,
                        "repaired_mesh_path": str(proxy_path),
                        "cache_hit": True,
                    }
                )
                return output_path, repair_info
        else:
            return output_path, repair_info

    log_phase("mesh load/normalize")
    mesh, normalization_center, normalization_scale = normalize_mesh_with_transform(
        load_mesh(mesh_path), copy=False
    )
    sdf_mesh = mesh
    sign_method = "normal"
    repair_info: dict[str, Any] = {
        "method": REPAIR_NONE,
        "surface_source": "original_mesh",
        "sdf_sign_method": sign_method,
        "original_mesh": mesh_summary(mesh),
    }
    scene = None

    if repair_config is not None and repair_config.method == REPAIR_MANIFOLDPLUS:
        if repair_config.repaired_mesh_dir is None:
            raise ValueError("repaired mesh directory is required")
        proxy_path = repaired_mesh_output_path(
            repair_config.repaired_mesh_dir,
            proxy_dataset_key,
            class_name,
            model_id,
        )
        log_phase("manifold repair")
        repair_result = repair_mesh_with_manifoldplus(mesh, proxy_path, repair_config)
        sdf_mesh = repair_result.mesh
        sign_method = "occupancy"
        repair_status = "reused cached proxy" if repair_result.used_cache else "generated repaired proxy"
        print(f"  repair: {repair_status}")
        repair_info = {
            "method": REPAIR_MANIFOLDPLUS,
            "sdf_sign_method": sign_method,
            "manifoldplus_depth": repair_config.manifoldplus_depth,
            "repaired_mesh_path": str(proxy_path),
            "cache_hit": repair_result.used_cache,
            "surface_source": (
                "repaired_mesh" if use_repaired_surface else "original_mesh"
            ),
            "original_mesh": mesh_summary(mesh),
            "repaired_mesh": mesh_summary(sdf_mesh),
        }
        if fidelity_config is not None:
            log_phase("repaired-mesh fidelity validation")
            fidelity_sidecar_path = repair_fidelity_output_path(proxy_path)
            cached_fidelity = (
                load_repair_fidelity(fidelity_sidecar_path)
                if reuse_repair_fidelity and repair_result.used_cache
                else None
            )
            if (
                cached_fidelity is not None
                and repair_fidelity_cache_compatible(
                    cached_fidelity, fidelity_config
                )
            ):
                fidelity = apply_repair_fidelity_config(
                    cached_fidelity, fidelity_config
                )
                print("  fidelity: reused cached validation")
            else:
                if reuse_repair_fidelity and repair_result.used_cache:
                    print("  fidelity: cached validation missing or incompatible; recomputing")
                original_scene = make_raycast_scene(mesh)
                scene = make_raycast_scene(sdf_mesh)
                fidelity = repaired_mesh_fidelity(
                    original_mesh=mesh,
                    repaired_mesh=sdf_mesh,
                    original_scene=original_scene,
                    repaired_scene=scene,
                    config=fidelity_config,
                    batch_size=batch_size,
                    rng=rng,
                )
                del original_scene
                save_repair_fidelity(fidelity_sidecar_path, fidelity)
            repair_info["fidelity"] = fidelity
            if not fidelity["accepted"]:
                print(
                    "  filter: rejected repaired mesh "
                    f"(original->repaired p95={fidelity['original_to_repaired']['p95']:.6f}, "
                    f"repaired->original p95={fidelity['repaired_to_original']['p95']:.6f})"
                )
                return None, repair_info
    else:
        log_phase("manifold repair skipped")

    log_phase("raycast scene creation")
    if scene is None:
        scene = make_raycast_scene(sdf_mesh)
    sampling_mesh = sdf_mesh if use_repaired_surface else mesh

    log_phase("COD surface and SDF supervision sampling")
    arrays = sample_cod_supervision(
        mesh=sampling_mesh,
        scene=scene,
        surface_point_count=surface_point_count,
        near_surface_stds=near_surface_stds,
        uniform_point_count=uniform_point_count,
        batch_size=batch_size,
        rng=rng,
        sign_method=sign_method,
    )
    arrays["normalization_center"] = normalization_center
    arrays["normalization_scale"] = np.asarray(normalization_scale, dtype=np.float32)
    arrays["surface_source"] = np.asarray(
        "repaired_mesh" if use_repaired_surface else "original_mesh"
    )
    if "fidelity" in repair_info:
        arrays.update(fidelity_arrays(repair_info["fidelity"]))
    if sdf_mesh is not mesh:
        del sdf_mesh
    del mesh
    del scene

    log_phase("COD/SDF NPZ write")
    save_cod_sdf(output_path, arrays)
    return output_path, repair_info


def model_seed(seed: int, model_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{model_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _isolated_model_worker(connection, kwargs: dict[str, Any]) -> None:
    try:
        seed = kwargs.pop("seed")
        data_path, repair_info = process_model(
            **kwargs, rng=np.random.default_rng(seed)
        )
        connection.send({
            "ok": True,
            "data_path": str(data_path) if data_path is not None else None,
            "repair_info": repair_info,
        })
    except BaseException:
        connection.send({"ok": False, "traceback": traceback.format_exc()})
    finally:
        connection.close()


def process_model_isolated(**kwargs) -> tuple[Path | None, dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_model_worker,
        args=(sender, kwargs),
    )
    process.start()
    sender.close()
    process.join()

    result = receiver.recv() if receiver.poll() else None
    receiver.close()
    if process.exitcode != 0:
        model_id = Path(kwargs["mesh_path"]).stem
        raise RuntimeError(
            f"isolated worker for {model_id} exited with code {process.exitcode}; "
            "the OS may have killed it for exceeding memory"
        )
    if result is None:
        raise RuntimeError("isolated preprocessing worker returned no result")
    if not result["ok"]:
        raise RuntimeError(result["traceback"])
    data_path = result["data_path"]
    return (Path(data_path) if data_path is not None else None), result["repair_info"]


def iter_model_results(model_ids, model_workers: int, process_model_fn):
    def capture(model_id):
        try:
            return model_id, process_model_fn(model_id), None
        except Exception as error:
            return model_id, None, error

    if model_workers <= 0:
        raise ValueError("--model-workers must be positive")
    if model_workers == 1:
        for model_id in model_ids:
            yield capture(model_id)
        return

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=model_workers,
        thread_name_prefix="abo-model",
    ) as executor:
        futures = {
            executor.submit(capture, model_id): model_id for model_id in model_ids
        }
        for future in concurrent.futures.as_completed(futures):
            yield future.result()


def load_input_metadata(metadata_path: Path | None) -> dict[str, Any]:
    if metadata_path is None or not metadata_path.is_file():
        return {"products": {}}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def stable_product_type_key(entry: dict[str, Any], default_value: str) -> str:
    value = entry.get("product_type_key")
    if isinstance(value, str) and value:
        return value
    return default_value


def write_manifest(manifest_path: Path, dataset_key: str, class_name: str, model_ids: list[str]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({dataset_key: {class_name: model_ids}}, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_geometry_files(
    model_ids: set[str],
    geometry_paths: dict[str, Path],
) -> tuple[set[str], dict[str, list[str]]]:
    """Keep one ID per byte-identical geometry and return duplicate groups."""
    by_size: dict[int, list[str]] = {}
    for model_id in sorted(model_ids):
        path = geometry_paths[model_id]
        if not path.is_file():
            raise FileNotFoundError(
                f"training-eligible model has no repaired geometry: {path}"
            )
        by_size.setdefault(path.stat().st_size, []).append(model_id)

    retained = set(model_ids)
    duplicate_groups: dict[str, list[str]] = {}
    for same_size_ids in by_size.values():
        if len(same_size_ids) < 2:
            continue
        by_digest: dict[str, list[str]] = {}
        for model_id in same_size_ids:
            digest = file_sha256(geometry_paths[model_id])
            by_digest.setdefault(digest, []).append(model_id)
        for identical_ids in by_digest.values():
            if len(identical_ids) < 2:
                continue
            members = sorted(identical_ids)
            canonical = members[0]
            duplicate_groups[canonical] = members
            retained.difference_update(members[1:])
    return retained, duplicate_groups


def validate_train_ratio(train_ratio: float) -> None:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train ratio must be between 0 and 1, got {train_ratio}")


def shuffled_model_ids(model_ids: list[str], rng: np.random.Generator, limit: int | None = None) -> list[str]:
    shuffled = list(model_ids)
    rng.shuffle(shuffled)
    if limit is not None:
        shuffled = shuffled[:limit]
    return shuffled


def compute_split_counts(total_count: int, train_ratio: float) -> tuple[int, int]:
    if total_count < 2:
        raise ValueError(f"need at least 2 models to create train/val splits, got {total_count}")

    train_count = int(total_count * train_ratio)
    train_count = min(max(train_count, 1), total_count - 1)
    val_count = total_count - train_count

    if val_count < 1:
        raise ValueError(
            "could not allocate at least one validation sample; "
            f"count={total_count}, train_ratio={train_ratio}"
        )

    return train_count, val_count


def split_model_ids(
    model_ids: list[str],
    train_ratio: float,
    rng: np.random.Generator,
    *,
    limit: int | None = None,
) -> dict[str, list[str]]:
    selected_ids = shuffled_model_ids(model_ids, rng, limit=limit)
    train_count, val_count = compute_split_counts(len(selected_ids), train_ratio)
    train_ids = sorted(selected_ids[:train_count])
    val_ids = sorted(selected_ids[train_count : train_count + val_count])
    return {
        "all": sorted(selected_ids),
        "train": train_ids,
        "val": val_ids,
    }


def build_split_sets(
    type_to_ids: dict[str, list[str]],
    train_ratio: float,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    validate_train_ratio(train_ratio)

    sorted_types = sorted(type_to_ids.items())
    if not sorted_types:
        raise ValueError("no product types available for split generation")

    min_count = min(len(model_ids) for _, model_ids in sorted_types)
    if min_count < 2:
        small_types = [product_type for product_type, model_ids in sorted_types if len(model_ids) < 2]
        raise ValueError(
            "need at least 2 models per product type for balanced aggregate splits; "
            f"too small: {', '.join(small_types)}"
        )

    per_type_rng = np.random.default_rng(seed)
    per_type_splits: dict[str, dict[str, list[str]]] = {}
    balanced_all = {"all": [], "train": [], "val": []}

    for product_type_key, type_model_ids in sorted_types:
        if len(type_model_ids) < 2:
            raise ValueError(
                f"product type '{product_type_key}' needs at least 2 models for train/val splits, "
                f"got {len(type_model_ids)}"
            )

        type_splits = split_model_ids(type_model_ids, train_ratio, per_type_rng)
        per_type_splits[product_type_key] = type_splits

        balanced_type_splits = split_model_ids(type_model_ids, train_ratio, per_type_rng, limit=min_count)
        for split_name, split_ids in balanced_type_splits.items():
            balanced_all[split_name].extend(split_ids)

    for split_name, split_ids in balanced_all.items():
        balanced_all[split_name] = sorted(split_ids)

    return balanced_all, per_type_splits


def write_split_manifests(
    splits_dir: Path,
    split_prefix: str,
    dataset_key: str,
    class_name: str,
    all_splits: dict[str, list[str]],
    per_type_splits: dict[str, dict[str, list[str]]],
    write_aggregate: bool = True,
) -> dict[str, Any]:
    split_paths: dict[str, Any] = {"all": {}, "by_product_type": {}}

    if write_aggregate:
        for split_name, split_ids in all_splits.items():
            manifest_path = splits_dir / f"{split_prefix}_all_{split_name}.json"
            write_manifest(manifest_path, dataset_key, class_name, split_ids)
            split_paths["all"][split_name] = str(manifest_path)
            print(f"wrote split: {manifest_path}")

    for product_type_key, type_splits in sorted(per_type_splits.items()):
        split_paths["by_product_type"][product_type_key] = {}
        for split_name, split_ids in type_splits.items():
            manifest_path = splits_dir / f"{split_prefix}_{product_type_key}_{split_name}.json"
            write_manifest(manifest_path, dataset_key, class_name, split_ids)
            split_paths["by_product_type"][product_type_key][split_name] = str(manifest_path)
            print(f"wrote split: {manifest_path}")

    return split_paths


def default_metadata_out(args: argparse.Namespace) -> Path:
    return args.datasets_root / args.dataset_key / "preprocessing_metadata.json"


def main() -> None:
    args = parse_args()
    if args.model_workers <= 0:
        raise ValueError("--model-workers must be positive")
    if args.no_model_isolation and args.model_workers != 1:
        raise ValueError(
            "--no-model-isolation cannot be combined with --model-workers greater than 1"
        )
    rng = np.random.default_rng(args.seed)
    repair_config = build_repair_config(args)
    use_repaired_surface = repair_config.method == REPAIR_MANIFOLDPLUS
    fidelity_config = (
        build_repair_fidelity_config(args) if use_repaired_surface else None
    )

    metadata_payload = load_input_metadata(args.metadata_in)
    source_products = metadata_payload.get("products", {})

    model_ids = list_model_ids(args.source_dir)
    if args.limit is not None:
        model_ids = model_ids[: args.limit]
    if not model_ids:
        raise RuntimeError(f"no .glb models found in {args.source_dir}")

    process_model_ids = model_ids
    if args.only_models_in is not None:
        requested_ids = model_ids_from_manifest(args.only_models_in)
        available_ids = set(model_ids)
        missing_ids = sorted(requested_ids - available_ids)
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(
                f"{len(missing_ids)} IDs from {args.only_models_in} have no source mesh; first IDs: {preview}"
            )
        process_model_ids = [model_id for model_id in model_ids if model_id in requested_ids]

    products: dict[str, dict[str, Any]] = {}
    for model_id in model_ids:
        source_entry = source_products.get(model_id, {})
        product_type_key = stable_product_type_key(source_entry, args.class_name.upper())
        products[model_id] = dict(source_entry)
        products[model_id]["instance_id"] = model_id
        products[model_id]["3dmodel_id"] = model_id
        products[model_id]["product_type_key"] = product_type_key

    failures: dict[str, str] = {}
    rejected_repairs: dict[str, dict[str, Any]] = {}
    accepted_ids: set[str] = set()
    if not args.manifest_only:
        model_indices = {
            model_id: idx for idx, model_id in enumerate(process_model_ids, start=1)
        }

        def run_model(model_id: str):
            mesh_path = args.source_dir / f"{model_id}.glb"
            print(
                f"[{model_indices[model_id]}/{len(process_model_ids)}] "
                f"processing {mesh_path.name}",
                flush=True,
            )
            model_kwargs = dict(
                mesh_path=mesh_path,
                datasets_root=args.datasets_root,
                dataset_key=args.dataset_key,
                class_name=args.class_name,
                surface_point_count=args.surface_point_count,
                near_surface_stds=tuple(args.near_surface_stds),
                uniform_point_count=args.uniform_point_count,
                batch_size=args.batch_size,
                skip_existing=args.skip_existing,
                repair_config=repair_config,
                use_repaired_surface=use_repaired_surface,
                fidelity_config=fidelity_config,
                reuse_repair_fidelity=args.reuse_repair_fidelity,
                repair_cache_dataset_key=args.repair_cache_dataset_key,
            )
            output_path = object_output_paths(
                args.datasets_root, args.dataset_key, args.class_name, model_id
            )
            if args.skip_existing and output_path.is_file():
                model_rng = (
                    rng
                    if args.model_workers == 1
                    else np.random.default_rng(model_seed(args.seed, model_id))
                )
                return process_model(**model_kwargs, rng=model_rng)
            if args.no_model_isolation:
                return process_model(**model_kwargs, rng=rng)
            return process_model_isolated(
                **model_kwargs,
                seed=model_seed(args.seed, model_id),
            )

        for model_id, result, error in iter_model_results(
            process_model_ids, args.model_workers, run_model
        ):
            if error is not None:
                failures[model_id] = str(error)
                products[model_id]["processed"] = False
                products[model_id]["training_eligible"] = False
                products[model_id]["filter_reason"] = "processing_failure"
                print(f"  failed {model_id}: {error}", flush=True)
                if args.continue_on_error:
                    continue
                raise error
            data_path, repair_info = result
            products[model_id]["preprocessing_repair"] = repair_info
            if data_path is None:
                products[model_id].pop("cod_sdf_path", None)
                products[model_id]["processed"] = False
                products[model_id]["training_eligible"] = False
                products[model_id]["filter_reason"] = "repair_fidelity"
                rejected_repairs[model_id] = repair_info["fidelity"]
                gc.collect()
                continue
            products[model_id]["cod_sdf_path"] = str(data_path)
            products[model_id]["processed"] = True
            products[model_id]["training_eligible"] = True
            accepted_ids.add(model_id)
            if repair_info.get("method") == REPAIR_MANIFOLDPLUS:
                repair_status = "reused cached proxy" if repair_info.get("cache_hit") else "generated repaired proxy"
                print(f"  repair: {repair_status}")
            elif repair_info.get("skipped_existing"):
                print("  skipped existing outputs")
            print(f"  wrote {data_path}")
            gc.collect()

    for model_id in model_ids:
        products.setdefault(model_id, {})
        products[model_id].setdefault("instance_id", model_id)
        products[model_id].setdefault("3dmodel_id", model_id)
        products[model_id].setdefault("local_glb_path", str(args.source_dir / f"{model_id}.glb"))
        products[model_id].setdefault("processed", False)
        if model_id in failures:
            products[model_id].pop("cod_sdf_path", None)
            continue
        if products[model_id].get("filter_reason") == "repair_fidelity":
            continue
        data_path = object_output_paths(
            args.datasets_root, args.dataset_key, args.class_name, model_id
        )
        if data_path.is_file() and model_id not in accepted_ids:
            if use_repaired_surface:
                source, fidelity = read_repaired_record_fidelity(data_path)
                eligible = (
                    source == "repaired_mesh"
                    and fidelity is not None
                    and fidelity_config is not None
                    and repair_fidelity_passes(fidelity, fidelity_config)
                )
            else:
                fidelity = None
                eligible = True
            if eligible:
                accepted_ids.add(model_id)
                products[model_id]["cod_sdf_path"] = str(data_path)
                products[model_id]["processed"] = True
                products[model_id]["training_eligible"] = True
                if fidelity is not None:
                    products[model_id].setdefault(
                        "preprocessing_repair", {"fidelity": fidelity}
                    )
            elif use_repaired_surface:
                products[model_id].pop("cod_sdf_path", None)
                products[model_id]["training_eligible"] = False
                products[model_id].setdefault(
                    "filter_reason", "legacy_or_rejected_record"
                )

    if args.manifest_only and not use_repaired_surface:
        accepted_ids.update(model_ids)

    duplicate_geometry_groups: dict[str, list[str]] = {}
    if use_repaired_surface:
        if repair_config.repaired_mesh_dir is None:
            raise ValueError("repaired mesh directory is required")
        geometry_paths = {
            model_id: repaired_mesh_output_path(
                repair_config.repaired_mesh_dir,
                args.repair_cache_dataset_key or args.dataset_key,
                args.class_name,
                model_id,
            )
            for model_id in accepted_ids
        }
        accepted_ids, duplicate_geometry_groups = deduplicate_geometry_files(
            accepted_ids, geometry_paths
        )
        for canonical, members in duplicate_geometry_groups.items():
            products[canonical]["geometry_group_id"] = canonical
            products[canonical]["geometry_group_members"] = members
            for duplicate in members[1:]:
                products[duplicate]["geometry_group_id"] = canonical
                products[duplicate]["duplicate_of"] = canonical
                products[duplicate]["training_eligible"] = False
                products[duplicate]["filter_reason"] = "duplicate_geometry"
        duplicate_count = sum(
            len(members) - 1 for members in duplicate_geometry_groups.values()
        )
        if duplicate_count:
            print(
                f"excluded {duplicate_count} duplicate repaired geometries "
                f"from {len(duplicate_geometry_groups)} groups"
            )

    type_to_ids: dict[str, list[str]] = {}
    for model_id in sorted(accepted_ids):
        product_type_key = products[model_id]["product_type_key"]
        type_to_ids.setdefault(product_type_key, []).append(model_id)
    if not type_to_ids:
        raise RuntimeError(
            "no training-eligible records remain after repaired-mesh fidelity filtering"
        )

    splits_dir = args.datasets_root / "splits"
    all_splits, per_type_splits = build_split_sets(
        type_to_ids, args.train_ratio, args.seed
    )
    split_paths = write_split_manifests(
        splits_dir=splits_dir,
        split_prefix=args.split_prefix,
        dataset_key=args.dataset_key,
        class_name=args.class_name,
        all_splits=all_splits,
        per_type_splits=per_type_splits,
        write_aggregate=not args.per_type_splits_only,
    )

    metadata_out = args.metadata_out or default_metadata_out(args)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "dataset_key": args.dataset_key,
        "class_name": args.class_name,
        "source_metadata": str(args.metadata_in) if args.metadata_in else None,
        "selection": metadata_payload.get("selection"),
        "source_summary": metadata_payload.get("summary"),
        "preprocessing": {
            "normalization": "x_prime = scale * (x - center), isotropic max_abs_0.999",
            "surface_point_count": args.surface_point_count,
            "near_surface_stds": list(args.near_surface_stds),
            "uniform_point_count": args.uniform_point_count,
            "coordinate_bounds": [-1.0, 1.0],
            "sdf_sign": {
                "method": "multi_ray_majority",
                "ray_count": SIGN_RAY_COUNT,
            },
            "repair": {
                "method": repair_config.method,
                "manifoldplus_bin": str(repair_config.manifoldplus_bin)
                if repair_config.manifoldplus_bin is not None
                else None,
                "manifoldplus_depth": repair_config.manifoldplus_depth,
                "repaired_mesh_dir": str(repair_config.repaired_mesh_dir)
                if repair_config.repaired_mesh_dir is not None
                else None,
                "cache_dataset_key": (
                    args.repair_cache_dataset_key or args.dataset_key
                ),
                "surface_source": (
                    "repaired_mesh" if use_repaired_surface else "original_mesh"
                ),
                "fidelity_filter": (
                    {
                        "sample_count": fidelity_config.sample_count,
                        "distance_threshold": fidelity_config.distance_threshold,
                        "max_p95_distance": fidelity_config.max_p95_distance,
                        "max_outlier_fraction": fidelity_config.max_outlier_fraction,
                    }
                    if fidelity_config is not None
                    else None
                ),
            },
        },
        "splits": split_paths,
        "failures": failures,
        "rejected_repairs": rejected_repairs,
        "duplicate_geometry_groups": duplicate_geometry_groups,
        "training_eligible_count": len(accepted_ids),
        "products": products,
    }
    metadata_out.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote metadata: {metadata_out}")
    if failures:
        raise RuntimeError(
            f"preprocessing failed for {len(failures)} models; "
            f"details were written to {metadata_out}"
        )


if __name__ == "__main__":
    main()
