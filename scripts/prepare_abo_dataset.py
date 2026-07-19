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
import gc
import json
import os
import shutil
import subprocess
import tempfile
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--dataset-key", default=DEFAULT_DATASET_KEY)
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
    parser.add_argument("--batch-size", type=int, default=200000)
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
) -> tuple[trimesh.Trimesh, np.ndarray, float]:
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
        return RepairResult(mesh=load_repaired_mesh(output_path), used_cache=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.obj"
        mesh.export(input_path)
        command = [
            str(config.manifoldplus_bin),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--depth",
            str(config.manifoldplus_depth),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "ManifoldPlus repair failed for "
                f"{output_path}: {result.stderr.strip() or result.stdout.strip()}"
            )

    return RepairResult(mesh=load_repaired_mesh(output_path), used_cache=False)


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
    triangles = np.asarray(mesh.triangles[face_indices], dtype=np.float32)
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
            sdf[start:stop] = scene.compute_signed_distance(tensor).numpy().reshape(-1, 1)
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


def mesh_summary(mesh: trimesh.Trimesh) -> dict[str, Any]:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "components": int(len(mesh.split(only_watertight=False))),
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
) -> tuple[Path, dict[str, Any]]:
    def log_phase(message: str) -> None:
        print(f"  phase: {message}")

    model_id = mesh_path.stem
    output_path = object_output_paths(datasets_root, dataset_key, class_name, model_id)

    if skip_existing and output_path.is_file():
        repair_info: dict[str, Any] = {"skipped_existing": True}
        if repair_config is not None and repair_config.method == REPAIR_MANIFOLDPLUS:
            if repair_config.repaired_mesh_dir is None:
                raise ValueError("repaired mesh directory is required")
            proxy_path = repaired_mesh_output_path(
                repair_config.repaired_mesh_dir, dataset_key, class_name, model_id
            )
            if not proxy_path.is_file():
                raise FileNotFoundError(
                    "existing COD/SDF record has no matching repaired proxy: "
                    f"{proxy_path}"
                )
            repair_info.update(
                {
                    "method": REPAIR_MANIFOLDPLUS,
                    "sdf_sign_method": "occupancy",
                    "manifoldplus_depth": repair_config.manifoldplus_depth,
                    "repaired_mesh_path": str(proxy_path),
                    "cache_hit": True,
                }
            )
        return output_path, repair_info

    log_phase("mesh load/normalize")
    mesh, normalization_center, normalization_scale = normalize_mesh_with_transform(
        load_mesh(mesh_path)
    )
    sdf_mesh = mesh
    sign_method = "normal"
    repair_info: dict[str, Any] = {
        "method": REPAIR_NONE,
        "sdf_sign_method": sign_method,
        "original_mesh": mesh_summary(mesh),
    }

    if repair_config is not None and repair_config.method == REPAIR_MANIFOLDPLUS:
        if repair_config.repaired_mesh_dir is None:
            raise ValueError("repaired mesh directory is required")
        proxy_path = repaired_mesh_output_path(repair_config.repaired_mesh_dir, dataset_key, class_name, model_id)
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
            "original_mesh": mesh_summary(mesh),
            "repaired_mesh": mesh_summary(sdf_mesh),
        }
    else:
        log_phase("manifold repair skipped")

    log_phase("raycast scene creation")
    scene = make_raycast_scene(sdf_mesh)
    if sdf_mesh is not mesh:
        del sdf_mesh

    log_phase("COD surface and SDF supervision sampling")
    arrays = sample_cod_supervision(
        mesh=mesh,
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
    del mesh
    del scene

    log_phase("COD/SDF NPZ write")
    save_cod_sdf(output_path, arrays)
    return output_path, repair_info


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
) -> dict[str, Any]:
    split_paths: dict[str, Any] = {"all": {}, "by_product_type": {}}

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
    rng = np.random.default_rng(args.seed)
    repair_config = build_repair_config(args)

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
    type_to_ids: dict[str, list[str]] = {}

    for model_id in model_ids:
        source_entry = source_products.get(model_id, {})
        product_type_key = stable_product_type_key(source_entry, args.class_name.upper())
        type_to_ids.setdefault(product_type_key, []).append(model_id)
        products[model_id] = dict(source_entry)
        products[model_id]["instance_id"] = model_id
        products[model_id]["3dmodel_id"] = model_id
        products[model_id]["product_type_key"] = product_type_key

    splits_dir = args.datasets_root / "splits"
    all_splits, per_type_splits = build_split_sets(type_to_ids, args.train_ratio, args.seed)
    split_paths = write_split_manifests(
        splits_dir=splits_dir,
        split_prefix=args.split_prefix,
        dataset_key=args.dataset_key,
        class_name=args.class_name,
        all_splits=all_splits,
        per_type_splits=per_type_splits,
    )

    if not args.manifest_only:
        for idx, model_id in enumerate(process_model_ids, start=1):
            mesh_path = args.source_dir / f"{model_id}.glb"
            print(f"[{idx}/{len(process_model_ids)}] processing {mesh_path.name}")
            data_path, repair_info = process_model(
                mesh_path=mesh_path,
                datasets_root=args.datasets_root,
                dataset_key=args.dataset_key,
                class_name=args.class_name,
                surface_point_count=args.surface_point_count,
                near_surface_stds=tuple(args.near_surface_stds),
                uniform_point_count=args.uniform_point_count,
                batch_size=args.batch_size,
                rng=rng,
                skip_existing=args.skip_existing,
                repair_config=repair_config,
            )
            products[model_id]["cod_sdf_path"] = str(data_path)
            products[model_id]["preprocessing_repair"] = repair_info
            products[model_id]["processed"] = True
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
        if "cod_sdf_path" not in products[model_id]:
            data_path = object_output_paths(args.datasets_root, args.dataset_key, args.class_name, model_id)
            if data_path.is_file():
                products[model_id]["cod_sdf_path"] = str(data_path)
                products[model_id]["processed"] = True

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
            "repair": {
                "method": repair_config.method,
                "manifoldplus_bin": str(repair_config.manifoldplus_bin)
                if repair_config.manifoldplus_bin is not None
                else None,
                "manifoldplus_depth": repair_config.manifoldplus_depth,
                "repaired_mesh_dir": str(repair_config.repaired_mesh_dir)
                if repair_config.repaired_mesh_dir is not None
                else None,
            },
        },
        "splits": split_paths,
        "products": products,
    }
    metadata_out.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote metadata: {metadata_out}")


if __name__ == "__main__":
    main()
