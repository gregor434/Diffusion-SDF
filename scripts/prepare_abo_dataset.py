#!/usr/bin/env python3

"""Prepare ABO meshes for Diffusion-SDF and emit split-aware metadata.

This generalizes the chair-specific ABO preparation flow:
- consumes downloaded `.glb` files
- converts each mesh into the repo's CSV layout
- writes split manifests for all products and per product type
- writes a metadata JSON keyed directly by `3dmodel_id`
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh


DEFAULT_SOURCE_DIR = Path("datasets/ABO/models_filtered")
DEFAULT_DATASETS_ROOT = Path("datasets")
DEFAULT_DATASET_KEY = "abo"
DEFAULT_CLASS_NAME = "ABO"
DEFAULT_SPLIT_PREFIX = "abo"
DEFAULT_METADATA_IN = Path("datasets/ABO/abo_selected_subset.json")
DEFAULT_TRAIN_RATIO = 0.8


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
    parser.add_argument("--near-surface-count", type=int, default=596000)
    parser.add_argument("--grid-count", type=int, default=468000)
    parser.add_argument(
        "--surface-ratio",
        type=float,
        default=0.2,
        help="Fraction of near-surface rows written with sdf=0.",
    )
    parser.add_argument(
        "--offset-scale-small",
        type=float,
        default=0.01,
        help="Small normal offset relative to the normalized cube size.",
    )
    parser.add_argument(
        "--offset-scale-large",
        type=float,
        default=0.05,
        help="Large normal offset relative to the normalized cube size.",
    )
    parser.add_argument("--batch-size", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N models after sorting.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip objects whose sdf_data.csv and grid_gt.csv already exist.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only write splits and metadata; do not generate CSV data.",
    )
    return parser.parse_args()


def list_model_ids(source_dir: Path) -> list[str]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    return sorted(path.stem for path in source_dir.glob("*.glb"))


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    asset = trimesh.load(mesh_path, force="scene")
    if isinstance(asset, trimesh.Scene):
        if not asset.geometry:
            raise ValueError(f"scene has no geometry: {mesh_path}")
        mesh = asset.dump(concatenate=True)
    elif isinstance(asset, trimesh.Trimesh):
        mesh = asset
    else:
        raise TypeError(f"unsupported mesh type {type(asset)!r} for {mesh_path}")

    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    bounds = mesh.bounds.astype(np.float32)
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    max_extent = float(extent.max())
    if max_extent <= 0:
        raise ValueError("mesh has zero extent")
    mesh.apply_translation(-center)
    mesh.apply_scale(2.0 / max_extent)
    return mesh


def make_raycast_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    vertices = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
    faces = o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32))
    tmesh = o3d.t.geometry.TriangleMesh(vertices, faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return scene


def sample_near_surface(
    mesh: trimesh.Trimesh,
    total_count: int,
    surface_ratio: float,
    offset_scale_small: float,
    offset_scale_large: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if total_count <= 0:
        raise ValueError("near-surface count must be positive")

    surface_count = max(1, int(total_count * surface_ratio))
    offset_count = max(0, total_count - surface_count)
    pair_count = int(math.ceil(offset_count / 2.0))

    surface_points, surface_face_idx = trimesh.sample.sample_surface(mesh, surface_count)
    surface_rows = np.concatenate(
        [surface_points.astype(np.float32), np.zeros((surface_count, 1), dtype=np.float32)],
        axis=1,
    )

    if pair_count == 0:
        return surface_rows

    base_points, base_face_idx = trimesh.sample.sample_surface(mesh, pair_count)
    base_normals = np.asarray(mesh.face_normals[base_face_idx], dtype=np.float32)
    lengths = rng.uniform(offset_scale_small, offset_scale_large, size=(pair_count, 1)).astype(np.float32)

    pos_points = base_points.astype(np.float32) + base_normals * lengths
    neg_points = base_points.astype(np.float32) - base_normals * lengths
    pos_rows = np.concatenate([pos_points, lengths], axis=1)
    neg_rows = np.concatenate([neg_points, -lengths], axis=1)

    rows = np.concatenate([surface_rows, pos_rows, neg_rows], axis=0)
    if rows.shape[0] > total_count:
        rows = rows[:total_count]

    rng.shuffle(rows, axis=0)
    return rows


def compute_grid_sdf(
    scene: o3d.t.geometry.RaycastingScene,
    count: int,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    query_points = rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32)
    sdf = np.empty((count, 1), dtype=np.float32)

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = query_points[start:stop]
        out = scene.compute_closest_points(o3d.core.Tensor(batch))
        closest_points = out["points"].numpy()
        normals = out["primitive_normals"].numpy()

        delta = batch - closest_points
        unsigned = np.linalg.norm(delta, axis=1, keepdims=True)
        sign = np.sign(np.sum(delta * normals, axis=1, keepdims=True)).astype(np.float32)
        sign[sign == 0] = 1.0
        sdf[start:stop] = unsigned * sign

    return np.concatenate([query_points, sdf], axis=1)


def save_csv(csv_path: Path, rows: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(csv_path, rows, delimiter=",", fmt="%.7f")


def object_output_paths(
    datasets_root: Path,
    dataset_key: str,
    class_name: str,
    model_id: str,
) -> tuple[Path, Path]:
    sdf_path = datasets_root / dataset_key / class_name / model_id / "sdf_data.csv"
    grid_path = datasets_root / "grid_data" / dataset_key / class_name / model_id / "grid_gt.csv"
    return sdf_path, grid_path


def process_model(
    mesh_path: Path,
    datasets_root: Path,
    dataset_key: str,
    class_name: str,
    near_surface_count: int,
    grid_count: int,
    surface_ratio: float,
    offset_scale_small: float,
    offset_scale_large: float,
    batch_size: int,
    rng: np.random.Generator,
    skip_existing: bool,
) -> tuple[Path, Path]:
    model_id = mesh_path.stem
    sdf_path, grid_path = object_output_paths(datasets_root, dataset_key, class_name, model_id)

    if skip_existing and sdf_path.is_file() and grid_path.is_file():
        return sdf_path, grid_path

    mesh = normalize_mesh(load_mesh(mesh_path))
    scene = make_raycast_scene(mesh)

    near_surface_rows = sample_near_surface(
        mesh=mesh,
        total_count=near_surface_count,
        surface_ratio=surface_ratio,
        offset_scale_small=offset_scale_small,
        offset_scale_large=offset_scale_large,
        rng=rng,
    )
    grid_rows = compute_grid_sdf(
        scene=scene,
        count=grid_count,
        batch_size=batch_size,
        rng=rng,
    )

    save_csv(sdf_path, near_surface_rows)
    save_csv(grid_path, grid_rows)
    return sdf_path, grid_path


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
    return args.datasets_root / "splits" / f"{args.split_prefix}_metadata.json"


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    metadata_payload = load_input_metadata(args.metadata_in)
    source_products = metadata_payload.get("products", {})

    model_ids = list_model_ids(args.source_dir)
    if args.limit is not None:
        model_ids = model_ids[: args.limit]
    if not model_ids:
        raise RuntimeError(f"no .glb models found in {args.source_dir}")

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
        for idx, model_id in enumerate(model_ids, start=1):
            mesh_path = args.source_dir / f"{model_id}.glb"
            print(f"[{idx}/{len(model_ids)}] processing {mesh_path.name}")
            sdf_path, grid_path = process_model(
                mesh_path=mesh_path,
                datasets_root=args.datasets_root,
                dataset_key=args.dataset_key,
                class_name=args.class_name,
                near_surface_count=args.near_surface_count,
                grid_count=args.grid_count,
                surface_ratio=args.surface_ratio,
                offset_scale_small=args.offset_scale_small,
                offset_scale_large=args.offset_scale_large,
                batch_size=args.batch_size,
                rng=rng,
                skip_existing=args.skip_existing,
            )
            products[model_id]["sdf_data_path"] = str(sdf_path)
            products[model_id]["grid_gt_path"] = str(grid_path)
            products[model_id]["processed"] = True
            print(f"  wrote {sdf_path}")
            print(f"  wrote {grid_path}")

    for model_id in model_ids:
        products.setdefault(model_id, {})
        products[model_id].setdefault("instance_id", model_id)
        products[model_id].setdefault("3dmodel_id", model_id)
        products[model_id].setdefault("local_glb_path", str(args.source_dir / f"{model_id}.glb"))
        products[model_id].setdefault("processed", False)
        if "sdf_data_path" not in products[model_id]:
            sdf_path, grid_path = object_output_paths(args.datasets_root, args.dataset_key, args.class_name, model_id)
            if sdf_path.is_file() and grid_path.is_file():
                products[model_id]["sdf_data_path"] = str(sdf_path)
                products[model_id]["grid_gt_path"] = str(grid_path)
                products[model_id]["processed"] = True

    metadata_out = args.metadata_out or default_metadata_out(args)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "dataset_key": args.dataset_key,
        "class_name": args.class_name,
        "source_metadata": str(args.metadata_in) if args.metadata_in else None,
        "selection": metadata_payload.get("selection"),
        "source_summary": metadata_payload.get("summary"),
        "splits": split_paths,
        "products": products,
    }
    metadata_out.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote metadata: {metadata_out}")


if __name__ == "__main__":
    main()
