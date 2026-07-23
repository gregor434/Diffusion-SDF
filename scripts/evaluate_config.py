#!/usr/bin/env python3
"""Evaluate stored COD reconstruction artifacts without loading trained models."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NUM_POINTS = 10000
DEFAULT_FSCORE_THRESHOLDS = (0.01, 0.02)
DEFAULT_JSD_RESOLUTION = 28
DEFAULT_DOMAIN = 1.0
DEFAULT_IOU_BATCH_SIZE = 100000
MAX_COMPONENT_ANALYSIS_FACES = 500000
CHAMFER_DISTANCE_DEFINITION = {
    "formula": (
        "mean nearest-neighbor Euclidean distance sample-to-reference + "
        "mean nearest-neighbor Euclidean distance reference-to-sample"
    ),
    "point_distance": "L2",
    "squared": False,
    "direction_reduction": "sum",
}


@dataclass(frozen=True)
class EvaluationOptions:
    num_points: int = DEFAULT_NUM_POINTS
    seed: int = 0
    workers: int = 1
    jsd_resolution: int = DEFAULT_JSD_RESOLUTION
    fscore_thresholds: tuple = DEFAULT_FSCORE_THRESHOLDS
    iou_batch_size: int = DEFAULT_IOU_BATCH_SIZE


def _stable_seed(seed, key):
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    return (int(seed) + int.from_bytes(digest[:8], "little")) % (2**32)


def _finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_float(value)
    return value


def _resolve_path(value, config_dir, must_exist=True):
    if value is None:
        return None
    value = Path(value).expanduser()
    candidates = [value] if value.is_absolute() else [
        Path.cwd() / value,
        REPOSITORY_ROOT / value,
        config_dir / value,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        return None
    return candidates[0].resolve()


def _load_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _flatten_split(split):
    records = []
    for dataset, classes in split.items():
        for class_name, instances in classes.items():
            for instance_name in instances:
                records.append(
                    {
                        "dataset": str(dataset),
                        "class_name": str(class_name),
                        "instance_name": str(instance_name),
                        "key": "{}/{}".format(class_name, instance_name),
                    }
                )
    return records


def resolve_configuration(config_dir):
    config_dir = Path(config_dir).expanduser().resolve()
    specs_path = config_dir / "specs.json"
    if not specs_path.is_file():
        raise FileNotFoundError("configuration has no specs.json: {}".format(config_dir))

    specs = _load_json(specs_path)
    split_path = _resolve_path(specs.get("TestSplit"), config_dir)
    if split_path is None:
        raise FileNotFoundError(
            "TestSplit does not resolve to a file: {!r}".format(specs.get("TestSplit"))
        )
    records = _flatten_split(_load_json(split_path))

    source_specs = specs
    source_specs_dir = config_dir
    data_source = _resolve_path(specs.get("DataSource"), config_dir)
    if data_source is None and specs.get("modulation_ckpt_path"):
        checkpoint_path = _resolve_path(
            specs["modulation_ckpt_path"], config_dir, must_exist=False
        )
        upstream_specs_path = checkpoint_path.parent / "specs.json"
        if upstream_specs_path.is_file():
            source_specs = _load_json(upstream_specs_path)
            source_specs_dir = upstream_specs_path.parent
            data_source = _resolve_path(
                source_specs.get("DataSource"), source_specs_dir
            )
    if data_source is None:
        raise FileNotFoundError(
            "DataSource does not resolve from this configuration or its "
            "modulation checkpoint configuration"
        )

    cod_specs = specs.get("CODVaeSpecs") or source_specs.get("CODVaeSpecs")
    if not isinstance(cod_specs, dict):
        raise ValueError("configuration has no CODVaeSpecs")
    for field in ("latent_tokens", "latent_dimension"):
        if field not in cod_specs:
            raise ValueError("CODVaeSpecs has no {}".format(field))

    task = specs.get("training_task")
    if task not in ("modulation", "diffusion", "combined"):
        raise ValueError("unsupported COD training_task: {!r}".format(task))
    if task in ("diffusion", "combined"):
        diffusion_specs = specs.get("diffusion_model_specs")
        if not isinstance(diffusion_specs, dict):
            raise ValueError(
                "{} configuration has no diffusion_model_specs".format(task)
            )
        for field in ("latent_tokens", "latent_dimension"):
            if int(diffusion_specs.get(field, -1)) != int(cod_specs[field]):
                raise ValueError(
                    "diffusion_model_specs.{} must match "
                    "CODVaeSpecs.{} ({})".format(
                        field, field, cod_specs[field]
                    )
                )

    conditional = bool(
        specs.get("conditioning")
        or specs.get("diffusion_model_specs", {}).get("cond", False)
    )
    return {
        "config_dir": config_dir,
        "specs": specs,
        "source_specs": source_specs,
        "split_path": split_path,
        "records": records,
        "data_source": data_source,
        "cod_specs": cod_specs,
        "task": task,
        "conditional": conditional,
    }


def _reference_path(record, data_source):
    return (
        data_source
        / record["dataset"]
        / record["class_name"]
        / record["instance_name"]
        / "cod_sdf.npz"
    )


def _sample_array(points, count, rng):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("point cloud must be a non-empty Nx3 array")
    if not np.all(np.isfinite(points)):
        raise ValueError("point cloud contains non-finite values")
    replace = len(points) < count
    indices = rng.choice(len(points), size=count, replace=replace)
    return points[indices].astype(np.float32)


def load_reference_points(record, data_source, options):
    path = _reference_path(record, data_source)
    result = {
        "key": record["key"],
        "record": record,
        "path": str(path) if path is not None else None,
        "points": None,
        "error": None,
    }
    if not path.is_file():
        result["error"] = "missing COD reference cod_sdf.npz"
        return result

    try:
        with np.load(path) as data:
            if "surface_points" not in data:
                raise ValueError("COD record has no surface_points array")
            surface = np.asarray(data["surface_points"], dtype=np.float32)
        rng = np.random.default_rng(
            _stable_seed(options.seed, "reference:" + record["key"])
        )
        result["points"] = _sample_array(surface, options.num_points, rng)
    except Exception as exc:
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
    return result


def _as_mesh(loaded):
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not geometries:
            raise ValueError("scene contains no triangle meshes")
        return trimesh.util.concatenate(geometries)
    raise ValueError("artifact is not a triangle mesh")


def _face_areas(vertices, faces, batch_size=250000):
    areas = np.empty(len(faces), dtype=np.float64)
    for start in range(0, len(faces), batch_size):
        stop = min(start + batch_size, len(faces))
        triangles = vertices[faces[start:stop]]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        areas[start:stop] = np.linalg.norm(cross, axis=1) * 0.5
    return areas


def _sample_mesh(mesh, count, rng, face_areas=None):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0:
        raise ValueError("mesh has no vertices")
    if len(faces) == 0:
        return _sample_array(vertices, count, rng)

    areas = (
        np.asarray(face_areas, dtype=np.float64)
        if face_areas is not None
        else _face_areas(vertices, faces)
    )
    valid = np.isfinite(areas) & (areas > 0)
    if not np.any(valid):
        raise ValueError("mesh has no finite, non-degenerate faces")
    probabilities = np.where(valid, areas, 0.0)
    probabilities /= probabilities.sum()
    chosen = rng.choice(len(faces), size=count, replace=True, p=probabilities)
    selected = vertices[faces[chosen]]
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    points = (
        (1.0 - u)[:, None] * selected[:, 0]
        + (u * (1.0 - v))[:, None] * selected[:, 1]
        + (u * v)[:, None] * selected[:, 2]
    )
    return points.astype(np.float32)


def mesh_occupancy_iou(mesh, reference_path, batch_size):
    """Compare mesh occupancy with COD's uniform signed-distance samples."""
    reference_path = Path(reference_path)
    if not reference_path.is_file():
        raise FileNotFoundError("missing COD reference cod_sdf.npz")
    if not mesh.is_watertight:
        raise ValueError("mesh is not watertight; occupancy is not well-defined")

    # Imported lazily so Chamfer-only evaluation still works without Open3D.
    import open3d as o3d

    with np.load(reference_path) as data:
        required = ("uniform_query_points", "uniform_sdf")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                "COD record is missing occupancy arrays: {}".format(
                    ", ".join(missing)
                )
            )
        query_points = np.asarray(data["uniform_query_points"], dtype=np.float32)
        reference_sdf = np.asarray(data["uniform_sdf"], dtype=np.float32).reshape(-1)
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("uniform_query_points must have shape [N,3]")
    if len(query_points) == 0 or len(reference_sdf) != len(query_points):
        raise ValueError("COD uniform query and SDF arrays are empty or misaligned")
    if not np.all(np.isfinite(query_points)) or not np.all(np.isfinite(reference_sdf)):
        raise ValueError("COD uniform query or SDF arrays contain non-finite values")
    reference_occupied = reference_sdf < 0.0

    vertices = o3d.core.Tensor(
        np.asarray(mesh.vertices, dtype=np.float32)
    )
    faces = o3d.core.Tensor(
        np.asarray(mesh.faces, dtype=np.uint32)
    )
    tensor_mesh = o3d.t.geometry.TriangleMesh(vertices, faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    sample_occupied = np.empty(len(query_points), dtype=bool)
    for start in range(0, len(query_points), batch_size):
        stop = min(start + batch_size, len(query_points))
        occupancy = scene.compute_occupancy(
            o3d.core.Tensor(query_points[start:stop])
        ).numpy()
        sample_occupied[start:stop] = occupancy.reshape(-1) > 0.5

    intersection = int(np.count_nonzero(sample_occupied & reference_occupied))
    union = int(np.count_nonzero(sample_occupied | reference_occupied))
    iou = 1.0 if union == 0 else intersection / union
    return {
        "iou": float(iou),
        "iou_intersection": intersection,
        "iou_union": union,
        "iou_query_points": int(len(query_points)),
    }


def load_mesh_artifact(path, options, domain, reference_path=None):
    path = Path(path)
    result = {
        "path": str(path),
        "points": None,
        "error": None,
        "occupancy": None,
        "iou_error": None,
        "health": {
            "valid": False,
            "vertices": 0,
            "faces": 0,
            "watertight": False,
            "components": None,
            "degenerate_face_fraction": None,
            "outside_domain_vertex_fraction": None,
        },
    }
    try:
        mesh = _as_mesh(trimesh.load(path, process=False))
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if len(vertices) == 0 or not np.all(np.isfinite(vertices)):
            raise ValueError("mesh vertices are empty or non-finite")
        if len(faces) == 0:
            raise ValueError("mesh has no faces")

        face_areas = _face_areas(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
        )
        if len(faces) > MAX_COMPONENT_ANALYSIS_FACES:
            # Face adjacency can require several times the source mesh size.
            # Omit this diagnostic for unusually dense meshes so evaluation
            # does not fail merely while calculating mesh-health metadata.
            components = None
        else:
            face_adjacency = mesh.face_adjacency
            if len(face_adjacency) == 0:
                components = int(len(faces))
            else:
                component_labels = trimesh.graph.connected_component_labels(
                    face_adjacency, node_count=len(faces)
                )
                components = len(np.unique(component_labels))
        result["health"] = {
            "valid": True,
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "watertight": bool(mesh.is_watertight),
            "components": int(components) if components is not None else None,
            "degenerate_face_fraction": float(np.mean(face_areas <= 1e-15)),
            "outside_domain_vertex_fraction": float(
                np.mean(np.any(np.abs(vertices) > domain + 1e-6, axis=1))
            ),
        }
        rng = np.random.default_rng(_stable_seed(options.seed, "mesh:" + str(path)))
        result["points"] = _sample_mesh(
            mesh, options.num_points, rng, face_areas=face_areas
        )
        if reference_path is not None:
            try:
                result["occupancy"] = mesh_occupancy_iou(
                    mesh, reference_path, options.iou_batch_size
                )
            except Exception as exc:
                result["iou_error"] = "{}: {}".format(type(exc).__name__, exc)
    except Exception as exc:
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
    return result


def _parallel_map(function, items, workers):
    if workers <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, items))


def _distance_vectors(sample_points, reference_points):
    sample_points = np.asarray(sample_points)
    reference_points = np.asarray(reference_points)
    sample_to_reference = cKDTree(reference_points).query(sample_points, k=1)[0]
    reference_to_sample = cKDTree(sample_points).query(reference_points, k=1)[0]
    return sample_to_reference, reference_to_sample


def paired_surface_metrics(sample_points, reference_points, thresholds):
    sample_to_reference, reference_to_sample = _distance_vectors(
        sample_points, reference_points
    )
    metrics = {
        "cd_sample_to_reference": float(np.mean(sample_to_reference)),
        "cd_reference_to_sample": float(np.mean(reference_to_sample)),
        "hausdorff95": float(
            max(
                np.percentile(sample_to_reference, 95),
                np.percentile(reference_to_sample, 95),
            )
        ),
    }
    metrics["chamfer"] = (
        metrics["cd_sample_to_reference"] + metrics["cd_reference_to_sample"]
    )
    for threshold in thresholds:
        precision = float(np.mean(sample_to_reference <= threshold))
        recall = float(np.mean(reference_to_sample <= threshold))
        fscore = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        suffix = "{:g}".format(threshold)
        metrics["precision@{}".format(suffix)] = precision
        metrics["recall@{}".format(suffix)] = recall
        metrics["fscore@{}".format(suffix)] = fscore
    return metrics


def chamfer_distance(sample_points, reference_points):
    sample_to_reference, reference_to_sample = _distance_vectors(
        sample_points, reference_points
    )
    return float(np.mean(sample_to_reference) + np.mean(reference_to_sample))


def pairwise_chamfer(left_clouds, right_clouds, workers=1, symmetric=False):
    left_clouds = list(left_clouds)
    right_clouds = list(right_clouds)
    matrix = np.empty((len(left_clouds), len(right_clouds)), dtype=np.float64)

    def compute_row(index):
        row = np.empty(len(right_clouds), dtype=np.float64)
        for right_index, right in enumerate(right_clouds):
            if symmetric and right_index < index:
                row[right_index] = np.nan
            elif symmetric and right_index == index:
                row[right_index] = 0.0
            else:
                row[right_index] = chamfer_distance(left_clouds[index], right)
        return index, row

    rows = _parallel_map(compute_row, range(len(left_clouds)), workers)
    for index, row in rows:
        matrix[index] = row
    if symmetric:
        lower = np.tril_indices(len(left_clouds), -1)
        matrix[lower] = matrix.T[lower]
    return matrix


def _grid_coordinates(resolution, domain):
    axis = np.linspace(-domain, domain, resolution, dtype=np.float64)
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _occupancy_histogram(clouds, resolution, domain):
    grid = _grid_coordinates(resolution, domain)
    tree = cKDTree(grid)
    counters = np.zeros(len(grid), dtype=np.float64)
    for cloud in clouds:
        indices = tree.query(np.asarray(cloud), k=1)[1]
        np.add.at(counters, indices, 1.0)
    total = counters.sum()
    return counters / total if total > 0 else counters


def occupancy_jsd(sample_clouds, reference_clouds, resolution, domain):
    sample = _occupancy_histogram(sample_clouds, resolution, domain)
    reference = _occupancy_histogram(reference_clouds, resolution, domain)
    midpoint = 0.5 * (sample + reference)

    def kl_divergence(values, target):
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / target[mask])))

    return 0.5 * (
        kl_divergence(sample, midpoint) + kl_divergence(reference, midpoint)
    )


def distribution_metrics(sample_clouds, reference_clouds, options, domain):
    if not sample_clouds or not reference_clouds:
        return None

    cross = pairwise_chamfer(
        sample_clouds, reference_clouds, workers=options.workers
    )
    sample_self = pairwise_chamfer(
        sample_clouds, sample_clouds, workers=options.workers, symmetric=True
    )
    reference_self = pairwise_chamfer(
        reference_clouds, reference_clouds, workers=options.workers, symmetric=True
    )

    mmd = float(np.min(cross, axis=0).mean())
    coverage = float(len(np.unique(np.argmin(cross, axis=1))) / len(reference_clouds))

    combined = np.block(
        [
            [reference_self, cross.T],
            [cross, sample_self],
        ]
    )
    np.fill_diagonal(combined, np.inf)
    labels = np.concatenate(
        [np.ones(len(reference_clouds), dtype=bool), np.zeros(len(sample_clouds), dtype=bool)]
    )
    predictions = labels[np.argmin(combined, axis=1)]
    reference_predictions = predictions[: len(reference_clouds)]
    sample_predictions = predictions[len(reference_clouds) :]

    return {
        "mmd_cd": mmd,
        "coverage_cd": coverage,
        "one_nn_cd_accuracy": float(np.mean(predictions == labels)),
        "one_nn_cd_reference_accuracy": float(np.mean(reference_predictions)),
        "one_nn_cd_sample_accuracy": float(np.mean(~sample_predictions)),
        "jsd": float(
            occupancy_jsd(
                sample_clouds,
                reference_clouds,
                options.jsd_resolution,
                domain,
            )
        ),
        "sample_count": len(sample_clouds),
        "reference_count": len(reference_clouds),
    }


def _aggregate_metric_rows(rows, metric_names):
    aggregate = {}
    for metric_name in metric_names:
        values = [
            float(row[metric_name])
            for row in rows
            if row.get(metric_name) is not None
            and math.isfinite(float(row[metric_name]))
        ]
        if values:
            aggregate[metric_name] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "count": len(values),
            }
    return aggregate


def _mesh_health_summary(mesh_results):
    valid = [item for item in mesh_results if item["health"]["valid"]]
    component_counts = [
        item["health"]["components"]
        for item in valid
        if item["health"]["components"] is not None
    ]
    return {
        "discovered": len(mesh_results),
        "valid": len(valid),
        "invalid": len(mesh_results) - len(valid),
        "watertight_fraction": (
            float(np.mean([item["health"]["watertight"] for item in valid]))
            if valid
            else None
        ),
        "multi_component_fraction": (
            float(np.mean([count > 1 for count in component_counts]))
            if component_counts
            else None
        ),
        "component_count_omitted": len(valid) - len(component_counts),
        "mean_outside_domain_vertex_fraction": (
            float(
                np.mean(
                    [
                        item["health"]["outside_domain_vertex_fraction"]
                        for item in valid
                    ]
                )
            )
            if valid
            else None
        ),
    }


def validate_modulations(context):
    specs = context["specs"]
    config_dir = context["config_dir"]
    if specs.get("training_task") == "modulation":
        modulation_dir = config_dir / "modulations"
    else:
        modulation_value = (
            specs.get("modulation_cache_path") or specs.get("data_path")
        )
        modulation_dir = (
            _resolve_path(modulation_value, config_dir, must_exist=False)
            if modulation_value
            else config_dir / "modulations"
        )

    cod_specs = context["cod_specs"]
    expected_shape = (
        int(cod_specs["latent_tokens"]),
        int(cod_specs["latent_dimension"]),
    )

    result = {
        "path": str(modulation_dir),
        "expected": len(context["records"]),
        "found": 0,
        "valid": 0,
        "missing": [],
        "invalid": {},
        "expected_shape": list(expected_shape),
        "observed_shapes": {},
        "latent_statistics": None,
    }
    if not modulation_dir.is_dir():
        result["missing"] = [record["key"] for record in context["records"]]
        return result

    for record in context["records"]:
        path = (
            modulation_dir
            / record["class_name"]
            / record["instance_name"]
            / "modulation.npz"
        )
        if not path.is_file():
            result["missing"].append(record["key"])
            continue
        result["found"] += 1
        try:
            with np.load(path) as data:
                required = ("object_id", "posterior_mean", "posterior_logvar")
                missing = [name for name in required if name not in data]
                if missing:
                    raise ValueError(
                        "missing arrays: {}".format(", ".join(missing))
                    )
                object_id = str(np.asarray(data["object_id"]).item())
                posterior_mean = np.asarray(data["posterior_mean"])
                posterior_logvar = np.asarray(data["posterior_logvar"])
            shape = tuple(posterior_mean.shape)
            shape_key = "x".join(str(value) for value in shape)
            result["observed_shapes"][shape_key] = (
                result["observed_shapes"].get(shape_key, 0) + 1
            )
            if object_id != record["instance_name"]:
                raise ValueError(
                    "object_id {!r} does not match {!r}".format(
                        object_id, record["instance_name"]
                    )
                )
            if shape != expected_shape:
                raise ValueError(
                    "posterior shape {} does not match expected {}".format(
                        shape, expected_shape
                    )
                )
            if posterior_logvar.shape != posterior_mean.shape:
                raise ValueError("posterior_logvar shape does not match posterior_mean")
            if not np.all(np.isfinite(posterior_mean)):
                raise ValueError("posterior_mean contains non-finite values")
            if not np.all(np.isfinite(posterior_logvar)):
                raise ValueError("posterior_logvar contains non-finite values")
            result["valid"] += 1
        except Exception as exc:
            result["invalid"][record["key"]] = "{}: {}".format(
                type(exc).__name__, exc
            )

    stats_path = (
        _resolve_path(
            specs["latent_stats_path"], config_dir, must_exist=False
        )
        if specs.get("latent_stats_path")
        else modulation_dir / "latent_stats.npz"
    )
    if stats_path.is_file():
        stats = {"path": str(stats_path), "valid": False, "error": None}
        try:
            with np.load(stats_path) as data:
                mean = np.asarray(data["mean"])
                std = np.asarray(data["std"])
            expected_stats_shape = (1, 1, expected_shape[1])
            if mean.shape != expected_stats_shape or std.shape != expected_stats_shape:
                raise ValueError(
                    "mean/std shapes must both be {}".format(expected_stats_shape)
                )
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
                raise ValueError("mean/std contain non-finite values")
            if np.any(std <= 0):
                raise ValueError("std must be strictly positive")
            stats["valid"] = True
        except Exception as exc:
            stats["error"] = "{}: {}".format(type(exc).__name__, exc)
        result["latent_statistics"] = stats
    return result


def _discover_artifacts(context):
    recon_dir = context["config_dir"] / "recon"
    task = context["task"]
    records = context["records"]
    artifacts = []

    if task == "modulation":
        for record in records:
            artifacts.append(
                {
                    "key": record["key"],
                    "record": record,
                    "sample": "reconstruct",
                    "path": recon_dir
                    / record["class_name"]
                    / record["instance_name"]
                    / "reconstruct.ply",
                    "expected": True,
                }
            )
        return artifacts

    if context["conditional"]:
        expected_keys = {record["key"] for record in records}
        record_lookup = {record["key"]: record for record in records}
        if recon_dir.is_dir():
            for path in sorted(recon_dir.glob("*/*/*_recon.ply")):
                key = "{}/{}".format(path.parent.parent.name, path.parent.name)
                artifacts.append(
                    {
                        "key": key,
                        "record": record_lookup.get(key),
                        "sample": path.stem[: -len("_recon")],
                        "path": path,
                        "expected": key in expected_keys,
                    }
                )
        discovered_keys = {artifact["key"] for artifact in artifacts}
        for key in sorted(expected_keys - discovered_keys):
            record = record_lookup[key]
            artifacts.append(
                {
                    "key": key,
                    "record": record,
                    "sample": None,
                    "path": recon_dir
                    / record["class_name"]
                    / record["instance_name"]
                    / "*_recon.ply",
                    "expected": True,
                }
            )
        return artifacts

    if recon_dir.is_dir():
        for path in sorted(recon_dir.glob("*_recon.ply")):
            artifacts.append(
                {
                    "key": path.stem,
                    "record": None,
                    "sample": path.stem[: -len("_recon")],
                    "path": path,
                    "expected": True,
                }
            )
    return artifacts


def _metric_names(options):
    names = [
        "chamfer",
        "cd_sample_to_reference",
        "cd_reference_to_sample",
        "hausdorff95",
        "iou",
    ]
    for threshold in options.fscore_thresholds:
        suffix = "{:g}".format(threshold)
        names.extend(
            [
                "precision@{}".format(suffix),
                "recall@{}".format(suffix),
                "fscore@{}".format(suffix),
            ]
        )
    return names


def evaluate_configuration(config_dir, options=None):
    options = options or EvaluationOptions()
    if options.num_points < 1:
        raise ValueError("num_points must be positive")
    if options.workers < 1:
        raise ValueError("workers must be positive")
    if options.jsd_resolution < 2:
        raise ValueError("jsd_resolution must be at least 2")
    if options.iou_batch_size < 1:
        raise ValueError("iou_batch_size must be positive")
    if not options.fscore_thresholds or any(
        threshold <= 0 for threshold in options.fscore_thresholds
    ):
        raise ValueError("F-score thresholds must be positive")

    context = resolve_configuration(config_dir)
    source_specs = context["source_specs"]
    domain = float(
        context["specs"].get(
            "ReconstructionDomain",
            source_specs.get("ReconstructionDomain", DEFAULT_DOMAIN),
        )
    )
    if domain <= 0:
        raise ValueError("ReconstructionDomain must be positive")
    warnings = []

    artifacts = _discover_artifacts(context)
    existing_artifacts = [artifact for artifact in artifacts if artifact["path"].is_file()]
    paired_evaluation = context["task"] == "modulation" or context["conditional"]
    mesh_results = _parallel_map(
        lambda artifact: load_mesh_artifact(
            artifact["path"],
            options,
            domain,
            reference_path=(
                _reference_path(
                    artifact["record"], context["data_source"]
                )
                if paired_evaluation and artifact["record"] is not None
                else None
            ),
        ),
        existing_artifacts,
        options.workers,
    )
    for artifact, mesh_result in zip(existing_artifacts, mesh_results):
        artifact["mesh"] = mesh_result

    missing_expected = [
        artifact["key"]
        for artifact in artifacts
        if artifact["expected"] and not artifact["path"].is_file()
    ]
    # Unconditional generations are a free sample set, not one reconstruction
    # per TestSplit object. Conditional and stage-one discovery add explicit
    # placeholders above for records that should have an output.
    missing_expected_count = len(missing_expected)
    if missing_expected_count:
        warnings.append(
            "{} expected reconstruction artifact(s) are missing".format(
                missing_expected_count
            )
        )
    invalid_meshes = [
        item["path"] for item in mesh_results if item["points"] is None
    ]
    if invalid_meshes:
        warnings.append("{} mesh artifact(s) are invalid".format(len(invalid_meshes)))
    iou_failures = [
        item
        for item in mesh_results
        if paired_evaluation
        and item["points"] is not None
        and item["occupancy"] is None
    ]
    if iou_failures:
        warnings.append(
            "{} paired reconstruction(s) have no valid IoU".format(
                len(iou_failures)
            )
        )

    reference_results = _parallel_map(
        lambda record: load_reference_points(record, context["data_source"], options),
        context["records"],
        options.workers,
    )
    references = {
        item["key"]: item["points"]
        for item in reference_results
        if item["points"] is not None
    }
    missing_references = [
        item["key"] for item in reference_results if item["points"] is None
    ]
    if missing_references:
        warnings.append(
            "{} test reference(s) could not be loaded".format(len(missing_references))
        )

    rows = []
    paired_rows = []
    for artifact in artifacts:
        path_exists = artifact["path"].is_file()
        mesh_result = artifact.get("mesh")
        row = {
            "key": artifact["key"],
            "sample": artifact["sample"],
            "path": str(artifact["path"]),
            "status": (
                "missing"
                if not path_exists
                else "invalid"
                if mesh_result is None or mesh_result["points"] is None
                else "valid"
            ),
            "error": mesh_result["error"] if mesh_result is not None else None,
        }
        if mesh_result is not None:
            row.update(
                {"mesh_{}".format(key): value for key, value in mesh_result["health"].items()}
            )
            row["iou_error"] = mesh_result["iou_error"]
            if mesh_result["occupancy"] is not None:
                row.update(mesh_result["occupancy"])
        if (
            mesh_result is not None
            and mesh_result["points"] is not None
            and artifact["key"] in references
            and (context["task"] == "modulation" or context["conditional"])
        ):
            metrics = paired_surface_metrics(
                mesh_result["points"],
                references[artifact["key"]],
                options.fscore_thresholds,
            )
            row.update(metrics)
            paired_rows.append(row)
        rows.append(row)

    valid_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("mesh", {}).get("points") is not None
    ]
    sample_clouds = [artifact["mesh"]["points"] for artifact in valid_artifacts]
    reference_clouds = list(references.values())

    modulation_summary = validate_modulations(context)
    if modulation_summary["missing"]:
        warnings.append(
            "{} canonical COD modulation artifact(s) are missing".format(
                len(modulation_summary["missing"])
            )
        )
    if modulation_summary["invalid"]:
        warnings.append(
            "{} COD modulation artifact(s) are invalid".format(
                len(modulation_summary["invalid"])
            )
        )
    latent_statistics = modulation_summary["latent_statistics"]
    if latent_statistics is not None and not latent_statistics["valid"]:
        warnings.append("COD latent statistics are invalid")

    summary = {
        "schema_version": 3,
        "metric_definitions": {
            "chamfer": CHAMFER_DISTANCE_DEFINITION,
        },
        "configuration": {
            "path": str(context["config_dir"]),
            "description": context["specs"].get("Description"),
            "training_task": context["task"],
            "conditional": context["conditional"],
            "conditioning": context["specs"].get("conditioning"),
            "test_split": str(context["split_path"]),
            "data_source": (
                str(context["data_source"]) if context["data_source"] is not None else None
            ),
            "data_format": "cod_sdf.npz",
        },
        "parameters": {
            "num_points": options.num_points,
            "seed": options.seed,
            "workers": options.workers,
            "jsd_resolution": options.jsd_resolution,
            "fscore_thresholds": list(options.fscore_thresholds),
            "iou_batch_size": options.iou_batch_size,
            "reconstruction_domain": domain,
        },
        "coverage": {
            "expected_test_instances": len(context["records"]),
            "discovered_reconstructions": len(existing_artifacts),
            "valid_reconstructions": len(valid_artifacts),
            "missing_expected_reconstructions": missing_expected_count,
            "usable_references": len(references),
            "missing_references": len(missing_references),
            "valid_iou_comparisons": sum(
                item["occupancy"] is not None for item in mesh_results
            ),
            "missing_iou_comparisons": len(iou_failures),
            "partial": bool(
                missing_expected_count
                or missing_references
                or invalid_meshes
                or iou_failures
                or modulation_summary["missing"]
                or modulation_summary["invalid"]
                or (
                    latent_statistics is not None
                    and not latent_statistics["valid"]
                )
            ),
        },
        "mesh_health": _mesh_health_summary(mesh_results),
        "modulations": modulation_summary,
        "warnings": warnings,
    }

    if paired_rows:
        summary["paired_fidelity"] = _aggregate_metric_rows(
            paired_rows, _metric_names(options)
        )

    if context["conditional"]:
        grouped_rows = {}
        for row in paired_rows:
            grouped_rows.setdefault(row["key"], []).append(row)
        conditional_rows = []
        diversity_values = []
        for key, group in sorted(grouped_rows.items()):
            condition = {"key": key, "sample_count": len(group)}
            for metric_name in _metric_names(options):
                values = [
                    float(row[metric_name])
                    for row in group
                    if row.get(metric_name) is not None
                ]
                if not values:
                    continue
                condition["mean_{}".format(metric_name)] = float(np.mean(values))
                if (
                    metric_name == "chamfer"
                    or metric_name == "hausdorff95"
                    or metric_name.startswith("cd_")
                ):
                    condition["best_{}".format(metric_name)] = float(np.min(values))
                else:
                    condition["best_{}".format(metric_name)] = float(np.max(values))

            clouds = [
                artifact["mesh"]["points"]
                for artifact in valid_artifacts
                if artifact["key"] == key
            ]
            if len(clouds) >= 2:
                pairwise = pairwise_chamfer(
                    clouds, clouds, workers=options.workers, symmetric=True
                )
                upper = pairwise[np.triu_indices(len(clouds), 1)]
                condition["diversity_pairwise_chamfer"] = float(np.mean(upper))
                diversity_values.append(condition["diversity_pairwise_chamfer"])
            conditional_rows.append(condition)
        summary["conditional"] = {
            "conditions_with_samples": len(grouped_rows),
            "missing_conditions": len(context["records"]) - len(grouped_rows),
            "samples_per_condition": {
                "min": min((len(group) for group in grouped_rows.values()), default=0),
                "median": (
                    float(np.median([len(group) for group in grouped_rows.values()]))
                    if grouped_rows
                    else 0
                ),
                "max": max((len(group) for group in grouped_rows.values()), default=0),
            },
            "mean_diversity_pairwise_chamfer": (
                float(np.mean(diversity_values)) if diversity_values else None
            ),
            "per_condition": conditional_rows,
        }

    if context["task"] in ("diffusion", "combined"):
        summary["distribution"] = distribution_metrics(
            sample_clouds, reference_clouds, options, domain
        )
        if summary["distribution"] is None:
            warnings.append(
                "distribution metrics require at least one valid sample and reference"
            )

    summary = _json_ready(summary)
    if not valid_artifacts:
        raise RuntimeError("no usable reconstruction artifacts were found")
    if not references:
        raise RuntimeError("no usable ground-truth references were found")
    return summary, rows


def write_results(summary, rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    temporary_summary = output_dir / ".summary.json.tmp"
    with temporary_summary.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_summary, summary_path)

    per_shape_path = output_dir / "per_shape.csv"
    temporary_csv = output_dir / ".per_shape.csv.tmp"
    fieldnames = sorted({key for row in rows for key in row})
    with temporary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, per_shape_path)
    return summary_path, per_shape_path


def print_summary(summary, summary_path=None):
    coverage = summary["coverage"]
    print("Configuration: {}".format(summary["configuration"]["path"]))
    print(
        "Reconstructions: {}/{} valid ({} missing expected)".format(
            coverage["valid_reconstructions"],
            coverage["discovered_reconstructions"],
            coverage["missing_expected_reconstructions"],
        )
    )
    print(
        "References: {}/{} usable".format(
            coverage["usable_references"], coverage["expected_test_instances"]
        )
    )
    if "paired_fidelity" in summary:
        chamfer = summary["paired_fidelity"]["chamfer"]
        print(
            "Paired Chamfer (unsquared L2 CD): "
            "mean={:.6g}, median={:.6g}, p95={:.6g}".format(
                chamfer["mean"], chamfer["median"], chamfer["p95"]
            )
        )
        if "iou" in summary["paired_fidelity"]:
            iou = summary["paired_fidelity"]["iou"]
            print(
                "Paired IoU: mean={:.6g}, median={:.6g}, p95={:.6g}".format(
                    iou["mean"], iou["median"], iou["p95"]
                )
            )
    if summary.get("distribution"):
        distribution = summary["distribution"]
        print(
            "Distribution (unsquared L2 CD): "
            "MMD-CD={:.6g}, COV-CD={:.3f}, 1-NN={:.3f}, JSD={:.6g}".format(
                distribution["mmd_cd"],
                distribution["coverage_cd"],
                distribution["one_nn_cd_accuracy"],
                distribution["jsd"],
            )
        )
    if summary["warnings"]:
        print("Partial/warnings:")
        for warning in summary["warnings"]:
            print("  - {}".format(warning))
    if summary_path is not None:
        print("Wrote {}".format(summary_path))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate stored COD Diffusion-SDF reconstructions and native COD "
            "modulations without loading checkpoints."
        )
    )
    parser.add_argument("config_dir", help="configuration directory containing specs.json")
    parser.add_argument(
        "--output-dir",
        help="result directory (default: CONFIG_DIR/evaluation)",
    )
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--jsd-resolution", type=int, default=DEFAULT_JSD_RESOLUTION
    )
    parser.add_argument(
        "--fscore-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_FSCORE_THRESHOLDS),
    )
    parser.add_argument(
        "--iou-batch-size",
        type=int,
        default=DEFAULT_IOU_BATCH_SIZE,
        help="number of COD uniform query points ray-cast per batch",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    options = EvaluationOptions(
        num_points=args.num_points,
        seed=args.seed,
        workers=args.workers,
        jsd_resolution=args.jsd_resolution,
        fscore_thresholds=tuple(args.fscore_thresholds),
        iou_batch_size=args.iou_batch_size,
    )
    try:
        summary, rows = evaluate_configuration(args.config_dir, options)
        output_dir = args.output_dir or str(
            Path(args.config_dir).expanduser().resolve() / "evaluation"
        )
        summary_path, _ = write_results(summary, rows, output_dir)
        print_summary(summary, summary_path)
        return 0
    except Exception as exc:
        print("Evaluation failed: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
