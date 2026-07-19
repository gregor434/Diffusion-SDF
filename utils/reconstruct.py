"""Reconstruction filtering shared by modulation extraction."""

import trimesh

from utils import evaluate


def filter_threshold(mesh_path, ground_truth_points, threshold):
    reconstructed = trimesh.load(mesh_path + ".ply")
    reconstructed_points, _ = trimesh.sample.sample_surface(
        reconstructed, ground_truth_points.shape[-2]
    )
    chamfer = evaluate.calc_cd(ground_truth_points, reconstructed_points)
    return chamfer <= threshold

