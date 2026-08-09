"""Reconstruction filtering shared by modulation extraction and caching."""

import numpy as np
import trimesh

from utils import evaluate


def reconstruction_chamfer(mesh_path, ground_truth_points, seed=None):
    """Return deterministic squared mesh Chamfer, or infinity for an invalid mesh."""
    path = str(mesh_path)
    if not path.endswith(".ply"):
        path += ".ply"
    try:
        reconstructed = trimesh.load(path)
        if not (
            isinstance(reconstructed, trimesh.Trimesh)
            and len(reconstructed.vertices) > 0
            and len(reconstructed.faces) > 0
            and np.isfinite(reconstructed.vertices).all()
        ):
            return float("inf")
        count = int(ground_truth_points.shape[-2])
        if seed is None:
            reconstructed_points, _ = trimesh.sample.sample_surface(
                reconstructed, count
            )
        else:
            # trimesh versions supported by this project use NumPy's global RNG.
            # Preserve caller state while making cache-quality scores reproducible.
            state = np.random.get_state()
            try:
                np.random.seed(int(seed) % (2**32))
                reconstructed_points, _ = trimesh.sample.sample_surface(
                    reconstructed, count
                )
            finally:
                np.random.set_state(state)
        chamfer = float(evaluate.calc_cd(ground_truth_points, reconstructed_points))
        return chamfer if np.isfinite(chamfer) else float("inf")
    except Exception:
        return float("inf")


def filter_threshold(mesh_path, ground_truth_points, threshold, seed=None):
    return reconstruction_chamfer(mesh_path, ground_truth_points, seed) <= threshold
