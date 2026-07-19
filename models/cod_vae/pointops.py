"""COD-VAE farthest-point sampling with an optional CUDA pointops fast path."""

import torch

try:  # COD-VAE's official extension (torch 2.1/CUDA); optional for CPU/tests.
    from pointops.functions import pointops as _pointops
except (ImportError, OSError):
    _pointops = None


def fps(points: torch.Tensor, count: int) -> torch.Tensor:
    """Return sampled coordinates, matching ``pointops.fps([B,N,3], count)``."""
    if _pointops is not None and points.is_cuda:
        return _pointops.fps(points, count)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"surface points must have shape [B,N,3], got {tuple(points.shape)}")
    if points.shape[1] < count:
        raise ValueError(f"COD-VAE requires at least {count} points, got {points.shape[1]}")

    batch, num_points, _ = points.shape
    centroids = torch.empty(batch, count, dtype=torch.long, device=points.device)
    distances = torch.full((batch, num_points), float("inf"), device=points.device)
    # A geometry-dependent deterministic seed avoids introducing an untracked RNG path.
    farthest = points.square().sum(dim=-1).argmax(dim=1)
    batch_indices = torch.arange(batch, device=points.device)
    for index in range(count):
        centroids[:, index] = farthest
        centroid = points[batch_indices, farthest].unsqueeze(1)
        distance = (points - centroid).square().sum(dim=-1)
        distances = torch.minimum(distances, distance)
        farthest = distances.argmax(dim=1)
    return points.gather(1, centroids.unsqueeze(-1).expand(-1, -1, 3))

