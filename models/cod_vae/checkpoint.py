from pathlib import Path
from typing import Mapping

import torch


def _state_dict(checkpoint):
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, Mapping):
        return checkpoint
    raise TypeError("COD checkpoint must be a state dict or contain a 'state_dict' entry")


def load_cod_checkpoint(model, checkpoint_path, strict=True):
    """Load official COD-VAE, COD solver, or Diffusion-SDF-wrapped weights."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    source = _state_dict(checkpoint)
    prefixes = (
        "model.",
        "cod_vae.",
        "sdf_model.cod_vae.",
        "module.",
    )
    target_keys = set(model.state_dict())
    candidates = [dict(source)]
    for prefix in prefixes:
        candidates.append({key[len(prefix):]: value for key, value in source.items() if key.startswith(prefix)})
    state = max(candidates, key=lambda item: len(target_keys.intersection(item)))
    if not target_keys.intersection(state):
        raise RuntimeError(f"no COD-VAE parameters found in {checkpoint_path}")
    return model.load_state_dict(state, strict=strict)

