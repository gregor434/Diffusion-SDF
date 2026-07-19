"""Checkpoint-compatible COD-VAE architecture from join16/COD-VAE (ICCV 2025)."""

from .autoencoder import CompactLatentAutoencoder
from .vae import CompactLatentVAE

__all__ = ["CompactLatentAutoencoder", "CompactLatentVAE"]
