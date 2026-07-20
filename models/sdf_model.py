#!/usr/bin/env python3

"""COD-VAE tri-planes followed by the Diffusion-SDF neural SDF head."""

from collections import OrderedDict

import torch
from torch import nn

from models.archs.sdf_decoder import SdfDecoder
from models.cod_vae import CompactLatentVAE
from models.cod_vae.checkpoint import load_cod_checkpoint


DEFAULT_COD_ENCODER = {
    "num_patches": 512,
    "num_blocks": 4,
    "num_layers_per_block": 3,
}
DEFAULT_COD_DECODER = {
    "output_resolution": 128,
    "output_patch_size": 8,
    "num_layers": 12,
    "num_init_layers": 1,
    "mlp_ratio": 2,
    # SDF adaptation refines all spatial tokens by default.
    "keep_ratio": 1.0,
    # Retained for strict official-checkpoint keys; full_sdf/disabled bypass it.
    "num_merged_tokens": 8,
    "use_conv_refine": False,
    "uncertainty_mode": "full_sdf",
}


class SdfModel(nn.Module):
    """Checkpoint-compatible COD-VAE representation with an SDF-only head."""

    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        cod_specs = dict(specs.get("CODVaeSpecs", {}))
        sdf_specs = dict(specs.get("SdfModelSpecs", {}))

        encoder_params = {**DEFAULT_COD_ENCODER, **cod_specs.get("encoder_params", {})}
        decoder_params = {**DEFAULT_COD_DECODER, **cod_specs.get("decoder_params", {})}
        uncertainty_mode = cod_specs.get("uncertainty_mode")
        if uncertainty_mode is None:
            uncertainty_mode = (
                "full_sdf"
                if cod_specs.get("disable_uncertainty_pruning", True)
                else "occupancy_pruning"
            )
        decoder_params["uncertainty_mode"] = uncertainty_mode
        if uncertainty_mode in {"full_sdf", "disabled"}:
            decoder_params["keep_ratio"] = 1.0

        self.latent_tokens = int(cod_specs.get("latent_tokens", 32))
        self.latent_dimension = int(cod_specs.get("latent_dimension", 32))
        self.triplane_dimension = int(cod_specs.get("triplane_dimension", 32))
        embed_dim = int(cod_specs.get("embed_dimension", 512))
        if decoder_params.get("query_dim") is not None:
            self.triplane_dimension = int(decoder_params.pop("query_dim"))

        self.cod_vae = CompactLatentVAE(
            num_latent_layers=int(cod_specs.get("latent_decoder_layers", 12)),
            latent_dim=self.latent_dimension,
            output_dim=1,  # Occupancy head exists only for official checkpoint keys.
            num_latents=self.latent_tokens,
            embed_dim=embed_dim,
            query_dim=self.triplane_dimension,
            use_learnable_pos=bool(cod_specs.get("use_learnable_positions", False)),
            encoder_params=encoder_params,
            decoder_params=decoder_params,
            mlp_ratio=float(cod_specs.get("latent_mlp_ratio", 4.0)),
            num_heads=int(cod_specs.get("num_heads", 8)),
            dropout=float(cod_specs.get("dropout", 0.1)),
        )

        sdf_feature_dim = int(sdf_specs.get("feature_dim", self.triplane_dimension))
        self.feature_adapter = (
            nn.Identity()
            if sdf_feature_dim == self.triplane_dimension
            else nn.Linear(self.triplane_dimension, sdf_feature_dim)
        )
        self.sdf_decoder = SdfDecoder(
            latent_size=sdf_feature_dim,
            hidden_dim=int(sdf_specs.get("hidden_dim", 512)),
            skip_connection=bool(sdf_specs.get("skip_connection", True)),
            tanh_act=bool(sdf_specs.get("tanh_act", False)),
        )

        checkpoint_path = cod_specs.get("checkpoint_path")
        if checkpoint_path:
            load_cod_checkpoint(
                self.cod_vae,
                checkpoint_path,
                strict=bool(cod_specs.get("strict_checkpoint_loading", True)),
            )

    def encode_surface(self, surface_points, sample_posterior=True):
        encoded = self.cod_vae.encode_embed(surface_points)
        latent, posterior = self.cod_vae.encode_latents(encoded)
        if posterior is not None and not sample_posterior:
            latent = posterior.mode()
        return latent, posterior, encoded

    def decode_latent(self, latent):
        decoded_latent = self.cod_vae.decode_latents(latent)
        planes, initial_planes, uncertainty = self.cod_vae.autoencoder.decoder.decode(decoded_latent)
        return {
            "planes": planes,
            "initial_planes": initial_planes,
            "uncertainty": uncertainty,
            "decoded_latent": decoded_latent,
        }

    def query_sdf(self, planes, query_points):
        # The official sampler clamps in place; preserve caller-owned query tensors.
        query_features = self.cod_vae.autoencoder.decoder.decode_queries(
            planes, query_points.clone()
        )
        query_features = self.feature_adapter(query_features)
        return self.sdf_decoder(torch.cat((query_points, query_features), dim=-1)).squeeze(-1)

    def query_uncertainty(self, uncertainty_planes, query_points):
        value = self.cod_vae.autoencoder.decoder.decode_uncertainty(
            uncertainty_planes, query_points.clone()
        )
        return value.squeeze(-1)

    def forward(
        self,
        surface_points,
        query_points,
        sample_posterior=True,
        return_initial_sdf=False,
        return_query_uncertainty=False,
    ):
        latent, posterior, encoded = self.encode_surface(surface_points, sample_posterior)
        decoded = self.decode_latent(latent)
        decoded.update(
            latent=latent,
            posterior=posterior,
            encoded_features=encoded,
            sdf=self.query_sdf(decoded["planes"], query_points),
        )
        if return_initial_sdf:
            decoded["initial_sdf"] = self.query_sdf(
                decoded["initial_planes"], query_points
            )
        if return_query_uncertainty:
            if decoded["uncertainty"] is None:
                raise ValueError("uncertainty queries are unavailable in disabled mode")
            decoded["query_uncertainty"] = self.query_uncertainty(
                decoded["uncertainty"], query_points
            )
        return decoded

    def component_modules(self):
        ae = self.cod_vae.autoencoder
        return OrderedDict(
            point_encoder=nn.ModuleList([ae.point_embed, ae.norm_latent, ae.encoder]),
            variational_block=self.cod_vae.latent_proj_in,
            latent_decoder=nn.ModuleList([self.cod_vae.latent_proj_out, self.cod_vae.latent_decoder]),
            triplane_decoder=ae.decoder,
            sdf_network=nn.ModuleList([self.feature_adapter, self.sdf_decoder]),
        )

    def set_trainable_components(self, names):
        names = set(names)
        components = self.component_modules()
        unknown = names.difference(components)
        if unknown:
            raise ValueError(f"unknown trainable COD components: {sorted(unknown)}")
        # Freeze unlisted parameters as well, including the checkpoint-only
        # occupancy head, before selectively enabling the SDF pipeline.
        self.requires_grad_(False)
        for name, module in components.items():
            module.requires_grad_(name in names)
        decoder = self.cod_vae.autoencoder.decoder
        if decoder.uncertainty_mode == "disabled":
            decoder.uncertainty_out.requires_grad_(False)
            if decoder.merging_module is not None:
                decoder.merging_module.requires_grad_(False)
        self._frozen_component_names = set(components).difference(names)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for name, module in self.component_modules().items():
                if name in getattr(self, "_frozen_component_names", set()):
                    module.eval()
            decoder = self.cod_vae.autoencoder.decoder
            if decoder.uncertainty_mode == "disabled":
                decoder.uncertainty_out.eval()
                if decoder.merging_module is not None:
                    decoder.merging_module.eval()
        return self

    # Compatibility with reconstruction helpers, now operating on tri-planes.
    def forward_with_plane_features(self, plane_features, xyz):
        if plane_features.ndim == 4:
            batch, channels, height, width = plane_features.shape
            if channels != 3 * self.triplane_dimension:
                raise ValueError("flattened planes must have 3 * triplane_dimension channels")
            plane_features = plane_features.view(
                batch, 3, self.triplane_dimension, height, width
            )
        return self.query_sdf(plane_features, xyz)
