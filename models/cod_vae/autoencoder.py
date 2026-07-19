import functools
from typing import List, Tuple

import torch
from torch import nn

from . import pointops
from .base import BaseAutoencoder
from .modules.pos import PointEmbed
from .modules.transformer import init_embedding
from .networks import CompactPointPatchEncoder, CompactTriplaneDecoder


class CompactLatentAutoencoder(BaseAutoencoder):

    def __init__(self,
                 output_dim: int,
                 num_latents: int,
                 embed_dim: int,
                 query_dim: int,
                 encoder_params: dict,
                 decoder_params: dict,
                 use_learnable_pos: bool = False,
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.query_dim = query_dim

        self.use_learnable_pos = use_learnable_pos

        ## layers
        self.point_embed = PointEmbed(dim=embed_dim)
        self.norm_latent = nn.LayerNorm(embed_dim)
        self.encoder = CompactPointPatchEncoder(embed_dim=embed_dim, **encoder_params)
        self.decoder = CompactTriplaneDecoder(embed_dim=embed_dim, query_dim=query_dim, **decoder_params)
        self.head = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, output_dim),
        )

        self.latent_pos = nn.Parameter(torch.empty(num_latents, embed_dim)) if use_learnable_pos else None
        self.reset_parameters()

    def reset_parameters(self):
        scale = self.embed_dim ** -0.5
        if self.latent_pos is not None:
            init_embedding(self.latent_pos, scale)

    def forward(self, pc, queries):
        z, posterior = self.encode(pc)
        patches, init_patches, uncertainty, z_recon = self.decode(z)
        out = self.decode_queries(patches, queries)

        outputs = {
            'logits': out,
        }
        if (init_patches is not None) and self.training:
            init_out = self.decode_queries(init_patches, queries)
            outputs['init_out'] = init_out
        if uncertainty is not None:
            uncertainty_queries = self.decoder.decode_uncertainty(uncertainty, queries)
            outputs['uncertainty'] = uncertainty_queries
            outputs['uncertainty_planes'] = uncertainty

        return outputs

    def encode_embed(self, pc):
        z = self._get_init_z(pc)
        z = self.norm_latent(z)
        return self.encoder(pc, z, self.point_embed)

    def encode_latents(self, z):
        ### autoencoder model without channel compression
        return z, None

    def decode_latents(self, z):
        ### autoencoder model without channel compression
        return z

    def decode_embed(self, z):
        context, init_context, uncertainty = self.decoder.decode(z)
        return context, init_context, uncertainty, z

    def decode_queries(self, context, queries):
        query_features = self.decoder.decode_queries(context, queries)
        out = self.head(query_features).squeeze(-1)
        return out

    def _get_init_z(self, pc):
        if self.latent_pos is None:
            z = pointops.fps(pc, self.num_latents)
            z = self.point_embed(z)
        else:
            z = self.latent_pos.unsqueeze(0).expand(pc.size(0), -1, -1)
        return z
