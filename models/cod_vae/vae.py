import torch
from torch import nn
from torch.cuda.amp import autocast

from .base import BaseAutoencoder
from .autoencoder import CompactLatentAutoencoder
from .modules.blocks import StandardTransformerBlock
from .modules.kl import DiagonalGaussianDistribution


class CompactLatentVAE(BaseAutoencoder):

    def __init__(self,
                 num_latent_layers: int,
                 latent_dim: int,

                 mlp_ratio: float = 4.0,
                 num_heads: int = 8,
                 dropout: float = 0.1,

                 ## params shared with autoencoder
                 embed_dim: int = -1,
                 **autoencoder_kwargs,
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.autoencoder = CompactLatentAutoencoder(embed_dim=embed_dim, **autoencoder_kwargs)
        self.latent_proj_in = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, latent_dim * 2),
        )
        self.latent_proj_out = nn.Sequential(
            nn.Linear(latent_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.latent_decoder = StandardTransformerBlock(embed_dim=embed_dim,
                                                       num_layers=num_latent_layers,
                                                       out_dim=embed_dim,
                                                       num_heads=num_heads,
                                                       mlp_ratio=mlp_ratio,
                                                       dropout=dropout,
                                                       num_register=-1,
                                                       )

        ## when to load checkpoint?
        pass

    def load_autoencoder_weights(self, state_dict):
        self.autoencoder.load_state_dict(state_dict)
        self.autoencoder.requires_grad_(False)

    def encode_embed(self, pc):
        return self.autoencoder.encode_embed(pc)

    def decode_embed(self, z):
        return self.autoencoder.decode_embed(z)

    def decode_queries(self, context, queries):
        return self.autoencoder.decode_queries(context, queries)

    @autocast(enabled=False)
    def encode_latents(self, z):
        z = self.latent_proj_in(z.float())
        posterior = DiagonalGaussianDistribution(z)
        z = posterior.sample()

        return z, posterior

    def decode_latents(self, z):
        z = self.latent_proj_out(z)
        z = self.latent_decoder(z)
        return z
