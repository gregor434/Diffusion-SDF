from abc import ABC, abstractmethod

from torch import nn

class BaseAutoencoder(nn.Module, ABC):
    embed_dim: int = -1

    def encode(self, pc):
        z = self.encode_embed(pc)
        return self.encode_latents(z)

    def decode(self, z):
        z = self.decode_latents(z)
        return self.decode_embed(z)

    @abstractmethod
    def encode_embed(self, pc):
        raise NotImplementedError

    @abstractmethod
    def encode_latents(self, z):
        raise NotImplementedError

    @abstractmethod
    def decode_latents(self, z):
        raise NotImplementedError

    @abstractmethod
    def decode_embed(self, z):
        raise NotImplementedError

    @abstractmethod
    def decode_queries(self, context, queries):
        raise NotImplementedError

    def load_autoencoder_weights(self, state_dict):
        # do nothing (will be implemented in VAE)
        pass
