"""Transformer EDM operating directly on native COD latent tokens."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.archs.condition_encoders import ConditionEncoderSet


def latent_set_distance_matrix(
    left,
    right=None,
    *,
    metric="sliced_wasserstein",
    mmd_bandwidth=1.0,
    projections=None,
):
    """Differentiable, permutation-invariant distances between latent sets.

    ``left`` and ``right`` contain batches of sets shaped ``[B, M, D]``.  The
    returned matrix has one distance for every pair of sets.  Sliced
    Wasserstein is the default because it is scale-stable and avoids assigning
    COD tokens to one another.  RBF MMD is available for ablations.
    """
    right = left if right is None else right
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("latent sets must have shape [batch, tokens, channels]")
    if left.shape[1:] != right.shape[1:]:
        raise ValueError(
            "latent-set distance requires equal token and channel dimensions, "
            f"got {tuple(left.shape[1:])} and {tuple(right.shape[1:])}"
        )

    if metric == "sliced_wasserstein":
        if projections is None:
            projections = torch.eye(
                left.shape[-1], device=left.device, dtype=left.dtype
            )
        projections = F.normalize(projections.to(left), dim=-1)
        left_sorted = torch.sort(left @ projections.T, dim=1).values
        right_sorted = torch.sort(right @ projections.T, dim=1).values
        squared = (
            left_sorted[:, None] - right_sorted[None, :]
        ).square().mean(dim=(-1, -2))
        # Self-distances lie exactly on zero.  sqrt has an infinite derivative
        # there, which can contaminate gradients even when callers later mask
        # the matrix diagonal (zero upstream gradient times infinity is NaN).
        return squared.clamp_min(1e-12).sqrt()

    if metric == "mmd":
        bandwidth = float(mmd_bandwidth)
        if bandwidth <= 0:
            raise ValueError("mmd_bandwidth must be positive")

        def kernel_mean(first, second):
            squared = torch.cdist(first.float(), second.float()).square()
            return torch.exp(-squared / (2.0 * bandwidth**2)).mean(dim=(-1, -2))

        left_self = kernel_mean(left, left)
        right_self = kernel_mean(right, right)
        batch_left, batch_right = left.shape[0], right.shape[0]
        expanded_left = left[:, None].expand(-1, batch_right, -1, -1)
        expanded_right = right[None, :].expand(batch_left, -1, -1, -1)
        cross = kernel_mean(
            expanded_left.reshape(-1, left.shape[1], left.shape[2]),
            expanded_right.reshape(-1, right.shape[1], right.shape[2]),
        ).reshape(batch_left, batch_right)
        mmd_squared = left_self[:, None] + right_self[None, :] - 2.0 * cross
        return mmd_squared.clamp_min(1e-12).sqrt()

    raise ValueError(f"unknown latent-set distance: {metric}")


def zero_module(module):
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class PositionalEmbedding(nn.Module):
    """Continuous EDM noise embedding used by the VecSet reference model."""

    def __init__(self, channels, max_positions=10000):
        super().__init__()
        self.channels = channels
        self.max_positions = max_positions

    def forward(self, value):
        frequencies = torch.arange(
            self.channels // 2, dtype=torch.float32, device=value.device
        )
        frequencies = frequencies / (self.channels // 2)
        frequencies = (1 / self.max_positions) ** frequencies
        embedding = value.float().ger(frequencies)
        return torch.cat((embedding.cos(), embedding.sin()), dim=1)


class AdaLayerNorm(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.affine = nn.Linear(dimension, dimension * 2)

    def forward(self, value, time_embedding):
        scale, shift = self.affine(time_embedding).chunk(2, dim=-1)
        return self.norm(value) * (1 + scale) + shift


class CrossAttention(nn.Module):
    """Attention layer matching the VecSet EDM transformer parameterization."""

    def __init__(self, query_dim, context_dim=None, heads=8, head_dim=64, dropout=0):
        super().__init__()
        context_dim = context_dim or query_dim
        inner_dim = heads * head_dim
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, value, context=None):
        context = value if context is None else context
        batch, query_count, _ = value.shape
        key_count = context.shape[1]
        query = self.to_q(value).view(batch, query_count, self.heads, -1).transpose(1, 2)
        key = self.to_k(context).view(batch, key_count, self.heads, -1).transpose(1, 2)
        content = self.to_v(context).view(batch, key_count, self.heads, -1).transpose(1, 2)
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        result = torch.matmul(attention.softmax(dim=-1), content)
        result = result.transpose(1, 2).reshape(batch, query_count, -1)
        return self.to_out(result)


class GEGLU(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.projection = nn.Linear(input_dimension, output_dimension * 2)

    def forward(self, value):
        value, gate = self.projection(value).chunk(2, dim=-1)
        return value * F.gelu(gate)


class EDMTransformerBlock(nn.Module):
    def __init__(self, dimension, heads, mlp_ratio=4, dropout=0, conditional=False):
        super().__init__()
        self.norm_self = AdaLayerNorm(dimension)
        head_dim = dimension // heads
        self.self_attention = CrossAttention(
            dimension, heads=heads, head_dim=head_dim, dropout=dropout
        )
        self.conditional = conditional
        if conditional:
            self.norm_cross = AdaLayerNorm(dimension)
            self.cross_attention = CrossAttention(
                dimension, heads=heads, head_dim=head_dim, dropout=dropout
            )
        self.norm_mlp = AdaLayerNorm(dimension)
        hidden = int(dimension * mlp_ratio)
        self.mlp = nn.Sequential(
            GEGLU(dimension, hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
        )

    def forward(self, value, time_embedding, context=None):
        normalized = self.norm_self(value, time_embedding)
        value = value + self.self_attention(normalized)
        if self.conditional and context is not None:
            normalized = self.norm_cross(value, time_embedding)
            value = value + self.cross_attention(normalized, context)
        return value + self.mlp(self.norm_mlp(value, time_embedding))


class CODLatentTransformer(nn.Module):
    """VecSet-style non-causal transformer preserving [B,tokens,channels]."""

    def __init__(
        self,
        latent_tokens=32,
        latent_dimension=32,
        width=512,
        depth=12,
        heads=8,
        mlp_ratio=4,
        dropout=0,
        cond=False,
        condition_dim=128,
        condition_encoders=None,
        use_learnable_slot_embeddings=False,
        **_,
    ):
        super().__init__()
        if width % heads:
            raise ValueError(
                f"transformer width ({width}) must be divisible by heads ({heads})"
            )
        self.latent_dimension = latent_dimension
        self.latent_tokens = int(latent_tokens)
        self.width = width
        self.conditional = bool(cond)
        self.input_projection = nn.Linear(latent_dimension, width, bias=False)
        self.slot_embedding = (
            nn.Parameter(torch.empty(1, self.latent_tokens, width))
            if use_learnable_slot_embeddings
            else None
        )
        if self.slot_embedding is not None:
            nn.init.normal_(self.slot_embedding, mean=0.0, std=0.02)
        self.noise_embedding = PositionalEmbedding(256)
        self.noise_mlp = nn.Sequential(
            nn.Linear(256, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            EDMTransformerBlock(
                width, heads, mlp_ratio=mlp_ratio, dropout=dropout,
                conditional=self.conditional,
            )
            for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = zero_module(
            nn.Linear(width, latent_dimension, bias=False)
        )
        if self.conditional:
            self.condition_encoder = ConditionEncoderSet(
                condition_dim=condition_dim,
                condition_encoders=condition_encoders,
            )
            self.condition_projection = (
                nn.Identity()
                if condition_dim == width
                else nn.Linear(condition_dim, width)
            )

    def forward(self, latent, noise, conditioning=None):
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dimension:
            raise ValueError(
                "COD transformer input must be [B,M,latent_dimension], "
                f"got {tuple(latent.shape)}"
            )
        time = self.noise_mlp(self.noise_embedding(noise)).unsqueeze(1)
        value = self.input_projection(latent)
        if self.slot_embedding is not None:
            if latent.shape[1] != self.latent_tokens:
                raise ValueError(
                    "slot-aware COD transformer expected "
                    f"{self.latent_tokens} tokens, got {latent.shape[1]}"
                )
            value = value + self.slot_embedding
        context = None
        if self.conditional and conditioning is not None:
            context = self.condition_projection(self.condition_encoder(conditioning))
        for block in self.blocks:
            value = block(value, time, context)
        return self.output_projection(self.output_norm(value))


class EDMLatentDiffusion(nn.Module):
    """EDM preconditioning, loss, clean estimate, and Heun sampler."""

    def __init__(self, model_specs, diffusion_specs):
        super().__init__()
        self.latent_tokens = int(model_specs.get("latent_tokens", 32))
        self.latent_dimension = int(model_specs.get("latent_dimension", 32))
        model_specs = dict(model_specs)
        model_specs.setdefault("latent_dimension", self.latent_dimension)
        self.model = CODLatentTransformer(**model_specs)

        self.sigma_data = float(diffusion_specs.get("sigma_data", 1.0))
        self.p_mean = float(diffusion_specs.get("P_mean", -1.2))
        self.p_std = float(diffusion_specs.get("P_std", 1.2))
        self.sigma_min = float(diffusion_specs.get("sigma_min", 0.002))
        self.sigma_max = float(diffusion_specs.get("sigma_max", 80.0))
        self.rho = float(diffusion_specs.get("rho", 7.0))
        self.sampling_steps = int(diffusion_specs.get("sampling_steps", 18))

    def forward(self, noisy, sigma, conditioning=None):
        sigma = torch.as_tensor(sigma, device=noisy.device, dtype=torch.float32)
        if sigma.ndim == 0:
            sigma = sigma.expand(noisy.shape[0])
        sigma_view = sigma.reshape(-1, 1, 1)
        sigma_data = self.sigma_data
        c_skip = sigma_data ** 2 / (sigma_view.square() + sigma_data ** 2)
        c_out = sigma_view * sigma_data / (
            sigma_view.square() + sigma_data ** 2
        ).sqrt()
        c_in = 1 / (sigma_data ** 2 + sigma_view.square()).sqrt()
        c_noise = sigma.clamp_min(1e-12).log() / 4
        residual = self.model(c_in * noisy.float(), c_noise, conditioning)
        return c_skip * noisy + c_out * residual.float()

    def training_loss(self, clean, conditioning=None, noise=None, sigma=None):
        if clean.ndim != 3:
            raise ValueError(f"COD latents must be [B,M,D], got {tuple(clean.shape)}")
        if sigma is None:
            rnd = torch.randn(clean.shape[0], device=clean.device)
            sigma = (rnd * self.p_std + self.p_mean).exp()
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = clean + noise * sigma.reshape(-1, 1, 1)
        denoised = self(noisy, sigma, conditioning)
        weight = (sigma.square() + self.sigma_data ** 2) / (
            sigma * self.sigma_data
        ).square()
        per_element = weight.reshape(-1, 1, 1) * (denoised - clean).square()
        return per_element.mean(), denoised, noisy, sigma

    @staticmethod
    def _expand_conditioning(conditioning, batch_size):
        if conditioning is None:
            return None

        is_mapping = isinstance(conditioning, dict)
        values = conditioning if is_mapping else {"conditioning": conditioning}
        expanded = {}
        for name, value in values.items():
            if not torch.is_tensor(value) or value.ndim == 0:
                raise TypeError(
                    f"condition '{name}' must be a tensor with a batch dimension"
                )
            if value.shape[0] == batch_size:
                expanded[name] = value
            elif value.shape[0] == 1:
                expanded[name] = value.expand(batch_size, *value.shape[1:])
            else:
                raise ValueError(
                    f"condition '{name}' has batch size {value.shape[0]}, "
                    f"but sampling requested {batch_size} samples"
                )
        return expanded if is_mapping else expanded["conditioning"]

    @torch.no_grad()
    def sample(self, batch_size, conditioning=None, noise=None, num_steps=None):
        device = next(self.parameters()).device
        conditioning = self._expand_conditioning(conditioning, batch_size)
        steps = int(num_steps or self.sampling_steps)
        if steps < 2:
            raise ValueError("EDM sampling requires at least two steps")
        indices = torch.arange(steps, dtype=torch.float64, device=device)
        t_steps = (
            self.sigma_max ** (1 / self.rho)
            + indices / (steps - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat((t_steps, t_steps.new_zeros(1)))
        if noise is None:
            noise = torch.randn(
                batch_size, self.latent_tokens, self.latent_dimension, device=device
            )
        value = noise.to(torch.float64) * t_steps[0]
        for index, (current, following) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            denoised = self(value.float(), current.float(), conditioning).to(torch.float64)
            derivative = (value - denoised) / current
            proposal = value + (following - current) * derivative
            if index < steps - 1:
                corrected = self(
                    proposal.float(), following.float(), conditioning
                ).to(torch.float64)
                next_derivative = (proposal - corrected) / following
                proposal = value + (following - current) * (
                    0.5 * derivative + 0.5 * next_derivative
                )
            value = proposal
        return value.float()


# The old public name is retained only for checkpoint/harness call sites.
DiffusionModel = EDMLatentDiffusion
