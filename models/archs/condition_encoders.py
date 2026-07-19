import torch
from torch import nn

class ConditionEncoder(nn.Module):
    def forward(self, condition):
        raise NotImplementedError


class PointCloudConditionEncoder(ConditionEncoder):
    def __init__(self, condition_dim, hidden_dim=128, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, condition_dim),
        )

    def forward(self, point_cloud):
        if point_cloud.ndim != 3 or point_cloud.shape[-1] != 3:
            raise ValueError(
                f"point-cloud condition must be [B,N,3], got {tuple(point_cloud.shape)}"
            )
        return self.encoder(point_cloud)


class ImageConditionEncoder(ConditionEncoder):
    def __init__(self, condition_dim, clip_feature_dim=512, **kwargs):
        super().__init__()
        self.clip_feature_dim = clip_feature_dim
        if clip_feature_dim == condition_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(clip_feature_dim, condition_dim)

    def forward(self, image_features):
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
        if image_features.dim() != 3:
            raise ValueError(
                f"image condition must have shape [B, 1, {self.clip_feature_dim}], got {tuple(image_features.shape)}"
            )
        if image_features.shape[-1] != self.clip_feature_dim:
            raise ValueError(
                f"image condition last dimension must be {self.clip_feature_dim}, got {image_features.shape[-1]}"
            )
        return self.proj(image_features)


class ConditionEncoderSet(nn.Module):
    def __init__(self, condition_dim, condition_encoders=None):
        super().__init__()
        if condition_encoders is None:
            condition_encoders = [{"type": "point_cloud"}]

        self.condition_dim = condition_dim
        self.encoders = nn.ModuleDict()
        for encoder_config in condition_encoders:
            encoder_config = dict(encoder_config)
            encoder_type = encoder_config.pop("type")
            self.encoders[encoder_type] = self._make_encoder(encoder_type, encoder_config)

    def _make_encoder(self, encoder_type, encoder_config):
        if encoder_type == "point_cloud":
            return PointCloudConditionEncoder(self.condition_dim, **encoder_config)
        if encoder_type == "image":
            return ImageConditionEncoder(self.condition_dim, **encoder_config)
        if encoder_type in ("text", "multiview_image"):
            raise NotImplementedError(f"{encoder_type} condition encoder is not implemented yet")
        raise ValueError(f"unknown condition encoder type: {encoder_type}")

    def forward(self, conditioning):
        if torch.is_tensor(conditioning):
            conditioning = {"point_cloud": conditioning}
        if not isinstance(conditioning, dict):
            raise TypeError("conditioning must be a tensor or a dict of modality tensors")

        tokens = []
        for name, encoder in self.encoders.items():
            condition = conditioning.get(name)
            if condition is not None:
                tokens.append(encoder(condition))

        if not tokens:
            available = ", ".join(conditioning.keys())
            expected = ", ".join(self.encoders.keys())
            raise ValueError(f"no configured condition encoders matched conditioning keys [{available}]; expected one of [{expected}]")

        return torch.cat(tokens, dim=1)
