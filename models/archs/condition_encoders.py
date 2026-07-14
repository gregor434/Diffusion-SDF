import torch
from torch import nn

from diff_utils.pointnet.conv_pointnet import ConvPointnet


class ConditionEncoder(nn.Module):
    def forward(self, condition):
        raise NotImplementedError


class PointCloudConditionEncoder(ConditionEncoder):
    def __init__(self, condition_dim, **kwargs):
        super().__init__()
        self.pointnet = ConvPointnet(c_dim=condition_dim, **kwargs)

    def forward(self, point_cloud):
        return self.pointnet(point_cloud, point_cloud)


class ImageConditionEncoder(ConditionEncoder):
    def __init__(self, condition_dim, patch_size=16, in_channels=3, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            condition_dim,
            kernel_size=patch_size,
            stride=patch_size,
            **kwargs,
        )

    def forward(self, image):
        if image.dim() != 4:
            raise ValueError(f"image condition must have shape [B, C, H, W], got {tuple(image.shape)}")
        tokens = self.proj(image)
        return tokens.flatten(2).transpose(1, 2)


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
