"""
Streamlined 3D Residual SE-ConvNet (`ResidualUNetSE3D`) for predicting protein
crystal solvent content from 3D Patterson maps + pre-phasing diffraction metadata.
"""

from typing import List, Optional, Sequence, Union
import torch
from torch import nn
from torch.nn import functional as F

from dl_common import (
    NUM_DL_METADATA,
    ContrastPool3D,
    MetadataFusionHead,
    RadialShellPool3D,
)


def _resolve_groups(num_channels: int, requested_groups: int = 8) -> int:
    g = min(requested_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(1, g)


def create_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    order: str = "gce",
    num_groups: int = 8,
    padding: int = 1,
    dropout_prob: float = 0.1,
    stride: int = 1,
):
    modules = []
    for i, char in enumerate(order):
        if char == "r":
            modules.append(("ReLU", nn.ReLU(inplace=True)))
        elif char == "l":
            modules.append(
                ("LeakyReLU", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            )
        elif char == "e":
            modules.append(("ELU", nn.ELU(inplace=True)))
        elif char == "c":
            bias = not ("g" in order or "b" in order)
            modules.append(
                (
                    "conv",
                    nn.Conv3d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        stride=stride,
                        bias=bias,
                    ),
                )
            )
        elif char == "g":
            is_before_conv = i < order.index("c")
            ch = in_channels if is_before_conv else out_channels
            modules.append(
                ("groupnorm", nn.GroupNorm(_resolve_groups(ch, num_groups), ch))
            )
        elif char == "b":
            is_before_conv = i < order.index("c")
            ch = in_channels if is_before_conv else out_channels
            modules.append(("batchnorm", nn.BatchNorm3d(ch)))
        elif char == "d":
            modules.append(("dropout", nn.Dropout3d(p=dropout_prob)))
        else:
            raise ValueError(f"Unsupported layer specifier '{char}'.")
    return modules


class SingleConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        order: str = "gce",
        num_groups: int = 8,
        padding: int = 1,
        stride: int = 1,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        for name, module in create_conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            order=order,
            num_groups=num_groups,
            padding=padding,
            dropout_prob=dropout_prob,
            stride=stride,
        ):
            self.add_module(name, module)


class ChannelSELayer3D(nn.Module):
    def __init__(self, num_channels: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        reduced = max(4, num_channels // reduction_ratio)
        self.fc1 = nn.Linear(num_channels, reduced, bias=True)
        self.fc2 = nn.Linear(reduced, num_channels, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        s = self.avg_pool(x).view(b, c)
        w = self.sigmoid(self.fc2(self.act(self.fc1(s)))).view(b, c, 1, 1, 1)
        return x * w


class SpatialSELayer3D(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(num_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if weights is not None:
            out = F.conv3d(x, weights.view(1, x.shape[1], 1, 1, 1))
        else:
            out = self.conv(x)
        return x * self.sigmoid(out)


class ChannelSpatialSELayer3D(nn.Module):
    def __init__(self, num_channels: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio=reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.cSE(x) + self.sSE(x))


class ResNetBlockSE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        order: str = "gce",
        num_groups: int = 8,
        reduction_ratio: int = 4,
        dropout_prob: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.proj = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(_resolve_groups(out_channels, num_groups), out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.conv1 = SingleConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            order=order,
            num_groups=num_groups,
            padding=kernel_size // 2,
        )
        n_order = order.replace("r", "").replace("e", "").replace("l", "")
        self.conv2 = SingleConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            order=n_order,
            num_groups=num_groups,
            padding=kernel_size // 2,
        )
        self.drop = nn.Dropout3d(dropout_prob) if dropout_prob > 0.0 else nn.Identity()
        self.act = nn.ELU(inplace=True) if "e" in order else nn.ReLU(inplace=True)
        self.se = ChannelSpatialSELayer3D(out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.conv1(residual)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.act(out + residual)
        return self.se(out)


class ResNetDownsampleBlockSE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        order: str = "gce",
        num_groups: int = 8,
        reduction_ratio: int = 4,
    ) -> None:
        super().__init__()
        # AvgPool + 1x1 conv anti-aliased residual shortcut (preserves smooth void contrast)
        self.shortcut = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_resolve_groups(out_channels, num_groups), out_channels),
        )
        self.conv1 = SingleConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            order=order,
            num_groups=num_groups,
            padding=kernel_size // 2,
            stride=2,
        )
        n_order = order.replace("r", "").replace("e", "").replace("l", "")
        self.conv2 = SingleConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            order=n_order,
            num_groups=num_groups,
            padding=kernel_size // 2,
            stride=1,
        )
        self.act = nn.ELU(inplace=True) if "e" in order else nn.ReLU(inplace=True)
        self.se = ChannelSpatialSELayer3D(out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv2(self.conv1(x))
        out = self.act(out + residual)
        return self.se(out)


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        apply_pooling: bool = True,
        pool_type: str = "avg",
        order: str = "gce",
        num_groups: int = 8,
        dropout_prob: float = 0.05,
    ) -> None:
        super().__init__()
        if apply_pooling:
            self.pool: nn.Module = (
                nn.AvgPool3d(kernel_size=2, stride=2)
                if pool_type == "avg"
                else nn.MaxPool3d(kernel_size=2, stride=2)
            )
        else:
            self.pool = nn.Identity()
        self.block = ResNetBlockSE(
            in_channels,
            out_channels,
            order=order,
            num_groups=num_groups,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


def _feature_progression(init_channels: int, num_levels: int) -> List[int]:
    return [init_channels * (2**k) for k in range(num_levels)]


class ResidualUNetSE3D(nn.Module):
    """
    Multi-scale 3D Residual SE-ConvNet with strided stem, `AvgPool3d` stages,
    concentric radial-shell pooling, `[Avg, Std, Max]` contrast pooling, and
    FiLM-modulated pre-phasing metadata fusion.
    """

    def __init__(
        self,
        in_channels: int = 1,
        f_maps: Union[int, Sequence[int]] = (16, 32, 64, 128, 256),
        layer_order: str = "gce",
        num_groups: int = 8,
        num_levels: int = 5,
        dropout_prob: float = 0.1,
        meta_dim: int = NUM_DL_METADATA,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(f_maps, int):
            channels = _feature_progression(f_maps, num_levels)
        else:
            channels = list(f_maps)
        assert len(channels) >= 3, "Expected at least 3 feature stages."

        # Strided 5x5x5 stem: 128^3 -> 64^3 immediately (4x-8x lower memory footprint)
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels, channels[0], kernel_size=5, stride=2, padding=2, bias=False
            ),
            nn.GroupNorm(_resolve_groups(channels[0], num_groups), channels[0]),
            nn.ELU(inplace=True),
            ResNetBlockSE(
                channels[0], channels[0], order=layer_order, num_groups=num_groups
            ),
        )

        # AvgPool3d stages: 64^3 -> 32^3 -> 16^3 -> 8^3 ...
        stages: List[nn.Module] = []
        for idx in range(1, len(channels)):
            stages.append(
                EncoderStage(
                    channels[idx - 1],
                    channels[idx],
                    apply_pooling=(idx < len(channels) - 1),
                    pool_type="avg",
                    order=layer_order,
                    num_groups=num_groups,
                    dropout_prob=0.5 * dropout_prob,
                )
            )
        self.encoders = nn.ModuleList(stages)

        # Concentric radial shell summary on Stage 1 (32^3 at 128^3 input = 4 A/voxel)
        self.radial_pool = RadialShellPool3D(spatial_size=32, num_shells=4)
        radial_dim = 4 * 2 * channels[1]

        # Distinct strided SE downsample stages (16^3 -> 8^3 -> 4^3)
        self.resnet_se_downsample1 = ResNetDownsampleBlockSE(
            channels[-1], channels[-1], order=layer_order, num_groups=num_groups
        )
        self.resnet_se_downsample2 = ResNetDownsampleBlockSE(
            channels[-1], channels[-1], order=layer_order, num_groups=num_groups
        )

        self.contrast_pool = ContrastPool3D()
        vol_dim = (3 * channels[-1]) + radial_dim

        self.fusion_head = MetadataFusionHead(
            vol_dim=vol_dim,
            meta_dim=meta_dim,
            hidden_dim=128,
            dropout_prob=dropout_prob,
        )

    def fit_metadata_scaler(self, meta_matrix) -> None:
        self.fusion_head.fit_metadata_scaler(meta_matrix)

    def forward(
        self,
        x: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.stem(x)
        radial_feat: Optional[torch.Tensor] = None
        for idx, stage in enumerate(self.encoders):
            x = stage(x)
            if idx == 0:
                radial_feat = self.radial_pool(x)

        x = self.resnet_se_downsample1(x)
        x = self.resnet_se_downsample2(x)
        contrast_feat = self.contrast_pool(x)

        assert radial_feat is not None
        vol_feat = torch.cat([contrast_feat, radial_feat], dim=1)
        return self.fusion_head(vol_feat, metadata)


# Alias preserved for backwards compatibility
UNet3D = ResidualUNetSE3D
