from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = max(1, min(8, out_channels // 8))
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UVAtlasUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 16,
        base_channels: int = 32,
    ):
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.stem = ConvBlock(input_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.down4 = DownBlock(c4, c5)

        self.up4 = UpBlock(c5, c4, c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.shared_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, c1 // 8)), c1),
            nn.GELU(),
        )
        self.basecolor_head = nn.Conv2d(c1, 3, kernel_size=1)
        self.detail_normal_head = nn.Conv2d(c1, 3, kernel_size=1)

    def forward(self, uv_input: torch.Tensor) -> dict[str, torch.Tensor]:
        x1 = self.stem(uv_input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y = self.up4(x5, x4)
        y = self.up3(y, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        y = self.shared_head(y)

        pred_basecolor = torch.sigmoid(self.basecolor_head(y))
        pred_detail_normal = torch.sigmoid(self.detail_normal_head(y))
        return {
            "basecolor": pred_basecolor,
            "detail_normal": pred_detail_normal,
        }
