import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


def compute_dense_output_channels(
    predict_basecolor: bool,
    predict_geo: bool,
    predict_normal: bool,
) -> int:
    dense_target_count = int(bool(predict_basecolor)) + int(bool(predict_geo)) + int(bool(predict_normal))
    if dense_target_count <= 0:
        raise ValueError("At least one of predict_basecolor, predict_geo, or predict_normal must be enabled.")
    return dense_target_count * 3 + 1


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for feature maps."""

    def __init__(self, d_model: int, max_h: int = 128, max_w: int = 128):
        super().__init__()
        d_model_half = d_model // 2

        y_pos = torch.arange(max_h).unsqueeze(1).float()
        x_pos = torch.arange(max_w).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model_half, 2).float()
            * (-math.log(10000.0) / max(d_model_half, 1))
        )

        pe_y = torch.zeros(max_h, d_model_half)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term)
        pe_y[:, 1::2] = torch.cos(y_pos * div_term)

        pe_x = torch.zeros(max_w, d_model_half)
        pe_x[:, 0::2] = torch.sin(x_pos * div_term)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term)

        self.register_buffer("pe_y", pe_y)
        self.register_buffer("pe_x", pe_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pe_y = self.pe_y[:h, :].unsqueeze(1).repeat(1, w, 1)
        pe_x = self.pe_x[:w, :].unsqueeze(0).repeat(h, 1, 1)
        pe = torch.cat([pe_x, pe_y], dim=2).permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
        if pe.shape[1] != c:
            pe = pe[:, :c]
        return pe


def _group_norm_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DepthwiseConvBlock(nn.Module):
    """Lightweight convolution block used by the FPN decoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_norm_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PredictionHead(nn.Module):
    """Small task-specific head for one dense prediction target."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden = max(48, in_channels // 2)
        self.block = nn.Sequential(
            DepthwiseConvBlock(in_channels, hidden),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DenseImageDecoder(nn.Module):
    """Lightweight FPN decoder with separate RGB / geo / normal / mask heads."""

    def __init__(
        self,
        top_in_channels: int,
        skip8_channels: int,
        skip4_channels: int,
        predict_basecolor: bool = True,
        predict_geo: bool = True,
        predict_normal: bool = True,
        output_size: int = 512,
        fpn_channels: int = 128,
    ):
        super().__init__()
        self.output_size = int(max(64, output_size))
        self.predict_basecolor = bool(predict_basecolor)
        self.predict_geo = bool(predict_geo)
        self.predict_normal = bool(predict_normal)
        self.out_channels = compute_dense_output_channels(
            predict_basecolor=self.predict_basecolor,
            predict_geo=self.predict_geo,
            predict_normal=self.predict_normal,
        )
        self.fpn_channels = int(max(64, fpn_channels))

        self.top_proj = nn.Sequential(
            nn.Conv2d(top_in_channels, self.fpn_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_norm_groups(self.fpn_channels), self.fpn_channels),
            nn.GELU(),
        )
        self.skip8_proj = nn.Sequential(
            nn.Conv2d(skip8_channels, self.fpn_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_norm_groups(self.fpn_channels), self.fpn_channels),
        )
        self.skip4_proj = nn.Sequential(
            nn.Conv2d(skip4_channels, self.fpn_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_norm_groups(self.fpn_channels), self.fpn_channels),
        )
        self.refine8 = DepthwiseConvBlock(self.fpn_channels, self.fpn_channels)
        self.refine4 = DepthwiseConvBlock(self.fpn_channels, self.fpn_channels)
        self.up4_to_2 = DepthwiseConvBlock(self.fpn_channels, self.fpn_channels)
        self.up2_to_1 = DepthwiseConvBlock(self.fpn_channels, self.fpn_channels)
        self.output_refine = DepthwiseConvBlock(self.fpn_channels, self.fpn_channels)

        self.rgb_head = PredictionHead(self.fpn_channels, 3) if self.predict_basecolor else None
        self.geo_head = PredictionHead(self.fpn_channels, 3) if self.predict_geo else None
        self.normal_head = PredictionHead(self.fpn_channels, 3) if self.predict_normal else None
        self.mask_head = PredictionHead(self.fpn_channels, 1)

    def forward(self, top: torch.Tensor, skip8: torch.Tensor, skip4: torch.Tensor) -> torch.Tensor:
        p16 = self.top_proj(top)
        p8 = self.skip8_proj(skip8) + F.interpolate(
            p16,
            size=skip8.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p8 = self.refine8(p8)

        p4 = self.skip4_proj(skip4) + F.interpolate(
            p8,
            size=skip4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p4 = self.refine4(p4)

        x = F.interpolate(p4, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.up4_to_2(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.up2_to_1(x)
        if x.shape[-2:] != (self.output_size, self.output_size):
            x = F.interpolate(
                x,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        x = self.output_refine(x)

        outputs = []
        if self.rgb_head is not None:
            outputs.append(torch.sigmoid(self.rgb_head(x)))
        if self.geo_head is not None:
            outputs.append(torch.sigmoid(self.geo_head(x)))
        if self.normal_head is not None:
            outputs.append(torch.sigmoid(self.normal_head(x)))
        outputs.append(self.mask_head(x))
        return torch.cat(outputs, dim=1)


class DenseImageTransformer(nn.Module):
    """
    ConvNeXt feature extractor + transformer + CNN decoder for
    basecolor + geo + normal + mandatory mask prediction.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        predict_basecolor: bool = True,
        predict_geo: bool = True,
        predict_normal: bool = True,
        output_size: int = 512,
        transformer_map_size: int = 32,
        backbone_weights: str = "imagenet",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.predict_basecolor = bool(predict_basecolor)
        self.predict_geo = bool(predict_geo)
        self.predict_normal = bool(predict_normal)
        self.output_channels = compute_dense_output_channels(
            predict_basecolor=self.predict_basecolor,
            predict_geo=self.predict_geo,
            predict_normal=self.predict_normal,
        )
        self.output_size = int(output_size)
        self.transformer_map_size = int(max(8, transformer_map_size))

        backbone = models.convnext_base(weights=None)
        if backbone_weights == "dinov3":
            weight_path = "models/dinov3_lvd1689m_torchvision.pth"
            print("Loading DINOv3 pretrained backbone...")
        else:
            weight_path = "models/convnext_base-6075fbad.pth"
            print("Loading ImageNet pretrained backbone...")

        if os.path.exists(weight_path):
            print(f"Loading weights from: {weight_path}")
            state_dict = torch.load(weight_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            backbone.load_state_dict(state_dict, strict=False)
        else:
            print(f"[Warning] Backbone weight not found: {weight_path}. Using random init.")

        return_nodes = {
            "features.1": "stride4",
            "features.3": "stride8",
            "features.5": "stride16",
        }
        self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)

        self.fusion_proj = nn.Conv2d(512, self.d_model, kernel_size=1)
        self.pos_embed = PositionalEncoding2D(self.d_model, max_h=128, max_w=128)

        self.token_pool = nn.AdaptiveAvgPool2d((self.transformer_map_size, self.transformer_map_size))
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=nhead,
                    dim_feedforward=self.d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    norm_first=True,
                    batch_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.transformer_norm = nn.LayerNorm(self.d_model)
        self.post_attn_proj = nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1)
        self.decoder = DenseImageDecoder(
            top_in_channels=self.d_model,
            skip8_channels=256,
            skip4_channels=128,
            predict_basecolor=self.predict_basecolor,
            predict_geo=self.predict_geo,
            predict_normal=self.predict_normal,
            output_size=self.output_size,
            fpn_channels=min(max(self.d_model // 2, 96), 192),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            [B, C, H, W], where C = 3 * (#enabled dense targets) + 1.
            Dense targets are emitted in this order: basecolor, geo, normal, then raw mask logit.
        """
        features = self.backbone(rgb)
        f4 = features["stride4"]
        f8 = features["stride8"]
        f16 = features["stride16"]

        x = self.fusion_proj(f16)
        x = x + self.pos_embed(x)

        token_map = self.token_pool(x)
        b, c, h, w = token_map.shape
        tokens = token_map.flatten(2).permute(0, 2, 1)  # [B, N, C]

        for layer in self.transformer_layers:
            tokens = layer(tokens)
        tokens = self.transformer_norm(tokens)

        token_map_out = tokens.permute(0, 2, 1).reshape(b, c, h, w)
        token_map_out = F.interpolate(
            token_map_out, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = x + self.post_attn_proj(token_map_out)

        return self.decoder(fused, f8, f4)
