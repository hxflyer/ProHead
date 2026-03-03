import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


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


class DenseImageDecoder(nn.Module):
    """CNN decoder from stride-4 feature map to full-resolution dense image."""

    def __init__(self, in_channels: int, out_channels: int = 3, output_size: int = 512):
        super().__init__()
        self.output_size = int(max(64, output_size))

        mid = max(64, in_channels // 2)
        low = max(32, mid // 2)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, low, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(low, low, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(low, low, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(low, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.up2(x)
        if x.shape[-2] != self.output_size or x.shape[-1] != self.output_size:
            x = F.interpolate(
                x,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return torch.sigmoid(self.head(x))


class DenseImageTransformer(nn.Module):
    """
    ConvNeXt feature extractor + 4-layer transformer + CNN decoder for dense image prediction.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        output_channels: int = 3,
        output_size: int = 512,
        transformer_map_size: int = 32,
        backbone_weights: str = "imagenet",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.output_channels = int(output_channels)
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

        self.fusion_proj = nn.Conv2d(128 + 256 + 512, self.d_model, kernel_size=1)
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
            in_channels=self.d_model,
            out_channels=self.output_channels,
            output_size=self.output_size,
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        features = self.backbone(rgb)
        f4 = features["stride4"]
        f8 = features["stride8"]
        f16 = features["stride16"]

        f8_up = F.interpolate(f8, size=f4.shape[-2:], mode="bilinear", align_corners=True)
        f16_up = F.interpolate(f16, size=f4.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([f4, f8_up, f16_up], dim=1)
        x = self.fusion_proj(x)
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

        return self.decoder(fused)
