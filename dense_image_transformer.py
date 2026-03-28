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
    dense_target_count = int(bool(predict_basecolor)) + int(bool(predict_geo))
    if predict_normal:
        dense_target_count += 2  # detail screen normal + geometry normal
    if dense_target_count <= 0:
        raise ValueError("At least one of predict_basecolor, predict_geo, or predict_normal must be enabled.")
    return dense_target_count * 3 + 1


def _activate_detail_normal(detail_raw: torch.Tensor) -> torch.Tensor:
    # Detail screen normal is a signed residual in [-1, 1].
    return detail_raw.tanh()


def _activate_geometry_normal(normal_raw: torch.Tensor) -> torch.Tensor:
    # Geometry normal remains a unit vector encoded in [0, 1].
    normal_unnorm = normal_raw.tanh()
    normal_normalized = F.normalize(normal_unnorm, dim=1, eps=1e-6)
    normal_01 = normal_normalized * 0.5 + 0.5
    return torch.cat(
        [
            1.0 - normal_01[:, 0:1],
            1.0 - normal_01[:, 1:2],
            normal_01[:, 2:3],
        ],
        dim=1,
    )


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


class IntegratedTaskBranch(nn.Module):
    """
    Memory-efficient integrated task branch with progressive channel reduction.
    Combines upsampling pathway and prediction head into one module.
    
    Flow: P4[48ch] → P2[48ch] → P1[32ch] → [16ch] → [out_ch]
    """

    def __init__(self, in_channels: int, out_channels: int, output_size: int = 1024):
        super().__init__()
        self.output_size = output_size
        
        # P4 → P2: 48ch → 48ch
        self.up_to_p2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseConvBlock(in_channels, 48),
        )
        
        # P2 → P1: 48ch → 32ch
        self.up_to_p1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseConvBlock(48, 32),
        )
        
        # P1 refinement + prediction: 32ch → 16ch → out_ch
        self.refine_and_predict = nn.Sequential(
            DepthwiseConvBlock(32, 16),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )
    
    def forward(self, p4_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p4_features: [B, 48, H/4, W/4]
        Returns:
            [B, out_channels, H, W]
        """
        p2 = self.up_to_p2(p4_features)  # [B, 48, H/2, W/2]
        p1 = self.up_to_p1(p2)            # [B, 32, H, W]
        
        # Resize to exact output size if needed
        if p1.shape[-2:] != (self.output_size, self.output_size):
            p1 = F.interpolate(
                p1,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        
        output = self.refine_and_predict(p1)  # [B, out_channels, H, W]
        return output


class MultiTaskFPNDecoder(nn.Module):
    """
    Memory-efficient multi-task FPN decoder with integrated task-specific branches.
    
    Architecture:
    - Shared bottom-up path: P16 → P8 → P4 (128ch → 64ch → 48ch)
    - Integrated task branches (progressive channel reduction):
      - RGB/Geo/Normal: 48ch → 48ch → 32ch → 16ch → 3ch
      - Mask: 48ch → 48ch → 32ch → 16ch → 1ch
    
    Memory-efficient design:
    - 70% less memory than separate branches + deep heads
    - Enables batch_size=3-4 at 1024×1024 resolution
    """

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
        
        # Shared bottom-up path channel dimensions
        self.ch16 = 128
        self.ch8 = 64
        self.ch4 = 48

        # ========================================
        # SHARED BOTTOM-UP PATH (P16 → P8 → P4)
        # ========================================
        self.top_proj = nn.Sequential(
            nn.Conv2d(top_in_channels, self.ch16, kernel_size=1, bias=True),
            nn.GroupNorm(_group_norm_groups(self.ch16), self.ch16),
            nn.GELU(),
        )
        self.skip8_proj = nn.Sequential(
            nn.Conv2d(skip8_channels, self.ch8, kernel_size=1, bias=True),
            nn.GroupNorm(_group_norm_groups(self.ch8), self.ch8),
        )
        self.skip4_proj = nn.Sequential(
            nn.Conv2d(skip4_channels, self.ch4, kernel_size=1, bias=True),
            nn.GroupNorm(_group_norm_groups(self.ch4), self.ch4),
        )
        
        self.lateral_16to8 = nn.Conv2d(self.ch16, self.ch8, kernel_size=1, bias=True)
        self.lateral_8to4 = nn.Conv2d(self.ch8, self.ch4, kernel_size=1, bias=True)
        
        self.refine8 = DepthwiseConvBlock(self.ch8, self.ch8)
        self.refine4 = DepthwiseConvBlock(self.ch4, self.ch4)
        
        # ========================================
        # TASK-SPECIFIC INTEGRATED BRANCHES (P4 → prediction)
        # ========================================
        if predict_basecolor:
            self.rgb_branch = IntegratedTaskBranch(self.ch4, 3, output_size)
        else:
            self.rgb_branch = None
            
        if predict_geo:
            self.geo_branch = IntegratedTaskBranch(self.ch4, 3, output_size)
        else:
            self.geo_branch = None
            
        if predict_normal:
            self.geometry_normal_branch = IntegratedTaskBranch(self.ch4, 3, output_size)
            self.normal_branch = IntegratedTaskBranch(self.ch4, 3, output_size)
        else:
            self.geometry_normal_branch = None
            self.normal_branch = None
        
        # Mask always trained
        self.mask_branch = IntegratedTaskBranch(self.ch4, 1, output_size)

    def forward(self, top: torch.Tensor, skip8: torch.Tensor, skip4: torch.Tensor) -> torch.Tensor:
        # ========================================
        # SHARED PATH: P16 → P8 → P4
        # ========================================
        p16 = self.top_proj(top)  # [B, 128, H/16, W/16]
        
        p16_up = self.lateral_16to8(p16)
        p8 = self.skip8_proj(skip8) + F.interpolate(
            p16_up,
            size=skip8.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p8 = self.refine8(p8)  # [B, 64, H/8, W/8]
        
        p8_up = self.lateral_8to4(p8)
        p4 = self.skip4_proj(skip4) + F.interpolate(
            p8_up,
            size=skip4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p4 = self.refine4(p4)  # [B, 48, H/4, W/4]
        
        # ========================================
        # TASK-SPECIFIC INTEGRATED BRANCHES: P4 → outputs
        # ========================================
        outputs = []
        
        if self.rgb_branch is not None:
            rgb_out = torch.sigmoid(self.rgb_branch(p4))  # [B, 3, H, W]
            outputs.append(rgb_out)
        
        if self.geo_branch is not None:
            geo_out = torch.sigmoid(self.geo_branch(p4))  # [B, 3, H, W]
            outputs.append(geo_out)
        
        if self.normal_branch is not None:
            outputs.append(_activate_geometry_normal(self.geometry_normal_branch(p4)))
            outputs.append(_activate_detail_normal(self.normal_branch(p4)))
        
        mask_logits = self.mask_branch(p4)  # [B, 1, H, W]
        outputs.append(mask_logits)
        
        return torch.cat(outputs, dim=1)


class DenseImageDecoder(nn.Module):
    """Lightweight FPN decoder with separate RGB / geo / normal / mask heads.
    
    Uses progressive channel reduction: P16(128) → P8(64) → P4(48) → P2(32) → P1(32)
    to reduce memory consumption by ~70% while maintaining quality.
    """

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
        
        # Progressive channel reduction for memory efficiency
        self.ch16 = 128
        self.ch8 = 64
        self.ch4 = 48
        self.ch2 = 32
        self.ch1 = 32

        self.top_proj = nn.Sequential(
            nn.Conv2d(top_in_channels, self.ch16, kernel_size=1, bias=True),
            nn.GroupNorm(_group_norm_groups(self.ch16), self.ch16),
            nn.GELU(),
        )
        self.skip8_proj = nn.Sequential(
            nn.Conv2d(skip8_channels, self.ch8, kernel_size=1, bias=True),
            nn.GroupNorm(_group_norm_groups(self.ch8), self.ch8),
        )
        self.skip4_proj = nn.Sequential(
            nn.Conv2d(skip4_channels, self.ch4, kernel_size=1, bias=True),
            nn.GroupNorm(_group_norm_groups(self.ch4), self.ch4),
        )
        
        # Channel transition layers for FPN lateral connections
        self.lateral_16to8 = nn.Conv2d(self.ch16, self.ch8, kernel_size=1, bias=True)
        self.lateral_8to4 = nn.Conv2d(self.ch8, self.ch4, kernel_size=1, bias=True)
        
        self.refine8 = DepthwiseConvBlock(self.ch8, self.ch8)
        self.refine4 = DepthwiseConvBlock(self.ch4, self.ch4)
        self.up4_to_2 = DepthwiseConvBlock(self.ch4, self.ch2)
        self.up2_to_1 = DepthwiseConvBlock(self.ch2, self.ch1)
        self.output_refine = DepthwiseConvBlock(self.ch1, self.ch1)

        self.rgb_head = PredictionHead(self.ch1, 3) if self.predict_basecolor else None
        self.geo_head = PredictionHead(self.ch1, 3) if self.predict_geo else None
        self.geometry_normal_head = PredictionHead(self.ch1, 3) if self.predict_normal else None
        self.normal_head = PredictionHead(self.ch1, 3) if self.predict_normal else None
        self.mask_head = PredictionHead(self.ch1, 1)

    def forward(self, top: torch.Tensor, skip8: torch.Tensor, skip4: torch.Tensor) -> torch.Tensor:
        # P16: [B, 128, H/16, W/16]
        p16 = self.top_proj(top)
        
        # P8: [B, 64, H/8, W/8] - reduce channels when upsampling from P16
        p16_up = self.lateral_16to8(p16)
        p8 = self.skip8_proj(skip8) + F.interpolate(
            p16_up,
            size=skip8.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p8 = self.refine8(p8)

        # P4: [B, 48, H/4, W/4] - reduce channels when upsampling from P8
        p8_up = self.lateral_8to4(p8)
        p4 = self.skip4_proj(skip4) + F.interpolate(
            p8_up,
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
            outputs.append(_activate_geometry_normal(self.geometry_normal_head(x)))
            outputs.append(_activate_detail_normal(self.normal_head(x)))
        outputs.append(self.mask_head(x))
        return torch.cat(outputs, dim=1)


class DenseImageTransformer(nn.Module):
    """
    ConvNeXt feature extractor + transformer + CNN decoder for
    basecolor + geo + geometry normal + detail normal + mandatory mask prediction.
    
    Supports two decoder types:
    - "shared": DenseImageDecoder (memory efficient, shared features)
    - "multitask": MultiTaskFPNDecoder (higher quality, task-specific branches)
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
        decoder_type: str = "multitask",
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
        backbone_mode = str(backbone_weights or "").strip().lower()
        weight_path = ""
        if backbone_mode == "dinov3":
            weight_path = "assets/pretrained/dinov3_lvd1689m_torchvision.pth"
            print("Loading DINOv3 pretrained backbone...")
        elif backbone_mode in {"", "none", "random"}:
            weight_path = ""
        else:
            weight_path = "assets/pretrained/convnext_base-6075fbad.pth"
            print("Loading ImageNet pretrained backbone...")

        if weight_path:
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
        
        # Select decoder type
        decoder_type = str(decoder_type).lower()
        if decoder_type == "multitask":
            print("[Model] Using MultiTaskFPNDecoder (task-specific branches)")
            self.decoder = MultiTaskFPNDecoder(
                top_in_channels=self.d_model,
                skip8_channels=256,
                skip4_channels=128,
                predict_basecolor=self.predict_basecolor,
                predict_geo=self.predict_geo,
                predict_normal=self.predict_normal,
                output_size=self.output_size,
                fpn_channels=min(max(self.d_model // 2, 96), 192),
            )
        else:
            print("[Model] Using DenseImageDecoder (shared features)")
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
            Dense targets are emitted in this order:
            basecolor, geo, geometry normal, detail normal, then raw mask logit.
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
