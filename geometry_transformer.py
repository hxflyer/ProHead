import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
import math


class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for spatial feature maps.
    """

    def __init__(self, d_model: int, max_h: int = 128, max_w: int = 128):
        super().__init__()
        self.d_model = d_model
        d_model_half = d_model // 2

        y_pos = torch.arange(max_h).unsqueeze(1).float()
        x_pos = torch.arange(max_w).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * (-math.log(10000.0) / d_model_half))

        pe_y = torch.zeros(max_h, d_model_half)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term)
        pe_y[:, 1::2] = torch.cos(y_pos * div_term)

        pe_x = torch.zeros(max_w, d_model_half)
        pe_x[:, 0::2] = torch.sin(x_pos * div_term)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term)

        self.register_buffer('pe_y', pe_y)
        self.register_buffer('pe_x', pe_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            pe: [B, C, H, W] positional encoding
        """
        B, C, H, W = x.shape
        pe_y = self.pe_y[:H, :].unsqueeze(1).repeat(1, W, 1)  # [H, W, C/2]
        pe_x = self.pe_x[:W, :].unsqueeze(0).repeat(H, 1, 1)  # [H, W, C/2]

        pe = torch.cat([pe_x, pe_y], dim=2)  # [H, W, C]
        pe = pe.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, C, H, W]
        return pe


class SimDRHead(nn.Module):
    """
    SimDR head for 5D coordinate offsets (x, y, z, u, v).
    """

    def __init__(
        self,
        d_model: int,
        k_bins: int = 256,
        output_dim: int = 5,
        trunk_hidden_dim: int = 256,
        range_3d: tuple[float, float] = (-1.0, 1.0),
        range_2d: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        if output_dim != 5:
            raise ValueError(f"SimDRHead currently supports output_dim=5, got {output_dim}.")

        self.k_bins = k_bins
        self.output_dim = output_dim

        trunk_hidden_dim = int(max(32, trunk_hidden_dim))
        self.trunk = nn.Sequential(
            nn.Linear(d_model, trunk_hidden_dim),
            nn.GELU(),
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim),
            nn.GELU(),
        )

        self.fc_x = nn.Linear(trunk_hidden_dim, k_bins)
        self.fc_y = nn.Linear(trunk_hidden_dim, k_bins)
        self.fc_z = nn.Linear(trunk_hidden_dim, k_bins)
        self.fc_u = nn.Linear(trunk_hidden_dim, k_bins)
        self.fc_v = nn.Linear(trunk_hidden_dim, k_bins)

        grid_3d = torch.linspace(range_3d[0], range_3d[1], k_bins)
        self.register_buffer('grid_3d', grid_3d)

        grid_2d = torch.linspace(range_2d[0], range_2d[1], k_bins)
        self.register_buffer('grid_2d', grid_2d)

    def forward(self, x: torch.Tensor, return_logits: bool = True):
        """
        Args:
            x: [B, N, d_model]
        Returns:
            offsets: [B, N, 5]
            logits: [B, N, 5, K] or None
        """
        # Keep SimDR logits/softmax in fp32 to avoid fp16 overflow in long training.
        # This block remains numerically stable even when outer forward runs under AMP.
        device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cuda"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            feat = self.trunk(x.float())

            logits_x = self.fc_x(feat)
            logits_y = self.fc_y(feat)
            logits_z = self.fc_z(feat)
            logits_u = self.fc_u(feat)
            logits_v = self.fc_v(feat)

            prob_x = F.softmax(logits_x, dim=-1)
            prob_y = F.softmax(logits_y, dim=-1)
            prob_z = F.softmax(logits_z, dim=-1)
            prob_u = F.softmax(logits_u, dim=-1)
            prob_v = F.softmax(logits_v, dim=-1)

            offset_x = (prob_x * self.grid_3d).sum(dim=-1)
            offset_y = (prob_y * self.grid_3d).sum(dim=-1)
            offset_z = (prob_z * self.grid_3d).sum(dim=-1)
            offset_u = (prob_u * self.grid_2d).sum(dim=-1)
            offset_v = (prob_v * self.grid_2d).sum(dim=-1)

        offsets = torch.stack([offset_x, offset_y, offset_z, offset_u, offset_v], dim=-1)
        logits = None
        if return_logits:
            logits = torch.stack([logits_x, logits_y, logits_z, logits_u, logits_v], dim=2)

        return offsets, logits


class FastOffsetHead(nn.Module):
    """Lightweight offset regression head for early-layer L1-only supervision."""

    def __init__(self, d_model: int, output_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        if output_dim != 5:
            raise ValueError(f"FastOffsetHead currently supports output_dim=5, got {output_dim}.")
        hidden_dim = int(max(16, hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionOffsetHead(nn.Module):
    """Shared lightweight regression head used by landmark and mesh branches."""

    def __init__(self, d_model: int, output_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        hidden_dim = int(max(16, hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeshVertexFeatureHead(nn.Module):
    """Predict per-vertex latent features for UV-space texture decoding."""

    def __init__(self, d_model: int, feature_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        feature_dim = int(max(4, feature_dim))
        hidden_dim = int(max(16, hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextureDecoder(nn.Module):
    """
    Decode UV-space feature map into RGB texture.
    Default path: 256x256 -> 512x512 -> 1024x1024.
    """

    def __init__(self, in_channels: int, output_size: int = 1024):
        super().__init__()
        self.output_size = int(max(64, output_size))
        mid = max(32, in_channels * 2)
        hi = max(64, in_channels * 4)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hi, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hi, hi, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(hi, mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(mid, max(32, mid // 2), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(max(32, mid // 2), max(32, mid // 2), kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.rgb_head = nn.Conv2d(max(32, mid // 2), 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.up2(x)
        if x.shape[-1] != self.output_size or x.shape[-2] != self.output_size:
            x = F.interpolate(
                x,
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=False,
            )
        return torch.sigmoid(self.rgb_head(x))


class DeformableCrossAttention(nn.Module):
    """
    Multi-scale deformable cross-attention.
    References are normalized [0, 1] image coordinates per query.
    """

    def __init__(self, d_model: int, nhead: int, num_points: int = 8, num_levels: int = 3):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        if num_points <= 0:
            raise ValueError("num_points must be > 0.")
        if num_levels <= 0:
            raise ValueError("num_levels must be > 0.")

        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_points = int(num_points)
        self.num_levels = int(num_levels)
        self.head_dim = self.d_model // self.nhead

        self.sampling_offsets = nn.Linear(self.d_model, self.nhead * self.num_levels * self.num_points * 2)
        self.attention_weights = nn.Linear(self.d_model, self.nhead * self.num_levels * self.num_points)
        self.value_proj = nn.Linear(self.d_model, self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.d_model)

        self.offset_scale = 1.0
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)

        # Deformable DETR-style radial bias init spreads initial samples around each reference point
        # instead of collapsing all points to the same location.
        thetas = torch.arange(self.nhead, dtype=torch.float32) * (2.0 * math.pi / float(self.nhead))
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)  # [nhead, 2]
        grid_init = grid_init / grid_init.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        grid_init = grid_init.view(self.nhead, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for p in range(self.num_points):
            grid_init[:, :, p, :] *= float(p + 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def set_offset_scale(self, scale: float) -> None:
        self.offset_scale = float(scale)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, Nq, C]
            memory: [B, S, C] flattened multi-scale features
            reference_points: [B, Nq, 2] in [0, 1]
            spatial_shapes: [L, 2] with (H_l, W_l)
            level_start_index: [L] flat start index for each level
        Returns:
            [B, Nq, C]
        """
        B, Nq, C = query.shape
        _, S, _ = memory.shape
        if spatial_shapes.shape[0] != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} feature levels, got {spatial_shapes.shape[0]}."
            )
        if level_start_index.shape[0] != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} level_start_index entries, got {level_start_index.shape[0]}."
            )

        value = self.value_proj(memory).view(B, S, self.nhead, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            B, Nq, self.nhead, self.num_levels, self.num_points, 2
        )
        if self.offset_scale != 1.0:
            sampling_offsets = sampling_offsets * float(self.offset_scale)
        attention_weights = self.attention_weights(query).view(
            B, Nq, self.nhead, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            B, Nq, self.nhead, self.num_levels, self.num_points
        )

        normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1).to(
            device=query.device, dtype=query.dtype
        )  # [L,2] = (W,H)

        sampling_locations = reference_points[:, :, None, None, None, :] + (
            sampling_offsets / normalizer[None, None, None, :, None, :]
        )

        output = query.new_zeros((B, Nq, self.nhead, self.head_dim))

        for level in range(self.num_levels):
            h_l = int(spatial_shapes[level, 0].item())
            w_l = int(spatial_shapes[level, 1].item())
            start = int(level_start_index[level].item())
            end = start + h_l * w_l

            value_l = value[:, start:end, :, :]  # [B, H_l*W_l, nhead, head_dim]
            value_l = value_l.permute(0, 2, 3, 1).contiguous().view(
                B * self.nhead, self.head_dim, h_l, w_l
            )

            grid_l = sampling_locations[:, :, :, level, :, :]  # [B, Nq, nhead, P, 2] in [0,1]
            grid_l = grid_l * 2.0 - 1.0
            grid_l = grid_l.permute(0, 2, 1, 3, 4).contiguous().view(
                B * self.nhead, Nq, self.num_points, 2
            )

            sampled_l = F.grid_sample(
                value_l,
                grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # [B*nhead, head_dim, Nq, P]
            sampled_l = sampled_l.view(B, self.nhead, self.head_dim, Nq, self.num_points)
            sampled_l = sampled_l.permute(0, 3, 1, 4, 2).contiguous()  # [B,Nq,nhead,P,head_dim]

            attn_l = attention_weights[:, :, :, level, :].unsqueeze(-1)  # [B,Nq,nhead,P,1]
            output = output + (sampled_l * attn_l).sum(dim=3)

        output = output.view(B, Nq, C)
        output = self.output_proj(output)
        return output


class DeformableDecoderLayer(nn.Module):
    """Decoder layer with self-attention + deformable cross-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_points: int = 8,
        num_levels: int = 3,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = DeformableCrossAttention(
            d_model=d_model,
            nhead=nhead,
            num_points=num_points,
            num_levels=num_levels,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.reference_point_proj = nn.Linear(d_model, 2)

    def set_deformable_offset_scale(self, scale: float) -> None:
        self.cross_attn.set_offset_scale(scale)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        q = self.norm1(tgt)
        q2, _ = self.self_attn(q, q, q)
        tgt = tgt + self.dropout1(q2)

        q = self.norm2(tgt)
        reference_points = self.reference_point_proj(q).sigmoid()
        q2 = self.cross_attn(
            query=q,
            memory=memory,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        tgt = tgt + self.dropout2(q2)

        q = self.norm3(tgt)
        q2 = self.linear2(self.dropout(self.activation(self.linear1(q))))
        tgt = tgt + self.dropout3(q2)
        return tgt


class GeometryTransformer(nn.Module):
    """
    Unified geometry transformer with selectable prediction head:
    - regression: direct coordinate offset regression (legacy behavior)
    - simdr: SimDR coordinate classification + expectation decoding
    """

    def __init__(
        self,
        num_landmarks: int,
        num_mesh: int,
        template_landmark: np.ndarray,
        template_mesh: np.ndarray,
        landmark2keypoint_knn_indices: np.ndarray,
        landmark2keypoint_knn_weights: np.ndarray,
        mesh2landmark_knn_indices: np.ndarray,
        mesh2landmark_knn_weights: np.ndarray,
        n_keypoint: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_dim: int = 5,
        backbone_weights: str = 'imagenet',
        model_type: str = 'regression',
        flatten_regression_outputs: bool = True,
        k_bins: int = 256,
        simdr_head_hidden_dim: int = 256,
        simdr_range_3d: tuple[float, float] = (-1.0, 1.0),
        simdr_range_2d: tuple[float, float] = (-1.0, 1.0),
        use_deformable_attention: bool = False,
        num_deformable_points: int = 16,
        use_fast_aux_regression_heads: bool = False,
        mesh_vertex_feature_dim: int = 16,
        texture_feature_map_size: int = 256,
        texture_output_size: int = 1024,
        flip_uv_v: bool = True,
        template_mesh_uv: np.ndarray | None = None,
    ):
        super().__init__()
        model_type = str(model_type).strip().lower()
        if model_type not in {"regression", "simdr"}:
            raise ValueError(f"Unsupported model_type: {model_type}. Expected 'regression' or 'simdr'.")
        if model_type == "simdr" and output_dim != 5:
            raise ValueError(f"SimDR mode currently supports output_dim=5, got {output_dim}.")

        self.num_landmarks = num_landmarks
        self.num_mesh = num_mesh
        self.n_keypoint = int(n_keypoint)
        self.d_model = int(d_model)
        self.output_dim = int(output_dim)
        self.model_type = model_type
        self.is_simdr_model = model_type == "simdr"
        self.flatten_regression_outputs = bool(flatten_regression_outputs)
        self.use_deformable_attention = bool(use_deformable_attention)
        self.num_feature_levels = 3
        self.use_fast_aux_regression_heads = bool(use_fast_aux_regression_heads and self.is_simdr_model)
        self.mesh_vertex_feature_dim = int(max(4, mesh_vertex_feature_dim))
        self.texture_feature_map_size = int(max(16, texture_feature_map_size))
        self.texture_output_size = int(max(64, texture_output_size))
        self.flip_uv_v = bool(flip_uv_v)

        self.register_buffer("template_landmark", torch.from_numpy(template_landmark).float())
        self.register_buffer("template_mesh", torch.from_numpy(template_mesh).float())
        if template_mesh_uv is None:
            if template_mesh.shape[1] >= 5:
                template_mesh_uv = template_mesh[:, 3:5]
            else:
                template_mesh_uv = np.zeros((num_mesh, 2), dtype=np.float32)
        template_mesh_uv = np.asarray(template_mesh_uv, dtype=np.float32)
        if template_mesh_uv.shape[0] != num_mesh or template_mesh_uv.shape[1] != 2:
            raise ValueError(
                f"template_mesh_uv must have shape [{num_mesh}, 2], got {template_mesh_uv.shape}."
            )
        template_mesh_uv = np.clip(template_mesh_uv, 0.0, 1.0)
        self.register_buffer("template_mesh_uv", torch.from_numpy(template_mesh_uv).float())

        self.register_buffer(
            "landmark2keypoint_knn_indices",
            torch.from_numpy(landmark2keypoint_knn_indices.astype(np.int64)),
        )
        self.register_buffer(
            "landmark2keypoint_knn_weights",
            torch.from_numpy(landmark2keypoint_knn_weights.astype(np.float32)),
        )
        self.register_buffer(
            "mesh2landmark_knn_indices",
            torch.from_numpy(mesh2landmark_knn_indices.astype(np.int64)),
        )
        self.register_buffer(
            "mesh2landmark_knn_weights",
            torch.from_numpy(mesh2landmark_knn_weights.astype(np.float32)),
        )

        backbone = models.convnext_base(weights=None)

        if backbone_weights == 'dinov3':
            print("Loading DINOv3 pretrained backbone...")
            weight_path = "models/dinov3_lvd1689m_torchvision.pth"
        else:
            print("Loading ImageNet pretrained backbone...")
            weight_path = "models/convnext_base-6075fbad.pth"

        print(f"Loading weights from: {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu")
        if 'model' in state_dict:
            state_dict = state_dict['model']
        backbone.load_state_dict(state_dict, strict=False)

        return_nodes = {
            'features.1': 'stride4',
            'features.3': 'stride8',
            'features.5': 'stride16',
        }
        self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)

        if self.use_deformable_attention:
            self.level_embed = nn.Parameter(torch.zeros(self.num_feature_levels, d_model))
            nn.init.normal_(self.level_embed, mean=0.0, std=0.02)
            self.level_proj_4 = nn.Conv2d(128, d_model, kernel_size=1)
            self.level_proj_8 = nn.Conv2d(256, d_model, kernel_size=1)
            self.level_proj_16 = nn.Conv2d(512, d_model, kernel_size=1)
        else:
            self.projection = nn.Conv2d(128 + 256 + 512, d_model, kernel_size=1)
        self.pos_embed = PositionalEncoding2D(d_model, max_h=128, max_w=128)

        if self.use_deformable_attention:
            self.decoder_layers = nn.ModuleList([
                DeformableDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    num_points=num_deformable_points,
                    num_levels=self.num_feature_levels,
                ) for _ in range(num_layers)
            ])
        else:
            self.decoder_layers = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    norm_first=True,
                    batch_first=True,
                ) for _ in range(num_layers)
            ])
        self.num_layers = num_layers
        self.decoder_norm = nn.LayerNorm(d_model)

        self.query_embed = nn.Embedding(self.n_keypoint, d_model)

        self.landmark_pos_embed = nn.Embedding(num_landmarks, d_model)
        self.mesh_pos_embed = nn.Embedding(num_mesh, d_model)

        self.landmark_pos_proj = nn.Linear(d_model, d_model)
        self.landmark_sub_proj = nn.Linear(d_model, d_model)

        if self.is_simdr_model:
            self.landmark_head = SimDRHead(
                d_model,
                k_bins=k_bins,
                output_dim=output_dim,
                trunk_hidden_dim=simdr_head_hidden_dim,
                range_3d=simdr_range_3d,
                range_2d=simdr_range_2d,
            )
            self.mesh_head = SimDRHead(
                d_model,
                k_bins=k_bins,
                output_dim=output_dim,
                trunk_hidden_dim=simdr_head_hidden_dim,
                range_3d=simdr_range_3d,
                range_2d=simdr_range_2d,
            )
            self.to_coord = None
        else:
            self.landmark_head = None
            self.mesh_head = None
            self.to_coord = RegressionOffsetHead(d_model, output_dim=output_dim, hidden_dim=64)

        if self.use_fast_aux_regression_heads:
            fast_hidden = min(64, int(simdr_head_hidden_dim))
            self.landmark_fast_head = FastOffsetHead(d_model, output_dim=output_dim, hidden_dim=fast_hidden)
            self.mesh_fast_head = FastOffsetHead(d_model, output_dim=output_dim, hidden_dim=fast_hidden)
        else:
            self.landmark_fast_head = None
            self.mesh_fast_head = None
        self.mesh_vertex_feature_head = MeshVertexFeatureHead(
            d_model=d_model,
            feature_dim=self.mesh_vertex_feature_dim,
            hidden_dim=64,
        )
        self.mesh_texture_decoder = TextureDecoder(
            in_channels=self.mesh_vertex_feature_dim,
            output_size=self.texture_output_size,
        )

    def set_deformable_offset_scale(self, scale: float) -> None:
        if not self.use_deformable_attention:
            return
        for layer in self.decoder_layers:
            if hasattr(layer, "set_deformable_offset_scale"):
                layer.set_deformable_offset_scale(scale)

    def _uv_to_pixel_coords(
        self,
        out_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uv = self.template_mesh_uv.to(device=device, dtype=dtype)
        u = uv[:, 0].clamp(0.0, 1.0)
        v = uv[:, 1].clamp(0.0, 1.0)
        if self.flip_uv_v:
            v = 1.0 - v
        x = u * float(max(out_size - 1, 1))
        y = v * float(max(out_size - 1, 1))
        return x, y

    def rasterize_mesh_vertex_attributes(
        self,
        vertex_attrs: torch.Tensor,
        out_size: int | None = None,
        return_coverage: bool = False,
    ):
        """
        Splat per-vertex attributes to UV map using bilinear weights.
        Args:
            vertex_attrs: [B, M, C]
            out_size: map size (H=W)
        Returns:
            uv_map: [B, C, H, W]
            coverage(optional): [B, 1, H, W]
        """
        if vertex_attrs.ndim != 3:
            raise ValueError(f"vertex_attrs must be [B, M, C], got {tuple(vertex_attrs.shape)}")
        B, M, C = vertex_attrs.shape
        if M != self.num_mesh:
            raise ValueError(f"vertex_attrs second dim {M} != num_mesh {self.num_mesh}.")

        H = int(self.texture_feature_map_size if out_size is None else out_size)
        W = H
        x, y = self._uv_to_pixel_coords(H, vertex_attrs.device, vertex_attrs.dtype)

        x0 = torch.floor(x).long().clamp(0, W - 1)
        y0 = torch.floor(y).long().clamp(0, H - 1)
        x1 = (x0 + 1).clamp(0, W - 1)
        y1 = (y0 + 1).clamp(0, H - 1)

        wx1 = x - x0.to(dtype=vertex_attrs.dtype)
        wy1 = y - y0.to(dtype=vertex_attrs.dtype)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        w00 = wx0 * wy0
        w01 = wx1 * wy0
        w10 = wx0 * wy1
        w11 = wx1 * wy1

        idx00 = y0 * W + x0
        idx01 = y0 * W + x1
        idx10 = y1 * W + x0
        idx11 = y1 * W + x1

        uv_map = vertex_attrs.new_zeros((B, C, H * W))
        uv_weight = vertex_attrs.new_zeros((B, 1, H * W))
        attrs_t = vertex_attrs.transpose(1, 2).contiguous()  # [B, C, M]
        ones = vertex_attrs.new_ones((B, 1, M))

        def _splat(idx: torch.Tensor, w: torch.Tensor) -> None:
            w_c = w.view(1, 1, M)
            src_val = attrs_t * w_c
            src_w = ones * w_c
            uv_map.scatter_add_(2, idx.view(1, 1, M).expand(B, C, M), src_val)
            uv_weight.scatter_add_(2, idx.view(1, 1, M).expand(B, 1, M), src_w)

        _splat(idx00, w00)
        _splat(idx01, w01)
        _splat(idx10, w10)
        _splat(idx11, w11)

        uv_map = uv_map / uv_weight.clamp(min=1e-6)
        coverage = (uv_weight > 1e-6).to(dtype=uv_map.dtype)
        uv_map = uv_map * coverage
        uv_map = uv_map.view(B, C, H, W)
        coverage = coverage.view(B, 1, H, W)
        if return_coverage:
            return uv_map, coverage
        return uv_map

    def forward_landmark_head(
        self,
        keypoint_feats: torch.Tensor,
        return_logits: bool = True,
        use_simdr_head: bool = True,
    ):
        """Project keypoint features to landmarks and decode template residuals."""
        idx = self.landmark2keypoint_knn_indices
        w = self.landmark2keypoint_knn_weights

        neighbor_feats = keypoint_feats[:, idx, :]  # [B, num_landmarks, k, d]
        landmark_base_feats = (neighbor_feats * w.unsqueeze(0).unsqueeze(-1)).sum(dim=2)

        landmark_pos_feats = self.landmark_pos_proj(landmark_base_feats)
        pos_emb = self.landmark_pos_embed(torch.arange(self.num_landmarks, device=landmark_pos_feats.device))
        z = landmark_pos_feats + pos_emb.unsqueeze(0)

        if self.is_simdr_model and use_simdr_head:
            offsets, logits = self.landmark_head(z, return_logits=return_logits)
        else:
            if self.is_simdr_model and self.landmark_fast_head is not None:
                offsets = self.landmark_fast_head(z)
                logits = None
            elif self.is_simdr_model:
                offsets, logits = self.landmark_head(z, return_logits=False)
            else:
                offsets = self.to_coord(z)
                logits = None
        coords = self.template_landmark.unsqueeze(0) + offsets
        return coords, logits

    def forward_mesh_head(
        self,
        keypoint_feats: torch.Tensor,
        return_logits: bool = True,
        use_simdr_head: bool = True,
    ):
        """Project keypoint features to mesh via landmarks and decode template residuals."""
        landmark_idx = self.landmark2keypoint_knn_indices
        landmark_w = self.landmark2keypoint_knn_weights

        neighbor_feats = keypoint_feats[:, landmark_idx, :]
        landmark_base_feats = (neighbor_feats * landmark_w.unsqueeze(0).unsqueeze(-1)).sum(dim=2)

        landmark_sub_feats = self.landmark_sub_proj(landmark_base_feats)

        mesh_idx = self.mesh2landmark_knn_indices
        mesh_w = self.mesh2landmark_knn_weights

        mesh_neighbor_feats = landmark_sub_feats[:, mesh_idx, :]
        mesh_feats = (mesh_neighbor_feats * mesh_w.unsqueeze(0).unsqueeze(-1)).sum(dim=2)

        pos_emb = self.mesh_pos_embed(torch.arange(self.num_mesh, device=mesh_feats.device))
        z = mesh_feats + pos_emb.unsqueeze(0)

        if self.is_simdr_model and use_simdr_head:
            offsets, logits = self.mesh_head(z, return_logits=return_logits)
        else:
            if self.is_simdr_model and self.mesh_fast_head is not None:
                offsets = self.mesh_fast_head(z)
                logits = None
            elif self.is_simdr_model:
                offsets, logits = self.mesh_head(z, return_logits=False)
            else:
                offsets = self.to_coord(z)
                logits = None
        coords = self.template_mesh.unsqueeze(0) + offsets
        mesh_vertex_features = self.mesh_vertex_feature_head(z)
        return coords, logits, mesh_vertex_features

    def forward(
        self,
        rgb: torch.Tensor,
        return_logits_mask=None,
        use_simdr_mask=None,
        predict_layer_mask=None,
        decode_texture_mask=None,
    ):
        """
        Args:
            rgb: [B, 3, 512, 512]
            return_logits_mask: optional list/tuple[bool] of len=num_layers.
                Controls whether each decoder layer returns SimDR logits.
            use_simdr_mask: optional list/tuple[bool] of len=num_layers.
                Controls whether each decoder layer uses SimDR head or lightweight
                regression head for offsets (when available).
            predict_layer_mask: optional list/tuple[bool] of len=num_layers.
                Controls whether each decoder layer computes landmark/mesh heads.
                Useful for validation/inference when only the final layer is consumed.
            decode_texture_mask: optional list/tuple[bool] of len=num_layers.
                Controls whether each decoder layer decodes UV texture map.
        Returns:
            List of dicts with landmark/mesh absolute coords and SimDR logits.
        """
        B = rgb.shape[0]
        if return_logits_mask is None:
            return_logits_mask = [self.is_simdr_model] * self.num_layers
        if len(return_logits_mask) != self.num_layers:
            raise ValueError(
                f"return_logits_mask length {len(return_logits_mask)} != num_layers {self.num_layers}"
            )
        if use_simdr_mask is None:
            use_simdr_mask = [self.is_simdr_model] * self.num_layers
        if len(use_simdr_mask) != self.num_layers:
            raise ValueError(
                f"use_simdr_mask length {len(use_simdr_mask)} != num_layers {self.num_layers}"
            )
        if predict_layer_mask is None:
            predict_layer_mask = [True] * self.num_layers
        if len(predict_layer_mask) != self.num_layers:
            raise ValueError(
                f"predict_layer_mask length {len(predict_layer_mask)} != num_layers {self.num_layers}"
            )
        if decode_texture_mask is None:
            decode_texture_mask = [False] * (self.num_layers - 1) + [True]
        if len(decode_texture_mask) != self.num_layers:
            raise ValueError(
                f"decode_texture_mask length {len(decode_texture_mask)} != num_layers {self.num_layers}"
            )

        features = self.backbone(rgb)
        f4 = features['stride4']
        f8 = features['stride8']
        f16 = features['stride16']

        if self.use_deformable_attention:
            x4 = self.level_proj_4(f4)
            x8 = self.level_proj_8(f8)
            x16 = self.level_proj_16(f16)

            x4 = x4 + self.pos_embed(x4) + self.level_embed[0].view(1, -1, 1, 1)
            x8 = x8 + self.pos_embed(x8) + self.level_embed[1].view(1, -1, 1, 1)
            x16 = x16 + self.pos_embed(x16) + self.level_embed[2].view(1, -1, 1, 1)

            x4 = x4.flatten(2).permute(0, 2, 1)
            x8 = x8.flatten(2).permute(0, 2, 1)
            x16 = x16.flatten(2).permute(0, 2, 1)

            src = torch.cat([x4, x8, x16], dim=1)
            spatial_shapes = torch.as_tensor(
                [
                    [f4.shape[2], f4.shape[3]],
                    [f8.shape[2], f8.shape[3]],
                    [f16.shape[2], f16.shape[3]],
                ],
                dtype=torch.long,
                device=rgb.device,
            )
            level_tokens = spatial_shapes[:, 0] * spatial_shapes[:, 1]
            level_start_index = torch.cat(
                [level_tokens.new_zeros((1,)), level_tokens.cumsum(dim=0)[:-1]],
                dim=0,
            )
        else:
            f8_up = F.interpolate(f8, size=f4.shape[-2:], mode='bilinear', align_corners=True)
            f16_up = F.interpolate(f16, size=f4.shape[-2:], mode='bilinear', align_corners=True)

            x_fused = torch.cat([f4, f8_up, f16_up], dim=1)
            x = self.projection(x_fused)
            src = (x + self.pos_embed(x)).flatten(2).permute(0, 2, 1)
            spatial_shapes = None
            level_start_index = None

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        outputs = []
        out = queries

        for i, layer in enumerate(self.decoder_layers):
            if self.use_deformable_attention:
                out = layer(out, src, spatial_shapes, level_start_index)
            else:
                out = layer(out, src)

            if i == self.num_layers - 1:
                out = self.decoder_norm(out)

            if bool(predict_layer_mask[i]):
                use_simdr_head = self.is_simdr_model and bool(use_simdr_mask[i])
                need_logits = self.is_simdr_model and bool(return_logits_mask[i]) and use_simdr_head
                lm_coords, lm_logits = self.forward_landmark_head(
                    out,
                    return_logits=need_logits,
                    use_simdr_head=use_simdr_head,
                )
                mesh_coords, mesh_logits, mesh_vertex_features = self.forward_mesh_head(
                    out,
                    return_logits=need_logits,
                    use_simdr_head=use_simdr_head,
                )
                if bool(decode_texture_mask[i]):
                    mesh_feature_map = self.rasterize_mesh_vertex_attributes(
                        mesh_vertex_features,
                        out_size=self.texture_feature_map_size,
                        return_coverage=False,
                    )
                    mesh_texture = self.mesh_texture_decoder(mesh_feature_map)
                else:
                    mesh_feature_map = None
                    mesh_texture = None
            else:
                lm_coords = None
                lm_logits = None
                mesh_coords = None
                mesh_logits = None
                mesh_vertex_features = None
                mesh_feature_map = None
                mesh_texture = None

            if (not self.is_simdr_model) and lm_coords is not None and self.flatten_regression_outputs:
                lm_coords = lm_coords.view(B, -1)
                mesh_coords = mesh_coords.view(B, -1)

            if self.is_simdr_model:
                outputs.append({
                    'landmark': lm_coords,
                    'landmark_logits': lm_logits,
                    'mesh': mesh_coords,
                    'mesh_logits': mesh_logits,
                    'mesh_vertex_features': mesh_vertex_features,
                    'mesh_feature_map': mesh_feature_map,
                    'mesh_texture': mesh_texture,
                })
            else:
                outputs.append({
                    'landmark': lm_coords,
                    'mesh': mesh_coords,
                    'mesh_vertex_features': mesh_vertex_features,
                    'mesh_feature_map': mesh_feature_map,
                    'mesh_texture': mesh_texture,
                })

        return outputs


class SimDRGeometryTransformer(GeometryTransformer):
    """Backward-compatible alias for the unified transformer in SimDR mode."""

    def __init__(self, *args, **kwargs):
        kwargs["model_type"] = "simdr"
        kwargs.setdefault("flatten_regression_outputs", False)
        kwargs.setdefault("use_deformable_attention", True)
        super().__init__(*args, **kwargs)


def create_geometry_transformer(
    num_landmarks: int,
    num_mesh: int,
    template_landmark: np.ndarray,
    template_mesh: np.ndarray,
    landmark2keypoint_knn_indices: np.ndarray,
    landmark2keypoint_knn_weights: np.ndarray,
    mesh2landmark_knn_indices: np.ndarray,
    mesh2landmark_knn_weights: np.ndarray,
    n_keypoint: int,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    output_dim: int = 5,
    backbone_weights: str = 'imagenet',
    model_type: str = 'regression',
    flatten_regression_outputs: bool = True,
    k_bins: int = 256,
    simdr_head_hidden_dim: int = 256,
    simdr_range_3d: tuple[float, float] = (-0.5, 0.5),
    simdr_range_2d: tuple[float, float] = (-0.5, 0.5),
    use_deformable_attention: bool = False,
    num_deformable_points: int = 16,
    use_fast_aux_regression_heads: bool = False,
    mesh_vertex_feature_dim: int = 16,
    texture_feature_map_size: int = 256,
    texture_output_size: int = 1024,
    flip_uv_v: bool = True,
    template_mesh_uv: np.ndarray | None = None,
) -> GeometryTransformer:
    return GeometryTransformer(
        num_landmarks=num_landmarks,
        num_mesh=num_mesh,
        template_landmark=template_landmark,
        template_mesh=template_mesh,
        landmark2keypoint_knn_indices=landmark2keypoint_knn_indices,
        landmark2keypoint_knn_weights=landmark2keypoint_knn_weights,
        mesh2landmark_knn_indices=mesh2landmark_knn_indices,
        mesh2landmark_knn_weights=mesh2landmark_knn_weights,
        n_keypoint=n_keypoint,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_dim=output_dim,
        backbone_weights=backbone_weights,
        model_type=model_type,
        flatten_regression_outputs=flatten_regression_outputs,
        k_bins=k_bins,
        simdr_head_hidden_dim=simdr_head_hidden_dim,
        simdr_range_3d=simdr_range_3d,
        simdr_range_2d=simdr_range_2d,
        use_deformable_attention=use_deformable_attention,
        num_deformable_points=num_deformable_points,
        use_fast_aux_regression_heads=use_fast_aux_regression_heads,
        mesh_vertex_feature_dim=mesh_vertex_feature_dim,
        texture_feature_map_size=texture_feature_map_size,
        texture_output_size=texture_output_size,
        flip_uv_v=flip_uv_v,
        template_mesh_uv=template_mesh_uv,
    )


def create_simdr_geometry_transformer(
    num_landmarks: int,
    num_mesh: int,
    template_landmark: np.ndarray,
    template_mesh: np.ndarray,
    landmark2keypoint_knn_indices: np.ndarray,
    landmark2keypoint_knn_weights: np.ndarray,
    mesh2landmark_knn_indices: np.ndarray,
    mesh2landmark_knn_weights: np.ndarray,
    n_keypoint: int,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    output_dim: int = 5,
    backbone_weights: str = 'imagenet',
    k_bins: int = 256,
    simdr_head_hidden_dim: int = 256,
    simdr_range_3d: tuple[float, float] = (-0.5, 0.5),
    simdr_range_2d: tuple[float, float] = (-0.5, 0.5),
    use_deformable_attention: bool = True,
    num_deformable_points: int = 16,
    use_fast_aux_regression_heads: bool = False,
    mesh_vertex_feature_dim: int = 16,
    texture_feature_map_size: int = 256,
    texture_output_size: int = 1024,
    flip_uv_v: bool = True,
    template_mesh_uv: np.ndarray | None = None,
) -> SimDRGeometryTransformer:
    return SimDRGeometryTransformer(
        num_landmarks=num_landmarks,
        num_mesh=num_mesh,
        template_landmark=template_landmark,
        template_mesh=template_mesh,
        landmark2keypoint_knn_indices=landmark2keypoint_knn_indices,
        landmark2keypoint_knn_weights=landmark2keypoint_knn_weights,
        mesh2landmark_knn_indices=mesh2landmark_knn_indices,
        mesh2landmark_knn_weights=mesh2landmark_knn_weights,
        n_keypoint=n_keypoint,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_dim=output_dim,
        backbone_weights=backbone_weights,
        flatten_regression_outputs=False,
        k_bins=k_bins,
        simdr_head_hidden_dim=simdr_head_hidden_dim,
        simdr_range_3d=simdr_range_3d,
        simdr_range_2d=simdr_range_2d,
        use_deformable_attention=use_deformable_attention,
        num_deformable_points=num_deformable_points,
        use_fast_aux_regression_heads=use_fast_aux_regression_heads,
        mesh_vertex_feature_dim=mesh_vertex_feature_dim,
        texture_feature_map_size=texture_feature_map_size,
        texture_output_size=texture_output_size,
        flip_uv_v=flip_uv_v,
        template_mesh_uv=template_mesh_uv,
    )
