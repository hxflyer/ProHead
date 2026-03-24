import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dense_image_transformer import (
    DenseImageTransformer,
    PositionalEncoding2D,
    compute_dense_output_channels,
)
from data_utils.obj_io import load_uv_obj_file
from train_visualize_helper import load_combined_mesh_uv


def _normalize_imagenet(rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=rgb.device).view(1, 3, 1, 1)
    return (rgb - mean) / std


def _split_dense_prediction(
    pred: torch.Tensor,
    *,
    predict_basecolor: bool,
    predict_geo: bool,
    predict_normal: bool,
) -> dict[str, torch.Tensor | None]:
    expected_channels = compute_dense_output_channels(
        predict_basecolor=predict_basecolor,
        predict_geo=predict_geo,
        predict_normal=predict_normal,
    )
    channels = int(pred.shape[1])
    if channels != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} dense output channels, got {channels}"
        )

    channel_idx = 0
    pred_basecolor = None
    pred_geo = None
    pred_normal = None
    if predict_basecolor:
        pred_basecolor = pred[:, channel_idx : channel_idx + 3]
        channel_idx += 3
    if predict_geo:
        pred_geo = pred[:, channel_idx : channel_idx + 3]
        channel_idx += 3
    if predict_normal:
        # Dense2Geometry keeps the legacy 10-channel structure and ignores the
        # extra detail-normal triplet from the new dense stage.
        pred_normal = pred[:, channel_idx : channel_idx + 3]
        channel_idx += 3
        channel_idx += 3

    return {
        "basecolor": pred_basecolor,
        "geo": pred_geo,
        "normal": pred_normal,
        "mask_logits": pred[:, channel_idx : channel_idx + 1],
    }


def _load_matching_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    model_state = model.state_dict()
    filtered_state = {}
    skipped_keys: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state or model_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        filtered_state[key] = value
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    return skipped_keys, list(missing_keys) + list(unexpected_keys)


def _sample_texture_at_uv(texture_chw: torch.Tensor, uv: np.ndarray, device: torch.device) -> torch.Tensor:
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"Expected uv [N, 2], got {tuple(uv.shape)}")
    uv_t = torch.from_numpy(np.asarray(uv, dtype=np.float32)).to(device=device)
    grid = uv_t.clone()
    grid[:, 1] = 1.0 - grid[:, 1]
    grid = grid.view(1, -1, 1, 2)
    grid = grid.mul(2.0).sub(1.0)
    tex = texture_chw.unsqueeze(0).to(device=device, dtype=torch.float32)
    samples = F.grid_sample(tex, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return samples.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()


def _load_combined_mesh_triangle_faces(model_dir: str = "assets/topology") -> np.ndarray:
    part_files = [
        "mesh_head.obj",
        "mesh_eye_l.obj",
        "mesh_eye_r.obj",
        "mesh_mouth.obj",
    ]
    tris = []
    offset = 0
    for file_name in part_files:
        obj_path = os.path.join(model_dir, file_name)
        verts, uvs, _, v_faces, _, _ = load_uv_obj_file(obj_path, triangulate=True)
        if verts is None or uvs is None or v_faces is None:
            raise ValueError(f"Failed to load verts/uv/faces from {obj_path}")
        tris.append(np.asarray(v_faces, dtype=np.int32) + int(offset))
        offset += int(len(verts))
    return np.concatenate(tris, axis=0).astype(np.int32, copy=False)


def _remap_triangle_faces_after_vertex_filter(
    triangle_faces: np.ndarray,
    kept_vertex_indices: np.ndarray,
    original_vertex_count: int,
    vertex_positions: np.ndarray | None = None,
) -> np.ndarray:
    tri = np.asarray(triangle_faces, dtype=np.int64)
    if tri.size == 0:
        return np.zeros((0, 3), dtype=np.int32)

    kept = np.asarray(kept_vertex_indices, dtype=np.int64)
    remap = np.full((int(original_vertex_count),), -1, dtype=np.int64)
    remap[kept] = np.arange(kept.shape[0], dtype=np.int64)

    all_indices = np.arange(int(original_vertex_count), dtype=np.int64)
    filtered = all_indices[remap < 0]
    if filtered.size > 0:
        if vertex_positions is not None and vertex_positions.shape[0] == int(original_vertex_count):
            pos = np.asarray(vertex_positions, dtype=np.float32)
            kept_pos = pos[kept]
            filt_pos = pos[filtered]
            diff = filt_pos[:, None, :] - kept_pos[None, :, :]
            nn_idx = (diff * diff).sum(axis=-1).argmin(axis=-1)
        else:
            nn_idx = np.searchsorted(kept, filtered).clip(0, kept.shape[0] - 1)
        remap[filtered] = nn_idx

    return remap[tri].astype(np.int32, copy=False)


def _load_filtered_template_mesh(model_dir: str = "assets/topology") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    template_mesh = np.load(os.path.join(model_dir, "mesh_template.npy")).astype(np.float32)
    template_mesh_uv = load_combined_mesh_uv(model_dir=model_dir, copy=True).astype(np.float32, copy=False)
    template_mesh_faces = _load_combined_mesh_triangle_faces(model_dir=model_dir)
    template_mesh_full = template_mesh.copy()
    mesh_indices_path = os.path.join(model_dir, "mesh_indices.npy")
    if os.path.exists(mesh_indices_path):
        mesh_indices = np.load(mesh_indices_path).astype(np.int64, copy=False)
        if mesh_indices.max() < template_mesh.shape[0]:
            template_mesh = template_mesh[mesh_indices]
        if mesh_indices.max() < template_mesh_uv.shape[0]:
            template_mesh_uv = template_mesh_uv[mesh_indices]
        template_mesh_faces = _remap_triangle_faces_after_vertex_filter(
            template_mesh_faces,
            mesh_indices,
            original_vertex_count=int(template_mesh_full.shape[0]),
            vertex_positions=template_mesh_full[:, :3],
        )
    return template_mesh, template_mesh_uv, template_mesh_faces


def _build_directed_mesh_edges(triangle_faces: np.ndarray, num_vertices: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = np.asarray(triangle_faces, dtype=np.int64)
    if tri.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.ones((num_vertices,), dtype=np.float32),
        )
    edges = np.concatenate(
        [
            tri[:, [0, 1]],
            tri[:, [1, 2]],
            tri[:, [2, 0]],
            tri[:, [1, 0]],
            tri[:, [2, 1]],
            tri[:, [0, 2]],
        ],
        axis=0,
    )
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(edges, axis=0)
    degree = np.bincount(edges[:, 1], minlength=num_vertices).astype(np.float32)
    degree = np.maximum(degree, 1.0)
    return edges[:, 0], edges[:, 1], degree


@dataclass
class DenseStageConfig:
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    output_size: int = 1024
    transformer_map_size: int = 32
    backbone_weights: str = "imagenet"
    decoder_type: str = "multitask"


class LandmarkAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, image_memory: torch.Tensor) -> torch.Tensor:
        q = self.cross_norm_q(x)
        kv = self.cross_norm_kv(image_memory)
        cross_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x + cross_out

        q = self.self_norm(x)
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        x = x + self_out

        x = x + self.ffn(self.ffn_norm(x))
        return x


class MeshRefineBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class MeshGraphBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.neighbor_norm = nn.LayerNorm(d_model)
        self.update = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        neighbor = torch.zeros_like(x)
        neighbor.index_add_(1, edge_dst, x[:, edge_src, :])
        neighbor = neighbor / degree.view(1, -1, 1).clamp_min(1.0)
        update = self.update(torch.cat([self.self_norm(x), self.neighbor_norm(neighbor)], dim=-1))
        return x + update


class Dense2Geometry(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dense_stage_cfg: DenseStageConfig | None = None,
        dense_checkpoint: str = "",
        freeze_dense_stage: bool = True,
        image_memory_size: int = 16,
        search_size: int = 512,
        landmark_search_size: int = 256,
        local_search_radius: int = 12,
        local_min_candidates: int = 9,
        search_chunk_size: int = 1024,
        search_distance_threshold: float = 0.05,
        search_distance_floor: float = 0.02,
        search_mad_scale: float = 3.0,
        search_mask_threshold: float = 0.30,
        search_min_geo_magnitude: float = 0.02,
        min_search_candidates: int = 512,
        subpixel_radius: int = 2,
        subpixel_temperature: float = 1e-7,
        subpixel_max_drift: float = 3.0,
        model_dir: str = "assets/topology",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.output_dim = 6
        self.image_memory_size = int(max(4, image_memory_size))
        self.search_size = int(max(32, search_size))
        self.landmark_search_size = int(max(32, landmark_search_size))
        self.local_search_radius = int(max(1, local_search_radius))
        self.local_min_candidates = int(max(0, local_min_candidates))
        self.search_chunk_size = int(max(32, search_chunk_size))
        self.search_distance_threshold = float(search_distance_threshold)
        self.search_distance_floor = float(search_distance_floor)
        self.search_mad_scale = float(search_mad_scale)
        self.search_mask_threshold = float(search_mask_threshold)
        self.search_min_geo_magnitude = float(search_min_geo_magnitude)
        self.min_search_candidates = int(max(8, min_search_candidates))
        self.freeze_dense_stage = bool(freeze_dense_stage)
        self.subpixel_radius = int(max(1, subpixel_radius))
        self.subpixel_temperature = float(max(1e-6, subpixel_temperature))
        self.subpixel_max_drift = float(max(0.5, subpixel_max_drift))

        template_mesh, template_mesh_uv, template_mesh_faces = _load_filtered_template_mesh(model_dir=model_dir)
        self.num_mesh = int(template_mesh.shape[0])
        self.register_buffer("template_mesh", torch.from_numpy(template_mesh.astype(np.float32)))
        self.register_buffer("template_mesh_uv", torch.from_numpy(template_mesh_uv.astype(np.float32)))
        edge_src, edge_dst, edge_degree = _build_directed_mesh_edges(template_mesh_faces, self.num_mesh)
        self.register_buffer("mesh_edge_src", torch.from_numpy(edge_src.astype(np.int64)))
        self.register_buffer("mesh_edge_dst", torch.from_numpy(edge_dst.astype(np.int64)))
        self.register_buffer("mesh_edge_degree", torch.from_numpy(edge_degree.astype(np.float32)))

        mesh2landmark_idx = np.load(os.path.join(model_dir, "mesh2landmark_knn_indices.npy")).astype(np.int64)
        mesh2landmark_w = np.load(os.path.join(model_dir, "mesh2landmark_knn_weights.npy")).astype(np.float32)
        self.num_landmarks = int(mesh2landmark_idx.max()) + 1
        self.register_buffer("mesh2landmark_knn_indices", torch.from_numpy(mesh2landmark_idx))
        self.register_buffer("mesh2landmark_knn_weights", torch.from_numpy(mesh2landmark_w))

        landmark2keypoint_idx_path = os.path.join(model_dir, "landmark2keypoint_knn_indices.npy")
        landmark2keypoint_w_path = os.path.join(model_dir, "landmark2keypoint_knn_weights.npy")
        if os.path.exists(landmark2keypoint_idx_path) and os.path.exists(landmark2keypoint_w_path):
            landmark2keypoint_idx = np.load(landmark2keypoint_idx_path).astype(np.int64)
            landmark2keypoint_w = np.load(landmark2keypoint_w_path).astype(np.float32)
            if landmark2keypoint_idx.shape[0] != self.num_landmarks or landmark2keypoint_w.shape != landmark2keypoint_idx.shape:
                raise ValueError(
                    "landmark2keypoint KNN shape mismatch: "
                    f"indices={tuple(landmark2keypoint_idx.shape)} weights={tuple(landmark2keypoint_w.shape)} "
                    f"num_landmarks={self.num_landmarks}"
                )
        else:
            landmark2keypoint_idx = np.arange(self.num_landmarks, dtype=np.int64)[:, None]
            landmark2keypoint_w = np.ones((self.num_landmarks, 1), dtype=np.float32)
        self.num_keypoints = int(landmark2keypoint_idx.max()) + 1
        self.register_buffer("landmark2keypoint_knn_indices", torch.from_numpy(landmark2keypoint_idx))
        self.register_buffer("landmark2keypoint_knn_weights", torch.from_numpy(landmark2keypoint_w))

        geo_atlas = np.load(os.path.join(model_dir, "geo_feature_atlas.npy")).astype(np.float32)
        if geo_atlas.ndim != 3 or geo_atlas.shape[0] != 3:
            raise ValueError(f"Expected geo atlas [3, H, W], got {tuple(geo_atlas.shape)}")
        atlas_t = torch.from_numpy(geo_atlas)
        mesh_geo_codes = _sample_texture_at_uv(atlas_t, template_mesh_uv, device=torch.device("cpu")).cpu().numpy()
        landmark_geo_codes = self._pool_mesh_to_landmarks_cpu(mesh_geo_codes, mesh2landmark_idx, mesh2landmark_w)
        self.register_buffer("mesh_geo_codes", torch.from_numpy(mesh_geo_codes.astype(np.float32)))
        self.register_buffer("landmark_geo_codes", torch.from_numpy(landmark_geo_codes.astype(np.float32)))

        dense_cfg = dense_stage_cfg or DenseStageConfig()
        self.dense_stage = DenseImageTransformer(
            d_model=int(dense_cfg.d_model),
            nhead=int(dense_cfg.nhead),
            num_layers=int(dense_cfg.num_layers),
            predict_basecolor=True,
            predict_geo=True,
            predict_normal=True,
            output_size=int(dense_cfg.output_size),
            transformer_map_size=int(dense_cfg.transformer_map_size),
            backbone_weights=str(dense_cfg.backbone_weights),
            decoder_type=str(dense_cfg.decoder_type),
        )
        if dense_checkpoint:
            self.load_dense_stage(dense_checkpoint)
        if self.freeze_dense_stage:
            for param in self.dense_stage.parameters():
                param.requires_grad = False

        self.image_proj_4 = nn.Conv2d(128, self.d_model, kernel_size=1)
        self.image_proj_8 = nn.Conv2d(256, self.d_model, kernel_size=1)
        self.image_proj_16 = nn.Conv2d(512, self.d_model, kernel_size=1)
        self.image_pos_embed = PositionalEncoding2D(self.d_model, max_h=256, max_w=256)
        self.level_embed = nn.Parameter(torch.zeros(3, self.d_model))

        self.vertex_image_proj = nn.Linear(128 + 256 + 512, self.d_model)
        self.vertex_dense_proj = nn.Sequential(
            nn.Linear(10, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.vertex_semantic_proj = nn.Linear(3, self.d_model)
        self.vertex_coord_proj = nn.Sequential(
            nn.Linear(4, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.vertex_embed = nn.Embedding(self.num_mesh, self.d_model)
        self.vertex_concat_fuse = nn.Sequential(
            nn.LayerNorm(self.d_model * 5),
            nn.Linear(self.d_model * 5, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.d_model),
        )

        self.landmark_semantic_proj = nn.Linear(3, self.d_model)
        self.landmark_coord_proj = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.landmark_embed = nn.Embedding(self.num_landmarks, self.d_model)
        self.landmark_blocks = nn.ModuleList(
            [LandmarkAttentionBlock(self.d_model, nhead=nhead) for _ in range(int(num_layers))]
        )
        self.landmark_norm = nn.LayerNorm(self.d_model)
        self.mesh_context_proj = nn.Linear(self.d_model, self.d_model)
        self.mesh_graph_blocks = nn.ModuleList(
            [MeshGraphBlock(self.d_model) for _ in range(2)]
        )
        self.mesh_refine_blocks = nn.ModuleList(
            [MeshRefineBlock(self.d_model) for _ in range(5)]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.output_dim),
        )

    @staticmethod
    def _pool_mesh_to_landmarks_cpu(
        mesh_values: np.ndarray,
        mesh2landmark_idx: np.ndarray,
        mesh2landmark_w: np.ndarray,
    ) -> np.ndarray:
        num_landmarks = int(mesh2landmark_idx.max()) + 1
        out = np.zeros((num_landmarks, mesh_values.shape[1]), dtype=np.float32)
        denom = np.zeros((num_landmarks, 1), dtype=np.float32)
        for k in range(mesh2landmark_idx.shape[1]):
            idx = mesh2landmark_idx[:, k]
            w = mesh2landmark_w[:, k : k + 1]
            np.add.at(out, idx, mesh_values * w)
            np.add.at(denom, idx, w)
        return out / np.clip(denom, 1e-6, None)

    def load_dense_stage(self, checkpoint_path: str) -> None:
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Dense checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        skipped_keys, load_notes = _load_matching_state_dict(self.dense_stage, state_dict)
        print(f"[Dense2Geometry] Loaded dense stage from {checkpoint_path}")
        if skipped_keys:
            print(f"[Dense2Geometry] Skipped dense-stage keys: {skipped_keys[:10]}")
        if load_notes:
            print(f"[Dense2Geometry] Dense-stage load notes: {load_notes[:10]}")

    def _run_dense_stage(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        rgb_in = _normalize_imagenet(rgb)

        def _forward_impl():
            features = self.dense_stage.backbone(rgb_in)
            f4 = features["stride4"].contiguous()
            f8 = features["stride8"].contiguous()
            f16 = features["stride16"].contiguous()

            x = self.dense_stage.fusion_proj(f16)
            x = x + self.dense_stage.pos_embed(x)

            token_map = self.dense_stage.token_pool(x)
            b, c, h, w = token_map.shape
            tokens = token_map.flatten(2).permute(0, 2, 1)
            for layer in self.dense_stage.transformer_layers:
                tokens = layer(tokens)
            tokens = self.dense_stage.transformer_norm(tokens)

            token_map_out = tokens.permute(0, 2, 1).reshape(b, c, h, w)
            token_map_out = F.interpolate(
                token_map_out,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = x + self.dense_stage.post_attn_proj(token_map_out)
            dense_pred = self.dense_stage.decoder(fused, f8, f4)
            parts = _split_dense_prediction(
                dense_pred,
                predict_basecolor=bool(getattr(self.dense_stage, "predict_basecolor", True)),
                predict_geo=bool(getattr(self.dense_stage, "predict_geo", True)),
                predict_normal=bool(getattr(self.dense_stage, "predict_normal", True)),
            )
            return f4, f8, f16, parts

        if self.freeze_dense_stage:
            with torch.no_grad():
                return _forward_impl()
        return _forward_impl()

    def _build_image_memory(self, f4: torch.Tensor, f8: torch.Tensor, f16: torch.Tensor) -> torch.Tensor:
        levels = []
        for level_idx, (feat, proj) in enumerate(
            [
                (f4, self.image_proj_4),
                (f8, self.image_proj_8),
                (f16, self.image_proj_16),
            ]
        ):
            x = proj(feat)
            x = F.adaptive_avg_pool2d(x, output_size=(self.image_memory_size, self.image_memory_size))
            x = x + self.image_pos_embed(x) + self.level_embed[level_idx].view(1, -1, 1, 1)
            levels.append(x.flatten(2).transpose(1, 2))
        return torch.cat(levels, dim=1)

    def _compute_adaptive_threshold(self, distances: torch.Tensor) -> torch.Tensor:
        finite = distances[torch.isfinite(distances)]
        if finite.numel() == 0:
            return distances.new_tensor(self.search_distance_threshold)
        median = finite.median()
        mad = (finite - median).abs().median()
        adaptive = median + self.search_mad_scale * mad
        threshold = torch.minimum(adaptive, distances.new_tensor(self.search_distance_threshold))
        threshold = torch.maximum(threshold, distances.new_tensor(self.search_distance_floor))
        return threshold

    def _nearest_feature_search(
        self,
        query_codes: torch.Tensor,
        candidate_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cand = candidate_codes.float()
        query = query_codes.float()
        cand_sq = (cand * cand).sum(dim=1)
        best_dist2 = []
        best_idx = []
        cand_t = cand.transpose(0, 1).contiguous()
        for start in range(0, query.shape[0], self.search_chunk_size):
            end = min(start + self.search_chunk_size, query.shape[0])
            q = query[start:end]
            q_sq = (q * q).sum(dim=1, keepdim=True)
            dist2 = q_sq + cand_sq.unsqueeze(0) - 2.0 * (q @ cand_t)
            dist2 = dist2.clamp_min_(0.0)
            chunk_dist2, chunk_idx = dist2.min(dim=1)
            best_dist2.append(chunk_dist2)
            best_idx.append(chunk_idx)
        return torch.cat(best_idx, dim=0), torch.cat(best_dist2, dim=0).sqrt()

    def _build_search_candidates(
        self,
        pred_geo: torch.Tensor,
        pred_mask_logits: torch.Tensor,
        search_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        search_geo = F.interpolate(
            pred_geo.unsqueeze(0),
            size=(search_size, search_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        search_mask = F.interpolate(
            pred_mask_logits.unsqueeze(0),
            size=(search_size, search_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        geo_flat = search_geo.permute(1, 2, 0).reshape(-1, 3)
        mask_prob = torch.sigmoid(search_mask[0]).reshape(-1)
        geo_mag = torch.linalg.norm(geo_flat, dim=1)
        valid = geo_mag > self.search_min_geo_magnitude
        valid = valid & (mask_prob > self.search_mask_threshold)
        if int(valid.sum().item()) < self.min_search_candidates:
            valid = geo_mag > self.search_min_geo_magnitude
        if int(valid.sum().item()) < self.min_search_candidates:
            topk = min(self.min_search_candidates, int(geo_mag.numel()))
            _, top_idx = torch.topk(geo_mag, k=max(topk, 1), largest=True, sorted=False)
            valid = torch.zeros_like(geo_mag, dtype=torch.bool)
            valid[top_idx] = True

        ys, xs = torch.meshgrid(
            torch.arange(search_size, device=pred_geo.device, dtype=torch.float32),
            torch.arange(search_size, device=pred_geo.device, dtype=torch.float32),
            indexing="ij",
        )
        uv_grid = torch.stack(
            [
                (xs + 0.5) / float(search_size),
                (ys + 0.5) / float(search_size),
            ],
            dim=-1,
        ).view(-1, 2)

        return geo_flat[valid], uv_grid[valid]

    def _broadcast_landmark_uv_to_mesh(
        self,
        landmark_uv: torch.Tensor,
        landmark_match: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.mesh2landmark_knn_indices
        w = self.mesh2landmark_knn_weights
        neighbor_uv = landmark_uv[idx]
        neighbor_match = landmark_match[idx]
        uv_weight = w * neighbor_match
        denom = uv_weight.sum(dim=1, keepdim=True)
        coarse_uv = (neighbor_uv * uv_weight.unsqueeze(-1)).sum(dim=1) / denom.clamp_min(1e-6)
        coarse_valid = denom.squeeze(-1) > 1e-6
        coarse_uv[~coarse_valid] = -1.0
        return coarse_uv, coarse_valid

    def _local_refine_mesh_search(
        self,
        pred_geo: torch.Tensor,
        pred_mask_logits: torch.Tensor,
        coarse_uv: torch.Tensor,
        coarse_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        search_geo = F.interpolate(
            pred_geo.unsqueeze(0),
            size=(self.search_size, self.search_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        search_mask = F.interpolate(
            pred_mask_logits.unsqueeze(0),
            size=(self.search_size, self.search_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        geo_hw = search_geo.permute(1, 2, 0).contiguous()
        mask_prob_hw = torch.sigmoid(search_mask[0]).contiguous()

        center_x = torch.round(coarse_uv[:, 0] * float(self.search_size) - 0.5).long()
        center_y = torch.round(coarse_uv[:, 1] * float(self.search_size) - 0.5).long()
        center_x = center_x.clamp(0, self.search_size - 1)
        center_y = center_y.clamp(0, self.search_size - 1)

        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-self.local_search_radius, self.local_search_radius + 1, device=pred_geo.device),
            torch.arange(-self.local_search_radius, self.local_search_radius + 1, device=pred_geo.device),
            indexing="ij",
        )
        offsets_y = offsets_y.reshape(1, -1)
        offsets_x = offsets_x.reshape(1, -1)

        patch_x = (center_x.unsqueeze(1) + offsets_x).clamp(0, self.search_size - 1)
        patch_y = (center_y.unsqueeze(1) + offsets_y).clamp(0, self.search_size - 1)

        candidate_codes = geo_hw[patch_y, patch_x]
        candidate_mask_prob = mask_prob_hw[patch_y, patch_x]
        candidate_geo_mag = torch.linalg.norm(candidate_codes, dim=-1)
        valid = candidate_geo_mag > self.search_min_geo_magnitude
        valid = valid & (candidate_mask_prob > self.search_mask_threshold)

        if self.local_min_candidates > 0:
            enough_valid = valid.sum(dim=1) >= self.local_min_candidates
            valid = torch.where(
                enough_valid.unsqueeze(1),
                valid,
                candidate_geo_mag > self.search_min_geo_magnitude,
            )

        still_empty = valid.sum(dim=1) <= 0
        valid = torch.where(still_empty.unsqueeze(1), torch.ones_like(valid, dtype=torch.bool), valid)
        valid = valid & coarse_valid.unsqueeze(1)

        query_codes = self.mesh_geo_codes.float()
        dist2 = ((candidate_codes.float() - query_codes.unsqueeze(1)) ** 2).sum(dim=-1)
        dist2 = dist2.masked_fill(~valid, float("inf"))
        best_dist2, best_idx = dist2.min(dim=1)
        distances = best_dist2.clamp_min(0.0).sqrt()
        threshold = self._compute_adaptive_threshold(distances)
        accept = coarse_valid & torch.isfinite(distances) & (distances <= threshold)

        candidate_uv = torch.stack(
            [
                (patch_x.float() + 0.5) / float(self.search_size),
                (patch_y.float() + 0.5) / float(self.search_size),
            ],
            dim=-1,
        )
        row_idx = torch.arange(candidate_uv.shape[0], device=pred_geo.device)
        searched_uv = candidate_uv[row_idx, best_idx].to(dtype=torch.float32)
        searched_uv[~accept] = -1.0
        distances = torch.where(
            torch.isfinite(distances),
            distances,
            torch.full_like(distances, self.search_distance_threshold),
        )
        return searched_uv, accept.float(), distances

    def _subpixel_refine_mesh_search(
        self,
        pred_geo: torch.Tensor,
        searched_uv: torch.Tensor,
        accept: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage 3: re-search at native 1024 resolution + subpixel refinement.

        Stage 2 searched a downsampled 512 grid, so its UV has 1/512 quantization.
        This stage:
          1) Re-searches a local patch at native 1024 resolution (hard best-pixel)
          2) Applies soft-argmin around that best pixel for subpixel precision
        """
        H = pred_geo.shape[1]
        W = pred_geo.shape[2]
        geo_hw = pred_geo.permute(1, 2, 0).contiguous()  # [H, W, 3]

        accept_bool = accept > 0.5
        out_uv = searched_uv.clone()
        out_dist = distances.clone()

        # --- Step 1: hard re-search at 1024 resolution ---
        # Stage-2 UV has 1/512 quantization. At 1024, each 512-pixel spans ~2 pixels.
        # Use subpixel_radius to define the search patch (default radius=2 -> 5x5).
        center_x = torch.round(searched_uv[:, 0] * float(W) - 0.5).long().clamp(0, W - 1)
        center_y = torch.round(searched_uv[:, 1] * float(H) - 0.5).long().clamp(0, H - 1)

        r = self.subpixel_radius
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-r, r + 1, device=pred_geo.device),
            torch.arange(-r, r + 1, device=pred_geo.device),
            indexing="ij",
        )
        offsets_y = offsets_y.reshape(1, -1)
        offsets_x = offsets_x.reshape(1, -1)

        patch_x = (center_x.unsqueeze(1) + offsets_x).clamp(0, W - 1)
        patch_y = (center_y.unsqueeze(1) + offsets_y).clamp(0, H - 1)

        # Sample geo codes at all patch locations [N, num_patch, 3]
        patch_geo = geo_hw[patch_y, patch_x]
        query_codes = self.mesh_geo_codes.float()  # [N, 3]

        # Find best discrete pixel in the patch
        diff = patch_geo.float() - query_codes.unsqueeze(1)  # [N, num_patch, 3]
        dist2 = (diff * diff).sum(dim=-1)  # [N, num_patch]
        best_dist2, best_idx = dist2.min(dim=1)

        # Best pixel UV at 1024 resolution
        row_idx = torch.arange(patch_x.shape[0], device=pred_geo.device)
        best_px_x = patch_x[row_idx, best_idx]
        best_px_y = patch_y[row_idx, best_idx]
        hard_uv = torch.stack(
            [(best_px_x.float() + 0.5) / float(W), (best_px_y.float() + 0.5) / float(H)],
            dim=-1,
        )
        hard_dist = best_dist2.clamp_min(0.0).sqrt()

        # --- Step 2: soft-argmin subpixel refinement around the best pixel ---
        # Small patch (3x3) centered on the best 1024-pixel for subpixel interpolation
        sub_offsets_y, sub_offsets_x = torch.meshgrid(
            torch.arange(-1, 2, device=pred_geo.device),
            torch.arange(-1, 2, device=pred_geo.device),
            indexing="ij",
        )
        sub_offsets_y = sub_offsets_y.reshape(1, -1)
        sub_offsets_x = sub_offsets_x.reshape(1, -1)

        sub_patch_x = (best_px_x.unsqueeze(1) + sub_offsets_x).clamp(0, W - 1)
        sub_patch_y = (best_px_y.unsqueeze(1) + sub_offsets_y).clamp(0, H - 1)

        sub_geo = geo_hw[sub_patch_y, sub_patch_x]  # [N, 9, 3]
        sub_diff = sub_geo.float() - query_codes.unsqueeze(1)
        sub_dist2 = (sub_diff * sub_diff).sum(dim=-1)  # [N, 9]

        # Soft-argmin weights
        weights = torch.exp(-sub_dist2 / self.subpixel_temperature)  # [N, 9]
        w_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

        sub_uv = torch.stack(
            [(sub_patch_x.float() + 0.5) / float(W), (sub_patch_y.float() + 0.5) / float(H)],
            dim=-1,
        )  # [N, 9, 2]
        refined_uv = (sub_uv * weights.unsqueeze(-1)).sum(dim=1) / w_sum  # [N, 2]
        refined_dist = (sub_dist2.sqrt() * weights).sum(dim=1) / w_sum.squeeze(-1)

        # Drift check: reject if refined UV moved too far from stage-2
        drift_px = (refined_uv - searched_uv) * torch.tensor([float(W), float(H)], device=pred_geo.device)
        drift = torch.linalg.norm(drift_px, dim=1)
        drift_ok = drift <= self.subpixel_max_drift

        # Apply: use subpixel result where drift is ok, else hard result
        update_mask = accept_bool & drift_ok
        hard_only_mask = accept_bool & ~drift_ok
        out_uv[update_mask] = refined_uv[update_mask]
        out_dist[update_mask] = refined_dist[update_mask]
        out_uv[hard_only_mask] = hard_uv[hard_only_mask]
        out_dist[hard_only_mask] = hard_dist[hard_only_mask]

        return out_uv, accept, out_dist

    def _search_single_sample(
        self,
        pred_geo: torch.Tensor,
        pred_mask_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_codes, candidate_uv = self._build_search_candidates(
            pred_geo=pred_geo,
            pred_mask_logits=pred_mask_logits,
            search_size=self.landmark_search_size,
        )
        if candidate_codes.numel() == 0:
            return (
                torch.full((self.num_mesh, 2), -1.0, device=pred_geo.device, dtype=torch.float32),
                torch.zeros((self.num_mesh,), device=pred_geo.device, dtype=torch.float32),
                torch.full((self.num_mesh,), self.search_distance_threshold, device=pred_geo.device, dtype=torch.float32),
            )

        nearest_idx, distances = self._nearest_feature_search(self.landmark_geo_codes, candidate_codes)
        threshold = self._compute_adaptive_threshold(distances)
        landmark_accept = torch.isfinite(distances) & (distances <= threshold)

        landmark_uv = candidate_uv[nearest_idx].to(dtype=torch.float32)
        landmark_uv[~landmark_accept] = -1.0
        coarse_uv, coarse_valid = self._broadcast_landmark_uv_to_mesh(landmark_uv, landmark_accept.float())
        searched_uv, accept, distances = self._local_refine_mesh_search(
            pred_geo=pred_geo,
            pred_mask_logits=pred_mask_logits,
            coarse_uv=coarse_uv,
            coarse_valid=coarse_valid,
        )
        return self._subpixel_refine_mesh_search(
            pred_geo=pred_geo,
            searched_uv=searched_uv,
            accept=accept,
            distances=distances,
        )

    def _search_mesh_positions(
        self,
        pred_geo: torch.Tensor,
        pred_mask_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        searched = []
        matched = []
        distances = []
        with torch.no_grad():
            for b in range(pred_geo.shape[0]):
                uv_b, match_b, dist_b = self._search_single_sample(pred_geo[b], pred_mask_logits[b])
                searched.append(uv_b)
                matched.append(match_b)
                distances.append(dist_b)
        return (
            torch.stack(searched, dim=0),
            torch.stack(matched, dim=0),
            torch.stack(distances, dim=0),
        )

    def _sample_feature_map(
        self,
        feat_map: torch.Tensor,
        uv: torch.Tensor,
        match_mask: torch.Tensor,
    ) -> torch.Tensor:
        uv_safe = torch.where(match_mask.unsqueeze(-1) > 0.5, uv, torch.zeros_like(uv))
        grid = uv_safe.mul(2.0).sub(1.0)
        grid = grid.view(feat_map.shape[0], uv.shape[1], 1, 2)
        sampled = F.grid_sample(
            feat_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).contiguous()
        return sampled * match_mask.unsqueeze(-1)

    def _pool_mesh_to_landmarks(
        self,
        mesh_tokens: torch.Tensor,
        searched_uv: torch.Tensor,
        match_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = mesh_tokens.device
        idx = self.mesh2landmark_knn_indices.to(device=device)
        w = self.mesh2landmark_knn_weights.to(device=device, dtype=mesh_tokens.dtype)

        landmark_tokens = torch.zeros(
            (mesh_tokens.shape[0], self.num_landmarks, mesh_tokens.shape[-1]),
            device=device,
            dtype=mesh_tokens.dtype,
        )
        landmark_uv = torch.zeros(
            (mesh_tokens.shape[0], self.num_landmarks, 2),
            device=device,
            dtype=searched_uv.dtype,
        )
        token_denom = torch.zeros(
            (mesh_tokens.shape[0], self.num_landmarks, 1),
            device=device,
            dtype=mesh_tokens.dtype,
        )
        uv_denom = torch.zeros(
            (mesh_tokens.shape[0], self.num_landmarks, 1),
            device=device,
            dtype=searched_uv.dtype,
        )

        for k in range(idx.shape[1]):
            idx_k = idx[:, k]
            w_k = w[:, k].view(1, -1, 1)
            landmark_tokens.index_add_(1, idx_k, mesh_tokens * w_k)
            token_denom.index_add_(1, idx_k, w_k.expand(mesh_tokens.shape[0], -1, -1))

            uv_weight = w_k.to(dtype=searched_uv.dtype) * match_mask.unsqueeze(-1).to(dtype=searched_uv.dtype)
            landmark_uv.index_add_(1, idx_k, searched_uv * uv_weight)
            uv_denom.index_add_(1, idx_k, uv_weight)

        landmark_tokens = landmark_tokens / token_denom.clamp_min(1e-6)
        landmark_uv = landmark_uv / uv_denom.clamp_min(1e-6)
        landmark_valid = (uv_denom.squeeze(-1) > 1e-6).float()
        return landmark_tokens, landmark_uv, landmark_valid

    def _pool_landmarks_to_keypoints(
        self,
        landmark_values: torch.Tensor,
        landmark_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.landmark2keypoint_knn_indices.to(device=landmark_values.device)
        w = self.landmark2keypoint_knn_weights.to(device=landmark_values.device, dtype=landmark_values.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)

        keypoint_values = torch.zeros(
            (landmark_values.shape[0], self.num_keypoints, landmark_values.shape[-1]),
            device=landmark_values.device,
            dtype=landmark_values.dtype,
        )
        keypoint_conf_sum = torch.zeros(
            (landmark_values.shape[0], self.num_keypoints, 1),
            device=landmark_values.device,
            dtype=landmark_values.dtype,
        )
        keypoint_weight_sum = torch.zeros(
            (landmark_values.shape[0], self.num_keypoints, 1),
            device=landmark_values.device,
            dtype=landmark_values.dtype,
        )
        conf = landmark_confidence.to(device=landmark_values.device, dtype=landmark_values.dtype).unsqueeze(-1)

        for k in range(idx.shape[1]):
            idx_k = idx[:, k]
            w_k = w[:, k].view(1, -1, 1)
            weighted_conf = w_k * conf
            keypoint_values.index_add_(1, idx_k, landmark_values * weighted_conf)
            keypoint_conf_sum.index_add_(1, idx_k, weighted_conf)
            keypoint_weight_sum.index_add_(1, idx_k, w_k.expand(landmark_values.shape[0], -1, -1))

        keypoint_values = keypoint_values / keypoint_conf_sum.clamp_min(1e-6)
        keypoint_conf = (keypoint_conf_sum / keypoint_weight_sum.clamp_min(1e-6)).squeeze(-1).clamp(0.0, 1.0)
        return keypoint_values, keypoint_conf

    def _pool_mesh_values_to_landmarks(
        self,
        mesh_values: torch.Tensor,
        mesh_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.mesh2landmark_knn_indices.to(device=mesh_values.device)
        w = self.mesh2landmark_knn_weights.to(device=mesh_values.device, dtype=mesh_values.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)

        landmark_values = torch.zeros(
            (mesh_values.shape[0], self.num_landmarks, mesh_values.shape[-1]),
            device=mesh_values.device,
            dtype=mesh_values.dtype,
        )
        landmark_conf_sum = torch.zeros(
            (mesh_values.shape[0], self.num_landmarks, 1),
            device=mesh_values.device,
            dtype=mesh_values.dtype,
        )
        landmark_weight_sum = torch.zeros(
            (mesh_values.shape[0], self.num_landmarks, 1),
            device=mesh_values.device,
            dtype=mesh_values.dtype,
        )
        conf = mesh_confidence.to(device=mesh_values.device, dtype=mesh_values.dtype).unsqueeze(-1)

        for k in range(idx.shape[1]):
            idx_k = idx[:, k]
            w_k = w[:, k].view(1, -1, 1)
            weighted_conf = w_k * conf
            landmark_values.index_add_(1, idx_k, mesh_values * weighted_conf)
            landmark_conf_sum.index_add_(1, idx_k, weighted_conf)
            landmark_weight_sum.index_add_(1, idx_k, w_k.expand(mesh_values.shape[0], -1, -1))

        landmark_values = landmark_values / landmark_conf_sum.clamp_min(1e-6)
        landmark_conf = (landmark_conf_sum / landmark_weight_sum.clamp_min(1e-6)).squeeze(-1).clamp(0.0, 1.0)
        return landmark_values, landmark_conf

    def _pool_mesh_values_to_keypoints(
        self,
        mesh_values: torch.Tensor,
        mesh_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        landmark_values, landmark_conf = self._pool_mesh_values_to_landmarks(mesh_values, mesh_confidence)
        return self._pool_landmarks_to_keypoints(landmark_values, landmark_conf)

    def _broadcast_keypoints_to_landmarks(self, keypoint_values: torch.Tensor) -> torch.Tensor:
        idx = self.landmark2keypoint_knn_indices.to(device=keypoint_values.device)
        w = self.landmark2keypoint_knn_weights.to(device=keypoint_values.device, dtype=keypoint_values.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        neighbor_values = keypoint_values[:, idx, :]
        return (neighbor_values * w.unsqueeze(0).unsqueeze(-1)).sum(dim=2)

    def _broadcast_landmarks_to_mesh(self, landmark_tokens: torch.Tensor) -> torch.Tensor:
        idx = self.mesh2landmark_knn_indices.to(device=landmark_tokens.device)
        w = self.mesh2landmark_knn_weights.to(device=landmark_tokens.device, dtype=landmark_tokens.dtype)
        neighbor_tokens = landmark_tokens[:, idx, :]
        return (neighbor_tokens * w.unsqueeze(0).unsqueeze(-1)).sum(dim=2)

    def refine_mesh_uv_with_search(
        self,
        mesh_coords: torch.Tensor,
        searched_uv: torch.Tensor,
        match_mask: torch.Tensor,
        search_distance: torch.Tensor,
        image_size: int,
        num_iters: int = 3,
        step_size: float = 0.9,
        max_step_px: float = 1.5,
        smooth_blend: float = 0.20,
        confidence_power: float = 1.5,
        keypoint_gain: float = 1.0,
        keypoint_max_step_px: float = 2.0,
        landmark_refine_blend: float = 0.60,
        keypoint_only: bool = False,
        direct_snap_gain: float = 1.0,
        direct_snap_max_step_px: float = 4.0,
        direct_diffuse_iters: int = 3,
        direct_diffuse_blend: float = 0.70,
    ) -> torch.Tensor:
        if int(num_iters) <= 0 or int(image_size) <= 0:
            return mesh_coords

        refined = mesh_coords.clone()
        uv = refined[..., 3:5]
        match = match_mask.to(device=uv.device, dtype=uv.dtype).clamp(0.0, 1.0)
        distance = torch.nan_to_num(
            search_distance.to(device=uv.device, dtype=uv.dtype),
            nan=float(self.search_distance_threshold),
            posinf=float(self.search_distance_threshold),
            neginf=0.0,
        )
        tau = float(max(self.search_distance_threshold, self.search_distance_floor, 1e-6))
        confidence = match * torch.exp(-distance / tau)
        if float(confidence_power) != 1.0:
            confidence = confidence.pow(float(max(confidence_power, 0.0)))
        if not bool((confidence > 0).any().item()):
            return refined

        idx = self.mesh2landmark_knn_indices.to(device=uv.device)
        w = self.mesh2landmark_knn_weights.to(device=uv.device, dtype=uv.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        edge_src = self.mesh_edge_src.to(device=uv.device)
        edge_dst = self.mesh_edge_dst.to(device=uv.device)
        edge_degree = self.mesh_edge_degree.to(device=uv.device, dtype=uv.dtype)

        batch_size = uv.shape[0]
        num_landmarks = self.num_landmarks
        max_step_uv = float(max(max_step_px, 0.0)) / float(max(int(image_size), 1))
        step = float(max(step_size, 0.0))
        blend = float(min(max(smooth_blend, 0.0), 1.0))
        keypoint_stage_gain = float(max(keypoint_gain, 0.0))
        keypoint_max_step_uv = float(max(keypoint_max_step_px, 0.0)) / float(max(int(image_size), 1))
        landmark_refine = float(min(max(landmark_refine_blend, 0.0), 1.0))
        use_keypoint_only = bool(keypoint_only)

        if use_keypoint_only and keypoint_stage_gain > 0.0 and self.num_keypoints > 0:
            base_uv = uv.clone()
            keypoint_offset = torch.zeros((batch_size, self.num_keypoints, 2), device=uv.device, dtype=uv.dtype)

            for _ in range(int(num_iters)):
                landmark_from_keypoint = self._broadcast_keypoints_to_landmarks(keypoint_offset)
                mesh_delta = self._broadcast_landmarks_to_mesh(landmark_from_keypoint)

                if blend > 0.0 and edge_src.numel() > 0:
                    neighbor = torch.zeros_like(mesh_delta)
                    neighbor.index_add_(1, edge_dst, mesh_delta[:, edge_src, :])
                    neighbor = neighbor / edge_degree.view(1, -1, 1).clamp_min(1.0)
                    mesh_delta = (1.0 - blend) * mesh_delta + blend * neighbor

                uv_candidate = (base_uv + mesh_delta).clamp(0.0, 1.0)
                residual = searched_uv.to(device=uv.device, dtype=uv.dtype) - uv_candidate
                residual = torch.where(match.unsqueeze(-1) > 0.5, residual, torch.zeros_like(residual))

                keypoint_update, _ = self._pool_mesh_values_to_keypoints(residual, confidence)
                keypoint_update = keypoint_update * keypoint_stage_gain
                if keypoint_max_step_uv > 0.0:
                    kp_norm = torch.linalg.norm(keypoint_update, dim=-1, keepdim=True)
                    kp_scale = (keypoint_max_step_uv / kp_norm.clamp_min(1e-6)).clamp(max=1.0)
                    keypoint_update = keypoint_update * kp_scale

                keypoint_offset = keypoint_offset + step * keypoint_update

            landmark_delta = self._broadcast_keypoints_to_landmarks(keypoint_offset)
            mesh_delta = self._broadcast_landmarks_to_mesh(landmark_delta)
            if blend > 0.0 and edge_src.numel() > 0:
                neighbor = torch.zeros_like(mesh_delta)
                neighbor.index_add_(1, edge_dst, mesh_delta[:, edge_src, :])
                neighbor = neighbor / edge_degree.view(1, -1, 1).clamp_min(1.0)
                mesh_delta = (1.0 - blend) * mesh_delta + blend * neighbor

            uv = (base_uv + mesh_delta).clamp(0.0, 1.0)
            refined[..., 3:5] = uv
            return refined

        for _ in range(int(num_iters)):
            residual = searched_uv.to(device=uv.device, dtype=uv.dtype) - uv
            residual = torch.where(match.unsqueeze(-1) > 0.5, residual, torch.zeros_like(residual))
            conf = confidence.unsqueeze(-1)

            landmark_delta = torch.zeros((batch_size, num_landmarks, 2), device=uv.device, dtype=uv.dtype)
            landmark_conf_sum = torch.zeros((batch_size, num_landmarks, 1), device=uv.device, dtype=uv.dtype)
            landmark_weight_sum = torch.zeros((batch_size, num_landmarks, 1), device=uv.device, dtype=uv.dtype)

            for k in range(idx.shape[1]):
                idx_k = idx[:, k]
                w_k = w[:, k].view(1, -1, 1)
                weighted_conf = w_k * conf
                landmark_delta.index_add_(1, idx_k, residual * weighted_conf)
                landmark_conf_sum.index_add_(1, idx_k, weighted_conf)
                landmark_weight_sum.index_add_(1, idx_k, w_k.expand(batch_size, -1, -1))

            landmark_target_delta = landmark_delta / landmark_conf_sum.clamp_min(1e-6)
            landmark_conf = (landmark_conf_sum / landmark_weight_sum.clamp_min(1e-6)).squeeze(-1).clamp(0.0, 1.0)

            if keypoint_stage_gain > 0.0 and self.num_keypoints > 0:
                keypoint_delta, keypoint_conf = self._pool_landmarks_to_keypoints(
                    landmark_target_delta,
                    landmark_conf,
                )
                keypoint_delta = keypoint_delta * keypoint_stage_gain
                if keypoint_max_step_uv > 0.0:
                    kp_norm = torch.linalg.norm(keypoint_delta, dim=-1, keepdim=True)
                    kp_scale = (keypoint_max_step_uv / kp_norm.clamp_min(1e-6)).clamp(max=1.0)
                    keypoint_delta = keypoint_delta * kp_scale

                landmark_from_keypoint = self._broadcast_keypoints_to_landmarks(keypoint_delta)
                keypoint_landmark_conf = self._broadcast_keypoints_to_landmarks(keypoint_conf.unsqueeze(-1)).squeeze(-1)
                if use_keypoint_only:
                    landmark_delta = landmark_from_keypoint
                else:
                    landmark_delta = landmark_from_keypoint + landmark_refine * (landmark_target_delta - landmark_from_keypoint)
                landmark_conf = torch.maximum(landmark_conf, keypoint_landmark_conf.clamp(0.0, 1.0))
            else:
                landmark_delta = landmark_target_delta

            if max_step_uv > 0.0:
                lm_norm = torch.linalg.norm(landmark_delta, dim=-1, keepdim=True)
                lm_scale = (max_step_uv / lm_norm.clamp_min(1e-6)).clamp(max=1.0)
                landmark_delta = landmark_delta * lm_scale

            mesh_delta = (landmark_delta[:, idx, :] * w.unsqueeze(0).unsqueeze(-1)).sum(dim=2)
            mesh_support = (landmark_conf[:, idx] * w.unsqueeze(0)).sum(dim=2).clamp(0.0, 1.0)
            mesh_delta = mesh_delta * mesh_support.unsqueeze(-1)

            if blend > 0.0 and edge_src.numel() > 0:
                neighbor = torch.zeros_like(mesh_delta)
                neighbor.index_add_(1, edge_dst, mesh_delta[:, edge_src, :])
                neighbor = neighbor / edge_degree.view(1, -1, 1).clamp_min(1.0)
                mesh_delta = (1.0 - blend) * mesh_delta + blend * neighbor

            uv = (uv + step * mesh_delta).clamp(0.0, 1.0)

        direct_gain = float(max(direct_snap_gain, 0.0))
        if use_keypoint_only:
            direct_gain = 0.0
        direct_blend = float(min(max(direct_diffuse_blend, 0.0), 1.0))
        direct_iters = int(max(direct_diffuse_iters, 0))
        direct_max_step_uv = float(max(direct_snap_max_step_px, 0.0)) / float(max(int(image_size), 1))
        if direct_gain > 0.0:
            anchor_delta = searched_uv.to(device=uv.device, dtype=uv.dtype) - uv
            anchor_delta = torch.where(match.unsqueeze(-1) > 0.5, anchor_delta, torch.zeros_like(anchor_delta))
            anchor_delta = anchor_delta * confidence.unsqueeze(-1) * direct_gain

            if direct_max_step_uv > 0.0:
                anchor_norm = torch.linalg.norm(anchor_delta, dim=-1, keepdim=True)
                anchor_scale = (direct_max_step_uv / anchor_norm.clamp_min(1e-6)).clamp(max=1.0)
                anchor_delta = anchor_delta * anchor_scale

            if direct_iters > 0 and edge_src.numel() > 0:
                disp = anchor_delta
                support = confidence.clone()
                matched = match.unsqueeze(-1) > 0.5
                anchor_hold = (0.75 + 0.25 * confidence).clamp(0.0, 1.0).unsqueeze(-1)
                for _ in range(direct_iters):
                    neighbor_disp = torch.zeros_like(disp)
                    neighbor_disp.index_add_(1, edge_dst, disp[:, edge_src, :])
                    neighbor_disp = neighbor_disp / edge_degree.view(1, -1, 1).clamp_min(1.0)

                    neighbor_support = torch.zeros_like(support)
                    neighbor_support.index_add_(1, edge_dst, support[:, edge_src])
                    neighbor_support = neighbor_support / edge_degree.view(1, -1).clamp_min(1.0)

                    mixed_disp = (1.0 - direct_blend) * disp + direct_blend * neighbor_disp
                    disp = torch.where(
                        matched,
                        anchor_hold * anchor_delta + (1.0 - anchor_hold) * mixed_disp,
                        mixed_disp * neighbor_support.unsqueeze(-1).clamp(0.0, 1.0),
                    )
                    support = torch.where(match > 0.5, confidence, neighbor_support.clamp(0.0, 1.0))
                uv = (uv + disp).clamp(0.0, 1.0)
            else:
                uv = (uv + anchor_delta).clamp(0.0, 1.0)

        refined[..., 3:5] = uv
        return refined

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        f4, f8, f16, dense_parts = self._run_dense_stage(rgb)
        if dense_parts["basecolor"] is None or dense_parts["geo"] is None or dense_parts["normal"] is None:
            raise ValueError(
                "Dense2Geometry requires dense-stage predictions for basecolor, geo, normal, and mask."
            )

        pred_basecolor = dense_parts["basecolor"]
        pred_geo = dense_parts["geo"]
        pred_normal = dense_parts["normal"]
        pred_mask_logits = dense_parts["mask_logits"]
        searched_uv, match_mask, search_dist = self._search_mesh_positions(
            pred_geo=pred_geo,
            pred_mask_logits=pred_mask_logits,
        )

        sampled_f4 = self._sample_feature_map(f4, searched_uv, match_mask)
        sampled_f8 = self._sample_feature_map(f8, searched_uv, match_mask)
        sampled_f16 = self._sample_feature_map(f16, searched_uv, match_mask)
        sampled_feat = torch.cat([sampled_f4, sampled_f8, sampled_f16], dim=-1)
        sampled_basecolor = self._sample_feature_map(pred_basecolor, searched_uv, match_mask)
        sampled_geo = self._sample_feature_map(pred_geo, searched_uv, match_mask)
        sampled_normal = self._sample_feature_map(pred_normal, searched_uv, match_mask)
        sampled_mask = self._sample_feature_map(torch.sigmoid(pred_mask_logits), searched_uv, match_mask)
        sampled_dense = torch.cat([sampled_geo, sampled_normal, sampled_basecolor, sampled_mask], dim=-1)

        semantic_token = self.vertex_semantic_proj(self.mesh_geo_codes.unsqueeze(0).expand(rgb.shape[0], -1, -1))
        coord_input = torch.cat([searched_uv.clamp_min(0.0), match_mask.unsqueeze(-1), search_dist.unsqueeze(-1)], dim=-1)
        vertex_parts = [
            self.vertex_image_proj(sampled_feat),
            self.vertex_dense_proj(sampled_dense),
            semantic_token,
            self.vertex_coord_proj(coord_input),
            self.vertex_embed.weight.unsqueeze(0).expand(rgb.shape[0], -1, -1),
        ]
        vertex_tokens = self.vertex_concat_fuse(torch.cat(vertex_parts, dim=-1))

        image_memory = self._build_image_memory(f4, f8, f16)
        landmark_tokens, landmark_uv, landmark_valid = self._pool_mesh_to_landmarks(
            vertex_tokens,
            searched_uv,
            match_mask,
        )
        landmark_tokens = landmark_tokens + self.landmark_embed.weight.unsqueeze(0)
        landmark_tokens = landmark_tokens + self.landmark_semantic_proj(
            self.landmark_geo_codes.unsqueeze(0).expand(rgb.shape[0], -1, -1)
        )
        landmark_tokens = landmark_tokens + self.landmark_coord_proj(
            torch.cat([landmark_uv.clamp_min(0.0), landmark_valid.unsqueeze(-1)], dim=-1)
        )
        for block in self.landmark_blocks:
            landmark_tokens = block(landmark_tokens, image_memory)
        landmark_tokens = self.landmark_norm(landmark_tokens)

        mesh_context = self._broadcast_landmarks_to_mesh(landmark_tokens)
        mesh_tokens = vertex_tokens + self.mesh_context_proj(mesh_context)
        for block in self.mesh_graph_blocks:
            mesh_tokens = block(
                mesh_tokens,
                self.mesh_edge_src,
                self.mesh_edge_dst,
                self.mesh_edge_degree,
            )
        intermediate_meshes = []
        template = self.template_mesh.unsqueeze(0)
        for block in self.mesh_refine_blocks:
            mesh_tokens = block(mesh_tokens)
            raw = self.output_head(mesh_tokens)
            # xyz (0:3) and depth (5:6): offset from template
            # uv (3:5): absolute prediction
            mesh_coords = template.clone().expand(rgb.shape[0], -1, -1).clone()
            mesh_coords[..., :3] = template[..., :3] + raw[..., :3]
            mesh_coords[..., 3:5] = raw[..., 3:5].clamp(0.0, 1.0)
            mesh_coords[..., 5:] = template[..., 5:] + raw[..., 5:]
            intermediate_meshes.append(mesh_coords)

        return {
            "mesh": intermediate_meshes[-1],
            "intermediate_meshes": intermediate_meshes[:-1],
            "searched_uv": searched_uv,
            "match_mask": match_mask,
            "search_distance": search_dist,
            "pred_geo": pred_geo,
            "pred_normal": pred_normal,
            "pred_basecolor": pred_basecolor,
            "pred_mask_logits": pred_mask_logits,
        }
