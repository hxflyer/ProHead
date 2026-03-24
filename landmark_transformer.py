import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ConvNeXt_Base_Weights
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

class LandmarkTransformer(nn.Module):
    """
    RGB-based landmark detection using ConvNeXt-Base backbone + Transformer decoder with Deep Supervision.
    Fuses Stride 4, 8, and 16 features.
    """
    
    def __init__(
        self,
        num_landmarks: int,
        template_landmark: np.ndarray,
        dense_knn_indices: np.ndarray,
        dense_knn_weights: np.ndarray,
        n_sparse: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_dim:int = 2,
        backbone_weights: str = 'imagenet'
    ):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        self.n_sparse = int(n_sparse)
        self.d_model = int(d_model)
        
        # Register template landmark
        self.register_buffer("template_landmark", torch.from_numpy(template_landmark).float())
        
        # Register KNN mapping
        idx_tensor = torch.from_numpy(dense_knn_indices.astype(np.int64))
        w_tensor = torch.from_numpy(dense_knn_weights.astype(np.float32))
        self.register_buffer("dense_knn_indices", idx_tensor)
        self.register_buffer("dense_knn_weights", w_tensor)
        
        # --- Backbone: ConvNeXt-Base ---
        backbone = models.convnext_base(weights=None)
        
        if backbone_weights == 'dinov3':
            print("Loading DINOv3 pretrained backbone (converted to Torchvision format)...")
            # Use the converted weights file
            weight_path = "assets/pretrained/dinov3_lvd1689m_torchvision.pth"
        else:
            print("Loading ImageNet pretrained backbone (Fusion Stride 4+8+16)...")
            weight_path = "assets/pretrained/convnext_base-6075fbad.pth"

        print(f"Loading weights from: {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu")
        
        # Handle state_dict if it has 'model' or similar keys (standard checkpoints sometimes do)
        if 'model' in state_dict:
            state_dict = state_dict['model']
            
        # DINOv3 converted weights are just the dict.
        # ImageNet weights are just the dict.
        
        # We use strict=False because ImageNet weights usually have a classifier head 
        # that we might ignore or matches (ConvNeXt uses 'classifier').
        # Our converted DINOv3 weights have 'classifier.0' (norm) but no fc.
        backbone.load_state_dict(state_dict, strict=False)
        # Extract features at Stride 4, 8, 16
        # features.1 -> Stage 0 (Stride 4, 128 ch)
        # features.3 -> Stage 1 (Stride 8, 256 ch)
        # features.5 -> Stage 2 (Stride 16, 512 ch)
        return_nodes = {
            'features.1': 'stride4',
            'features.3': 'stride8',
            'features.5': 'stride16',
        }
        self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
        
        # Projection Layer Calculation:
        # Stride 4 (128) + Stride 8 (256) + Stride 16 (512) = 896 channels
        self.projection = nn.Conv2d(128 + 256 + 512, d_model, kernel_size=1)
        
        # 2D Positional encoding (128x128 resolution)
        self.pos_embed = PositionalEncoding2D(d_model, max_h=128, max_w=128)
        
        # --- Transformer Decoder with Deep Supervision ---
        # We manually create layers to support deep supervision
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
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
        
        # Norm after decoder
        self.decoder_norm = nn.LayerNorm(d_model)
        
        self.query_embed = nn.Embedding(self.n_sparse, d_model)
        self.landmark_pos_embed = nn.Embedding(num_landmarks, d_model)
        
        # Prediction Head (Shared across layers)
        self.to_coord = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )
        
        # Initialize output layer
        nn.init.constant_(self.to_coord[-1].weight, 0)
        nn.init.constant_(self.to_coord[-1].bias, 0)

    def forward_head(self, sparse_feats):
        """Helper to project sparse features to dense landmarks"""
        idx = self.dense_knn_indices
        w = self.dense_knn_weights
        
        neighbor_feats = sparse_feats[:, idx, :]
        w_expanded = w.unsqueeze(0).unsqueeze(-1)
        dense_feats = (neighbor_feats * w_expanded).sum(dim=2)
        
        pos_emb = self.landmark_pos_embed(torch.arange(self.num_landmarks, device=dense_feats.device))
        pos_emb = pos_emb.unsqueeze(0)
        z = dense_feats + pos_emb
        
        offsets = self.to_coord(z)
        
        # Add normalized template landmark to the predicted normalized offsets
        # Template is [N, 5], offsets is [B, N, 5]
        coords = self.template_landmark.unsqueeze(0) + offsets
            
        return coords

    def forward(self, rgb: torch.Tensor):
        """
        Args:
            rgb: [B, 3, 512, 512] RGB image
        Returns:
            coords_list: List of [B, num_landmarks*2] tensors (one per layer)
        """
        B = rgb.shape[0]
        
        # --- Backbone ---
        features = self.backbone(rgb)
        f4 = features['stride4']   # [B, 128, 128, 128]
        f8 = features['stride8']   # [B, 256, 64, 64]
        f16 = features['stride16'] # [B, 512, 32, 32]
        
        # --- Feature Fusion ---
        # Upsample to 128x128
        f8_up = F.interpolate(f8, size=f4.shape[-2:], mode='bilinear', align_corners=True)
        f16_up = F.interpolate(f16, size=f4.shape[-2:], mode='bilinear', align_corners=True)
        
        # Concatenate: 128 + 256 + 512 = 896 channels
        x_fused = torch.cat([f4, f8_up, f16_up], dim=1)
        
        # --- Projection ---
        x = self.projection(x_fused) # [B, d_model, 128, 128]
        
        # Positional Encoding
        pos = self.pos_embed(x)
        src = x + pos
        
        # Flatten [B, d_model, 128, 128] -> [B, 16384, d_model]
        src = src.flatten(2).permute(0, 2, 1)
        
        # Queries [B, n_sparse, d_model]
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # --- Decoder with Deep Supervision ---
        outputs = []
        out = queries
        
        for i, layer in enumerate(self.decoder_layers):
            out = layer(out, src)
            
            # Normalize and predict for this layer
            if i == self.num_layers - 1:
                out = self.decoder_norm(out)
                
            # Predict coords
            coords = self.forward_head(out)
            outputs.append(coords.view(B, -1))
            
        return outputs

def create_landmark_transformer(
    num_landmarks: int,
    template_landmark: np.ndarray,
    dense_knn_indices: np.ndarray,
    dense_knn_weights: np.ndarray,
    n_sparse: int,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    output_dim: int = 2,
    backbone_weights: str = 'imagenet'
) -> LandmarkTransformer:
    return LandmarkTransformer(
        num_landmarks=num_landmarks,
        template_landmark=template_landmark,
        dense_knn_indices=dense_knn_indices,
        dense_knn_weights=dense_knn_weights,
        n_sparse=n_sparse,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_dim=output_dim,
        backbone_weights=backbone_weights
    )
