# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ProHead** is a deep learning system for 3D head geometry and appearance prediction from 2D images. It predicts:
- Facial landmarks and 3D mesh vertex positions (geometry transformer)
- Dense texture maps: basecolor, geometry coordinates, surface normals, masks (dense image transformer)

Training data comes from Unreal Engine / MetaHuman captures with ground-truth 3D meshes and camera metadata.

## Common Commands

### Training (Windows)
```bash
python windows_train_geometry_transformer.py --epochs 200 --batch_size 8
python windows_train_dense_image_transformer.py --epochs 50 --basecolor --geo --normal
```

### Training (Linux / distributed)
```bash
python linux_train_geometry_transformer.py --epochs 200 --batch_size 8
python linux_train_dense_image_transformer.py --epochs 50 --basecolor --geo --normal
```

### Inference
```bash
python inference_geometry.py --image_path input.jpg --model_path best_geometry_transformer_dim6.pth --output_path output/
python inference_dense_image.py --image_path input.jpg --model_path best_dense_image_transformer_ch3.pth --output_path output/
```

### Test Scripts
```bash
python test_script/test_render_quick.py
python test_script/test_geo_exr.py
python test_script/test_nvdiffrast_uv_random_vertex_color.py
```

### Dataset Inspection
```bash
python examine_dense_image_dataset.py
```

## Architecture

### Two-Pipeline Design

**Pipeline 1 — Geometry Transformer** (`geometry_transformer.py` + `geometry_train_core.py`):
- Input: 512×512 RGB images
- Backbone: ConvNeXt-Base (pretrained, `models/convnext_base-6075fbad.pth`)
- Decoder: Transformer with 512 hidden dims, 8 heads, 4 layers, deformable attention
- Output: 5D per vertex/landmark (x, y, z, u, v UV coords)
- Loss: SimDR (KL divergence distribution regression) + L1 regression + optional texture rendering supervision via nvdiffrast
- Also predicts 2D landmarks via `landmark_transformer.py`

**Pipeline 2 — Dense Image Transformer** (`dense_image_transformer.py` + `dense_image_train_core.py`):
- Input: 1024×1024 RGB images
- Backbone: ConvNeXt or ResNet + FPN-style decoder with depthwise convolutions
- Output: 10+ channels — basecolor RGB, geometry XYZ, normal XYZ, face mask
- Multi-task learning with learnable uncertainty weighting per task
- Losses: L1 for color/geo/normal channels, BCE+Dice for mask

### Key Helpers

| File | Purpose |
|------|---------|
| `train_loss_helper.py` | SimDRLoss, WingLoss, metric accumulators |
| `train_visualize_helper.py` | UV topology, mesh rendering, combined mesh visualization |
| `mat_load_helper.py` | Parses Unreal Engine `MatrixData.txt` (camera, head transforms) |
| `tex_pack_helper.py` | Composes combined UV texture maps from multiple sources |
| `align_5pt_helper.py` | 5-point face alignment for crop/warp preprocessing |
| `obj_load_helper.py` | OBJ file loading with triangulation |
| `build_combined_knn.py` | Builds KNN spatial index between landmarks/mesh/keypoints |

### Datasets

- `metahuman_geometry_dataset.py` — RGB + geometry with texture packing, 5-point alignment, augmentation
- `dense_image_dataset.py` — Dense supervision: RGB → basecolor/geo/normal/mask maps

### Data Paths

- **Windows**: `G:/CapturedFrames_final*_processed`, textures at `G:/textures`
- **Linux**: `/hy-tmp/CapturedFrames_final*_processed`, textures at `/hy-tmp/textures`
- Platform-specific training entry points (`windows_*` vs `linux_*`) handle these path differences

### Template Assets (`model/` directory)

Pre-built topology files used at runtime:
- `*.obj` — Head, eye, mouth mesh templates
- `*.npy` — Vertex templates, KNN indices, UV inverse maps
- `*.png` — UV layout masks
- `landmark_mask.txt` — Landmark topology

### Training Infrastructure

- PyTorch DDP for multi-GPU distributed training
- Automatic Mixed Precision (fp16/bf16)
- TensorBoard logging to `runs/`
- Optional nvdiffrast for differentiable texture rendering during training

## Pre-trained Checkpoints

| File | Description |
|------|------------|
| `best_geometry_transformer_dim6.pth` | Geometry model, 6D output (1.07GB) |
| `best_geometry_transformer_dim5.pth` | Geometry model, 5D output (1.09GB) |
| `best_dense_image_transformer_ch3.pth` | Dense image model, 3-channel (784MB) |

## Optional Dependencies

Some features are optional with fallbacks:
- `nvdiffrast` — Differentiable rendering (training texture supervision)
- `dlib` — Face detection preprocessing
- `OpenEXR` / `imageio` — EXR file format support
