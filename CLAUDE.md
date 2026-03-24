# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Response Rules
The user is not native English speaker, you need optimize his input, correct grammar, wording, spelling, and send this optimized prompt at begining of you reply, let the user could improve his English by learn from you. Keep it simple.

## Working Rules
You should try to use Language Server Protocol (LSP) first instead of use grep

## Code Simplification Principles

- Keep everyting as simple as possible, you are not writing code for handling every corner case.
- Assume inputs are valid. Only validate at real system boundaries.
- No error handling for cases that cannot happen.
- No validity flags, stats, or tracking unless genuinely needed.
- No debug code, print statements, or visualization inside classes.
- Hardcode constants. Do not pass parameters that never change.
- Inline thin wrappers. Do not add methods that only delegate.

## Project Overview

**ProHead** is a deep learning system for 3D head geometry and appearance prediction from 2D images. It predicts:
- Facial landmarks and 3D mesh vertex positions (geometry transformer)
- Dense texture maps: basecolor, geometry coordinates, surface normals, masks (dense image transformer)
- Combined dense-to-geometry refinement via geo-code mesh search (dense2geometry)

Training data comes from Unreal Engine / MetaHuman captures with ground-truth 3D meshes and camera metadata. Real-world images (FFHQ) can also be used with pseudo-GT generation.

## Common Commands

### Training (Windows)
```bash
python train_geometry_transformer.py --epochs 200 --batch_size 8
python train_dense_image_transformer.py --epochs 50 --basecolor --geo --normal
```

### Training (Linux / distributed)
```bash
python train_geometry_transformer.py --epochs 200 --batch_size 8
python train_dense_image_transformer.py --epochs 50 --basecolor --geo --normal
python train_dense2geometry.py --epochs 50 --batch_size 2
```

### Inference
```bash
python inference_geometry.py --image_path input.jpg --model_path artifacts/checkpoints/best_geometry_transformer_dim6.pth --output_dir output/
python inference_dense_image.py --image_path input.jpg --checkpoint artifacts/checkpoints/best_dense_image_transformer_ch10.pth --output_dir output/
python inference_dense2geometry.py --image_path samples/ --checkpoint artifacts/checkpoints/best_dense2geometry.pth --output_dir artifacts/test_result_dense2geometry/
```

### Test Scripts
```bash
python scripts/debug/test_render_quick.py
python scripts/debug/test_geo_exr.py
python scripts/debug/test_nvdiffrast_uv_random_vertex_color.py
python scripts/debug/test_search_stage_speed.py
```

### Dataset Preparation
```bash
python scripts/debug/examine_dense_image_dataset.py
python scripts/data/prepare_geometry_dataset_from_ffhq.py
python scripts/data/predict_geometry_dataset.py
python scripts/data/precompute_template_depth.py
```

## Architecture

### Three-Pipeline Design

**Pipeline 1 — Geometry Transformer** (`geometry_transformer.py` + `geometry_train_core.py`):
- Input: 512x512 RGB images
- Backbone: ConvNeXt-Base (pretrained, `assets/pretrained/convnext_base-6075fbad.pth`)
- Decoder: Transformer with 512 hidden dims, 8 heads, 4 layers, deformable attention
- Output: 5D or 6D per vertex/landmark (x, y, z, u, v + optional depth)
- Loss: SimDR (KL divergence distribution regression) + L1 regression + optional texture rendering supervision via nvdiffrast
- Also predicts 2D landmarks via `landmark_transformer.py`

**Pipeline 2 — Dense Image Transformer** (`dense_image_transformer.py` + `dense_image_train_core.py`):
- Input: 1024x1024 RGB images
- Backbone: ConvNeXt or ResNet + FPN-style decoder with depthwise convolutions
- Output: 10+ channels — basecolor RGB, geometry XYZ, normal XYZ, face mask
- Multi-task learning with learnable uncertainty weighting per task
- Losses: L1 for color/geo/normal channels, BCE+Dice for mask

**Pipeline 3 — Dense2Geometry** (`dense2geometry.py` + `dense2geometry_train_core.py`):
- Combines frozen dense image model with a trainable geometry refinement transformer
- 3-stage mesh search: landmark@256 -> mesh@512 -> subpixel@1024 (L2 geo-code matching)
- Transformer decoder: graph convolution + self-attention refinement (5 mesh_refine_blocks)
- Deep supervision: shared output_head applied at every refine block, intermediate losses weighted 0.3x
- Output: 6D per vertex — xyz offset from template, absolute UV prediction, depth offset from template
- Supports legacy checkpoint compatibility via forward method patching in inference
- Test folder prediction at each epoch end with dlib 5pt alignment (saves to `artifacts/test_predictions/`)

### Key Helpers

| File | Purpose |
|------|---------|
| `train_loss_helper.py` | SimDRLoss, WingLoss, MeshSmoothnessLoss, metric accumulators, weighted L1 |
| `train_visualize_helper.py` | UV topology, mesh rendering, combined mesh visualization |
| `data_utils/camera_io.py` | Parses Unreal Engine `MatrixData.txt` (camera, head transforms) |
| `data_utils/texture_pack.py` | Composes combined UV texture maps from multiple EXR sources |
| `align_5pt_helper.py` | 5-point face alignment with pose-aware scaling and direction shift |
| `data_utils/obj_io.py` | OBJ file loading with UV/normal support and n-gon triangulation |
| `scripts/data/build_combined_knn.py` | Builds KNN spatial index between landmarks/mesh/keypoints |
| `scripts/data/build_template.py` | Constructs template mesh topology and landmark files |
| `scripts/debug/project_mesh_to_screen.py` | Projects 3D mesh to 2D screen space using camera parameters |
| `scripts/debug/real_dataset_point_search.py` | Geo feature matching for real dataset pseudo-GT generation |

### Datasets

- `geometry_dataset.py` — RGB + geometry with texture packing, 5-point alignment, augmentation (GeometryDataset)
- `dense_image_dataset.py` — Dense supervision: RGB -> basecolor/geo/normal/mask maps with EXR support
- `dense2geometry_dataset.py` — Synthetic-only wrapper around GeometryDataset for dense2geometry training

### Data Paths

- **Windows**: `G:/CapturedFrames_final*_processed`, textures at `G:/textures`
- **Linux**: `/hy-tmp/CapturedFrames_final*_processed`, textures at `/hy-tmp/textures`
- Unified train entry points detect the platform and apply the right presets automatically

### Template Assets (`assets/topology/` directory)

Pre-built topology files used at runtime:
- `*.obj` — Head, eye, mouth mesh/landmark/keypoint templates
- `*.npy` — Vertex templates, KNN indices, UV inverse maps, geo feature atlas
- `*.exr` — Baked texture maps (skin, eyes, teeth, geo features)
- `*.png` — UV layout masks, face masks, hair masks
- `landmark_mask.txt`, `mesh_mask.txt` — Topology configuration

### Training Infrastructure

- PyTorch DDP for multi-GPU distributed training
- Automatic Mixed Precision (fp16/bf16)
- TensorBoard logging to `artifacts/runs/`
- Optional nvdiffrast for differentiable texture rendering during training
- Deep supervision for dense2geometry pipeline (intermediate layer losses)
- Epoch-end test folder prediction with visualization panels

## Pre-trained Checkpoints

| File | Description |
|------|------------|
| `artifacts/checkpoints/best_geometry_transformer_dim6.pth` | Geometry model, 6D output (1.07GB) |
| `artifacts/checkpoints/best_geometry_transformer_dim6.pth` | Geometry model, 5D output (1.09GB) |
| `artifacts/checkpoints/best_dense_image_transformer_ch10.pth` | Dense image model, 10-channel (basecolor+geo+normal+mask) |
| `artifacts/checkpoints/best_dense2geometry.pth` | Dense2Geometry combined model |

## Optional Dependencies

Some features are optional with fallbacks:
- `nvdiffrast` — Differentiable rendering (training texture supervision)
- `dlib` — Face detection preprocessing (5-point landmarks)
- `OpenEXR` / `imageio` — EXR file format support
- `cv2.FaceDetectorYN` (YuNet) — Alternative face detector for inference
