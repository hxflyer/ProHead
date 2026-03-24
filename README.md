# ProHead

ProHead is a research codebase for predicting head geometry and dense appearance from images.

It includes three main pipelines:
- Geometry transformer: predicts landmarks and mesh vertices from RGB.
- Dense image transformer: predicts basecolor, geometry, normals, and masks.
- Dense2Geometry: refines geometry using dense predictions and atlas search.

## Repo Layout

```text
.
|-- assets/
|   |-- topology/        # Canonical mesh, UV, template, and atlas assets
|   |-- pretrained/      # Backbone weights and detector models
|   `-- demo/            # Static demo visualization assets
|-- artifacts/
|   |-- checkpoints/     # Saved model checkpoints
|   |-- runs/            # TensorBoard / experiment outputs
|   `-- training_*       # Generated previews during training
|-- data_utils/          # Shared dataset and IO helpers
|-- scripts/
|   |-- data/            # Dataset preparation and preprocessing tools
|   |-- debug/           # Manual test and debugging scripts
|   `-- visualize/       # Visualization-only utilities
|-- samples/             # Example input images
|-- legacy/              # Old code kept for reference only
|-- train_geometry_transformer.py
|-- train_dense_image_transformer.py
|-- train_dense2geometry.py
|-- inference_geometry.py
|-- inference_dense_image.py
`-- inference_dense2geometry.py
```

## Setup

Install the common dependencies first:

```bash
pip install -r requirements.txt
```

Some optional features need extra packages:
- `nvdiffrast` for differentiable rendering
- `openexr` and `Imath` for EXR loading through OpenEXR
- `dlib` if you want the optional 68-point landmark predictor path

## Main Commands

### Training

```bash
python train_geometry_transformer.py --epochs 200 --batch_size 8
python train_dense_image_transformer.py --epochs 50 --basecolor --geo --normal
python train_dense2geometry.py --epochs 50 --batch_size 2
```

### Inference

```bash
python inference_geometry.py --image_path input.jpg --model_path artifacts/checkpoints/best_geometry_transformer_dim6.pth --output_dir output/
python inference_dense_image.py --image_path input.jpg --checkpoint artifacts/checkpoints/best_dense_image_transformer_ch10.pth --output_dir output/
python inference_dense2geometry.py --image_path samples/ --checkpoint artifacts/checkpoints/best_dense2geometry.pth --output_dir output_dense2geometry/
```

### Data / Precompute

```bash
python scripts/data/predict_geometry_dataset.py --help
python scripts/data/prepare_geometry_dataset_from_ffhq.py --help
python scripts/data/precompute_geo_normal.py --help
python scripts/data/precompute_template_depth.py --help
```

### Debug / Visualization

```bash
python scripts/debug/test_render_quick.py
python scripts/debug/test_visualize_dataset_samples.py
python scripts/visualize/visualize_geo_normal.py --help
```

## Important Paths

- Topology assets: `assets/topology/`
- Pretrained backbones and detectors: `assets/pretrained/`
- Example images: `samples/`
- Default checkpoints: `artifacts/checkpoints/`

## Notes

- The main train and inference entrypoints stay at the repo root on purpose.
- Most utility scripts were moved under `scripts/` to keep the root focused.
- `legacy/` is not part of the active pipeline.
