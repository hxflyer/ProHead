# gsplat Training Renderer

`render_head.py` is now a library-only differentiable renderer for training.

It exposes:

- `DifferentiableRendererConfig`
- `DifferentiableHeadRenderer`
- `matrix_data_to_torch(...)`
- `stack_mat_tensors(...)`
- `default_sampling_cache_path(...)`

The training-facing interface is:

```python
from pathlib import Path

import torch

from gsplat.render_head import (
    DifferentiableHeadRenderer,
    DifferentiableRendererConfig,
    matrix_data_to_torch,
)

renderer = DifferentiableHeadRenderer(
    reference_vertices=reference_vertices,   # [N, 3] template/local mesh
    uvs=uvs,                                 # [N, 2]
    faces=faces,                             # [F, 3]
    uv_faces=faces,                          # usually matches faces here
    config=DifferentiableRendererConfig(
        image_width=512,
        image_height=512,
    ),
    sampling_cache_path=Path("gsplat/cache/training_sampling_cache.npz"),
)

mat = matrix_data_to_torch(matrix_data, device="cuda")
mesh_positions = torch.from_numpy(local_vertices).to("cuda")          # [N, 3]
texture = torch.from_numpy(texture_rgb).permute(2, 0, 1).to("cuda")  # [3, H, W]

render = renderer(mat, mesh_positions, texture)  # [3, H, W]
```

The static subdivision / even-sampling plan is cached on disk once. During training, only the dynamic pieces are recomputed:

- current world-space vertex positions from `mat["head_matrix"]`
- current face normals / face-aligned Gaussian orientation
- current Gaussian scales from the deformed face areas
- sampled colors from the input texture tensor

`render_dataset_sample.py` is the demo CLI that loads a few dataset samples, converts mats / meshes / textures to torch tensors, calls the renderer, and saves the color renders:

```powershell
python gsplat/render_dataset_sample.py --data-root G:\CapturedFrames_final8_processed --num-samples 3 --width 512 --height 512
```
