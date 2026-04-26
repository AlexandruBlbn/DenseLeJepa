# Topic 5: Project Structure Plan -Frangi-Masked SIGReg-
**Date:** 2026-04-26

## Proposed Directory Structure
The standalone project `d:/DenseJepa/` should be organized for modularity and scalability:

```text
d:/DenseJepa/
├── configs/                # Hydra/YAML configuration files
│   └── pretrain.yaml       # Main pretraining hyperparams
├── data/                   # Dataset management
│   ├── dataset.py          # Grayscale image loader
│   └── transforms.py       # Augmentation (LeJEPA views style)
├── model/                  # Model definitions
│   ├── backbone.py         # ViT with hierarchical levels (V-JEPA 2.1)
│   ├── predictor.py        # Latent predictor head
│   └── head.py             # Dense projection head
├── loss/                   # Loss functions
│   ├── sigreg.py           # BHEP / Sketched CF implementation
│   └── contrastive.py      # InfoNCE or similar for vessel patches
├── preprocessing/          # Offline/Online vessel masking
│   └── frangi_masker.py    # Frangi filter wrapper (skimage)
├── utils/                  # Logging, Distributed, Masks logic
│   ├── masking.py          # Patch-level masking utilities
│   └── checkpointing.py    # Optimizer and model saving
├── train.py                # Main training entry point
└── requirements.txt        # Minimal dependency list
```

## Module Decomposition
- **`preprocessing/frangi_masker.py`**: Computes vesselness score using `skimage.filters.frangi`. Output: Binary mask or soft probability map.
- **`model/backbone.py`**: A ViT (e.g., `vit_small`) that returns features from multiple layers (e.g., layers 2, 5, 8, 11) to support dense segmentation.
- **`loss/sigreg.py`**: Re-implements the `BHEP` multivariate test from LeJEPA.
- **`utils/masking.py`**: Splits image into patches and classifies them based on Frangi score density (Vessel vs. Background).

## Dependencies (Minimal for 24GB GPU)
- `torch>=2.1` (with `bfloat16` and `SDPA` support)
- `torchvision`
- `scikit-image` (for Frangi)
- `timm` (for baseline backbones)
- `numpy`, `opencv-python`
- `hydra-core` (for config)
- `wandb` (optional, for logging)
- `einops` (for tensor manipulation)

## Training Loop Pseudocode
```python
# Integration of Frangi pre-compute -> Patch extraction -> Conditional loss
for images in dataloader:
    # 1. Pre-compute or load masks
    vessel_maps = frangi_filter(images)
    
    # 2. Extract patches & classify 
    # v_mask: 1 for vessel patches, 0 for background
    v_mask = get_patch_masks(vessel_maps, threshold=0.05) 
    
    # 3. Augmentations
    v1, v2 = augment_views(images) # Global/Local views
    
    # 4. Forward pass
    z1, h1 = model(v1) # Get projections and hierarchical features
    z2, h2 = model(v2)
    
    # 5. Conditional Loss
    # Background patches: apply SIGReg (Gaussian regularization)
    bg_loss = sigreg_loss(z1[~v_mask]) + sigreg_loss(z2[~v_mask])
    
    # Vessel patches: apply Contrastive/Invariance pull
    v_loss = mse_loss(z1[v_mask], z2[v_mask])
    
    total_loss = (1 - lamb) * v_loss + lamb * bg_loss
    total_loss.backward()
    optimizer.step()
```

## Configuration Schema
```yaml
model:
  name: "vit_small"
  patch_size: 16
  hierarchical_layers: [2, 5, 8, 11]

training:
  batch_size: 64
  lr: 5e-4
  lamb: 0.02
  precision: "bf16"

frangi:
  sigmas: [1, 3, 5]
  beta: 0.5
  vessel_patch_threshold: 0.1 # % of pixels in patch
```

## Concerns & Implementation Caveats
- **Offline Masking:** Computing Frangi filters during every iteration of training is slow. It is recommended to run a preprocessing script to save `.npy` masks to disk first.
- **Single GPU:** With 24GB VRAM, use `ViT-Small` or `ViT-Base` with `batch_size=64/128` and `bf16` to maintain throughput.
- **Grayscale:** Ensure all filters and augmentations are verified for single-channel input.
