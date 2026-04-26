# Anatomy-Conditioned Sub-Manifold JEPA (ACM-JEPA)
## For X-Ray Coronary Angiography (XCA) Self-Supervised Learning

---

## 1. Problem Diagnosis

| Issue | Root Cause | Impact |
|-------|-----------|--------|
| **Background dominance** (>90% pixels) | All SSL methods treat patches uniformly | Model collapses to "background-only" latent |
| **DINOv3 collapse** | Teacher EMA + student compete on uniform background | High VRAM, representation ignores vessels |
| **LeJEPA collapse** | SIGReg forces single N(0,I) Gaussian on ALL patches | Vessel manifold gets mathematically blended into ribs |
| **V-JEPA 2.1** | EMA target encoder + slow momentum update | Memory-intensive, still no structure prior |

**Core Insight**: The latent space needs *anatomy-conditioned geometry* — background should spread isotropically (Gaussian), vessels should form a tight sub-manifold.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Anatomy-Conditioned Sub-Manifold JEPA             │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────┐   │
│  │  Frangi   │───▶│ Patch Cls    │───▶│ Conditional Loss        │   │
│  │  Filter   │    │ (Vessel/BG)  │    │ ├── SIGReg (BG patches) │   │
│  └──────────┘    └──────────────┘    │ ├── VesselContr (Vessel) │   │
│                                       │ └── L2 Predict (All)    │   │
│  ┌──────────────────────────────┐    └──────────┬──────────────┘   │
│  │ Shared Encoder (ViT/Swin)    │               ▲                  │
│  │ ├── Hierarchical outputs     │───────────────┘                  │
│  │ │   (V-JEPA 2.1 dense feat)  │                                  │
│  │ └── Predictor (MLP/ViT)      │                                  │
│  └──────────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 Design Philosophy

| Principle | Implementation |
|-----------|---------------|
| **No EMA / Teacher-Student** | Single shared encoder (LeJEPA style) |
| **Dense features** | V-JEPA 2.1 hierarchical multi-level outputs |
| **Anatomically guided** | Frangi vesselness mask splits regularization |
| **Collapse-proof** | VICReg-like variance maintained via SIGReg on BG |
| **Vessel-preserving** | Contrastive pull-to-centroid on vessel patches |

---

## 3. Module-by-Module Specification

### 3.1 Frangi Vesselness Preprocessor

**File**: [`modules/frangi_processor.py`]

```python
class FrangiVesselness(nn.Module):
    """
    Deterministic Frangi vesselness filter.
    Input:  [B, 1, H, W]  (grayscale XCA)
    Output: [B, 1, H, W]  (vesselness prob in [0,1])
    
    Uses skimage.filters.frangi or a custom implementation.
    Parameters tuned for coronary arteries:
    - scales=[1, 2, 3, 4]  (vessel diameter range in pixels)
    - beta=0.5  (blobness suppression)
    - gamma=15  (background suppression)
    """
```

**Key properties**:
- **Deterministic & differentiable** (use ReLU + sqrt for eigenvalue computation)
- **No learned parameters** — pure classical CV prior
- **GPU-friendly** — can run on-device in the dataloader or inside training loop
- **Scale range**: coronary arteries are 2–8 pixels wide at 512×512 resolution

**Integration options**:

| Option | Pros | Cons |
|--------|------|------|
| **A. Precompute offline** | Zero compute cost during training | Storage, cannot augment |
| **B. On-the-fly in dataloader** | Augmentation-compatible | CPU-GPU transfer bottleneck |
| **C. nn.Module in forward pass** | End-to-end, GPU-accelerated | Requires differentiable Frangi |

**Recommendation**: Option C — implement a differentiable Frangi using [`torch` operations] (eigenvalues via `torch.linalg.eigh` on the Hessian).

---

### 3.2 Patch-Level Vessel Classification

**File**: [`modules/frangi_masking.py`]

```python
class FrangiPatchClassifier(nn.Module):
    """
    Maps pixel-level Frangi mask to patch-level vessel/background labels.
    
    Input:
        frangi_mask: [B, 1, H, W]  (vesselness prob)
        patch_size: int (e.g., 16 for ViT, 4 for Swin)
    Output:
        vessel_mask:  [B, N]  bool tensor (True = vessel patch)
        vessel_score: [B, N]  float tensor (mean vesselness per patch)
    
    Classification rule:
        pool = avg_pool(frangi_mask, kernel=patch_size, stride=patch_size)
        threshold = 0.1  (configurable; lower = more conservative vessel capture)
        vessel_mask = pool > threshold
    """
```

**Why avg_pool with threshold instead of learnable**:
- Keights the frangi prior hard — prevents the encoder from "cheating" by ignoring the mask.
- The threshold creates a clean separation between the two regularization regimes.

**Token splitting**: After classification, we have:

```python
# Inside the forward pass:
# predicted tokens:  [B, N, D]
# vessel_mask:       [B, N]  bool

# Split by mask
bg_tokens = pred_tokens[~vessel_mask]       # [N_bg, D]
vessel_tokens = pred_tokens[vessel_mask]    # [N_vessel, D]
```

---

### 3.3 Conditional Regularization: Frangi-Masked SIGReg

**File**: [`regularizers/frangi_sigreg.py`]

#### 3.3.1 Background Branch: SIGReg (Gaussian Constraint)

Same as existing [`GlobalSIGReg`](usables/sigreg.py) but only applied to `bg_tokens`:

```python
class BackgroundSIGReg(nn.Module):
    """
    Forces background tokens into N(0, I_d) via characteristic function matching.
    
    Input:  [N_bg, D]  (selected background tokens across batch)
    Output: scalar loss
    
    Projects onto K random directions, matches cos/sin moments 
    to Gaussian kernel phi(t) = exp(-0.5*t^2).
    """
    def forward(self, bg_tokens: Tensor) -> Tensor:
        bg_global = self._to_global(bg_tokens)  # [N_bg, D]
        # ... standard SIGReg projection + CF matching ...
```

**Why this works**: Background is chaotic (noise, bone, air, soft tissue). A Gaussian spread is the maximum-entropy distribution — it forces the background to occupy the latent space broadly without collapse. Since background is >90% of tokens, this alone prevents global collapse.

#### 3.3.2 Vessel Branch: VesselCentroidContrastive

```python
class VesselCentroidContrastive(nn.Module):
    """
    Pulls vessel tokens toward a learned/adaptive centroid.
    
    Input:  [N_vessel, D]  (selected vessel tokens across batch)
    Output: scalar loss
    
    Two variants:
    1. Fixed centroid: centroid = learnable nn.Parameter(D,)
    2. Adaptive centroid: centroid = mean(vessel_tokens).detach()
    
    Loss: L_vessel = ||v_tokens - centroid||^2  (weighted by vessel_score)
    
    Optional: add intra-cluster push (variance regularization) 
    to prevent total collapse of vessel cluster.
    """
    def forward(self, vessel_tokens: Tensor, vessel_scores: Tensor) -> Tensor:
        # Adaptive centroid (stop-gradient to avoid trivial solution)
        centroid = vessel_tokens.mean(dim=0).detach()
        
        # Weighted MSE pull
        diff = (vessel_tokens - centroid).pow(2).sum(dim=-1)  # [N_vessel]
        loss = (diff * vessel_scores).mean()  # weighted by vesselness
        
        # Variance regularization: ensure vessel cluster has min std
        std = vessel_tokens.std(dim=0).mean()
        variance_reg = F.relu(self.target_std - std)
        
        return loss + self.lambda_var * variance_reg
```

**Why this works**: Vessels are sparse (<<10% of patches). Without explicit clustering, they get absorbed into the Gaussian background. The centroid pull creates a tight sub-manifold that the predictor can learn to reconstruct, while variance regularization prevents sub-collapse.

#### 3.3.3 Combined Loss

```python
def frangi_masked_loss(
    pred_tokens: Tensor,       # [B, N, D]  predicted
    target_tokens: Tensor,     # [B, N, D]  target
    frangi_mask: Tensor,       # [B, 1, H, W]
    bg_sigreg: BackgroundSIGReg,
    vessel_contrast: VesselCentroidContrastive,
    patch_size: int,
    lambda_sig: float = 0.02,
    lambda_vessel: float = 0.05,
    lambda_pred: float = 1.0,
) -> Dict[str, Tensor]:
    
    B, N, D = pred_tokens.shape
    grid_h = H // patch_size
    grid_w = W // patch_size
    
    # 1. Patch-level vessel classification
    vessel_mask = classify_patches(frangi_mask, patch_size)  # [B, N]
    
    # 2. Prediction loss on ALL tokens (standard JEPA)
    loss_pred = F.mse_loss(pred_tokens, target_tokens)
    
    # 3. Split tokens by vessel mask
    bg_pred = pred_tokens[~vessel_mask]       # [N_bg, D]
    bg_target = target_tokens[~vessel_mask]   # [N_bg, D]
    v_pred = pred_tokens[vessel_mask]         # [N_v, D]
    v_scores = vessel_mask_float[vessel_mask] # [N_v]
    
    # 4. Conditional regularization
    loss_sig = bg_sigreg(bg_pred)
    loss_vessel = vessel_contrast(v_pred, v_scores)
    
    return {
        "loss_total": lambda_pred * loss_pred 
                    + lambda_sig * loss_sig 
                    + lambda_vessel * loss_vessel,
        "loss_pred": loss_pred,
        "loss_sigreg": loss_sig,
        "loss_vessel": loss_vessel,
        "vessel_ratio": vessel_mask.float().mean(),
    }
```

---

### 3.4 Encoder with V-JEPA 2.1 Dense Features

**File**: [`models/vjepa_encoder.py`]

The V-JEPA 2.1 VisionTransformer (from [`repos/vjepa2/app/vjepa_2_1/models/vision_transformer.py`](repos/vjepa2/app/vjepa_2_1/models/vision_transformer.py:148-151)) outputs **hierarchical dense features** — 4 levels concatenated along the channel dimension:

| Encoder Depth | Hierarchical Layers | Output dim |
|--------------|-------------------|------------|
| 12 (ViT-B) | `[2, 5, 8, 11]` | 4 × 768 = 3072 |
| 24 (ViT-L) | `[5, 11, 17, 23]` | 4 × 1024 = 4096 |
| 40 (ViT-G) | `[9, 19, 29, 39]` | 4 × 1408 = 5632 |

```python
class HierarchicalEncoder(nn.Module):
    """
    Wraps a ViT to output hierarchical tokens (V-JEPA 2.1 style).
    
    Input:  [B, C, H, W]
    Output: {
        "tokens":       [B, N, D*4]   # hierarchical concatenation
        "layer_tokens": [[B, N, D]] * 4  # individual layers
        "spatial_hw":   (H//patch, W//patch)
    }
    """
    def __init__(self, backbone: VisionTransformer):
        super().__init__()
        self.backbone = backbone
        self.backbone.out_layers = backbone.hierarchical_layers
        # Each hierarchical output gets its own norm (already in ViT)
    
    def forward(self, x):
        # ViT forward with out_layers set → returns 4 tensors
        layer_outs = self.backbone(x)  # [[B, N, D]] * 4
        
        # Concatenate along channel dim (V-JEPA 2.1 style)
        tokens = torch.cat(layer_outs, dim=-1)  # [B, N, D*4]
        
        return {
            "tokens": tokens,
            "layer_tokens": layer_outs,
            "spatial_hw": (x.shape[-2] // patch_size, x.shape[-1] // patch_size),
        }
```

**Integration with existing DenseLeJEPA** (from [`usables/model.py`](usables/model.py:102)):

Replace `SwinTokenEncoder` with `HierarchicalEncoder`:

```python
class AcmJepaEncoder(nn.Module):
    """
    ACM-JEPA encoder with V-JEPA 2.1 dense features.
    Uses VisionTransformer with hierarchical outputs.
    
    The predictor predicts all 4 hierarchical levels independently,
    and the loss is computed on each level with deep supervision.
    """
```

---

### 3.5 Predictor (Dense)

**File**: [`models/predictor.py`]

```python
class DensePredictor(nn.Module):
    """
    Predicts target tokens from context tokens.
    
    Input:  context_tokens [B, N_ctx, D*4]  (4 hierarchical levels concatenated)
    Output: predicted_tokens [B, N_tgt, D*4]
    
    Architecture options:
    1. MLP (LeJEPA style): LayerNorm → Linear → GELU → Linear
    2. Small ViT (V-JEPA style): 4-8 transformer blocks
    """
```

**For XCA with limited data**, Option 1 (MLP) is preferred — fewer parameters, less overfitting. The hierarchical prediction is handled by:

1. Split input into 4 level-specific chunks
2. Apply 4 independent MLP predictors (weight-shared for efficiency)
3. Concatenate back to `[B, N, D*4]`

---

## 4. Training Pipeline

### 4.1 Data Flow

```
┌─────────┐     ┌──────────────┐     ┌──────────────────┐
│ XCA     │────▶│ Frangi       │────▶│ Frangi Mask      │
│ Image   │     │ Processor    │     │ (H, W)           │
└─────────┘     └──────────────┘     └────────┬─────────┘
                                               │
                                               ▼
┌───────────────────────────────────────────────────────────┐
│ Mask Generator (random block mask)                        │
│ - Creates token_mask [B, N]  (True = masked/predict)     │
│ - Context view = image × (1 - mask) + fill_value × mask  │
└─────────────────────────────────────────┬─────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
           ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
           │ Context View    │   │ Target View     │   │ Frangi Mask     │
           │ [B, C, H, W]    │   │ (same as input) │   │ [B, 1, H, W]    │
           └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
                     │                     │                     │
                     ▼                     ▼                     │
           ┌──────────────────┐           │                     │
           │ Shared Encoder   │           │                     │
           │ (Hierarchical)   │           │                     │
           └────────┬─────────┘           │                     │
                     │                     │                     │
                     ▼                     ▼                     │
           ┌──────────────────┐  ┌──────────────────┐           │
           │ Context Tokens   │  │ Target Tokens    │           │
           │ [B, N_ctx, D*4]  │  │ [B, N, D*4]      │           │
           └────────┬─────────┘  └────────┬─────────┘           │
                     │                     │                     │
                     ▼                     │                     │
           ┌──────────────────┐            │                     │
           │ Predictor        │            │                     │
           └────────┬─────────┘            │                     │
                     │                     │                     │
                     ▼                     ▼                     │
           ┌──────────────────┐  ┌──────────────────┐           │
           │ Pred Tokens      │  │ Target Tokens    │◄──────────┘
           │ [B, N_tgt, D*4]  │  │ [B, N_tgt, D*4]  │
           └────────┬─────────┘  └──────────────────┘
                     │
                     ▼
           ┌──────────────────────────────────────────────────┐
           │ Frangi-Masked Conditional Loss                   │
           │ ├── L_pred on ALL masked tokens (hierarchical)   │
           │ ├── SIGReg on background tokens                  │
           │ └── VesselContrastive on vessel tokens           │
           └──────────────────────────────────────────────────┘
```

### 4.2 Augmentation Strategy for XCA

Standard ImageNet augmentations (color jitter, Gaussian blur) can destroy vessel information in XCA. Specialized augmentations:

| Augmentation | Applied to | Rationale |
|-------------|-----------|-----------|
| RandomResizedCrop | Both views | Invariance to translation/scale |
| Horizontal Flip | Both views | Coronary anatomy is symmetric |
| **Random Contrast/Brightness** | Context only | Invariance to X-ray exposure |
| **Random Gamma** | Context only | Invariance to acquisition settings |
| **No Color Jitter** | — | XCA is grayscale; no color info |
| **No Gaussian Blur** | — | Blurs thin vessels out of existence |

**Critical**: The target view should have **minimal augmentation** (only crop + flip) to preserve the vessel structure the predictor must learn to reconstruct.

### 4.3 Optimization Strategy

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Optimizer | AdamW | Standard for ViT |
| Learning rate | 3e-4 → 3e-6 (cosine) | Warmup 10 epochs |
| Weight decay | 0.05 → 0.5 (cosine) | Strong regularization for small data |
| Batch size | 64 (adjust per VRAM) | Small enough for 24GB cards |
| Mask ratio | 0.75 | Higher than 0.65 for stronger task |
| Epochs | 300 | Plateaus around ~200 |
| `lambda_sig` | 0.02 | From LeJEPA default |
| `lambda_vessel` | 0.05 → 0.01 (anneal) | High early, decay to prevent over-constraining |
| `target_std` (vessel var) | 0.5 | Maintains healthy sub-cluster spread |

### 4.4 Training Schedule (3 Phases)

#### Phase 1: Warmup (epochs 0–20)
- Freeze Frangi mask classification (fixed threshold)
- `lambda_sig = 0.01`, `lambda_vessel = 0.1` (strong vessel clustering early)
- LR warmup from 1e-6 → 3e-4

#### Phase 2: Main Training (epochs 20–200)
- Full loss with `lambda_sig = 0.02`, `lambda_vessel = 0.05`
- Cosine LR decay
- Monitor `vessel_ratio` metric — should stabilize at 5–15%

#### Phase 3: Fine-tuning (epochs 200–300)
- Anneal `lambda_vessel` to 0.01 (relax vessel constraint)
- Reduce mask ratio to 0.65 (easier task for refinement)
- Final LR 3e-6

---

## 5. Implementation Plan (File Structure)

```
xca-ssl/
├── configs/
│   └── acmjepa_pretrain.yaml         # Training config
│
├── data/                              # XCA dataset handling
│   ├── xca_dataset.py                 # PyTorch Dataset for XCA DICOM/PNG
│   └── transforms_xca.py              # XCA-specific augmentations
│
├── modules/                           # Core algorithmic modules
│   ├── frangi_processor.py            # Frangi vesselness filter (differentiable)
│   ├── frangi_masking.py              # Frangi → patch-level classifier
│   └── __init__.py
│
├── regularizers/                      # Regularization modules
│   ├── bg_sigreg.py                   # BackgroundSIGReg (modified GlobalSIGReg)
│   ├── vessel_contrastive.py          # VesselCentroidContrastive
│   ├── frangi_masked_loss.py          # Combined loss function
│   └── __init__.py
│
├── models/                            # Neural architectures
│   ├── encoder.py                     # HierarchicalEncoder (V-JEPA 2.1 style)
│   ├── predictor.py                   # DensePredictor (hierarchical prediction)
│   ├── acmjepa.py                     # ACM-JEPA: encoder + predictor + loss
│   └── __init__.py
│
├── segmentation/                      # Downstream segmentation
│   ├── segmentation_head.py           # UNet-like decoder
│   └── trainer.py                     # Segmentation fine-tuning
│
├── train.py                           # Pre-training script
├── train_segmentation.py              # Segmentation fine-tuning script
└── eval_segmentation.py               # Evaluation on XCA segmentation
```

---

## 6. Key Design Decisions

### 6.1 Why ViT over Swin for the Encoder?

| Aspect | Swin (current) | ViT (V-JEPA 2.1) | Decision |
|--------|---------------|-------------------|----------|
| Hierarchical feats | ✅ Built-in (window merging) | ✅ Via intermediate layers | **ViT** — proven for SSL, simpler |
| Position encoding | ✅ Relative bias | ✅ RoPE or absolute | **ViT** — RoPE generalizes better |
| Dense feature quality | ❌ Window attention limits context | ✅ Global attention | **ViT** — needed for anatomy |
| V-JEPA 2.1 compat | ❌ | ✅ Direct match | **ViT** — reuse pretrained weights |

**Final**: Use `vit_small` (12 layers, 384 dim, 6 heads) or `vit_base` (12 layers, 768 dim, 12 heads) depending on VRAM budget.

### 6.2 Why Not Use EMA Target Encoder?

**Because Frangi-Masked SIGReg replaces the EMA collapse prevention mechanism:**

| Mechanism | EMA Teacher | Frangi-Masked SIGReg |
|-----------|------------|---------------------|
| Prevents collapse | ✅ Momentum target provides stable targets | ✅ Gaussian constraint spreads background |
| Memory cost | ❌ 2× model params (student + teacher) | ✅ 1× model params |
| VRAM | ❌ ~2× for optimizer states | ✅ ~1× |
| Vessel preservation | ❌ No structure prior | ✅ Explicit vessel clustering |
| Convergence speed | ❌ Slow (momentum = 0.99+ in later epochs) | ✅ Fast (direct optimization) |

### 6.3 Why Frangi + Contrastive Instead of GMM?

| Approach | Complexity | Vessel Preservation | Training Stability |
|----------|-----------|-------------------|-------------------|
| **GMM Prior** (proposed in prompt) | Medium — requires EM-like updates | ✅ Soft assignment | ❌ GMM can collapse to single mode |
| **VICReg** (proposed in prompt) | Low — variance + covariance | ❌ Global variance ignores structure | ✅ Very stable |
| **Frangi-Masked SIGReg** ✅ | Low — hard assignment from CV prior | ✅ Explicit vessel cluster | ✅ Hard mask prevents confusion |

**GMM with EM** would learn the assignments online, but could still collapse the vessel Gaussian into the background one. The **Frangi hard mask** is a stronger prior that guarantees separation.

### 6.4 V-JEPA 2.1 Dense Feature Integration

The key innovation from V-JEPA 2.1 (from [`repos/vjepa2/app/vjepa_2_1/models/vision_transformer.py:297-340`](repos/vjepa2/app/vjepa_2_1/models/vision_transformer.py:297-340)) is:

1. **Multi-level output**: The encoder returns `[B, N, D*4]` where each `D`-sized chunk comes from a different transformer depth
2. **Shallow layers** capture low-level vessel edges and textures
3. **Deep layers** capture high-level vessel topology and branching

This multi-scale representation is **critical for downstream segmentation** — a UNet-like decoder can fuse these levels for pixel-accurate vessel segmentation.

```python
# V-JEPA 2.1 style: encoder outputs hierarchical tokens
hier = torch.cat(hier, dim=2)  # [B, N, D*4]

# For segmentation: reshape to spatial grid and use as multi-scale features
feats = hier.view(B, H_patches, W_patches, D*4).permute(0, 3, 1, 2)
# → UNet decoder can upsample and concatenate with skip connections
```

---

## 7. Downstream Segmentation Transfer

### 7.1 Segmentation Head Architecture

```python
class VesselSegmentationHead(nn.Module):
    """
    Takes hierarchical tokens [B, N, D*4] → [B, C, H, W] segmentation.
    
    Architecture:
    1. Reshape tokens to spatial grid [B, D*4, H//p, W//p]
    2. Split into 4 level-specific feature maps [B, D, H//p, W//p] × 4
    3. Lightweight UNet decoder:
        - Level 0 (shallow):  preserve high-res, skip connect to final
        - Level 1-2 (mid):    upsample + fuse
        - Level 3 (deepest):  global context, upsample
    4. Output: [B, 1, H, W]  (binary vessel segmentation)
    """
```

### 7.2 Fine-tuning Protocol

| Protocol | Description |
|----------|-------------|
| **Linear probing** | Freeze encoder, train head only — evaluates representation quality |
| **Full fine-tuning** | Unfreeze all, low LR (1e-5) — best segmentation accuracy |
| **Adapter tuning** | Insert LoRA adapters — best compromise for small labeled datasets |

---

## 8. Risk Mitigation

| Risk | Mitigation | Contingency |
|------|-----------|-------------|
| Frangi threshold not robust | Use percentile-based threshold per-image | Fall back to Otsu thresholding |
| Vessel cluster collapses to point | `target_std` variance regularization | Switch to InfoNCE-style contrastive |
| SIGReg on too few bg tokens | If vessel_ratio > 50%, fall back to global SIGReg | Adaptive threshold |
| Frangi misses thin vessels | Multi-scale Frangi with finer scales | Combine with Hessian-based enhancement |
| Small dataset overfitting | Strong weight decay, dropout 0.1, stochastic depth 0.05 | Pretrain on synthetic XCA data |

---

## 9. Evaluation Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Dice Score** | Vessel segmentation overlap | >0.75 |
| **HD95** (Hausdorff Distance) | Vessel boundary accuracy | <5 pixels |
| **ClDice** | Centerline preservation | >0.80 |
| **AUPRC** | Precision-recall curve | >0.85 |
| **Latent Collapse Score** | `min(eigvals(cov(z)))` — smallest eigenvalue of token covariance | >0.1 |
| **Vessel-BG Separation** | `||mean(z_v) - mean(z_bg)||` in latent space | >1.0 |
| **Background Isotropy** | `std(eigvals(cov(z_bg))) / mean(eigvals(cov(z_bg)))` | <0.3 |

---

## 10. Implementation Roadmap

### Phase A: Foundation (Week 1)
1. Implement Frangi processor module
2. Implement Frangi patch classification
3. Implement BackgroundSIGReg (progressive ratio)
4. Test on synthetic vessel images

### Phase B: Core Training (Week 2)
5. Implement ACM-JEPA model (encoder + predictor + conditional loss)
6. Implement XCA dataset and augmentations
7. Implement training loop
8. Debug training stability

### Phase C: Validation (Week 3)
9. Train on XCA dataset (100+ cases)
10. Monitor latent space metrics (collapse, separation)
11. Evaluate on segmentation downstream task
12. Ablate each component (Frangi, SIGReg-bg, vessel-centroid)

### Phase D: Refinement (Week 4)
13. Hyperparameter sweep (lambdas, thresholds, mask ratios)
14. Integration with V-JEPA 2.1 pretrained weights for initialization
15. Final evaluation and comparison against:
    - LeJEPA (baseline)
    - DINOv3 (baseline)
    - V-JEPA 2.1 (baseline)
    - ACM-JEPA (ours)
