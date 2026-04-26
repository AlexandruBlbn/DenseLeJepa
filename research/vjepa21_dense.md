# Topic 2: V-JEPA 2.1 Dense Features
**Date:** 2026-04-26

## Key Findings
- **Deep Self-Supervision:** V-JEPA 2.1 introduces a hierarchy where the self-supervised loss is applied to multiple intermediate representations of the encoder, not just the final layer.
- **Dense Predictive Loss:** A masking-based objective where both visible (context) and masked tokens contribute to the loss, preventing "collapse" of local features.
- **Temporally Consistent Features:** Optimized for dense prediction tasks (segmentation, tracking) by ensuring that features in consecutive frames or neighboring patches are coherent.
- **Multiseq Masking:** Uses a 3D multi-block masking strategy to handle spatial and temporal dimensions simultaneously.

## Architecture: Dense Prediction Head
- **Hierarchical Attachment:** The predictor is attached to specific "hierarchical layers" of the ViT backbone.
  - For ViT-L (depth 24): Layers `[5, 11, 17, 23]`.
  - For depth 12: Layers `[2, 5, 8, 11]`.
- **Bottleneck Projection:** The predictor maps context tokens to a lower dimension (`predictor_embed_dim`, e.g., 384) before processing.
- **Predictor Structure:** A standard ViT but specialized for latent prediction. It uses `mask_tokens` to represent missing regions and predicts the teacher tokens at those locations.
- **Feature Extraction:** Downstream segmentation can extract per-patch features by accessing the hidden states of the target encoder at the specified hierarchical levels.

## Loss Formulation
V-JEPA 2.1 combines two primary terms:
1. **Prediction Loss:** $L_{pred} = \sum |z_{pred} - h_{target}|^p$ (typically $p=2$, but config suggests `loss_exp` varies).
2. **Context Loss (Optional):** When `predict_all` is True, the model also predicts features for visible tokens to ensure the context representation is as rich as the target representation.
   - **Distance Weighting:** Loss can be scaled by the spatial/temporal distance between the context patch and the target patch.

## Masking Strategy: Multiblock 3D
- **Spatial Scale:** 0.2 to 0.8 (covers a significant portion of the image).
- **Temporal Scale:** Typically 1.0 (covers all frames in a clip for video, or 1.0 for static images).
- **Aspect Ratio:** 0.3 to 3.0.
- **Blocks:** Multiple blocks (`npred`) are sampled per image to create a complex occluded view.

## V-JEPA 2.0 vs 2.1 Highlights
- **V-JEPA 2.0:** Focused on global representations and action anticipation.
- **V-JEPA 2.1:** specifically optimized for **dense features** (PCA visualizations show much clearer object/part separation).
- **Levels:** 2.1 uses `levels_predictor` (default 4) to combine multi-layer features, whereas 2.0 often used single-layer distillation.

## Extraction for Downstream Segmentation
- To get dense features from a pretrained V-JEPA 2.1 model:
```python
# Assuming encoder is loaded
# Set return_hierarchical=True or out_layers=[layers]
hierarchical_features = encoder(images, training=False) 
# Result: concatenated features from [2, 5, 8, 11] (for depth 12)
# Shape: (B, N_patches, D * 4)
```

## Concerns for XCA Domain
- **Patch Size:** Default is 16. For thin coronary arteries, a smaller patch size (e.g., 8) might be necessary to avoid "averaging out" small vessels.
- **Interpolation:** The `MultiSeqWrapper` uses `bicubic` interpolation for gram-mode (upscaling). This might introduce artifacts in medical images; `bilinear` or no upscaling might be safer.
