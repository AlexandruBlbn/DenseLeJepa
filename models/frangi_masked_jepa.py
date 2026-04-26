"""
Frangi-Masked JEPA: Top-level model combining backbone + projection head.

Computes per-patch vessel weights from Frangi masks and produces
projected embeddings for the SSL loss.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .backbone import HierarchicalViT
from .projection_head import ProjectionHead


class FrangiMaskedJEPA(nn.Module):
    """
    Frangi-Masked Joint Embedding Predictive Architecture (student only).

    Single encoder (no EMA teacher) with:
    1. Hierarchical ViT backbone
    2. MLP projection head
    3. Per-patch vessel weight computation from Frangi masks

    Args:
        backbone_cfg: dict for HierarchicalViT (keys: img_size, patch_size, in_chans,
                      embed_dim, depth, num_heads, out_layers)
        proj_cfg: dict for ProjectionHead (keys: in_dim, hidden_dim, out_dim)
        frangi_cfg: dict for Frangi parameters (keys: patch_vessel_weight_scale)

    Shapes:
        Input:  images [B, 1, H, W], frangi_masks [B, 1, H, W]
        Output dict:
            'proj': Tensor[B, N_patches, D_proj]
            'patch_features': Tensor[B, N_patches, D]
            'hierarchical': list[Tensor[B, N_patches, D]]
            'vessel_weights': Tensor[B, N_patches]
    """

    def __init__(
        self,
        backbone_cfg: dict,
        proj_cfg: dict,
        frangi_cfg: dict,
    ):
        super().__init__()
        self.frangi_cfg = frangi_cfg

        # Build backbone
        self.backbone = HierarchicalViT(**backbone_cfg)

        # Build projection head
        self.projection_head = ProjectionHead(**proj_cfg)

        # Patch size from backbone
        self.patch_size = backbone_cfg.get("patch_size", 8)
        self.embed_dim = backbone_cfg.get("embed_dim", 384)

    def compute_patch_weights(self, frangi_masks: torch.Tensor) -> torch.Tensor:
        """
        Given per-pixel Frangi masks, compute per-patch vessel weights w_p.

        Args:
            frangi_masks: [B, 1, H, W] per-pixel Frangi vesselness in [0, 1]

        Returns:
            w_p: [B, N_patches] vessel weights in [0, 1]
                w_p = min(1.0, mean_frangi / patch_vessel_weight_scale)
        """
        B, _, H, W = frangi_masks.shape
        scale = self.frangi_cfg.get("patch_vessel_weight_scale", 0.1)
        ps = self.patch_size

        # Average pool over patches: [B, 1, H/ps, W/ps]
        pooled = nn.functional.avg_pool2d(frangi_masks, kernel_size=ps, stride=ps)
        # [B, H/ps * W/ps]
        pooled_flat = pooled.reshape(B, -1)

        # w_p = min(1.0, mean_frangi / scale)
        w_p = torch.clamp(pooled_flat / max(scale, 1e-8), max=1.0)
        return w_p  # [B, N]

    def forward(
        self,
        images: torch.Tensor,
        frangi_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: [B, 1, H, W] input images normalized to [0, 1]
            frangi_masks: [B, 1, H, W] Frangi vesselness maps

        Returns:
            dict:
                'proj': Tensor[B, N_patches, D_proj]  projected patch tokens
                'patch_features': Tensor[B, N_patches, D]  raw patch tokens (no CLS)
                'hierarchical': list[Tensor[B, N_patches, D]]  features from out_layers
                'vessel_weights': Tensor[B, N_patches]  per-patch w_p
        """
        B, C, H, W = images.shape

        # Validate shapes
        if C != 1:
            raise ValueError(f"Expected 1-channel input, got {C} channels")
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input dims {H}x{W} must be divisible by patch_size {self.patch_size}"
            )

        # Backbone forward
        out = self.backbone(images)  # dict with 'patch_features', 'hierarchical'

        # Get patch tokens (remove CLS token if present)
        patch_features = out["patch_features"]  # [B, N_patches, D] (timm ViT has no CLS)

        # Project
        projected = self.projection_head(patch_features)  # [B, N_patches, D_proj]

        # Compute vessel weights
        vessel_weights = self.compute_patch_weights(frangi_masks)  # [B, N_patches]

        # Hierarchical features (strip CLS if present)
        hierarchical = out.get("hierarchical", [])

        return {
            "proj": projected,
            "patch_features": patch_features,
            "hierarchical": hierarchical,
            "vessel_weights": vessel_weights,
        }
