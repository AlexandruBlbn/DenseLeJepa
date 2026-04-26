"""
Hierarchical ViT-S/8 backbone with intermediate feature extraction.

Uses timm's vit_small_patch8_224 as base, modified to:
1. Accept grayscale (1-channel) input — weight-repeat the first conv
2. Return features from specified intermediate layers
3. Return [CLS] token + patch tokens as flat sequence

Adapted from V-JEPA 2.1's VisionTransformer architecture.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import timm


class HierarchicalViT(nn.Module):
    """
    ViT-S/8 backbone with hierarchical feature extraction.

    Args:
        img_size: int (assumes square input)
        patch_size: int (default 8)
        in_chans: int (default 1, grayscale)
        embed_dim: int (default 384)
        depth: int (default 12)
        num_heads: int (default 6)
        out_layers: list of int (layers to extract features from)
        mlp_ratio: float (default 4.0)

    Shapes:
        Input:  [B, in_chans, H, W]
        Output dict:
            'patch_features': Tensor[B, N_patches+1, embed_dim]  # final layer [CLS + patches]
            'hierarchical': list[Tensor[B, N_patches+1, embed_dim]]  # from out_layers
            'layer_outputs': dict layer_idx -> Tensor[B, N_patches+1, embed_dim]
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        out_layers: Optional[List[int]] = None,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.out_layers = out_layers if out_layers is not None else [depth - 1]

        # Build ViT using timm
        # vit_small_patch8_224 has embed_dim=384, depth=12, num_heads=6, patch_size=8
        vit = timm.create_model(
            "vit_small_patch8_224",
            pretrained=True,
            num_classes=0,  # no classification head
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dynamic_img_size=True,
        )

        # Modify first conv from 3-channel to 1-channel by averaging weights
        old_conv = vit.patch_embed.proj
        new_conv = nn.Conv2d(
            in_chans,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            if in_chans == 1 and old_conv.in_channels == 3:
                # Average over RGB channels
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            elif in_chans == old_conv.in_channels:
                new_conv.weight.data = old_conv.weight.data
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data
        vit.patch_embed.proj = new_conv

        # Store submodules
        self.patch_embed = vit.patch_embed
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm

        # Handle positional embedding — timm includes CLS token in pos_embed
        # We separate CLS pos_embed from spatial pos_embed
        if hasattr(vit, 'pos_embed') and vit.pos_embed is not None:
            # vit_small_patch8_224 pos_embed shape: [1, 1 + 28*28, D]
            # First token is CLS, rest are spatial
            self.register_buffer("pos_embed_cls", vit.pos_embed[:, :1, :])       # [1, 1, D]
            self.register_buffer("pos_embed_spatial", vit.pos_embed[:, 1:, :])   # [1, 784, D]
            self.orig_grid_size = int(vit.pos_embed.shape[1] - 1) ** 0.5  # e.g., 28
        else:
            self.register_buffer("pos_embed_cls", None)
            self.register_buffer("pos_embed_spatial", None)
            self.orig_grid_size = 0

        # Register forward hooks for intermediate layers
        self._hooks: List = []
        self._hook_outputs: Dict[int, torch.Tensor] = {}

        def _make_hook(layer_idx: int):
            def _hook(module, input, output):
                self._hook_outputs[layer_idx] = output
            return _hook

        for idx in self.out_layers:
            if idx < len(self.blocks):
                hook = self.blocks[idx].register_forward_hook(_make_hook(idx))
                self._hooks.append(hook)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] input images

        Returns:
            dict with 'patch_features', 'hierarchical', 'layer_outputs'
        """
        B, C, H, W = x.shape
        self._hook_outputs.clear()

        # Patch embedding: returns [B, N_patches, D] (no CLS token)
        x = self.patch_embed(x)  # [B, N, D] where N = (H/8)*(W/8)
        N_patches = x.shape[1]

        # Add positional embedding (interpolate spatial part only)
        if self.pos_embed_spatial is not None:
            target_h = H // self.patch_size
            target_w = W // self.patch_size
            pos_spatial = self._interpolate_pos_embed(
                self.pos_embed_spatial, target_h, target_w
            )  # [1, target_h * target_w, D]
            x = x + pos_spatial

        x = self.pos_drop(x)

        # Forward through blocks, collecting hooks
        for blk in self.blocks:
            x = blk(x)

        # Final norm
        x = self.norm(x)  # [B, N, D]

        # Collect hierarchical features
        hierarchical: List[torch.Tensor] = []
        for idx in self.out_layers:
            if idx in self._hook_outputs:
                # Hook outputs are pre-norm, apply norm
                h = self.norm(self._hook_outputs[idx])  # [B, N, D]
                hierarchical.append(h)
            elif idx == len(self.blocks) - 1:
                hierarchical.append(x)

        return {
            "patch_features": x,        # [B, N, D]
            "hierarchical": hierarchical,  # list of [B, N, D]
            "layer_outputs": dict(self._hook_outputs),
        }

    def _interpolate_pos_embed(
        self,
        pos_embed: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """
        Interpolate 2D spatial positional embedding to target grid size.

        Args:
            pos_embed: [1, N_orig, D] — spatial-only pos embed (no CLS)
            target_h: int, target grid height in patches
            target_w: int, target grid width in patches

        Returns:
            [1, target_h * target_w, D] interpolated positional embedding
        """
        D = pos_embed.shape[-1]
        orig_side = int(pos_embed.shape[1] ** 0.5)

        # [1, N_orig, D] -> [1, D, orig_side, orig_side]
        pe = pos_embed.reshape(1, orig_side, orig_side, D).permute(0, 3, 1, 2)
        # Interpolate to target grid size
        pe = nn.functional.interpolate(
            pe,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )
        # [1, D, Hp, Wp] -> [1, Hp*Wp, D]
        pe = pe.permute(0, 2, 3, 1).reshape(1, -1, D)
        return pe

    def get_num_layers(self) -> int:
        return len(self.blocks)

    def no_weight_decay(self) -> List[str]:
        return []


def vit_small_patch8(**kwargs) -> HierarchicalViT:
    """Convenience factory for ViT-S/8."""
    return HierarchicalViT(
        patch_size=8,
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs
    )
