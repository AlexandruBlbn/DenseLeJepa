import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Tuple


class SwinTokenEncoder(nn.Module):
    """
    Step-1 encoder:
    image -> feature map -> token sequence [B, N, D]
    """

    def __init__(
        self,
        backbone_name: str = "swinv2_tiny_window8_256",
        pretrained: bool = False,
        out_index: int = 3,
        token_dim: int = 384,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(out_index,),
        )
        backbone_dim = self.backbone.feature_info.channels()[-1]

        self.token_proj = nn.Linear(backbone_dim, token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _to_nchw(feat: torch.Tensor) -> torch.Tensor:
        """
        Robust conversion:
        - if feature map is NHWC, convert to NCHW
        - if already NCHW, keep as-is
        """
        if feat.ndim != 4:
            raise ValueError(f"Expected 4D feature map, got shape {tuple(feat.shape)}")

        # NHWC heuristic: last dim is channel-like and larger than spatial dims
        if feat.shape[-1] > feat.shape[1] and feat.shape[-1] > feat.shape[2]:
            feat = feat.permute(0, 3, 1, 2).contiguous()

        return feat

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)[-1]                 # [B, C, H, W] or [B, H, W, C]
        feat = self._to_nchw(feat)                 # [B, C, H, W]
        b, c, h, w = feat.shape

        tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B, N=H*W, C]
        tokens = self.token_proj(tokens)                         # [B, N, D]
        tokens = self.token_norm(tokens)
        tokens = self.token_dropout(tokens)

        global_tokens = tokens.mean(dim=1)                       # [B, D]

        return {
            "tokens": tokens,
            "global_tokens": global_tokens,
            "spatial_h": torch.tensor(h, device=tokens.device),
            "spatial_w": torch.tensor(w, device=tokens.device),
        }


class TokenPredictor(nn.Module):
    """
    Step-1 dense predictor:
    context tokens -> predicted dense target tokens
    """

    def __init__(self, token_dim: int = 384, hidden_mult: int = 4):
        super().__init__()
        hidden_dim = token_dim * hidden_mult

        self.net = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class DenseLeJepaStep1(nn.Module):
    """
    Step-1 full module:
    - shared encoder for context and target views
    - predictor acts on context tokens
    - returns tensors needed for dense loss and later SIGReg integration
    """

    def __init__(
        self,
        backbone_name: str = "swinv2_tiny_window8_256",
        pretrained: bool = False,
        token_dim: int = 384,
        predictor_hidden_mult: int = 4,
    ):
        super().__init__()
        self.encoder = SwinTokenEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            token_dim=token_dim,
        )
        self.predictor = TokenPredictor(
            token_dim=token_dim,
            hidden_mult=predictor_hidden_mult,
        )

    def forward(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        detach_target: bool = True,
    ) -> Dict[str, torch.Tensor]:
        # Shared-weights encoder branches
        context_pack = self.encoder(x_context)
        target_pack = self.encoder(x_target)

        context_tokens = context_pack["tokens"]
        target_tokens = target_pack["tokens"]
        target_global = target_pack["global_tokens"]

        # Optional detach in early debugging phase
        if detach_target:
            target_tokens = target_tokens.detach()
            target_global = target_global.detach()

        pred_dense = self.predictor(context_tokens)       # [B, N, D]
        pred_global = pred_dense.mean(dim=1)              # [B, D]

        return {
            "context_tokens": context_tokens,
            "target_tokens": target_tokens,
            "pred_dense": pred_dense,
            "pred_global": pred_global,
            "target_global": target_global,
            "spatial_h": context_pack["spatial_h"],
            "spatial_w": context_pack["spatial_w"],
        }


def dense_prediction_loss(
    pred_dense: torch.Tensor,
    target_dense: torch.Tensor,
    normalize_tokens: bool = True,
) -> torch.Tensor:
    """
    Step-1 dense loss:
    token-level MSE between predicted and target tokens
    """
    if normalize_tokens:
        pred_dense = F.layer_norm(pred_dense, (pred_dense.shape[-1],))
        target_dense = F.layer_norm(target_dense, (target_dense.shape[-1],))
    return F.mse_loss(pred_dense, target_dense)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DenseLeJepaStep1(
        backbone_name="swinv2_tiny_window8_256",
        pretrained=False,
        token_dim=384,
    ).to(device)

    
    x_target = torch.randn(2, 3, 256, 256, device=device)
    x_context = x_target.clone()

    out = model(x_context, x_target, detach_target=True)
    loss_dense = dense_prediction_loss(out["pred_dense"], out["target_tokens"])

    print("pred_dense:", out["pred_dense"].shape)
    print("target_tokens:", out["target_tokens"].shape)
    print("pred_global:", out["pred_global"].shape)
    print("dense_loss:", float(loss_dense))