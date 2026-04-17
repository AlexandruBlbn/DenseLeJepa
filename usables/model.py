import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Optional, Sequence, Tuple
from sigreg import GlobalSIGReg

class SwinTokenEncoder(nn.Module):
    """
    Step-1 encoder:
    image -> multiple feature maps -> token sequences [B, N_i, D]
    """

    def __init__(
        self,
        backbone_name: str = "swinv2_tiny_window8_256",
        pretrained: bool = False,
        out_indices: Tuple[int, ...] = (1, 2, 3),
        token_dim: int = 384,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        stage_dims = self.backbone.feature_info.channels()

        self.token_proj = nn.ModuleList([nn.Linear(d, token_dim) for d in stage_dims])
        self.token_norm = nn.ModuleList([nn.LayerNorm(token_dim) for _ in stage_dims])
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
        feats = self.backbone(x)

        token_layers = []
        spatial_shapes = []
        for feat, proj, norm in zip(feats, self.token_proj, self.token_norm):
            feat = self._to_nchw(feat)
            _, _, h, w = feat.shape

            tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B, N=H*W, C]
            tokens = proj(tokens)                                   # [B, N, D]
            tokens = norm(tokens)
            tokens = self.token_dropout(tokens)

            token_layers.append(tokens)
            spatial_shapes.append((h, w))

        global_tokens = token_layers[-1].mean(dim=1)  # [B, D]

        return {
            "token_layers": token_layers,
            "tokens": token_layers[-1],
            "global_tokens": global_tokens,
            "spatial_shapes": spatial_shapes,
            "spatial_h": torch.tensor(spatial_shapes[-1][0], device=token_layers[-1].device),
            "spatial_w": torch.tensor(spatial_shapes[-1][1], device=token_layers[-1].device),
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
    - predictor acts on all context token layers
    - returns tensors for deep dense supervision + SIGReg
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

        pred_dense_layers = [self.predictor(t) for t in context_pack["token_layers"]]
        target_layers = target_pack["token_layers"]
        target_global = target_pack["global_tokens"]

        # Optional detach in early debugging phase
        if detach_target:
            target_layers = [t.detach() for t in target_layers]
            target_global = target_global.detach()

        pred_dense = pred_dense_layers[-1]
        pred_global = pred_dense.mean(dim=1)

        return {
            "context_tokens": context_pack["tokens"],
            "pred_dense_layers": pred_dense_layers,
            "target_layers": target_layers,
            "pred_dense": pred_dense,
            "target_tokens": target_layers[-1],
            "pred_global": pred_global,
            "target_global": target_global,
            "spatial_shapes": context_pack["spatial_shapes"],
            "spatial_h": context_pack["spatial_h"],
            "spatial_w": context_pack["spatial_w"],
        }


def sample_block_token_mask(
    batch_size: int,
    grid_h: int,
    grid_w: int,
    mask_ratio: float = 0.65,
    min_block: int = 2,
    max_block: int = 4,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Returns boolean mask [B, N], where True means masked token.
    """
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError(f"mask_ratio must be in (0,1), got {mask_ratio}")

    total_tokens = grid_h * grid_w
    target_masked = max(1, int(round(total_tokens * mask_ratio)))
    mask = torch.zeros(batch_size, total_tokens, dtype=torch.bool, device=device)

    for b in range(batch_size):
        for _ in range(256):
            if int(mask[b].sum().item()) >= target_masked:
                break

            bh = int(torch.randint(min_block, max_block + 1, (1,), device=device).item())
            bw = int(torch.randint(min_block, max_block + 1, (1,), device=device).item())
            bh = min(bh, grid_h)
            bw = min(bw, grid_w)

            top = int(torch.randint(0, grid_h - bh + 1, (1,), device=device).item())
            left = int(torch.randint(0, grid_w - bw + 1, (1,), device=device).item())

            block = torch.zeros((grid_h, grid_w), dtype=torch.bool, device=device)
            block[top:top + bh, left:left + bw] = True
            mask[b] |= block.flatten()

        cur = int(mask[b].sum().item())
        if cur > target_masked:
            idx = torch.nonzero(mask[b], as_tuple=False).squeeze(1)
            keep = idx[torch.randperm(cur, device=device)[:target_masked]]
            mask[b].fill_(False)
            mask[b, keep] = True
        elif cur < target_masked:
            free = torch.nonzero(~mask[b], as_tuple=False).squeeze(1)
            need = target_masked - cur
            extra = free[torch.randperm(free.numel(), device=device)[:need]]
            mask[b, extra] = True

    return mask


def mask_image_with_token_mask(
    x: torch.Tensor,
    token_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """
    Upsamples token mask to image and masks pixels for context view.
    """
    if token_mask.dtype != torch.bool:
        token_mask = token_mask.bool()

    b, _, h, w = x.shape
    if token_mask.shape != (b, grid_h * grid_w):
        raise ValueError(
            f"Expected token_mask shape {(b, grid_h * grid_w)}, got {tuple(token_mask.shape)}"
        )

    mask_2d = token_mask.view(b, 1, grid_h, grid_w).float()
    mask_img = F.interpolate(mask_2d, size=(h, w), mode="nearest")
    return x * (1.0 - mask_img) + fill_value * mask_img


def resize_token_mask(
    token_mask: torch.Tensor,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Resize a boolean token mask from src grid to dst grid.
    """
    if token_mask.dtype != torch.bool:
        token_mask = token_mask.bool()

    b, n = token_mask.shape
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw

    if n != src_h * src_w:
        raise ValueError(
            f"Mask has {n} tokens but src grid is {src_h}x{src_w}"
        )

    m = token_mask.view(b, 1, src_h, src_w).float()
    m = F.interpolate(m, size=(dst_h, dst_w), mode="nearest")
    return m.view(b, dst_h * dst_w).bool()


def dense_prediction_loss(
    pred_dense: torch.Tensor,
    target_dense: torch.Tensor,
    normalize_tokens: bool = True,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    If token_mask is provided [B, N] bool, compute loss only on masked tokens.
    """
    if normalize_tokens:
        pred_dense = F.layer_norm(pred_dense, (pred_dense.shape[-1],))
        target_dense = F.layer_norm(target_dense, (target_dense.shape[-1],))

    per_token = (pred_dense - target_dense).pow(2).mean(dim=-1)  # [B, N]

    if token_mask is None:
        return per_token.mean()

    if token_mask.dtype != torch.bool:
        token_mask = token_mask.bool()
    if token_mask.shape != per_token.shape:
        raise ValueError(
            f"token_mask shape {tuple(token_mask.shape)} does not match {tuple(per_token.shape)}"
        )

    if bool(token_mask.any()):
        return per_token[token_mask].mean()
    return per_token.mean()


def deep_dense_loss(
    pred_layers: Sequence[torch.Tensor],
    target_layers: Sequence[torch.Tensor],
    base_token_mask: torch.Tensor,
    base_hw: Tuple[int, int],
    layer_hws: Sequence[Tuple[int, int]],
    normalize_tokens: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dense supervision over multiple layers.
    Computes both masked-token and visible-token reconstruction losses.
    """
    if not (len(pred_layers) == len(target_layers) == len(layer_hws)):
        raise ValueError("pred_layers, target_layers, and layer_hws must have same length")

    loss_masked = pred_layers[0].new_tensor(0.0)
    loss_visible = pred_layers[0].new_tensor(0.0)
    n_layers = len(pred_layers)

    for pred_layer, target_layer, hw in zip(pred_layers, target_layers, layer_hws):
        layer_mask = resize_token_mask(base_token_mask, src_hw=base_hw, dst_hw=hw)
        visible_mask = ~layer_mask

        loss_masked = loss_masked + dense_prediction_loss(
            pred_layer,
            target_layer,
            normalize_tokens=normalize_tokens,
            token_mask=layer_mask,
        )
        loss_visible = loss_visible + dense_prediction_loss(
            pred_layer,
            target_layer,
            normalize_tokens=normalize_tokens,
            token_mask=visible_mask,
        )

    loss_masked = loss_masked / n_layers
    loss_visible = loss_visible / n_layers
    loss_dense = loss_masked + loss_visible
    return loss_dense, loss_masked, loss_visible


def dense_lejepa_step3_losses(
    outputs: Dict[str, torch.Tensor],
    sigreg: nn.Module,
    token_mask: torch.Tensor,
    lambda_sig: float = 0.02,
    normalize_tokens: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Step-3 objective:
    - deep dense supervision on masked + visible tokens
    - global SIGReg regularization (single lambda, LeJEPA-style simplicity)
    """
    base_hw = tuple(outputs["spatial_shapes"][-1])

    loss_dense, loss_masked, loss_visible = deep_dense_loss(
        outputs["pred_dense_layers"],
        outputs["target_layers"],
        base_token_mask=token_mask,
        base_hw=base_hw,
        layer_hws=outputs["spatial_shapes"],
        normalize_tokens=normalize_tokens,
    )

    loss_sigreg = sigreg(outputs["pred_dense_layers"][-1])
    loss_total = loss_dense + lambda_sig * loss_sigreg

    return {
        "loss_total": loss_total,
        "loss_dense": loss_dense,
        "loss_masked": loss_masked,
        "loss_visible": loss_visible,
        "loss_sigreg": loss_sigreg,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DenseLeJepaStep1(
        backbone_name="swinv2_tiny_window8_256",
        pretrained=False,
        token_dim=384,
    ).to(device)

    sigreg = GlobalSIGReg(knots=17, t_max=3.0, num_projections=256).to(device)

    x_target = torch.randn(2, 3, 256, 256, device=device)

    # Probe token grid once (for mask generation)
    with torch.no_grad():
        probe = model.encoder(x_target)
    grid_h, grid_w = probe["spatial_shapes"][-1]

    token_mask = sample_block_token_mask(
        batch_size=x_target.size(0),
        grid_h=grid_h,
        grid_w=grid_w,
        mask_ratio=0.65,
        min_block=2,
        max_block=4,
        device=device,
    )

    x_context = mask_image_with_token_mask(
        x_target, token_mask=token_mask, grid_h=grid_h, grid_w=grid_w, fill_value=0.0
    )

    out = model(x_context, x_target, detach_target=True)
    losses = dense_lejepa_step3_losses(
        out,
        sigreg=sigreg,
        token_mask=token_mask,
        lambda_sig=0.02,
    )

    print("pred_dense:", out["pred_dense"].shape)
    print("target_tokens:", out["target_tokens"].shape)
    print("pred_global:", out["pred_global"].shape)
    print("supervised_layers:", len(out["pred_dense_layers"]))
    print("masked_ratio:", float(token_mask.float().mean()))
    print("loss_dense:", losses["loss_dense"].detach().item())
    print("loss_masked:", losses["loss_masked"].detach().item())
    print("loss_visible:", losses["loss_visible"].detach().item())
    print("loss_sigreg:", losses["loss_sigreg"].detach().item())
    print("loss_total:", losses["loss_total"].detach().item())