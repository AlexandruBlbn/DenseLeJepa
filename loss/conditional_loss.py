"""
Frangi-Masked Conditional Loss.

Combines SIGReg (background regularization) + VICReg (vessel structure) with:
1. Soft weighting: each patch contributes to both losses via w_p
2. Adaptive normalization: each loss term normalized by its own EMA running mean

Algorithm:
    - SIGReg applied to ALL patches weighted by (1 - w_p).
      Background patches (low w_p) dominate the Gaussianity test.
    - VICReg applied only to vessel-heavy patches (w_p > 0.5).
      Preserves vessel manifold structure.
    - Each raw loss is divided by its EMA running mean to balance scales.
"""

from typing import Dict

import torch
import torch.nn as nn

from .sigreg import SIGReg
from .vicreg import vessel_vicreg_loss


class FrangiMaskedConditionalLoss(nn.Module):
    """
    Frangi-masked conditional SSL loss: SIGReg on background + VICReg on vessels.

    Args:
        sigreg_cfg: dict for SIGReg (keys: proj_dim, knots, beta)
        vicreg_cfg: dict for VICReg weights (keys: sim_coeff, std_coeff, cov_coeff,
                     target_std)
        lambda_bg: float, SIGReg base weight
        lambda_vessel: float, VICReg base weight
        loss_ema_decay: float, EMA decay for running mean normalization

    Forward:
        proj1: [B, N, D] — projections from view 1
        proj2: [B, N, D] — projections from view 2
        w_p:   [B, N]    — vessel weights in [0, 1]

    Returns dict:
        'loss': scalar — total weighted loss
        'loss_sigreg': scalar — normalized SIGReg
        'loss_vicreg': scalar — normalized VICReg
        'loss_sigreg_raw': scalar — unnormalized SIGReg
        'loss_vicreg_raw': scalar — unnormalized VICReg
        'sigreg_weight': float — effective SIGReg weight (lambda / ema)
        'vicreg_weight': float — effective VICReg weight (lambda / ema)
        'n_vessel_patches': int — number of vessel patches (w_p > 0.5)
        'n_total_patches': int — total patches
    """

    def __init__(
        self,
        sigreg_cfg: dict,
        vicreg_cfg: dict,
        lambda_bg: float = 1.0,
        lambda_vessel: float = 1.0,
        loss_ema_decay: float = 0.9,
    ):
        super().__init__()
        self.lambda_bg = lambda_bg
        self.lambda_vessel = lambda_vessel
        self.loss_ema_decay = loss_ema_decay

        # SIGReg module
        self.sigreg = SIGReg(**sigreg_cfg)

        # VICReg config
        self.vicreg_cfg = vicreg_cfg

        # EMA running means for adaptive normalization
        self.register_buffer("ema_sigreg", torch.zeros(1))
        self.register_buffer("ema_vicreg", torch.zeros(1))
        # Track whether EMAs have been initialized
        self.register_buffer("_ema_initialized", torch.tensor(0, dtype=torch.bool))

    def forward(
        self,
        proj1: torch.Tensor,
        proj2: torch.Tensor,
        w_p: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Frangi-masked conditional loss.

        Args:
            proj1: [B, N, D] projections from view 1
            proj2: [B, N, D] projections from view 2
            w_p:   [B, N] vessel weights in [0, 1]

        Returns:
            dict of scalar losses and metadata
        """
        B, N, D = proj1.shape

        # --- Step 1: Prepare flattened tensors ---
        # [B, N, D] -> [B*N, D]
        z1_flat = proj1.reshape(-1, D)
        z2_flat = proj2.reshape(-1, D)
        w_flat = w_p.reshape(-1)  # [B*N]

        # --- Step 2: SIGReg loss (background regularization) ---
        # SIGReg tests if z ~ N(0, I). We run it on all patches.
        # The loss is naturally dominated by background patches (majority).
        # To explicitly weight: we scale embeddings by sqrt(1 - w_p).
        # Background patches (w_p ≈ 0) contribute fully.
        # Vessel patches (w_p ≈ 1) contribute negligibly.
        z_bg_weighted = z1_flat * torch.sqrt(
            (1.0 - w_flat).clamp(min=0.0).unsqueeze(-1)
        )
        # Use only the first view for SIGReg (both views should be N(0,I))
        loss_sigreg_raw = self.sigreg(z_bg_weighted)

        # --- Step 3: VICReg loss (vessel structure) ---
        # Select vessel-heavy patches: w_p > 0.5
        vessel_mask = w_flat > 0.5  # [B*N]
        n_vessel = vessel_mask.sum().item()

        if n_vessel > 1:  # Need at least 2 samples for covariance
            z1_vessel = z1_flat[vessel_mask]  # [N_ves, D]
            z2_vessel = z2_flat[vessel_mask]  # [N_ves, D]
            vicreg_out = vessel_vicreg_loss(
                z1_vessel, z2_vessel, **self.vicreg_cfg
            )
            loss_vicreg_raw = vicreg_out["loss"]
        else:
            loss_vicreg_raw = torch.tensor(0.0, device=proj1.device)

        # --- Step 4: Adaptive normalization by EMA running mean ---
        with torch.no_grad():
            if not self._ema_initialized:
                # Initialize EMAs with first batch values
                self.ema_sigreg.copy_(loss_sigreg_raw.detach())
                self.ema_vicreg.copy_(loss_vicreg_raw.detach() if n_vessel > 1
                                      else torch.ones_like(self.ema_vicreg))
                self._ema_initialized.fill_(True)
            else:
                decay = self.loss_ema_decay
                self.ema_sigreg.mul_(decay).add_(
                    (1.0 - decay) * loss_sigreg_raw.detach()
                )
                if n_vessel > 1:
                    self.ema_vicreg.mul_(decay).add_(
                        (1.0 - decay) * loss_vicreg_raw.detach()
                    )

        # Normalize
        eps = 1e-6
        loss_sigreg = loss_sigreg_raw / self.ema_sigreg.clamp(min=eps)
        loss_vicreg = loss_vicreg_raw / self.ema_vicreg.clamp(min=eps)

        # --- Step 5: Combine ---
        total_loss = (
            self.lambda_bg * loss_sigreg
            + self.lambda_vessel * loss_vicreg
        )

        return {
            "loss": total_loss,
            "loss_sigreg": loss_sigreg,
            "loss_vicreg": loss_vicreg,
            "loss_sigreg_raw": loss_sigreg_raw,
            "loss_vicreg_raw": loss_vicreg_raw,
            "sigreg_weight": self.lambda_bg / self.ema_sigreg.clamp(min=eps).item(),
            "vicreg_weight": self.lambda_vessel / self.ema_vicreg.clamp(min=eps).item(),
            "n_vessel_patches": n_vessel,
            "n_total_patches": B * N,
        }
