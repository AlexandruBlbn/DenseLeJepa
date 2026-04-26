"""
VICReg-style loss for vessel patches.

Three terms:
1. Invariance: MSE between two views of the same patch
2. Variance: hinge loss on std dev per dimension (prevents collapse)
3. Covariance: penalizes off-diagonal covariance (decorrelates dims)

Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization"
"""

import torch
import torch.nn.functional as F
from typing import Dict


def vessel_vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    target_std: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    VICReg loss computed on vessel patch embeddings.

    Args:
        z1: [N_ves, D] — first view vessel patch projections
        z2: [N_ves, D] — second view vessel patch projections
        sim_coeff: weight for invariance (MSE) term
        std_coeff: weight for variance (hinge) term
        cov_coeff: weight for covariance (decorrelation) term
        target_std: target standard deviation for variance hinge

    Returns:
        dict with keys:
            'loss': scalar — weighted total
            'invariance': scalar — unweighted MSE
            'variance': scalar — unweighted variance loss
            'covariance': scalar — unweighted covariance loss

    Shapes:
        z1, z2: [N_ves, D_proj]
    """
    N, D = z1.shape

    # --- Invariance: MSE between two views ---
    invariance_loss = F.mse_loss(z1, z2)

    # --- Variance: hinge loss on std ---
    # Center embeddings
    z1_centered = z1 - z1.mean(dim=0)    # [N, D]
    z2_centered = z2 - z2.mean(dim=0)    # [N, D]

    std_z1 = torch.sqrt(z1_centered.var(dim=0, unbiased=False) + 1e-4)  # [D]
    std_z2 = torch.sqrt(z2_centered.var(dim=0, unbiased=False) + 1e-4)  # [D]

    variance_loss = (
        F.relu(target_std - std_z1).mean()
        + F.relu(target_std - std_z2).mean()
    ) / 2.0

    # --- Covariance: penalize off-diagonal elements ---
    def _covariance_loss(x: torch.Tensor) -> torch.Tensor:
        """
        Compute covariance regularization for one view.

        Args:
            x: [N, D] centered embeddings

        Returns:
            scalar: mean squared off-diagonal covariance
        """
        # Covariance matrix: [D, D]
        # Clamp N-1 to min 1 to avoid division by zero when N_ves = 1
        n_effective = max(N - 1, 1)
        cov = (x.T @ x) / n_effective
        # Zero out diagonal
        off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
        return off_diag / D

    cov_loss = (_covariance_loss(z1_centered) + _covariance_loss(z2_centered)) / 2.0

    # Total
    total_loss = (
        sim_coeff * invariance_loss
        + std_coeff * variance_loss
        + cov_coeff * cov_loss
    )

    return {
        "loss": total_loss,
        "invariance": invariance_loss,
        "variance": variance_loss,
        "covariance": cov_loss,
    }
