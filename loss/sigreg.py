"""
SIGReg: Sketched Isotropic Gaussian Regularization via BHEP characteristic function.

Computes the BHEP test statistic efficiently using random projections (sketching)
and quadrature integration. This avoids the O(N²) pairwise distance computation
of the full BHEP, yielding O(N * proj_dim) complexity.

Reference: LeJEPA (repos/lejepa/lejepa/multivariate/bhep.py)

The sketch works as follows:
    x_t = (z @ A) * t   with random A ~ N(0, 1/D) and quadrature knots t
    err = |mean(cos(x_t)) - phi|² + |mean(sin(x_t))|²
    statistic = ∫ err(t) * w(t) dt
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization.

    Penalizes deviation of the empirical embedding distribution from N(0, I_d).

    Args:
        proj_dim: int, dimension to project to for sketching (default 256)
        knots: int, number of quadrature knots (default 17)
        beta: float, bandwidth parameter (default 0.1)

    Shapes:
        Input:  [N, D] — batch of embeddings
        Output: scalar tensor

    Reference:
        LeJEPA BHEP implementation.
    """

    def __init__(
        self,
        proj_dim: int = 256,
        knots: int = 17,
        beta: float = 0.1,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.knots = knots
        self.beta = beta

        # Register random projection matrix A ~ N(0, 1) as buffer.
        # Shape: [D, proj_dim] — populated on first forward when D is known.
        self.register_buffer("A", None)

        # Create quadrature knots t in [0, t_max]
        # Gauss-Legendre quadrature on [0, 4/beta] covers the relevant CF range.
        t_max = 4.0 / max(beta, 1e-8)
        t = torch.linspace(0, t_max, knots).unsqueeze(-1)  # [knots, 1]

        # Weights for quadrature (simple trapezoidal rule)
        dt = t_max / (knots - 1) if knots > 1 else 1.0
        weights = torch.ones(knots) * dt
        weights[0] *= 0.5
        weights[-1] *= 0.5

        # phi(t) = exp(-beta² * t² / 2)  — CF of N(0, 1)
        phi = torch.exp(-(beta ** 2) * (t.squeeze(-1) ** 2) / 2.0)

        self.register_buffer("t", t)          # [knots, 1]
        self.register_buffer("weights", weights)  # [knots]
        self.register_buffer("phi", phi)       # [knots]

    def _init_projection(self, D: int, device: torch.device, dtype: torch.dtype):
        """Initialize the random projection matrix on first forward."""
        if self.A is not None and self.A.shape[0] == D:
            return
        A = torch.randn(D, self.proj_dim, device=device, dtype=dtype) / math.sqrt(D)
        self.register_buffer("A", A)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss.

        Args:
            z: [N, D] batch of embeddings

        Returns:
            scalar tensor (SIGReg statistic)

        Algorithm:
            1. x_t = (z @ A) * t^T   → shape [N, proj_dim, knots]
            2. mean_cos = mean(cos(x_t), dim=0)  → [proj_dim, knots]
            3. mean_sin = mean(sin(x_t), dim=0)  → [proj_dim, knots]
            4. err = |mean_cos - phi|² + |mean_sin|²  → [proj_dim, knots]
            5. statistic = mean(err @ weights) * sqrt(N)
        """
        N, D = z.shape

        # Initialize projection matrix if needed
        self._init_projection(D, z.device, z.dtype)

        # Project: [N, D] @ [D, K] -> [N, K] where K = proj_dim
        z_proj = z @ self.A  # [N, K]

        # Compute x_t = z_proj * t^T: [N, K] * [1, knots] -> [N, K, knots]
        x_t = z_proj.unsqueeze(-1) * self.t.unsqueeze(0).unsqueeze(0)  # [N, K, knots]

        # Mean cos and sin across batch
        cos_mean = torch.cos(x_t).mean(dim=0)  # [K, knots]
        sin_mean = torch.sin(x_t).mean(dim=0)  # [K, knots]

        # Error per projection dimension and knot
        err = (cos_mean - self.phi.unsqueeze(0)).pow(2) + sin_mean.pow(2)  # [K, knots]

        # Integrate over knots: [K, knots] @ [knots] -> [K]
        statistic_per_proj = err @ self.weights  # [K]

        # Mean over projections and scale by N for consistent test power.
        # err = |mean(cos) - phi|^2 + |mean(sin)|^2 is O(1/N) because mean reduces over N.
        # Multiplying by N makes the statistic O(1), matching the reference BHEP formulation
        # where the double-sum is divided by N^2 and then re-scaled.
        # Reference: LeJEPA MINIMAL.md line 74: statistic = (err @ weights) * proj.size(-2)
        statistic = statistic_per_proj.mean() * N

        return statistic
