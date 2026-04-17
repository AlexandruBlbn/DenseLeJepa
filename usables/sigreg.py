import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalSIGReg(nn.Module):
    """
    Global-only SIGReg for Dense-LeJEPA Step 2.
    Accepts either:
    - [B, N, D] token tensors
    - [B, D] global embedding tensors
    """

    def __init__(self, knots: int = 17, t_max: float = 3.0, num_projections: int = 256):
        super().__init__()
        if knots < 3:
            raise ValueError("knots must be >= 3")
        self.num_projections = int(num_projections)

        t = torch.linspace(0.0, float(t_max), int(knots), dtype=torch.float32)
        dt = float(t_max) / float(knots - 1)

        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt

        phi = torch.exp(-0.5 * t.square())

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def _to_global(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.mean(dim=1)  # [B, D]
        elif x.ndim != 2:
            raise ValueError(f"Expected [B, N, D] or [B, D], got shape {tuple(x.shape)}")

        x = F.layer_norm(x, (x.shape[-1],))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self._to_global(x)  # [B, D]
        bsz, dim = g.shape
        device = g.device
        dtype = g.dtype

        # Random normalized slicing directions A in R^{D x K}
        A = torch.randn(dim, self.num_projections, device=device, dtype=dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)

        # Project embeddings -> [B, K]
        proj = g @ A

        # Characteristic-function matching
        t = self.t.to(device=device, dtype=dtype)
        phi = self.phi.to(device=device, dtype=dtype)
        weights = self.weights.to(device=device, dtype=dtype)

        x_t = proj.unsqueeze(-1) * t  # [B, K, T]
        cos_mean = torch.cos(x_t).mean(dim=0)  # [K, T]
        sin_mean = torch.sin(x_t).mean(dim=0)  # [K, T]

        err = (cos_mean - phi).pow(2) + sin_mean.pow(2)
        stat = (err @ weights) * bsz  # [K]
        return stat.mean()