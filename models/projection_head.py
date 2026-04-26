"""
MLP projection head. Maps encoder patch tokens to D_proj space.

Architecture:
    Linear(in_dim, hidden_dim) → BN → ReLU → Linear(hidden_dim, out_dim) → BN
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    MLP projection head for SSL training.

    Args:
        in_dim: int, input dimension (embed_dim from encoder)
        hidden_dim: int (default 512)
        out_dim: int (default 128)
        n_layers: int, number of MLP layers (default 2)

    Shapes:
        Input:  [B, N, in_dim]
        Output: [B, N, out_dim]
    """

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 512,
        out_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        layers = []
        prev_dim = in_dim
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim

        # Final layer (no ReLU)
        layers.extend([
            nn.Linear(prev_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        ])

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, in_dim] patch tokens

        Returns:
            [B, N, out_dim] projected tokens

        Note:
            BatchNorm1d is applied on flattened [B*N, D] which computes
            statistics over all patches across the batch. This is standard
            practice in SSL (SimCLR, VICReg, LeJEPA) and provides consistent
            normalization regardless of spatial position.
        """
        B, N, D = x.shape

        # Reshape for BatchNorm1d: [B*N, D]
        x_flat = x.reshape(-1, D)
        out_flat = self.mlp(x_flat)  # [B*N, out_dim]
        # Reshape back
        out = out_flat.reshape(B, N, -1)
        return out
