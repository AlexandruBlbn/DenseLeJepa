import torch
import torch.nn.functional as F
import repos.lejepa as lejepa

class GlobalSIGReg(torch.nn.Module):
    def __init__(self, num_slices=256, n_points=17):
        super().__init__()
        self.test = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(n_points=n_points),
            num_slices=num_slices,
            reduction="mean",
        )

    def forward(self, tokens):
        # tokens: [B, N, D]
        g = tokens.mean(dim=1)                   # [B, D]
        g = F.layer_norm(g, (g.shape[-1],))
        return self.test(g)