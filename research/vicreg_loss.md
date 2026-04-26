# Topic 4: VICReg Loss Formulation
**Date:** 2026-04-26

## Key Findings
- **VICReg (Variance-Invariance-Covariance Regularization):** A self-supervised method that prevents collapse without requiring asymmetric architectures (teachers), stop-gradients, or clustering.
- **Explicit Regularization:** Unlike SIGReg which uses Gaussianity tests, VICReg explicitly forces the variance of each dimension to be above a threshold and the covariance between different dimensions to be zero.

## The Three Loss Terms
Total Loss: $\mathcal{L} = \lambda s(Z, Z') + \mu v(Z, Z') + \nu c(Z, Z')$

### 1. Invariance ($s$)
Mean-squared error between representations of two augmented views of the same image:
$$s(Z, Z') = \frac{1}{N} \sum_i \|z_i - z'_i\|^2$$

### 2. Variance ($v$)
A hinge loss on the standard deviation of each dimension across the batch, ensuring that the embedding space is utilized:
$$v(Z) = \frac{1}{d} \sum_{j=1}^d \max(0, \gamma - S(z^j, \epsilon))$$
Where $S$ is the standard deviation, $\gamma$ is a target (typically 1.0), and $\epsilon$ is a small scalar for numerical stability.

### 3. Covariance ($c$)
Penalizes the off-diagonal elements of the covariance matrix to decorrelate dimensions and maximize information capacity:
$$C(Z) = \frac{1}{N-1} \sum_{i=1}^N (z_i - \bar{z})(z_i - \bar{z})^T$$
$$c(Z) = \frac{1}{d} \sum_{j \neq k} [C(Z)]^2_{j,k}$$

## Hyperparameters (Default)
- $\lambda$ (Invariance): 25.0
- $\mu$ (Variance): 25.0
- $\nu$ (Covariance): 1.0
- $\gamma$ (Target Std): 1.0
- $\epsilon$: 1e-4

## Comparison: VICReg vs SIGReg
| Feature | VICReg | SIGReg (BHEP/LeJEPA) |
|---|---|---|
| **Objective** | Force Std Dev and Decorrelation | Test for Isotropic Gaussianity |
| **Complexity** | $O(d^2)$ for Covariance | $O(kd)$ for Sketched CF |
| **Stability** | Highly dependent on $\mu, \nu$ weights | Very stable with single $\lambda$ |
| **Manifold** | Spreads features linearly | More flexible non-parametric fit |

**When to prefer VICReg:** When strict decorrelation of features is required for linear separability in downstream tasks.
**When to prefer SIGReg:** When training budget is limited (lower complexity) or when the manifold structure is highly complex (non-Gaussian local structure).

## PyTorch Implementation Pattern
```python
def vicreg_loss(z, z_prime, sim_coeff=25, std_coeff=25, cov_coeff=1):
    # Invariance
    repr_loss = F.mse_loss(z, z_prime)
    
    # Variance
    z = z - z.mean(dim=0)
    z_prime = z_prime - z_prime.mean(dim=0)
    std_z = torch.sqrt(z.var(dim=0) + 1e-04)
    std_z_prime = torch.sqrt(z_prime.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1 - std_z)) + torch.mean(F.relu(1 - std_z_prime))
    
    # Covariance
    def covariance(x):
        n, d = x.shape
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (n - 1)
        off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
        return off_diag / d

    cov_loss = covariance(z) + covariance(z_prime)
    
    return sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
```
