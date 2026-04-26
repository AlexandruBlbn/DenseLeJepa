# Topic 1: LeJEPA SIGReg Implementation
**Date:** 2026-04-26

## Key Findings
- **SIGReg (Sketched Isotropic Gaussian Regularization):** A heuristics-free objective that constrains learned embeddings to an optimal isotropic Gaussian distribution.
- **Single-Student Architecture:** Unlike standard JEPA or BYOL/DINO, LeJEPA does NOT use an EMA teacher, stop-gradients, or momentum encoders. It is a "lean" architecture with a single trade-off hyperparameter ($\lambda$).
- **BHEP (Beta-Henze Energy-based Projection):** The core multivariate normality test used to enforce the Gaussian constraint.
- **Characteristic Function Integration:** Implementation often uses a sketched version of the characteristic function (CF) to improve efficiency via quadrature.

## Mathematical Formulation
The BHEP statistic $T_{N,\beta}$ measures the distance between the empirical characteristic function of the data and the characteristic function of a standard normal distribution:
$$T_{N,\beta} = \text{LHS} - \text{RHS} + \text{Constant}$$
Where:
- **LHS:** $\frac{1}{N^2} \sum_{i,j} \exp\left(-\frac{\beta^2}{2} \|x_i - x_j\|^2\right)$
- **RHS:** $\frac{2}{N(1+\beta^2)^{D/2}} \sum_i \exp\left(-\frac{\beta^2}{2(1+\beta^2)} \|x_i\|^2\right)$
- **Constant:** $(1 + 2\beta^2)^{-D/2}$

In the "sketched" implementation (`SIGReg` class in `MINIMAL.md`), it uses random projections $A$ and a set of knots $t$:
$$\text{err} = | \text{mean}(\cos(X A t)) - \phi |^2 + | \text{mean}(\sin(X A t)) |^2$$

## Implementation Details
### Single-Student Forward Pass
The model consists of a backbone encoder and a projection head.
1. **Views:** Generate $V$ views of the same image (e.g., 2 global + 6 local).
2. **Embed:** Pass all views through the single encoder.
3. **Loss:** 
   - **Invariance Loss:** Minimize variance across views of the same sample: `(proj.mean(0) - proj).square().mean()`.
   - **SIGReg Loss:** Regularize the distribution of the projected embeddings toward Isotropic Gaussian.
   - **Total Loss:** $(1-\lambda) \cdot \text{Invariance} + \lambda \cdot \text{SIGReg}$.

### Augmentation Strategy
- **Global Views:** 224x224, scale (0.3, 1.0).
- **Local Views:** 98x98, scale (0.05, 0.3).
- **Transforms:** RandomResizedCrop, HorizontalFlip, ColorJitter, Grayscale, GaussianBlur, Solarize.

### Loss Signature
- **Inputs:** `proj` tensor of shape `(N, D)` or `(V, N, D)` where $V$ is number of views, $N$ is batch size, $D$ is embedding dimension.
- **Outputs:** Scalar tensor (loss).
- **Shapes:** The BHEP statistic typically reduces over the sample dimension $N$.

## Hyperparameters (Default)
- `beta`: 0.1 (bandwidth for Gaussian kernel). Smaller values increase sensitivity to local deviations.
- `lamb` ($\lambda$): 0.02 (trade-off between invariance and regularization).
- `proj_dim`: 128 or 256.
- `knots`: 17 (for sketched implementation).

## Memory Efficiency
- **Linear complexity:** The sketched version uses random projections to avoid the $O(N^2)$ cost of the full BHEP pairwise distance matrix.
- **Mixed Precision:** Uses `bfloat16` natively for memory savings and speed on modern GPUs.

## Concerns for XCA Domain
- **Grayscale Nature:** XCA images are grayscale; standard ImageNet color-jittering might need adjustment (mapped to intensity/contrast variations only).
- **Vessel Structure:** Local views might be too small to capture connectivity if the scale is $ < 0.05$.
- **High Resolution:** Medical images often benefit from larger input sizes than 224x224.
