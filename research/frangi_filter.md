# Topic 3: Frangi Vesselness Filter
**Date:** 2026-04-26

## Key Findings
- **Purpose:** Specifically designed for the detection of tube-like structures (vessels, neurites) in 2D or 3D images.
- **Mechanism:** Relies on the eigenvalues of the Hessian matrix $(\mathcal{H})$ to determine the local likelihood of a ridge (tube) at multiple scales.
- **Multiscale:** The filter is applied over a range of Gaussian scales ($\sigma$) to capture vessels of varying diameters.

## Mathematical Formulation
For a 2D image, let $\lambda_1, \lambda_2$ be the eigenvalues of the Hessian matrix at scale $\sigma$, where $|\lambda_1| \leq |\lambda_2|$.
The vesselness score $V(\sigma)$ is calculated as:
$$V(\sigma) = \begin{cases} 0 & \text{if } \lambda_2 > 0 \text{ (for bright ridges)} \\ \exp\left(-\frac{R_B^2}{2\beta^2}\right) \left[1 - \exp\left(-\frac{S^2}{2c^2}\right)\right] & \text{otherwise} \end{cases}$$
Where:
- $R_B = |\lambda_1 / \lambda_2|$ : Blobness measure (how much the structure resembles a blob vs. a line).
- $S = \sqrt{\lambda_1^2 + \lambda_2^2}$ : Structureness measure (differentiates vessel from background noise).
- **Final Output:** The maximum response across all scales $V = \max_{\sigma} V(\sigma)$.

## Parameters (`skimage.filters.frangi`)
- `sigmas` (range/list): Range of scales. For XCA, this should match vessel widths in pixels (e.g., `range(1, 10, 2)`).
- `beta` (default 0.5): Weighting for the blobness term. Lower values make it more sensitive to elongated shapes.
- `c` (default 0.5 or half max Hessian norm): Weighting for the structureness term. Controls sensitivity to background noise.
- `black_ridges` (bool): If `True`, detects dark vessels on bright background. If `False` (XCA default), detects bright vessels on dark background.

## Computational Cost
- **Scaling:** $O(S \cdot N \log N)$ where $S$ is the number of scales and $N$ is number of pixels (due to FFT-based Gaussian blurring).
- **Benchmark (Approx):**
  - **512x512:** ~50-100ms.
  - **1024x1024:** ~300-500ms.
  - *Note:* Pre-computing these masks for the entire dataset is highly recommended before SSL training.

## Patch Classification (Vessel vs. Background)
To classify a patch for **Frangi-Masked SIGReg**:
1. Compute the Frangi vesselness map $M$.
2. Segment the map $M$ using a threshold (e.g., $M > 0.05$ or Otsu's method).
3. **Thresholding Patches:**
   - **Vessel Patch:** If more than $T\%$ (e.g., 10%) of pixels in the patch are classified as vessel.
   - **Background Patch:** If vessel pixel coverage is $< 1\%$.
   - **Discard/Neutral:** Patches in between can be ignored or treated with lower weight.

## Limitations for Coronary Arteries (XCA)
- **Calcium Deposits:** High-intensity coronary calcium can trigger high "blobness" and may be suppressed by the filter if $\beta$ is too low.
- **Crossing Vessels:** At points where vessels cross, the tubular assumption fails locally (eigenvalues reflect a blob), causing a "gap" in the mask.
- **Bifurcations:** Similar to crossings, bifurcations may have lower scores than straight segments.
