"""
XCA-specific augmentation transforms for SSL pretraining.

Generates V=2 global views per image with medical-appropriate augmentations:
- RandomResizedCrop with configurable scale
- Random horizontal flip (p=0.5)
- Intensity jitter (brightness, contrast) instead of color jitter
- Gaussian blur (p=0.1, sigma=[0.1, 2.0])

Spatial transforms (crop, flip) are applied identically to both
the image and the Frangi mask. Intensity transforms only affect the image.
"""

import math
import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor
from torchvision import transforms


class _RandomResizedCropPair:
    """
    RandomResizedCrop applied identically to image and mask.

    Args:
        size: target output size
        scale: range of the proportion of the original image to crop
        ratio: range of aspect ratio of the crop
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        scale: Tuple[float, float] = (0.3, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.333),
    ):
        self.size = (size, size) if isinstance(size, int) else size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            image: [C, H, W]
            mask:  [1, H, W]

        Returns:
            image_cropped: [C, H', W']
            mask_cropped:  [1, H', W']
        """
        _, H, W = image.shape
        # Sample crop parameters
        for _ in range(10):
            target_area = random.uniform(*self.scale) * H * W
            aspect_ratio = random.uniform(*self.ratio)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= W and 0 < h <= H:
                i = random.randint(0, H - h)
                j = random.randint(0, W - w)
                break
        else:
            # Fallback: center crop
            i, j, h, w = 0, 0, H, W

        image = TF.resized_crop(image, i, j, h, w, self.size, TF.InterpolationMode.BILINEAR)
        mask = TF.resized_crop(mask, i, j, h, w, self.size, TF.InterpolationMode.NEAREST)
        return image, mask


class _RandomHorizontalFlipPair:
    """Random horizontal flip applied identically to image and mask."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class _IntensityJitter:
    """
    Intensity jitter for grayscale images.
    Applies random brightness and contrast adjustments.

    Args:
        brightness: max brightness factor delta (e.g., 0.4 means [0.6, 1.4])
        contrast: max contrast factor delta
    """

    def __init__(self, brightness: float = 0.4, contrast: float = 0.4):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: Tensor) -> Tensor:
        """image: [C, H, W]"""
        # Brightness
        factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        image = TF.adjust_brightness(image, factor)
        # Contrast
        factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        image = TF.adjust_contrast(image, factor)
        return image.clamp(0.0, 1.0)


class _GaussianBlur:
    """Gaussian blur for grayscale images."""

    def __init__(self, p: float = 0.1, sigma_range: Tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, image: Tensor) -> Tensor:
        """image: [C, H, W]"""
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma_range)
            # Kernel size: 2*ceil(3*sigma) + 1
            ksize = int(2 * math.ceil(3 * sigma) + 1)
            if ksize % 2 == 0:
                ksize += 1
            kernel = self._gaussian_kernel(ksize, sigma).to(image.device, image.dtype)
            # Apply per channel
            C = image.shape[0]
            kernel = kernel.expand(C, 1, ksize, ksize)
            padding = ksize // 2
            image = F.conv2d(image.unsqueeze(0), kernel, padding=padding, groups=C).squeeze(0)
        return image

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> Tensor:
        """Create 1D Gaussian kernel and outer product to 2D."""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g[:, None] * g[None, :]
        return kernel.unsqueeze(0)  # [1, 1, size, size]


class XcaViewGenerator:
    """
    Generates multiple views for SSL training with corresponding Frangi mask crops.

    Each view is created by applying the same spatial transform (crop, flip)
    to both the image and mask, followed by intensity transforms (to image only).

    Args:
        image_size: target size for global views
        global_scale: [min, max] for global crop area ratio
        n_global: number of global views (default 2)
        intensity_jitter: max factor for brightness/contrast
        blur_prob: probability of Gaussian blur
        blur_sigma: (min, max) sigma for Gaussian blur

    Returns:
        List[Tensor]: views — list of [C, H', W'] tensors
        List[Tensor]: mask_views — list of [1, H', W'] tensors
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 512,
        global_scale: Tuple[float, float] = (0.3, 1.0),
        n_global: int = 2,
        intensity_jitter: float = 0.4,
        blur_prob: float = 0.1,
        blur_sigma: Tuple[float, float] = (0.1, 2.0),
    ):
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.n_global = n_global
        self.crop = _RandomResizedCropPair(self.image_size, scale=global_scale)
        self.flip = _RandomHorizontalFlipPair(p=0.5)
        self.intensity_jitter = _IntensityJitter(brightness=intensity_jitter, contrast=intensity_jitter)
        self.blur = _GaussianBlur(p=blur_prob, sigma_range=blur_sigma)

    def __call__(self, images: Tensor, masks: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            images: [B, 1, H, W] input batch
            masks:  [B, 1, H, W] frangi masks

        Returns:
            views: list of n_global [B, 1, H', W'] tensors
            mask_views: list of n_global [B, 1, H', W'] tensors
        """
        views: List[Tensor] = []
        mask_views: List[Tensor] = []

        B, C, H, W = images.shape

        for _ in range(self.n_global):
            view_batch: List[Tensor] = []
            mask_batch: List[Tensor] = []
            for b in range(B):
                img = images[b]   # [C, H, W]
                msk = masks[b]    # [1, H, W]

                # Spatial transforms (shared)
                img, msk = self.crop(img, msk)
                img, msk = self.flip(img, msk)

                # Intensity transforms (image only)
                img = self.intensity_jitter(img)
                img = self.blur(img)

                view_batch.append(img)
                mask_batch.append(msk)

            views.append(torch.stack(view_batch, dim=0))
            mask_views.append(torch.stack(mask_batch, dim=0))

        return views, mask_views


def xca_transform(
    image_size: int = 512,
    global_scale: Tuple[float, float] = (0.3, 1.0),
    intensity_jitter: float = 0.4,
    blur_prob: float = 0.1,
) -> transforms.Compose:
    """
    Returns a composed transform for training.
    Operates on dicts with 'image' and 'frangi_mask' keys.
    """
    # This is a simplified transform for single-image (non-batch) usage.
    # For batched view generation, use XcaViewGenerator.
    crop = _RandomResizedCropPair(image_size, scale=global_scale)
    flip = _RandomHorizontalFlipPair(p=0.5)
    jitter = _IntensityJitter(brightness=intensity_jitter, contrast=intensity_jitter)
    blur = _GaussianBlur(p=blur_prob)

    def _transform(sample: dict) -> dict:
        img = sample["image"]   # [C, H, W]
        msk = sample["frangi_mask"]  # [1, H, W]
        img, msk = crop(img, msk)
        img, msk = flip(img, msk)
        img = jitter(img)
        img = blur(img)
        sample["image"] = img
        sample["frangi_mask"] = msk
        return sample

    return transforms.Compose([transforms.Lambda(_transform)])
