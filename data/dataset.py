"""
XCA Dataset: loads grayscale X-ray Coronary Angiography images
with pre-computed Frangi vesselness masks.

Expects directory structure:
    train/
        img_001.png  (grayscale, any resolution)
        img_001_frangi.npy  (pre-computed Frangi map, same H×W)

If a Frangi .npy file is not found, computes it on-the-fly
using skimage.filters.frangi (with a logged warning).
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import frangi
from skimage.io import imread
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Supported image file extensions
_IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm")


class XcaDataset(Dataset):
    """
    PyTorch Dataset for XCA images with Frangi vesselness masks.

    Args:
        root: Path to image directory
        transform: callable applied to both image and mask (receives dict)
        frangi_suffix: suffix for Frangi .npy files (e.g., '_frangi')
        sigmas: scales for on-the-fly Frangi computation
        frangi_beta: blobness sensitivity for on-the-fly Frangi
        frangi_threshold: vesselness floor when computing patch weights

    Shapes:
        image:       [C, H, W] torch.float32, normalized to [0, 1]
        frangi_mask: [1, H, W] torch.float32, values in [0, 1]

    Returns dict:
        'image': Tensor[C, H, W]
        'frangi_mask': Tensor[1, H, W]
        'path': str
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        frangi_suffix: str = "_frangi",
        sigmas: Sequence[float] = (1, 3, 5, 7),
        frangi_beta: float = 0.5,
        frangi_threshold: float = 0.05,
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.frangi_suffix = frangi_suffix
        self.sigmas = sigmas
        self.frangi_beta = frangi_beta
        self.frangi_threshold = frangi_threshold

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        # Scan for image files
        self.image_paths: list[Path] = []
        for ext in _IMAGE_EXTENSIONS:
            self.image_paths.extend(sorted(self.root.glob(f"*{ext}")))
            self.image_paths.extend(sorted(self.root.glob(f"*/**/*{ext}")))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.root} with extensions {_IMAGE_EXTENSIONS}")

        logger.info(f"Found {len(self.image_paths)} images in {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_frangi_mask(self, image_path: Path) -> np.ndarray:
        """Load pre-computed Frangi mask or compute on-the-fly."""
        frangi_path = image_path.with_suffix("").with_name(
            image_path.stem + self.frangi_suffix + ".npy"
        )
        if frangi_path.exists():
            return np.load(str(frangi_path)).astype(np.float32)

        # Compute on-the-fly
        logger.warning(
            f"Frangi mask not found for {image_path.name}, "
            f"computing on-the-fly (consider pre-computing)"
        )
        image = imread(str(image_path))
        if image.ndim == 3:
            image = image.mean(axis=-1)  # grayscale
        image = image.astype(np.float32)
        image = (image - image.min()) / max(image.max() - image.min(), 1e-8)

        mask = frangi(
            image,
            sigmas=list(self.sigmas),
            beta=self.frangi_beta,
            black_ridges=False,
        )
        return mask.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]

        # Load grayscale image
        image = imread(str(image_path))
        if image.ndim == 3:
            image = image.mean(axis=-1)  # -> [H, W]
        image = image.astype(np.float32)
        # Normalize to [0, 1]
        image = (image - image.min()) / max(image.max() - image.min(), 1e-8)

        # Load / compute Frangi mask
        frangi_mask = self._load_frangi_mask(image_path)

        # Ensure spatial dims match
        if frangi_mask.shape != image.shape:
            raise RuntimeError(
                f"Frangi mask shape {frangi_mask.shape} != image shape {image.shape} "
                f"for {image_path}"
            )

        # Add channel dims: [H, W] -> [1, H, W]
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()       # [1, H, W]
        frangi_tensor = torch.from_numpy(frangi_mask).unsqueeze(0).float()  # [1, H, W]

        sample = {"image": image_tensor, "frangi_mask": frangi_tensor, "path": str(image_path)}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
