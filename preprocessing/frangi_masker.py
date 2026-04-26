"""
Pre-compute Frangi vesselness masks for a directory of XCA images.

Computes Frangi vesselness filter responses and saves them as .npy files
alongside the original images for efficient loading during training.

Usage:
    python -m preprocessing.frangi_masker --input_dir ./data/xca/train --sigmas 1 3 5 7
"""

import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from skimage.filters import frangi
from skimage.io import imread, imsave

logger = logging.getLogger(__name__)

# Image extensions to process
_IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def compute_frangi_mask(
    image_path: Path,
    sigmas: Sequence[float] = (1, 3, 5, 7),
    beta: float = 0.5,
    black_ridges: bool = False,
) -> np.ndarray:
    """
    Compute Frangi vesselness for a single grayscale image.

    Args:
        image_path: path to grayscale image
        sigmas: scales for Hessian computation (matches vessel widths)
        beta: blobness sensitivity (lower = more sensitive to elongated structures)
        black_ridges: False for bright vessels on dark background (XCA default)

    Returns:
        mask: [H, W] float32, values in [0, 1]
    """
    image = imread(str(image_path))
    if image.ndim == 3:
        image = image.mean(axis=-1)  # grayscale

    image = image.astype(np.float32)
    # Normalize to [0, 1]
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)
    else:
        image = image - img_min

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = frangi(
            image,
            sigmas=list(sigmas),
            beta=beta,
            black_ridges=black_ridges,
        )

    return mask.astype(np.float32)


def precompute_dataset(
    input_dir: Path,
    sigmas: Sequence[float] = (1, 3, 5, 7),
    beta: float = 0.5,
    suffix: str = "_frangi",
    debug_visualize: bool = False,
    recursive: bool = True,
) -> None:
    """
    Iterate over all images in input_dir, compute Frangi mask, save as .npy.

    For each image {stem}.{ext}, saves {stem}{suffix}.npy alongside it.
    Skips images that already have a corresponding .npy file.

    Args:
        input_dir: directory containing images
        sigmas: Frangi filter scales
        beta: blobness sensitivity
        suffix: suffix for output .npy files
        debug_visualize: if True, save side-by-side PNG visualization
        recursive: if True, search subdirectories recursively
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Collect image files
    image_paths: list[Path] = []
    for ext in _IMAGE_EXTENSIONS:
        if recursive:
            image_paths.extend(sorted(input_dir.rglob(f"*{ext}")))
        else:
            image_paths.extend(sorted(input_dir.glob(f"*{ext}")))

    logger.info(f"Found {len(image_paths)} images in {input_dir}")

    # Determine number of distinct processes (if no output dir nesting)
    processed = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    for img_path in image_paths:
        frangi_path = img_path.with_suffix("").with_name(
            img_path.stem + suffix + ".npy"
        )

        if frangi_path.exists():
            skipped += 1
            continue

        try:
            mask = compute_frangi_mask(img_path, sigmas=sigmas, beta=beta)
            np.save(str(frangi_path), mask)
            processed += 1

            if debug_visualize:
                viz_path = frangi_path.with_suffix(".png")
                _save_visualization(img_path, mask, viz_path)

            if processed % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Processed {processed}/{len(image_paths)} images "
                    f"({elapsed:.1f}s elapsed)"
                )

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            errors += 1

    elapsed = time.time() - start_time
    logger.info(
        f"Done: {processed} processed, {skipped} skipped, {errors} errors "
        f"in {elapsed:.1f}s"
    )


def _save_visualization(
    image_path: Path,
    mask: np.ndarray,
    output_path: Path,
) -> None:
    """Save a side-by-side visualization of original image and Frangi mask."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        image = imread(str(image_path))
        if image.ndim == 3:
            image = image.mean(axis=-1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(mask, cmap="hot")
        axes[1].set_title("Frangi Vesselness")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        logger.warning("matplotlib not available; skipping visualization")
    except Exception as e:
        logger.warning(f"Visualization failed for {image_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute Frangi vesselness masks for XCA images"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing XCA images",
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[1, 3, 5, 7],
        help="Frangi filter scales",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Blobness sensitivity",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_frangi",
        help="Suffix for output .npy files",
    )
    parser.add_argument(
        "--debug_visualize",
        action="store_true",
        help="Save side-by-side visualization PNGs",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    precompute_dataset(
        input_dir=Path(args.input_dir),
        sigmas=args.sigmas,
        beta=args.beta,
        suffix=args.suffix,
        debug_visualize=args.debug_visualize,
    )


if __name__ == "__main__":
    main()
