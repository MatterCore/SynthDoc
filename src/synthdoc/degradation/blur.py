"""Blur degradation transforms — Gaussian and motion blur."""

from __future__ import annotations

import cv2
import numpy as np


def apply_gaussian_blur(image: np.ndarray, kernel_max: int) -> np.ndarray:
    """Apply Gaussian blur with a random odd kernel size up to kernel_max.

    Args:
        image: Input image as float64 array in [0, 1].
        kernel_max: Maximum kernel size (will be forced to odd).

    Returns:
        Blurred image.
    """
    # Ensure kernel is odd and at least 3
    k = max(3, kernel_max)
    if k % 2 == 0:
        k -= 1
    return cv2.GaussianBlur(image, (k, k), 0)


def apply_motion_blur(
    image: np.ndarray, kernel_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Apply motion blur in a random direction.

    Args:
        image: Input image as float64 array in [0, 1].
        kernel_size: Size of the motion blur kernel.
        rng: NumPy random generator.

    Returns:
        Motion-blurred image.
    """
    k = max(3, kernel_size)
    if k % 2 == 0:
        k -= 1

    # Create a motion blur kernel
    kernel = np.zeros((k, k), dtype=np.float64)

    # Random direction: horizontal, vertical, or diagonal
    direction = int(rng.integers(0, 4))
    if direction == 0:
        # Horizontal
        kernel[k // 2, :] = 1.0
    elif direction == 1:
        # Vertical
        kernel[:, k // 2] = 1.0
    elif direction == 2:
        # Diagonal (top-left to bottom-right)
        for i in range(k):
            kernel[i, i] = 1.0
    else:
        # Diagonal (top-right to bottom-left)
        for i in range(k):
            kernel[i, k - 1 - i] = 1.0

    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)
