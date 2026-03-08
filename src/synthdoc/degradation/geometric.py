"""Geometric degradation transforms — rotation, perspective, scaling."""

from __future__ import annotations

import cv2
import numpy as np


def apply_rotation(
    image: np.ndarray, max_degrees: float, rng: np.random.Generator
) -> np.ndarray:
    """Apply random rotation within [-max_degrees, +max_degrees].

    Args:
        image: Input image as float64 array in [0, 1] with shape (H, W, C).
        max_degrees: Maximum rotation angle in degrees.
        rng: NumPy random generator.

    Returns:
        Rotated image (same size, with white fill for borders).
    """
    angle = float(rng.uniform(-max_degrees, max_degrees))
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1.0, 1.0, 1.0),
    )
    return rotated


def apply_perspective(
    image: np.ndarray, strength: float, rng: np.random.Generator
) -> np.ndarray:
    """Apply a random perspective transform.

    Args:
        image: Input image as float64 array in [0, 1].
        strength: Maximum pixel displacement for corners (in degrees equivalent).
        rng: NumPy random generator.

    Returns:
        Perspective-transformed image.
    """
    h, w = image.shape[:2]
    # Convert strength from degrees-like to pixel displacement
    max_disp = max(1, int(strength * w / 100))

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = src_pts.copy()

    for i in range(4):
        dst_pts[i][0] += rng.uniform(-max_disp, max_disp)
        dst_pts[i][1] += rng.uniform(-max_disp, max_disp)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1.0, 1.0, 1.0),
    )
    return warped


def apply_scaling(
    image: np.ndarray, rng: np.random.Generator, scale_range: tuple[float, float] = (0.95, 1.05)
) -> np.ndarray:
    """Apply random scaling and re-center to original dimensions.

    Args:
        image: Input image as float64 array in [0, 1].
        rng: NumPy random generator.
        scale_range: Min and max scale factors.

    Returns:
        Scaled image at original dimensions.
    """
    scale = float(rng.uniform(scale_range[0], scale_range[1]))
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create output canvas (white)
    output = np.ones_like(image)

    # Center the resized image
    x_off = (w - new_w) // 2
    y_off = (h - new_h) // 2

    # Compute overlap regions
    src_x_start = max(0, -x_off)
    src_y_start = max(0, -y_off)
    dst_x_start = max(0, x_off)
    dst_y_start = max(0, y_off)

    copy_w = min(new_w - src_x_start, w - dst_x_start)
    copy_h = min(new_h - src_y_start, h - dst_y_start)

    if copy_w > 0 and copy_h > 0:
        output[dst_y_start:dst_y_start + copy_h, dst_x_start:dst_x_start + copy_w] = \
            resized[src_y_start:src_y_start + copy_h, src_x_start:src_x_start + copy_w]

    return output
