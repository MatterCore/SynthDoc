"""Paper texture overlay — generates realistic paper background."""

from __future__ import annotations

import numpy as np


def _generate_perlin_octave(
    height: int, width: int, frequency: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate a single octave of Perlin-like noise using interpolated random grid.

    Args:
        height: Output height.
        width: Output width.
        frequency: Grid frequency (higher = more detail).
        rng: NumPy random generator.

    Returns:
        Noise array in [0, 1] with shape (height, width).
    """
    grid_h = max(2, int(height * frequency) + 2)
    grid_w = max(2, int(width * frequency) + 2)

    # Random values at grid points
    grid = rng.uniform(0, 1, (grid_h, grid_w))

    # Interpolation coordinates
    y_coords = np.linspace(0, grid_h - 1, height)
    x_coords = np.linspace(0, grid_w - 1, width)

    # Bilinear interpolation
    y_idx = np.clip(y_coords.astype(int), 0, grid_h - 2)
    x_idx = np.clip(x_coords.astype(int), 0, grid_w - 2)
    y_frac = y_coords - y_idx
    x_frac = x_coords - x_idx

    # Create meshgrid for vectorized interpolation
    yi, xi = np.meshgrid(y_idx, x_idx, indexing="ij")
    yf, xf = np.meshgrid(y_frac, x_frac, indexing="ij")

    top_left = grid[yi, xi]
    top_right = grid[yi, xi + 1]
    bottom_left = grid[yi + 1, xi]
    bottom_right = grid[yi + 1, xi + 1]

    top = top_left * (1 - xf) + top_right * xf
    bottom = bottom_left * (1 - xf) + bottom_right * xf
    result = top * (1 - yf) + bottom * yf

    return result


def generate_paper_texture(
    height: int, width: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate a paper-like texture.

    Combines multiple octaves of noise with yellowish tint.

    Args:
        height: Texture height.
        width: Texture width.
        rng: NumPy random generator.

    Returns:
        Texture array of shape (height, width, 3) in [0, 1].
    """
    # Multi-octave noise for paper grain
    texture = np.zeros((height, width), dtype=np.float64)
    for octave, (freq, amp) in enumerate([(0.005, 0.5), (0.01, 0.3), (0.03, 0.2)]):
        texture += _generate_perlin_octave(height, width, freq, rng) * amp

    # Normalize to [0, 1]
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)

    # Paper base color (slightly warm/yellowish)
    base_r = 0.97
    base_g = 0.95
    base_b = 0.90

    # Create RGB texture
    rgb = np.zeros((height, width, 3), dtype=np.float64)
    # Modulate the base color by the noise texture
    variation = texture * 0.06  # Subtle variation
    rgb[:, :, 0] = base_r - variation
    rgb[:, :, 1] = base_g - variation
    rgb[:, :, 2] = base_b - variation * 1.2  # Slightly more variation in blue channel

    return np.clip(rgb, 0, 1)


def apply_paper_texture(
    image: np.ndarray, rng: np.random.Generator, alpha: float = 0.15
) -> np.ndarray:
    """Overlay paper texture on a document image.

    Args:
        image: Input image as float64 array in [0, 1] with shape (H, W, 3).
        rng: NumPy random generator.
        alpha: Blending strength (0 = no texture, 1 = full texture).

    Returns:
        Image with paper texture overlay.
    """
    h, w = image.shape[:2]
    texture = generate_paper_texture(h, w, rng)

    # Alpha blend: only apply texture where the image is white-ish (paper areas)
    # This preserves ink/content while texturing the background
    brightness = image.mean(axis=2, keepdims=True)
    # Weight: higher for bright areas (paper), lower for dark areas (ink)
    weight = np.clip(brightness - 0.3, 0, 1) / 0.7
    effective_alpha = alpha * weight

    blended = image * (1 - effective_alpha) + texture * effective_alpha
    return np.clip(blended, 0, 1)
