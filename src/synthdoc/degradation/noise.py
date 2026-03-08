"""Noise degradation transforms — Gaussian, salt-and-pepper, and speckle noise."""

from __future__ import annotations

import numpy as np


def apply_gaussian_noise(
    image: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Add Gaussian noise to an image.

    Args:
        image: Input image as float64 array in [0, 1].
        sigma: Standard deviation of the noise.
        rng: NumPy random generator.

    Returns:
        Noisy image clipped to [0, 1].
    """
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def apply_salt_pepper(
    image: np.ndarray, density: float, rng: np.random.Generator
) -> np.ndarray:
    """Add salt-and-pepper noise.

    Args:
        image: Input image as float64 array in [0, 1].
        density: Fraction of pixels to corrupt (0-1).
        rng: NumPy random generator.

    Returns:
        Noisy image.
    """
    result = image.copy()
    num_pixels = image.shape[0] * image.shape[1]
    density = min(density, 0.1)  # Cap to avoid destroying the image

    # Salt (white pixels)
    num_salt = int(num_pixels * density / 2)
    salt_y = rng.integers(0, image.shape[0], size=num_salt)
    salt_x = rng.integers(0, image.shape[1], size=num_salt)
    result[salt_y, salt_x] = 1.0

    # Pepper (black pixels)
    num_pepper = int(num_pixels * density / 2)
    pepper_y = rng.integers(0, image.shape[0], size=num_pepper)
    pepper_x = rng.integers(0, image.shape[1], size=num_pepper)
    result[pepper_y, pepper_x] = 0.0

    return result


def apply_speckle(
    image: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Add multiplicative speckle noise.

    Speckle: output = image + image * noise

    Args:
        image: Input image as float64 array in [0, 1].
        sigma: Standard deviation of the noise multiplier.
        rng: NumPy random generator.

    Returns:
        Noisy image clipped to [0, 1].
    """
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image + image * noise, 0.0, 1.0)
