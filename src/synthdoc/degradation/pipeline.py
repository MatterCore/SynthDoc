"""Degradation pipeline orchestrator — chains transforms on final page images."""

from __future__ import annotations

import numpy as np
from PIL import Image

from synthdoc.config import DegradationConfig
from synthdoc.degradation.blur import apply_gaussian_blur, apply_motion_blur
from synthdoc.degradation.compression import apply_jpeg_compression
from synthdoc.degradation.geometric import apply_perspective, apply_rotation, apply_scaling
from synthdoc.degradation.noise import apply_gaussian_noise, apply_salt_pepper, apply_speckle
from synthdoc.degradation.texture import apply_paper_texture


class DegradationPipeline:
    """Chains degradation transforms on document images.

    The pipeline applies transforms in a fixed order:
    1. Geometric (rotation, perspective, scaling)
    2. Noise (gaussian, salt-pepper, speckle)
    3. Blur (gaussian, motion)
    4. Texture overlay (paper)
    5. Compression (JPEG artifacts)
    """

    def __init__(self, config: DegradationConfig) -> None:
        self.config = config
        self.params = config.get_effective_params()

    def apply(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        """Apply the full degradation pipeline to an image.

        Args:
            image: Input PIL Image (RGB).
            rng: NumPy random generator for reproducibility.

        Returns:
            Degraded PIL Image.
        """
        if self.config.level == "none":
            return image

        img = image.copy()
        img_array = np.array(img, dtype=np.float64) / 255.0

        # 1. Geometric transforms
        skew_deg = float(self.params["skew_max_degrees"])
        if skew_deg > 0.1:
            img_array = apply_rotation(img_array, skew_deg, rng)
            if rng.random() > 0.5:
                img_array = apply_perspective(img_array, skew_deg * 0.3, rng)
            if rng.random() > 0.7:
                img_array = apply_scaling(img_array, rng)

        # 2. Noise
        noise_sigma = float(self.params["noise_sigma"])
        if noise_sigma > 0.001:
            noise_choice = int(rng.integers(0, 3))
            if noise_choice == 0:
                img_array = apply_gaussian_noise(img_array, noise_sigma, rng)
            elif noise_choice == 1:
                img_array = apply_salt_pepper(img_array, noise_sigma * 0.5, rng)
            else:
                img_array = apply_speckle(img_array, noise_sigma, rng)

        # 3. Blur
        blur_max = int(self.params["blur_kernel_max"])
        if blur_max >= 3:
            if rng.random() > 0.5:
                img_array = apply_gaussian_blur(img_array, blur_max)
            else:
                img_array = apply_motion_blur(img_array, blur_max, rng)

        # 4. Paper texture
        if self.params["paper_texture"]:
            img_array = apply_paper_texture(img_array, rng)

        # 5. JPEG compression
        jpeg_quality = int(self.params["jpeg_quality_min"])
        if jpeg_quality < 95:
            # Convert back to PIL for JPEG compression
            img_uint8 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_uint8)
            img_pil = apply_jpeg_compression(img_pil, jpeg_quality, rng)
            return img_pil

        # Convert back to PIL
        img_uint8 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_uint8)
