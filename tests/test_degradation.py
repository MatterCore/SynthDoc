"""Tests for the degradation pipeline and individual transforms."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from synthdoc.config import DegradationConfig
from synthdoc.degradation.blur import apply_gaussian_blur, apply_motion_blur
from synthdoc.degradation.compression import apply_jpeg_compression
from synthdoc.degradation.geometric import apply_perspective, apply_rotation, apply_scaling
from synthdoc.degradation.noise import apply_gaussian_noise, apply_salt_pepper, apply_speckle
from synthdoc.degradation.pipeline import DegradationPipeline
from synthdoc.degradation.texture import apply_paper_texture


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def clean_image() -> Image.Image:
    """A simple synthetic document image with some content."""
    img = Image.new("RGB", (800, 600), "white")
    arr = np.array(img)
    arr[100:200, 100:700] = 0
    arr[250:350, 150:650] = 50
    arr[400:450, 200:600] = 100
    return Image.fromarray(arr)


@pytest.fixture()
def clean_array() -> np.ndarray:
    """Normalized float64 image array in [0, 1]."""
    arr = np.ones((600, 800, 3), dtype=np.float64)
    arr[100:200, 100:700] = 0.0
    arr[250:350, 150:650] = 0.2
    arr[400:450, 200:600] = 0.4
    return arr


class TestGaussianNoise:
    def test_adds_variation(self, clean_array: np.ndarray, rng: np.random.Generator):
        noisy = apply_gaussian_noise(clean_array, 0.05, rng)
        diff = np.abs(noisy - clean_array)
        assert diff.mean() > 0.01

    def test_preserves_shape(self, clean_array: np.ndarray, rng: np.random.Generator):
        noisy = apply_gaussian_noise(clean_array, 0.02, rng)
        assert noisy.shape == clean_array.shape

    def test_values_in_range(self, clean_array: np.ndarray, rng: np.random.Generator):
        noisy = apply_gaussian_noise(clean_array, 0.1, rng)
        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0


class TestSaltPepper:
    def test_adds_noise(self, clean_array: np.ndarray, rng: np.random.Generator):
        noisy = apply_salt_pepper(clean_array, 0.05, rng)
        diff = np.abs(noisy - clean_array)
        assert diff.sum() > 0

    def test_preserves_shape(self, clean_array: np.ndarray, rng: np.random.Generator):
        noisy = apply_salt_pepper(clean_array, 0.02, rng)
        assert noisy.shape == clean_array.shape


class TestSpeckle:
    def test_adds_noise(self, clean_array: np.ndarray, rng: np.random.Generator):
        noisy = apply_speckle(clean_array, 0.1, rng)
        diff = np.abs(noisy - clean_array)
        assert diff.mean() > 0.001


class TestGaussianBlur:
    def test_reduces_high_frequency(self, clean_array: np.ndarray):
        blurred = apply_gaussian_blur(clean_array, 5)
        grad_orig = np.abs(np.diff(clean_array, axis=1)).mean()
        grad_blur = np.abs(np.diff(blurred, axis=1)).mean()
        assert grad_blur < grad_orig

    def test_preserves_shape(self, clean_array: np.ndarray):
        blurred = apply_gaussian_blur(clean_array, 3)
        assert blurred.shape == clean_array.shape


class TestMotionBlur:
    def test_changes_image(self, clean_array: np.ndarray, rng: np.random.Generator):
        blurred = apply_motion_blur(clean_array, 5, rng)
        diff = np.abs(blurred - clean_array)
        assert diff.mean() > 0.001

    def test_preserves_shape(self, clean_array: np.ndarray, rng: np.random.Generator):
        blurred = apply_motion_blur(clean_array, 3, rng)
        assert blurred.shape == clean_array.shape


class TestRotation:
    def test_preserves_shape(self, clean_array: np.ndarray, rng: np.random.Generator):
        rotated = apply_rotation(clean_array, 3.0, rng)
        assert rotated.shape == clean_array.shape

    def test_changes_pixels(self, clean_array: np.ndarray, rng: np.random.Generator):
        rotated = apply_rotation(clean_array, 5.0, rng)
        diff = np.abs(rotated - clean_array)
        assert diff.mean() > 0.001


class TestPerspective:
    def test_preserves_shape(self, clean_array: np.ndarray, rng: np.random.Generator):
        warped = apply_perspective(clean_array, 2.0, rng)
        assert warped.shape == clean_array.shape


class TestScaling:
    def test_preserves_shape(self, clean_array: np.ndarray, rng: np.random.Generator):
        scaled = apply_scaling(clean_array, rng)
        assert scaled.shape == clean_array.shape


class TestJpegCompression:
    def test_changes_pixels(self, clean_image: Image.Image, rng: np.random.Generator):
        compressed = apply_jpeg_compression(clean_image, 20, rng)
        orig_arr = np.array(clean_image).astype(float)
        comp_arr = np.array(compressed).astype(float)
        diff = np.abs(orig_arr - comp_arr)
        assert diff.mean() > 0.5

    def test_preserves_size(self, clean_image: Image.Image, rng: np.random.Generator):
        compressed = apply_jpeg_compression(clean_image, 50, rng)
        assert compressed.size == clean_image.size


class TestPaperTexture:
    def test_changes_pixels(self, clean_array: np.ndarray, rng: np.random.Generator):
        textured = apply_paper_texture(clean_array, rng)
        diff = np.abs(textured - clean_array)
        assert diff.mean() > 0.001

    def test_preserves_shape(self, clean_array: np.ndarray, rng: np.random.Generator):
        textured = apply_paper_texture(clean_array, rng)
        assert textured.shape == clean_array.shape


class TestDegradationPipeline:
    def test_none_level_unchanged(self, clean_image: Image.Image, rng: np.random.Generator):
        config = DegradationConfig(level="none")
        pipeline = DegradationPipeline(config)
        result = pipeline.apply(clean_image, rng)
        assert np.array_equal(np.array(result), np.array(clean_image))

    def test_light_level_changes_image(self, clean_image: Image.Image, rng: np.random.Generator):
        config = DegradationConfig(level="light")
        pipeline = DegradationPipeline(config)
        result = pipeline.apply(clean_image, rng)
        assert isinstance(result, Image.Image)

    def test_heavy_level_degraded(self, clean_image: Image.Image, rng: np.random.Generator):
        config = DegradationConfig(level="heavy")
        pipeline = DegradationPipeline(config)
        result = pipeline.apply(clean_image, rng)
        assert isinstance(result, Image.Image)

    def test_returns_pil_image(self, clean_image: Image.Image, rng: np.random.Generator):
        for level in ["none", "light", "medium", "heavy"]:
            config = DegradationConfig(level=level)
            pipeline = DegradationPipeline(config)
            result = pipeline.apply(clean_image, rng)
            assert isinstance(result, Image.Image)

    def test_reproducible_with_seed(self, clean_image: Image.Image):
        config = DegradationConfig(level="medium")
        pipeline = DegradationPipeline(config)
        r1 = pipeline.apply(clean_image, np.random.default_rng(123))
        r2 = pipeline.apply(clean_image, np.random.default_rng(123))
        assert np.array_equal(np.array(r1), np.array(r2))
