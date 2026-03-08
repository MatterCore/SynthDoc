"""Tests for the SynthDoc generation engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from synthdoc.config import DegradationConfig, GenerationConfig, LayoutConfig, PageConfig
from synthdoc.engine import GeneratedPage, SynthDocEngine


@pytest.fixture()
def tmp_output(tmp_path: Path) -> Path:
    out = tmp_path / "synthdoc_test_output"
    out.mkdir()
    return out


@pytest.fixture()
def base_config(tmp_output: Path) -> GenerationConfig:
    return GenerationConfig(
        page=PageConfig(width_px=800, height_px=1000, dpi=100),
        layout=LayoutConfig(template="academic"),
        degradation=DegradationConfig(level="none"),
        count=2,
        output_dir=str(tmp_output),
        seed=42,
    )


class TestSynthDocEngine:
    def test_generate_page_returns_generated_page(self, base_config: GenerationConfig):
        engine = SynthDocEngine(base_config)
        page = engine.generate_page(0)
        assert isinstance(page, GeneratedPage)
        assert isinstance(page.image, Image.Image)
        assert page.page_num == 0

    def test_generate_page_correct_dimensions(self, base_config: GenerationConfig):
        engine = SynthDocEngine(base_config)
        page = engine.generate_page(0)
        assert page.image.size == (base_config.page.width_px, base_config.page.height_px)

    def test_generate_page_has_annotation(self, base_config: GenerationConfig):
        engine = SynthDocEngine(base_config)
        page = engine.generate_page(0)
        assert page.annotation is not None
        assert len(page.annotation.regions) >= 1

    def test_generate_creates_correct_count(self, base_config: GenerationConfig):
        engine = SynthDocEngine(base_config)
        pages = engine.generate()
        assert len(pages) == base_config.count

    def test_generate_saves_images(self, base_config: GenerationConfig):
        engine = SynthDocEngine(base_config)
        pages = engine.generate()
        for page in pages:
            assert page.image_path is not None
            assert page.image_path.exists()

    def test_generate_saves_annotations(self, base_config: GenerationConfig):
        engine = SynthDocEngine(base_config)
        engine.generate()
        ann_file = Path(base_config.output_dir) / "annotations.json"
        assert ann_file.exists()

    def test_seed_produces_reproducible_output(self, tmp_path: Path):
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"

        for out in (out1, out2):
            config = GenerationConfig(
                page=PageConfig(width_px=400, height_px=500, dpi=72),
                layout=LayoutConfig(template="academic"),
                degradation=DegradationConfig(level="none"),
                count=1,
                output_dir=str(out),
                seed=12345,
            )
            SynthDocEngine(config).generate()

        img1 = np.array(Image.open(out1 / "images" / "page_00000.png"))
        img2 = np.array(Image.open(out2 / "images" / "page_00000.png"))
        assert np.array_equal(img1, img2)

    def test_page_not_blank(self, base_config: GenerationConfig):
        engine = SynthDocEngine(base_config)
        page = engine.generate_page(0)
        arr = np.array(page.image)
        assert arr.std() > 1.0, "Page appears blank"

    @pytest.mark.parametrize("template", ["academic", "legal", "form", "report", "notebook"])
    def test_different_templates(self, template: str, tmp_path: Path):
        config = GenerationConfig(
            page=PageConfig(width_px=600, height_px=800, dpi=72),
            layout=LayoutConfig(template=template),
            degradation=DegradationConfig(level="none"),
            count=1,
            output_dir=str(tmp_path / template),
            seed=42,
        )
        engine = SynthDocEngine(config)
        page = engine.generate_page(0)
        assert isinstance(page.image, Image.Image)
        assert page.image.size == (600, 800)

    def test_html_output(self, base_config: GenerationConfig):
        engine = SynthDocEngine(base_config)
        page = engine.generate_page(0)
        assert len(page.html) > 0
        assert "<html>" in page.html
