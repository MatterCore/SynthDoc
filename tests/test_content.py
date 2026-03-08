"""Tests for content generators."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from synthdoc.content.figure import FigureGenerator, FigureResult
from synthdoc.content.formula import FormulaGenerator, FormulaResult
from synthdoc.content.handwriting import HandwritingGenerator, HandwritingResult
from synthdoc.content.signature import SignatureGenerator, SignatureResult
from synthdoc.content.table import TableGenerator, TableResult
from synthdoc.content.text import TextGenerator, TextResult


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestTextGenerator:
    def test_generates_image_and_text(self, rng: np.random.Generator):
        gen = TextGenerator()
        result = gen.generate(800, 400, rng)
        assert isinstance(result, TextResult)
        assert isinstance(result.image, Image.Image)
        assert len(result.text) > 0

    def test_image_dimensions(self, rng: np.random.Generator):
        gen = TextGenerator()
        result = gen.generate(600, 300, rng)
        assert result.image.size == (600, 300)

    def test_title_mode(self, rng: np.random.Generator):
        gen = TextGenerator()
        result = gen.generate(800, 100, rng, is_title=True)
        assert isinstance(result.image, Image.Image)
        assert len(result.text) > 0

    def test_different_seeds_different_text(self):
        gen = TextGenerator()
        t1 = gen.generate(400, 200, np.random.default_rng(1))
        t2 = gen.generate(400, 200, np.random.default_rng(999))
        assert t1.text != t2.text


class TestFormulaGenerator:
    def test_produces_formula_result(self, rng: np.random.Generator):
        gen = FormulaGenerator()
        result = gen.generate(400, 200, rng)
        assert isinstance(result, FormulaResult)
        assert isinstance(result.image, Image.Image)
        assert len(result.latex) > 0

    def test_correct_dimensions(self, rng: np.random.Generator):
        gen = FormulaGenerator()
        result = gen.generate(500, 150, rng)
        assert result.image.size == (500, 150)

    def test_not_blank(self, rng: np.random.Generator):
        gen = FormulaGenerator()
        result = gen.generate(600, 200, rng)
        arr = np.array(result.image)
        assert arr.std() > 1.0, "Formula image appears blank"

    @pytest.mark.parametrize("complexity", ["simple", "medium", "complex"])
    def test_complexity_levels(self, complexity: str, rng: np.random.Generator):
        gen = FormulaGenerator(complexity=complexity)
        result = gen.generate(500, 200, rng)
        assert isinstance(result.image, Image.Image)


class TestTableGenerator:
    def test_produces_table_result(self, rng: np.random.Generator):
        gen = TableGenerator()
        result = gen.generate(800, 400, rng)
        assert isinstance(result, TableResult)
        assert isinstance(result.image, Image.Image)
        assert result.rows >= 2
        assert result.cols >= 2

    def test_correct_dimensions(self, rng: np.random.Generator):
        gen = TableGenerator()
        result = gen.generate(600, 300, rng)
        assert result.image.size == (600, 300)

    def test_text_has_content(self, rng: np.random.Generator):
        gen = TableGenerator()
        result = gen.generate(800, 400, rng)
        assert len(result.text) > 0
        assert "\t" in result.text


class TestFigureGenerator:
    def test_produces_figure_result(self, rng: np.random.Generator):
        gen = FigureGenerator()
        result = gen.generate(600, 400, rng)
        assert isinstance(result, FigureResult)
        assert isinstance(result.image, Image.Image)
        assert len(result.chart_type) > 0
        assert len(result.title) > 0

    def test_correct_dimensions(self, rng: np.random.Generator):
        gen = FigureGenerator()
        result = gen.generate(500, 350, rng)
        assert result.image.size == (500, 350)

    def test_not_blank(self, rng: np.random.Generator):
        gen = FigureGenerator()
        result = gen.generate(600, 400, rng)
        arr = np.array(result.image)
        assert arr.std() > 2.0, "Figure image appears blank"


class TestHandwritingGenerator:
    def test_produces_result(self, rng: np.random.Generator):
        gen = HandwritingGenerator()
        result = gen.generate(600, 300, rng)
        assert isinstance(result, HandwritingResult)
        assert isinstance(result.image, Image.Image)
        assert len(result.text) > 0

    def test_correct_dimensions(self, rng: np.random.Generator):
        gen = HandwritingGenerator()
        result = gen.generate(500, 250, rng)
        assert result.image.size == (500, 250)

    def test_has_visible_strokes(self, rng: np.random.Generator):
        gen = HandwritingGenerator()
        result = gen.generate(600, 300, rng)
        arr = np.array(result.image)
        assert arr.std() > 1.0, "Handwriting image has no visible strokes"


class TestSignatureGenerator:
    def test_produces_result(self, rng: np.random.Generator):
        gen = SignatureGenerator()
        result = gen.generate(400, 150, rng)
        assert isinstance(result, SignatureResult)
        assert isinstance(result.image, Image.Image)

    def test_correct_dimensions(self, rng: np.random.Generator):
        gen = SignatureGenerator()
        result = gen.generate(350, 120, rng)
        assert result.image.size == (350, 120)

    def test_has_ink(self, rng: np.random.Generator):
        gen = SignatureGenerator()
        result = gen.generate(400, 150, rng)
        arr = np.array(result.image)
        assert arr.min() < 200, "Signature appears blank (no ink)"


class TestAllGeneratorsDimensions:
    @pytest.mark.parametrize("width,height", [(300, 150), (600, 400), (1000, 500)])
    def test_formula_dimensions(self, width: int, height: int, rng: np.random.Generator):
        result = FormulaGenerator().generate(width, height, rng)
        assert result.image.size == (width, height)

    @pytest.mark.parametrize("width,height", [(400, 200), (800, 400)])
    def test_table_dimensions(self, width: int, height: int, rng: np.random.Generator):
        result = TableGenerator().generate(width, height, rng)
        assert result.image.size == (width, height)

    @pytest.mark.parametrize("width,height", [(300, 200), (600, 400)])
    def test_figure_dimensions(self, width: int, height: int, rng: np.random.Generator):
        result = FigureGenerator().generate(width, height, rng)
        assert result.image.size == (width, height)
