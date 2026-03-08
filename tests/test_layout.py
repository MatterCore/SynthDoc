"""Tests for the layout composer and templates."""

from __future__ import annotations

import numpy as np
import pytest

from synthdoc.config import LayoutConfig, PageConfig
from synthdoc.layout.composer import LayoutComposer
from synthdoc.layout.templates import (
    TEMPLATE_FUNCTIONS,
    RegionSlot,
    academic_template,
    legal_template,
    notebook_template,
)


@pytest.fixture()
def page_config() -> PageConfig:
    return PageConfig()


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestAcademicTemplate:
    def test_returns_slots(self, page_config: PageConfig, rng: np.random.Generator):
        slots = academic_template(
            page_config.width_px, page_config.height_px,
            page_config.margin_top, page_config.margin_bottom,
            page_config.margin_left, page_config.margin_right, rng,
        )
        assert len(slots) >= 4

    def test_has_title(self, page_config: PageConfig, rng: np.random.Generator):
        slots = academic_template(
            page_config.width_px, page_config.height_px,
            page_config.margin_top, page_config.margin_bottom,
            page_config.margin_left, page_config.margin_right, rng,
        )
        types = {s.region_type for s in slots}
        assert "title" in types

    def test_has_body(self, page_config: PageConfig, rng: np.random.Generator):
        slots = academic_template(
            page_config.width_px, page_config.height_px,
            page_config.margin_top, page_config.margin_bottom,
            page_config.margin_left, page_config.margin_right, rng,
        )
        types = {s.region_type for s in slots}
        assert "body" in types


class TestLegalTemplate:
    def test_has_signature(self, page_config: PageConfig, rng: np.random.Generator):
        slots = legal_template(
            page_config.width_px, page_config.height_px,
            page_config.margin_top, page_config.margin_bottom,
            page_config.margin_left, page_config.margin_right, rng,
        )
        types = {s.region_type for s in slots}
        assert "signature" in types

    def test_has_header(self, page_config: PageConfig, rng: np.random.Generator):
        slots = legal_template(
            page_config.width_px, page_config.height_px,
            page_config.margin_top, page_config.margin_bottom,
            page_config.margin_left, page_config.margin_right, rng,
        )
        types = {s.region_type for s in slots}
        assert "header" in types


class TestNotebookTemplate:
    def test_has_handwriting(self, page_config: PageConfig, rng: np.random.Generator):
        slots = notebook_template(
            page_config.width_px, page_config.height_px,
            page_config.margin_top, page_config.margin_bottom,
            page_config.margin_left, page_config.margin_right, rng,
        )
        types = {s.region_type for s in slots}
        assert "handwriting" in types


class TestLayoutComposer:
    @pytest.mark.parametrize("template_name", list(TEMPLATE_FUNCTIONS.keys()))
    def test_compose_returns_slots(self, template_name: str, page_config: PageConfig, rng: np.random.Generator):
        config = LayoutConfig(template=template_name)
        composer = LayoutComposer(config, page_config)
        slots = composer.compose(rng)
        assert isinstance(slots, list)
        assert len(slots) >= 1
        assert all(isinstance(s, RegionSlot) for s in slots)

    @pytest.mark.parametrize("template_name", list(TEMPLATE_FUNCTIONS.keys()))
    def test_bboxes_within_page(self, template_name: str, page_config: PageConfig, rng: np.random.Generator):
        config = LayoutConfig(template=template_name)
        composer = LayoutComposer(config, page_config)
        slots = composer.compose(rng)
        for slot in slots:
            x1, y1, x2, y2 = slot.bbox
            assert 0 <= x1 < x2 <= page_config.width_px
            assert 0 <= y1 < y2 <= page_config.height_px

    @pytest.mark.parametrize("template_name", list(TEMPLATE_FUNCTIONS.keys()))
    def test_reading_order_unique(self, template_name: str, page_config: PageConfig, rng: np.random.Generator):
        config = LayoutConfig(template=template_name)
        composer = LayoutComposer(config, page_config)
        slots = composer.compose(rng)
        orders = [s.reading_order for s in slots]
        assert len(orders) == len(set(orders)), "Duplicate reading orders found"

    def test_unknown_template_raises(self, page_config: PageConfig, rng: np.random.Generator):
        config = LayoutConfig(template="nonexistent")
        composer = LayoutComposer(config, page_config)
        with pytest.raises(ValueError, match="Unknown template"):
            composer.compose(rng)

    def test_column_override(self, page_config: PageConfig, rng: np.random.Generator):
        config = LayoutConfig(template="academic", columns=3)
        composer = LayoutComposer(config, page_config)
        slots = composer.compose(rng)
        columns_used = {s.column for s in slots if s.column > 0}
        assert len(columns_used) >= 2


class TestRegionSlot:
    def test_width_height(self):
        slot = RegionSlot(region_type="body", bbox=(10, 20, 110, 220), reading_order=0)
        assert slot.width == 100
        assert slot.height == 200
