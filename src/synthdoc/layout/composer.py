"""Layout composition engine — selects a template and generates region slots."""

from __future__ import annotations

import numpy as np

from synthdoc.config import LayoutConfig, PageConfig
from synthdoc.layout.templates import TEMPLATE_FUNCTIONS, RegionSlot


class LayoutComposer:
    """Composes a page layout by selecting a template and generating region slots."""

    def __init__(self, layout_config: LayoutConfig, page_config: PageConfig) -> None:
        self.layout_config = layout_config
        self.page_config = page_config

    def compose(self, rng: np.random.Generator) -> list[RegionSlot]:
        """Generate region slots for a single page.

        Args:
            rng: NumPy random generator for reproducibility.

        Returns:
            List of RegionSlot instances describing content placement.
        """
        template_name = self.layout_config.template
        if template_name not in TEMPLATE_FUNCTIONS:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Available: {list(TEMPLATE_FUNCTIONS.keys())}"
            )

        template_fn = TEMPLATE_FUNCTIONS[template_name]
        slots = template_fn(
            width=self.page_config.width_px,
            height=self.page_config.height_px,
            margin_top=self.page_config.margin_top,
            margin_bottom=self.page_config.margin_bottom,
            margin_left=self.page_config.margin_left,
            margin_right=self.page_config.margin_right,
            rng=rng,
            columns=self.layout_config.columns,
        )

        # Validate all bboxes are within page bounds
        for slot in slots:
            x1, y1, x2, y2 = slot.bbox
            assert 0 <= x1 < x2 <= self.page_config.width_px, f"Invalid x bounds: {slot.bbox}"
            assert 0 <= y1 < y2 <= self.page_config.height_px, f"Invalid y bounds: {slot.bbox}"

        return slots
