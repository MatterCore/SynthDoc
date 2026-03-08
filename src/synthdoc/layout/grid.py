"""CSS grid generation utilities for HTML rendering."""

from __future__ import annotations

from synthdoc.layout.templates import RegionSlot


def generate_grid_css(
    slots: list[RegionSlot],
    page_width: int,
    page_height: int,
) -> str:
    """Generate CSS for absolutely positioned content regions.

    Each region slot becomes an absolutely positioned div inside a
    relative container matching the page dimensions.

    Args:
        slots: Region slots with bounding boxes.
        page_width: Page width in pixels.
        page_height: Page height in pixels.

    Returns:
        CSS string defining the page container and region positions.
    """
    css_parts = [
        f".page-container {{\n"
        f"  position: relative;\n"
        f"  width: {page_width}px;\n"
        f"  height: {page_height}px;\n"
        f"  background: white;\n"
        f"  overflow: hidden;\n"
        f"  margin: 0;\n"
        f"  padding: 0;\n"
        f"}}\n"
    ]

    for i, slot in enumerate(slots):
        x1, y1, x2, y2 = slot.bbox
        w = x2 - x1
        h = y2 - y1
        css_parts.append(
            f".region-{i} {{\n"
            f"  position: absolute;\n"
            f"  left: {x1}px;\n"
            f"  top: {y1}px;\n"
            f"  width: {w}px;\n"
            f"  height: {h}px;\n"
            f"  overflow: hidden;\n"
            f"}}\n"
        )

    return "\n".join(css_parts)


def generate_grid_html_divs(slots: list[RegionSlot]) -> list[str]:
    """Generate HTML div elements for each region slot.

    Args:
        slots: Region slots to generate divs for.

    Returns:
        List of HTML div strings, one per slot.
    """
    divs = []
    for i, slot in enumerate(slots):
        divs.append(
            f'<div class="region-{i}" data-type="{slot.region_type}" '
            f'data-order="{slot.reading_order}">{{{{ region_{i}_content }}}}</div>'
        )
    return divs
