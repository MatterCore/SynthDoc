"""Built-in layout templates for document generation.

Each template function takes page dimensions and returns a list of RegionSlots
describing where content should be placed on the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.random import Generator


@dataclass
class RegionSlot:
    """A rectangular region on the page that will hold content."""

    region_type: str  # body, formula, table, figure, header, footer, title, handwriting, signature
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    reading_order: int
    column: int = 0
    optional: bool = False

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


def _split_column_body(
    x_start: int,
    y_start: int,
    col_width: int,
    col_height: int,
    rng: Generator,
    column_idx: int,
    order_offset: int,
    content_types: Sequence[str],
) -> list[RegionSlot]:
    """Split a column area into multiple body/content regions stacked vertically."""
    slots: list[RegionSlot] = []
    y = y_start
    order = order_offset
    remaining = col_height
    min_block = 200

    while remaining > min_block:
        block_h = int(rng.integers(min_block, min(remaining, max(min_block + 1, remaining // 2 + 1))))
        if remaining - block_h < min_block:
            block_h = remaining

        rtype = str(rng.choice(content_types))
        slots.append(
            RegionSlot(
                region_type=rtype,
                bbox=(x_start, y, x_start + col_width, y + block_h),
                reading_order=order,
                column=column_idx,
            )
        )
        y += block_h
        remaining -= block_h
        order += 1

    return slots


def academic_template(
    width: int,
    height: int,
    margin_top: int,
    margin_bottom: int,
    margin_left: int,
    margin_right: int,
    rng: Generator,
    columns: int | None = None,
) -> list[RegionSlot]:
    """Academic paper layout: title, abstract, 2-column body with formulas and figures."""
    slots: list[RegionSlot] = []
    content_w = width - margin_left - margin_right
    order = 0

    # Title
    title_h = 120
    slots.append(
        RegionSlot(
            region_type="title",
            bbox=(margin_left, margin_top, margin_left + content_w, margin_top + title_h),
            reading_order=order,
        )
    )
    order += 1

    # Abstract (full width)
    abstract_y = margin_top + title_h + 30
    abstract_h = int(rng.integers(200, 350))
    slots.append(
        RegionSlot(
            region_type="body",
            bbox=(margin_left, abstract_y, margin_left + content_w, abstract_y + abstract_h),
            reading_order=order,
        )
    )
    order += 1

    # Two-column body
    num_cols = columns if columns is not None else 2
    gap = 60
    col_w = (content_w - gap * (num_cols - 1)) // num_cols
    body_y = abstract_y + abstract_h + 40
    body_h = height - margin_bottom - body_y

    content_types = ["body", "body", "body", "formula", "figure"]

    for c in range(num_cols):
        x = margin_left + c * (col_w + gap)
        col_slots = _split_column_body(x, body_y, col_w, body_h, rng, c, order, content_types)
        slots.extend(col_slots)
        order += len(col_slots)

    return slots


def legal_template(
    width: int,
    height: int,
    margin_top: int,
    margin_bottom: int,
    margin_left: int,
    margin_right: int,
    rng: Generator,
    columns: int | None = None,
) -> list[RegionSlot]:
    """Legal document: single column, numbered clauses, signature block at bottom."""
    slots: list[RegionSlot] = []
    content_w = width - margin_left - margin_right
    order = 0

    # Header
    header_h = 100
    slots.append(
        RegionSlot(
            region_type="header",
            bbox=(margin_left, margin_top, margin_left + content_w, margin_top + header_h),
            reading_order=order,
        )
    )
    order += 1

    # Title
    title_y = margin_top + header_h + 20
    title_h = 80
    slots.append(
        RegionSlot(
            region_type="title",
            bbox=(margin_left, title_y, margin_left + content_w, title_y + title_h),
            reading_order=order,
        )
    )
    order += 1

    # Body clauses
    body_y = title_y + title_h + 30
    sig_block_h = 250
    footer_h = 60
    available_h = height - margin_bottom - body_y - sig_block_h - footer_h - 40

    col_slots = _split_column_body(
        margin_left, body_y, content_w, available_h, rng, 0, order, ["body", "body", "body", "table"]
    )
    slots.extend(col_slots)
    order += len(col_slots)

    # Signature block
    sig_y = body_y + available_h + 20
    sig_w = content_w // 3
    slots.append(
        RegionSlot(
            region_type="signature",
            bbox=(margin_left + content_w - sig_w, sig_y, margin_left + content_w, sig_y + sig_block_h),
            reading_order=order,
            optional=True,
        )
    )
    order += 1

    # Footer
    footer_y = height - margin_bottom - footer_h
    slots.append(
        RegionSlot(
            region_type="footer",
            bbox=(margin_left, footer_y, margin_left + content_w, footer_y + footer_h),
            reading_order=order,
        )
    )

    return slots


def notebook_template(
    width: int,
    height: int,
    margin_top: int,
    margin_bottom: int,
    margin_left: int,
    margin_right: int,
    rng: Generator,
    columns: int | None = None,
) -> list[RegionSlot]:
    """Notebook layout: handwritten notes, margin annotations, scattered content."""
    slots: list[RegionSlot] = []
    order = 0

    # Main area (wider margin on left for annotations)
    note_margin_left = margin_left + 200
    content_w = width - margin_right - note_margin_left

    # Margin annotation column
    margin_h = height - margin_top - margin_bottom
    num_margin_notes = int(rng.integers(1, 4))
    margin_note_h = 150
    for i in range(num_margin_notes):
        y = margin_top + int(rng.integers(0, max(1, margin_h - margin_note_h)))
        slots.append(
            RegionSlot(
                region_type="handwriting",
                bbox=(margin_left, y, note_margin_left - 20, y + margin_note_h),
                reading_order=order,
                optional=True,
            )
        )
        order += 1

    # Main handwriting blocks
    y = margin_top
    available_h = height - margin_top - margin_bottom
    content_types = ["handwriting", "handwriting", "handwriting", "formula", "figure"]

    col_slots = _split_column_body(
        note_margin_left, y, content_w, available_h, rng, 0, order, content_types
    )
    slots.extend(col_slots)

    return slots


def form_template(
    width: int,
    height: int,
    margin_top: int,
    margin_bottom: int,
    margin_left: int,
    margin_right: int,
    rng: Generator,
    columns: int | None = None,
) -> list[RegionSlot]:
    """Form layout: fields, tables, headers."""
    slots: list[RegionSlot] = []
    content_w = width - margin_left - margin_right
    order = 0

    # Header with logo area
    header_h = 130
    slots.append(
        RegionSlot(
            region_type="header",
            bbox=(margin_left, margin_top, margin_left + content_w, margin_top + header_h),
            reading_order=order,
        )
    )
    order += 1

    # Title
    title_y = margin_top + header_h + 20
    title_h = 70
    slots.append(
        RegionSlot(
            region_type="title",
            bbox=(margin_left, title_y, margin_left + content_w, title_y + title_h),
            reading_order=order,
        )
    )
    order += 1

    # Form body — alternating body text and tables
    y = title_y + title_h + 30
    bottom_limit = height - margin_bottom - 200
    block_idx = 0
    while y < bottom_limit:
        if block_idx % 3 == 2:
            rtype = "table"
            h = int(rng.integers(200, 400))
        else:
            rtype = "body"
            h = int(rng.integers(100, 250))

        if y + h > bottom_limit:
            h = bottom_limit - y
            if h < 80:
                break

        slots.append(
            RegionSlot(
                region_type=rtype,
                bbox=(margin_left, y, margin_left + content_w, y + h),
                reading_order=order,
                column=0,
            )
        )
        order += 1
        y += h + 20
        block_idx += 1

    # Signature at bottom
    sig_y = height - margin_bottom - 160
    sig_w = content_w // 3
    slots.append(
        RegionSlot(
            region_type="signature",
            bbox=(margin_left + content_w - sig_w, sig_y, margin_left + content_w, sig_y + 120),
            reading_order=order,
            optional=True,
        )
    )

    return slots


def report_template(
    width: int,
    height: int,
    margin_top: int,
    margin_bottom: int,
    margin_left: int,
    margin_right: int,
    rng: Generator,
    columns: int | None = None,
) -> list[RegionSlot]:
    """Business report: header, charts, tables, body text."""
    slots: list[RegionSlot] = []
    content_w = width - margin_left - margin_right
    order = 0

    # Header
    header_h = 100
    slots.append(
        RegionSlot(
            region_type="header",
            bbox=(margin_left, margin_top, margin_left + content_w, margin_top + header_h),
            reading_order=order,
        )
    )
    order += 1

    # Title
    title_y = margin_top + header_h + 20
    title_h = 90
    slots.append(
        RegionSlot(
            region_type="title",
            bbox=(margin_left, title_y, margin_left + content_w, title_y + title_h),
            reading_order=order,
        )
    )
    order += 1

    # Executive summary (body text)
    body_y = title_y + title_h + 30
    body_h = int(rng.integers(250, 400))
    slots.append(
        RegionSlot(
            region_type="body",
            bbox=(margin_left, body_y, margin_left + content_w, body_y + body_h),
            reading_order=order,
        )
    )
    order += 1

    # Chart area — side by side
    chart_y = body_y + body_h + 30
    chart_h = int(rng.integers(350, 500))
    half_w = (content_w - 40) // 2
    slots.append(
        RegionSlot(
            region_type="figure",
            bbox=(margin_left, chart_y, margin_left + half_w, chart_y + chart_h),
            reading_order=order,
            column=0,
        )
    )
    order += 1
    slots.append(
        RegionSlot(
            region_type="figure",
            bbox=(margin_left + half_w + 40, chart_y, margin_left + content_w, chart_y + chart_h),
            reading_order=order,
            column=1,
        )
    )
    order += 1

    # Table below charts
    table_y = chart_y + chart_h + 30
    remaining = height - margin_bottom - table_y
    if remaining > 300:
        table_h = min(remaining // 2, 500)
        slots.append(
            RegionSlot(
                region_type="table",
                bbox=(margin_left, table_y, margin_left + content_w, table_y + table_h),
                reading_order=order,
            )
        )
        order += 1

        # More body text
        text_y = table_y + table_h + 20
        text_remaining = height - margin_bottom - text_y
        if text_remaining > 100:
            slots.append(
                RegionSlot(
                    region_type="body",
                    bbox=(margin_left, text_y, margin_left + content_w, text_y + text_remaining),
                    reading_order=order,
                )
            )

    # Footer
    footer_h = 60
    footer_y = height - margin_bottom - footer_h
    slots.append(
        RegionSlot(
            region_type="footer",
            bbox=(margin_left, footer_y, margin_left + content_w, footer_y + footer_h),
            reading_order=order + 1,
        )
    )

    return slots


def mixed_template(
    width: int,
    height: int,
    margin_top: int,
    margin_bottom: int,
    margin_left: int,
    margin_right: int,
    rng: Generator,
    columns: int | None = None,
) -> list[RegionSlot]:
    """Mixed layout: randomly pick from other templates."""
    templates = [academic_template, legal_template, form_template, report_template]
    chosen = templates[int(rng.integers(0, len(templates)))]
    return chosen(width, height, margin_top, margin_bottom, margin_left, margin_right, rng, columns)


TEMPLATES: dict[str, type] = {}  # populated below for reference

TEMPLATE_FUNCTIONS = {
    "academic": academic_template,
    "legal": legal_template,
    "notebook": notebook_template,
    "mixed": mixed_template,
    "form": form_template,
    "report": report_template,
}
