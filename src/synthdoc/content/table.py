"""Table content generation — structured tables with headers and data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFont


@dataclass
class TableResult:
    """Result of table generation."""

    image: Image.Image
    text: str
    rows: int
    cols: int


class TableGenerator:
    """Generates structured tables rendered as images."""

    def __init__(self, max_rows: int = 10, max_cols: int = 6) -> None:
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.fake = Faker()
        self._font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if size not in self._font_cache:
            try:
                self._font_cache[size] = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
                )
            except OSError:
                try:
                    self._font_cache[size] = ImageFont.truetype(
                        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size
                    )
                except OSError:
                    self._font_cache[size] = ImageFont.load_default(size=size)
        return self._font_cache[size]

    def _generate_header(self, num_cols: int, rng: np.random.Generator) -> list[str]:
        """Generate table column headers."""
        header_pools = [
            ["Name", "Date", "Amount", "Status", "Category", "ID", "Description", "Total"],
            ["Item", "Qty", "Price", "Subtotal", "Tax", "Discount", "Net"],
            ["Metric", "Q1", "Q2", "Q3", "Q4", "YTD", "Change"],
            ["Parameter", "Value", "Unit", "Min", "Max", "Mean", "Std Dev"],
            ["Country", "Population", "GDP", "Area", "Capital", "Currency"],
        ]
        pool = header_pools[int(rng.integers(0, len(header_pools)))]
        if len(pool) >= num_cols:
            indices = rng.choice(len(pool), size=num_cols, replace=False)
            return [pool[i] for i in sorted(indices)]
        return pool[:num_cols] + [f"Col {i}" for i in range(len(pool), num_cols)]

    def _generate_cell(self, col_idx: int, rng: np.random.Generator) -> str:
        """Generate a single cell value."""
        cell_type = int(rng.integers(0, 5))
        if cell_type == 0:
            return f"{rng.uniform(0, 10000):.2f}"
        elif cell_type == 1:
            return str(int(rng.integers(1, 1000)))
        elif cell_type == 2:
            return self.fake.date()
        elif cell_type == 3:
            return self.fake.word().capitalize()
        else:
            return f"${rng.uniform(10, 9999):.2f}"

    def generate(
        self,
        width: int,
        height: int,
        rng: np.random.Generator,
    ) -> TableResult:
        """Generate a table rendered as an image.

        Args:
            width: Target image width in pixels.
            height: Target image height in pixels.
            rng: NumPy random generator.

        Returns:
            TableResult with rendered image and table data as text.
        """
        num_cols = int(rng.integers(2, min(self.max_cols + 1, max(3, width // 200 + 1))))
        row_height = 40
        max_rows_fit = max(2, (height - 20) // row_height - 1)
        num_rows = int(rng.integers(2, min(self.max_rows + 1, max_rows_fit + 1)))

        font_size = max(14, min(22, width // (num_cols * 5)))
        font = self._get_font(font_size)
        bold_font = font  # Use same font; bold styling via color

        headers = self._generate_header(num_cols, rng)
        data: list[list[str]] = []
        for _ in range(num_rows):
            row = [self._generate_cell(c, rng) for c in range(num_cols)]
            data.append(row)

        # Calculate column widths
        padding = 10
        col_width = (width - 2 * padding) // num_cols
        actual_row_h = min(row_height, (height - 2 * padding) // (num_rows + 1))

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Alternating row colors
        use_alt_colors = rng.random() > 0.3
        alt_color = (240, 245, 250)

        y = padding

        # Draw header row
        header_bg = (50, 70, 120)
        draw.rectangle([padding, y, width - padding, y + actual_row_h], fill=header_bg)
        for c, header in enumerate(headers):
            x = padding + c * col_width + 6
            text = header[:col_width // (font_size // 2 + 1)]
            draw.text((x, y + 4), text, fill="white", font=bold_font)
        y += actual_row_h

        # Draw grid lines
        line_color = (180, 180, 180)

        # Data rows
        for r, row in enumerate(data):
            if use_alt_colors and r % 2 == 1:
                draw.rectangle([padding, y, width - padding, y + actual_row_h], fill=alt_color)

            for c, cell in enumerate(row):
                x = padding + c * col_width + 6
                text = cell[:col_width // (font_size // 2 + 1)]
                draw.text((x, y + 4), text, fill="black", font=font)

            # Horizontal line
            draw.line(
                [(padding, y + actual_row_h), (width - padding, y + actual_row_h)],
                fill=line_color,
                width=1,
            )
            y += actual_row_h

        # Vertical column lines
        for c in range(num_cols + 1):
            x = padding + c * col_width
            draw.line(
                [(x, padding), (x, padding + actual_row_h * (num_rows + 1))],
                fill=line_color,
                width=1,
            )

        # Outer border
        draw.rectangle(
            [padding, padding, width - padding, padding + actual_row_h * (num_rows + 1)],
            outline=(100, 100, 100),
            width=2,
        )

        # Build text representation
        text_lines = ["\t".join(headers)]
        for row in data:
            text_lines.append("\t".join(row))
        text = "\n".join(text_lines)

        return TableResult(image=img, text=text, rows=num_rows, cols=num_cols)
