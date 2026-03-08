"""Handwriting simulation — renders text with natural-looking variation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFont


@dataclass
class HandwritingResult:
    """Result of handwriting generation."""

    image: Image.Image
    text: str


class HandwritingGenerator:
    """Generates handwritten-looking text by rendering characters with random perturbations."""

    def __init__(self) -> None:
        self.fake = Faker()
        self._font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Get a font, preferring cursive/handwriting-style fonts."""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
        ]
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default(size=size)

    def _render_char_with_variation(
        self,
        draw: ImageDraw.ImageDraw,
        char: str,
        x: float,
        y: float,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        rng: np.random.Generator,
        ink_color: tuple[int, int, int],
    ) -> float:
        """Render a single character with random position/size perturbation.

        Returns the x advance for the next character.
        """
        # Baseline wobble
        y_offset = rng.normal(0, 1.5)
        x_offset = rng.normal(0, 0.5)

        draw.text(
            (x + x_offset, y + y_offset),
            char,
            fill=ink_color,
            font=font,
        )

        # Get character width for advance
        bbox = font.getbbox(char)
        char_w = bbox[2] - bbox[0]
        spacing_var = rng.normal(0, 1.0)
        return char_w + spacing_var

    def generate(
        self,
        width: int,
        height: int,
        rng: np.random.Generator,
    ) -> HandwritingResult:
        """Generate handwritten text as an image.

        Args:
            width: Target image width in pixels.
            height: Target image height in pixels.
            rng: NumPy random generator.

        Returns:
            HandwritingResult with rendered image and source text.
        """
        font_size = max(18, min(32, height // 8))
        font = self._get_font(font_size)
        line_height = int(font_size * 2.0)
        padding = 15

        max_lines = max(1, (height - 2 * padding) // line_height)
        num_sentences = max(1, max_lines)
        text = " ".join(self.fake.sentence(nb_words=int(rng.integers(5, 12))) for _ in range(num_sentences))

        # Ink color variation (dark blue/black)
        ink_r = int(rng.integers(0, 40))
        ink_g = int(rng.integers(0, 40))
        ink_b = int(rng.integers(30, 80))
        ink_color = (ink_r, ink_g, ink_b)

        # Slightly off-white background to simulate paper
        bg_val = int(rng.integers(245, 256))
        bg_color = (bg_val, bg_val, max(bg_val - 5, 240))

        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Draw ruled lines if there's space
        if max_lines > 2:
            for line_idx in range(max_lines + 1):
                ly = padding + line_idx * line_height + line_height - 4
                if ly < height - padding:
                    draw.line(
                        [(padding, ly), (width - padding, ly)],
                        fill=(200, 210, 220),
                        width=1,
                    )

        # Render text character by character with perturbation
        x = padding + rng.uniform(0, 10)
        y = padding
        lines_written = 0
        rendered_chars: list[str] = []

        for char in text:
            if lines_written >= max_lines:
                break

            if char == " ":
                space_w = font_size * 0.4 + rng.normal(0, 1)
                x += space_w
                rendered_chars.append(" ")
            else:
                advance = self._render_char_with_variation(draw, char, x, y, font, rng, ink_color)
                x += advance
                rendered_chars.append(char)

            # Line wrap
            if x > width - padding - font_size:
                x = padding + rng.uniform(0, 10)
                y += line_height + rng.normal(0, 2)
                lines_written += 1

        rendered_text = "".join(rendered_chars).strip()
        return HandwritingResult(image=img, text=rendered_text)
