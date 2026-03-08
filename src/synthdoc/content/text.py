"""Text content generation using Faker and domain-specific templates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFont


@dataclass
class TextResult:
    """Result of text generation."""

    image: Image.Image
    text: str


# Domain-specific sentence templates
ACADEMIC_TEMPLATES = [
    "The experimental results demonstrate that {noun} significantly affects {noun2}.",
    "We propose a novel approach to {verb}ing {noun} using {noun2} methods.",
    "Our analysis reveals a strong correlation between {noun} and {noun2}.",
    "The proposed framework achieves state-of-the-art performance on {noun} benchmarks.",
    "Previous work on {noun} has focused primarily on {noun2} techniques.",
    "We evaluate our method on {number} datasets and report {noun} metrics.",
    "The theoretical analysis shows that {noun} converges under {noun2} conditions.",
    "Figure {number} illustrates the relationship between {noun} and {noun2}.",
]

LEGAL_TEMPLATES = [
    "The parties hereby agree to the terms and conditions set forth in this {noun}.",
    "Notwithstanding the foregoing, the {noun} shall remain in full force and effect.",
    "In the event of a breach of this {noun}, the non-breaching party shall be entitled to {noun2}.",
    "This {noun} shall be governed by and construed in accordance with the laws of {noun2}.",
    "The {noun} shall indemnify and hold harmless the {noun2} from any claims.",
    "All notices required under this {noun} shall be delivered in writing to {noun2}.",
    "This {noun} constitutes the entire agreement between the parties regarding {noun2}.",
]

BUSINESS_TEMPLATES = [
    "Revenue for Q{number} increased by {percentage}% compared to the previous quarter.",
    "The {noun} division reported strong growth driven by {noun2} investments.",
    "Key performance indicators show improvement across all {noun} segments.",
    "Management recommends allocating additional resources to {noun} development.",
    "The strategic initiative to expand {noun} operations yielded positive {noun2} results.",
    "Market analysis indicates growing demand for {noun} in the {noun2} sector.",
]


class TextGenerator:
    """Generates text content and renders it as PIL images."""

    def __init__(self, language: str = "en") -> None:
        self.fake = Faker(language)
        self._font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Get a font at the specified size, using cache."""
        if size not in self._font_cache:
            try:
                self._font_cache[size] = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", size)
            except OSError:
                try:
                    self._font_cache[size] = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", size)
                except OSError:
                    self._font_cache[size] = ImageFont.load_default(size=size)
        return self._font_cache[size]

    def _fill_template(self, template: str, rng: np.random.Generator) -> str:
        """Fill a template with random nouns, verbs, numbers."""
        result = template
        while "{noun2}" in result:
            result = result.replace("{noun2}", self.fake.word(), 1)
        while "{noun}" in result:
            result = result.replace("{noun}", self.fake.word(), 1)
        while "{verb}" in result:
            result = result.replace("{verb}", self.fake.word(), 1)
        while "{number}" in result:
            result = result.replace("{number}", str(int(rng.integers(1, 100))), 1)
        while "{percentage}" in result:
            result = result.replace("{percentage}", f"{rng.uniform(1, 50):.1f}", 1)
        return result

    def _generate_domain_text(self, domain: str, num_sentences: int, rng: np.random.Generator) -> str:
        """Generate domain-specific text."""
        templates_map = {
            "academic": ACADEMIC_TEMPLATES,
            "legal": LEGAL_TEMPLATES,
            "report": BUSINESS_TEMPLATES,
            "business": BUSINESS_TEMPLATES,
        }
        templates = templates_map.get(domain, [])

        sentences = []
        for _ in range(num_sentences):
            if templates and rng.random() < 0.4:
                template = templates[int(rng.integers(0, len(templates)))]
                sentences.append(self._fill_template(template, rng))
            else:
                sentences.append(self.fake.sentence(nb_words=int(rng.integers(8, 20))))
        return " ".join(sentences)

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int) -> list[str]:
        """Word-wrap text to fit within max_width pixels."""
        words = text.split()
        lines: list[str] = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = font.getbbox(test_line)
            text_width = bbox[2] - bbox[0]
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def generate(
        self,
        width: int,
        height: int,
        rng: np.random.Generator,
        domain: str = "academic",
        font_size: int = 28,
        is_title: bool = False,
        is_header: bool = False,
        is_footer: bool = False,
    ) -> TextResult:
        """Generate text content rendered as an image.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            rng: NumPy random generator.
            domain: Text domain (academic, legal, report).
            font_size: Font size in pixels.
            is_title: If True, generate title-style text.
            is_header: If True, generate header text.
            is_footer: If True, generate footer text.

        Returns:
            TextResult with rendered image and source text.
        """
        if is_title:
            font_size = max(36, font_size + 16)
        elif is_header:
            font_size = max(24, font_size + 4)
        elif is_footer:
            font_size = max(18, font_size - 6)

        font = self._get_font(font_size)
        line_height = int(font_size * 1.5)
        max_lines = max(1, height // line_height)
        padding = 10

        if is_title:
            text = self.fake.sentence(nb_words=int(rng.integers(4, 10))).rstrip(".")
        elif is_header:
            text = f"{self.fake.company()} — {self.fake.catch_phrase()}"
        elif is_footer:
            text = f"Page {{page}} | {self.fake.date()} | {self.fake.company()}"
        else:
            num_sentences = max(1, max_lines // 2)
            text = self._generate_domain_text(domain, num_sentences, rng)

        lines = self._wrap_text(text, font, width - 2 * padding)
        lines = lines[:max_lines]

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        y = padding
        if is_title:
            # Center title
            for line in lines:
                bbox = font.getbbox(line)
                tw = bbox[2] - bbox[0]
                x = max(0, (width - tw) // 2)
                draw.text((x, y), line, fill="black", font=font)
                y += line_height
        else:
            for line in lines:
                draw.text((padding, y), line, fill="black", font=font)
                y += line_height

        rendered_text = "\n".join(lines)
        return TextResult(image=img, text=rendered_text)
