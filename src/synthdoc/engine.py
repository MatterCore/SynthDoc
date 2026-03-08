"""Main SynthDoc generation engine — orchestrates layout, content, rendering, and degradation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from synthdoc.annotation.generator import AnnotationGenerator, PageAnnotation
from synthdoc.config import GenerationConfig
from synthdoc.content.figure import FigureGenerator
from synthdoc.content.formula import FormulaGenerator
from synthdoc.content.handwriting import HandwritingGenerator
from synthdoc.content.signature import SignatureGenerator
from synthdoc.content.table import TableGenerator
from synthdoc.content.text import TextGenerator
from synthdoc.degradation.pipeline import DegradationPipeline
from synthdoc.layout.composer import LayoutComposer
from synthdoc.layout.templates import RegionSlot
from synthdoc.render.html import HTMLDocumentBuilder


@dataclass
class GeneratedPage:
    """Result of generating a single page."""

    page_num: int
    image: Image.Image
    annotation: PageAnnotation
    image_path: Path | None = None
    html: str = ""


class SynthDocEngine:
    """Main engine that generates synthetic document images with annotations.

    The generation pipeline for each page:
    1. Layout composition — select template, generate region slots
    2. Content filling — render text, formulas, tables, figures into slot images
    3. Compositing — paste content images onto a blank page
    4. Degradation — apply noise, blur, texture, compression
    5. Annotation — record bounding boxes, types, and text for each region
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.composer = LayoutComposer(config.layout, config.page)
        self.annotator = AnnotationGenerator()
        self.degradation = DegradationPipeline(config.degradation)
        self.html_builder = HTMLDocumentBuilder(config.page.width_px, config.page.height_px)

        # Content generators
        self.text_gen = TextGenerator(config.content.text_language)
        self.formula_gen = FormulaGenerator(config.content.formula_complexity)
        self.table_gen = TableGenerator(config.content.table_max_rows, config.content.table_max_cols)
        self.figure_gen = FigureGenerator()
        self.handwriting_gen = HandwritingGenerator()
        self.signature_gen = SignatureGenerator()

        # Seed management
        self._base_seed = config.seed if config.seed is not None else int(np.random.SeedSequence().entropy)  # type: ignore[arg-type]

    def _make_rng(self, page_num: int) -> np.random.Generator:
        """Create a deterministic RNG for a specific page number."""
        seed = self._base_seed + page_num
        return np.random.default_rng(seed)

    def _domain_for_template(self) -> str:
        """Map template name to text domain."""
        mapping = {
            "academic": "academic",
            "legal": "legal",
            "report": "report",
            "form": "report",
            "notebook": "academic",
            "mixed": "academic",
        }
        return mapping.get(self.config.layout.template, "academic")

    def _fill_slot(
        self,
        slot: RegionSlot,
        rng: np.random.Generator,
    ) -> tuple[Image.Image | None, str, str]:
        """Fill a single region slot with appropriate content.

        Returns:
            Tuple of (content_image, text_content, latex_content).
        """
        w = slot.width
        h = slot.height
        if w < 20 or h < 20:
            return None, "", ""

        region_type = slot.region_type
        allowed_types = self.config.content.types
        domain = self._domain_for_template()

        if region_type in ("title", "header", "footer"):
            result = self.text_gen.generate(
                w, h, rng,
                domain=domain,
                is_title=(region_type == "title"),
                is_header=(region_type == "header"),
                is_footer=(region_type == "footer"),
            )
            return result.image, result.text, ""

        elif region_type == "body":
            if "text" in allowed_types:
                result = self.text_gen.generate(w, h, rng, domain=domain)
                return result.image, result.text, ""
            return None, "", ""

        elif region_type == "formula":
            if "formula" in allowed_types:
                result = self.formula_gen.generate(w, h, rng)
                return result.image, result.latex, result.latex
            # Fall back to text
            result = self.text_gen.generate(w, h, rng, domain=domain)
            return result.image, result.text, ""

        elif region_type == "table":
            if "table" in allowed_types:
                result = self.table_gen.generate(w, h, rng)
                return result.image, result.text, ""
            result = self.text_gen.generate(w, h, rng, domain=domain)
            return result.image, result.text, ""

        elif region_type == "figure":
            if "figure" in allowed_types:
                result = self.figure_gen.generate(w, h, rng)
                return result.image, f"[Figure: {result.title}]", ""
            result = self.text_gen.generate(w, h, rng, domain=domain)
            return result.image, result.text, ""

        elif region_type == "handwriting":
            result = self.handwriting_gen.generate(w, h, rng)
            return result.image, result.text, ""

        elif region_type == "signature":
            result = self.signature_gen.generate(w, h, rng)
            return result.image, "[signature]", ""

        else:
            # Unknown type — fill with text
            result = self.text_gen.generate(w, h, rng, domain=domain)
            return result.image, result.text, ""

    def generate_page(self, page_num: int) -> GeneratedPage:
        """Generate a single page: layout -> fill content -> composite -> degrade -> annotate.

        Args:
            page_num: Zero-indexed page number.

        Returns:
            GeneratedPage with image, annotation, and metadata.
        """
        rng = self._make_rng(page_num)
        image_filename = f"page_{page_num:05d}.png"

        # 1. Layout composition
        slots = self.composer.compose(rng)

        # 2. Content filling
        content_images: dict[int, Image.Image] = {}
        page_ann = self.annotator.start_page(
            image_filename,
            self.config.page.width_px,
            self.config.page.height_px,
        )

        for i, slot in enumerate(slots):
            content_img, text, latex = self._fill_slot(slot, rng)
            if content_img is None and slot.optional:
                continue
            if content_img is None:
                # Generate placeholder for required slots
                content_img = Image.new("RGB", (slot.width, slot.height), "white")

            content_images[i] = content_img
            self.annotator.add_region(
                page_ann,
                region_type=slot.region_type,
                bbox=slot.bbox,
                reading_order=slot.reading_order,
                text=text,
                latex=latex,
            )

        # 3. Compositing (direct pixel paste for perfect alignment)
        page_image = self.html_builder.build_composite_image(slots, content_images)

        # Also build HTML for optional PDF export
        html = self.html_builder.build(slots, content_images)

        # 4. Degradation
        page_image = self.degradation.apply(page_image, rng)

        return GeneratedPage(
            page_num=page_num,
            image=page_image,
            annotation=page_ann,
            html=html,
        )

    def generate(self) -> list[GeneratedPage]:
        """Generate all pages according to config.

        Returns:
            List of GeneratedPage instances.
        """
        output_dir = Path(self.config.output_dir)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        pages: list[GeneratedPage] = []

        for page_num in range(self.config.count):
            generated = self.generate_page(page_num)

            # Save image
            image_path = images_dir / f"page_{page_num:05d}.png"
            generated.image.save(str(image_path))
            generated.image_path = image_path

            pages.append(generated)

        # Save annotations
        self.annotator.save_annotations(output_dir)
        self.annotator.save_per_page(output_dir / "annotations")

        return pages
