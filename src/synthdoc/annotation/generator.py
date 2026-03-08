"""COCO-format annotation generator — records region placements as ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RegionAnnotation:
    """A single region annotation with bbox, type, and optional text."""

    id: int
    region_type: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    reading_order: int
    text: str = ""
    latex: str = ""
    confidence: float = 1.0


@dataclass
class PageAnnotation:
    """All annotations for a single generated page."""

    image_filename: str
    width: int
    height: int
    regions: list[RegionAnnotation] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to COCO-style dictionary."""
        return {
            "image": self.image_filename,
            "width": self.width,
            "height": self.height,
            "regions": [
                {
                    "id": r.id,
                    "type": r.region_type,
                    "bbox": list(r.bbox),
                    "reading_order": r.reading_order,
                    "text": r.text,
                    **({"latex": r.latex} if r.latex else {}),
                    "confidence": r.confidence,
                }
                for r in self.regions
            ],
        }


class AnnotationGenerator:
    """Collects region annotations during page generation and exports them."""

    def __init__(self) -> None:
        self._pages: list[PageAnnotation] = []
        self._next_region_id: int = 1

    def start_page(self, image_filename: str, width: int, height: int) -> PageAnnotation:
        """Begin annotation collection for a new page.

        Args:
            image_filename: Output image filename.
            width: Page width in pixels.
            height: Page height in pixels.

        Returns:
            New PageAnnotation instance.
        """
        page = PageAnnotation(
            image_filename=image_filename,
            width=width,
            height=height,
        )
        self._pages.append(page)
        return page

    def add_region(
        self,
        page: PageAnnotation,
        region_type: str,
        bbox: tuple[int, int, int, int],
        reading_order: int,
        text: str = "",
        latex: str = "",
    ) -> RegionAnnotation:
        """Add a region annotation to a page.

        Args:
            page: The PageAnnotation to add to.
            region_type: Type of region (body, formula, table, etc.).
            bbox: Bounding box (x1, y1, x2, y2) in pixels.
            reading_order: Reading order index.
            text: OCR text content.
            latex: LaTeX source (for formula regions).

        Returns:
            The created RegionAnnotation.
        """
        annotation = RegionAnnotation(
            id=self._next_region_id,
            region_type=region_type,
            bbox=bbox,
            reading_order=reading_order,
            text=text,
            latex=latex,
        )
        self._next_region_id += 1
        page.regions.append(annotation)
        return annotation

    @property
    def pages(self) -> list[PageAnnotation]:
        """All collected page annotations."""
        return self._pages

    def save_annotations(self, output_dir: str | Path) -> Path:
        """Save all annotations as a single JSON file.

        Args:
            output_dir: Directory to save annotations.json.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        annotations_file = output_path / "annotations.json"
        data = {
            "dataset": "synthdoc",
            "num_pages": len(self._pages),
            "pages": [p.to_dict() for p in self._pages],
        }

        with open(annotations_file, "w") as f:
            json.dump(data, f, indent=2)

        return annotations_file

    def save_per_page(self, output_dir: str | Path) -> list[Path]:
        """Save one annotation JSON file per page.

        Args:
            output_dir: Directory to save individual annotation files.

        Returns:
            List of paths to saved files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        for page in self._pages:
            stem = Path(page.image_filename).stem
            ann_file = output_path / f"{stem}.json"
            with open(ann_file, "w") as f:
                json.dump(page.to_dict(), f, indent=2)
            paths.append(ann_file)

        return paths
