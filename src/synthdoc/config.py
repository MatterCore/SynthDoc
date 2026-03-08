"""Pydantic configuration models for SynthDoc generation pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PageConfig(BaseModel):
    """Physical page dimensions and margins in pixels."""

    width_px: int = 2550  # 8.5" at 300 DPI
    height_px: int = 3300  # 11" at 300 DPI
    dpi: int = 300
    margin_top: int = 150
    margin_bottom: int = 150
    margin_left: int = 150
    margin_right: int = 150

    @property
    def content_width(self) -> int:
        return self.width_px - self.margin_left - self.margin_right

    @property
    def content_height(self) -> int:
        return self.height_px - self.margin_top - self.margin_bottom


class LayoutConfig(BaseModel):
    """Layout template and column configuration."""

    template: str = "academic"  # academic, legal, notebook, mixed, form, report
    columns: int | None = None  # Override column count


class ContentConfig(BaseModel):
    """Content types and parameters."""

    types: list[str] = Field(default_factory=lambda: ["text", "formula", "table", "figure"])
    text_language: str = "en"
    formula_complexity: str = "medium"  # simple, medium, complex
    table_max_rows: int = 10
    table_max_cols: int = 6


class DegradationConfig(BaseModel):
    """Degradation pipeline configuration."""

    level: str = "light"  # none, light, medium, heavy
    noise_sigma: float = 0.02
    skew_max_degrees: float = 2.0
    blur_kernel_max: int = 3
    jpeg_quality_min: int = 70
    paper_texture: bool = True

    def get_effective_params(self) -> dict[str, float | int | bool]:
        """Return parameters scaled by degradation level."""
        multipliers = {"none": 0.0, "light": 1.0, "medium": 2.0, "heavy": 3.5}
        m = multipliers.get(self.level, 1.0)
        return {
            "noise_sigma": self.noise_sigma * m,
            "skew_max_degrees": self.skew_max_degrees * m,
            "blur_kernel_max": max(1, int(self.blur_kernel_max * m)),
            "jpeg_quality_min": max(10, int(self.jpeg_quality_min - 15 * m)),
            "paper_texture": self.paper_texture and m > 0,
        }


class GenerationConfig(BaseModel):
    """Top-level configuration for document generation."""

    page: PageConfig = Field(default_factory=PageConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    content: ContentConfig = Field(default_factory=ContentConfig)
    degradation: DegradationConfig = Field(default_factory=DegradationConfig)
    count: int = 10
    output_dir: str = "./output"
    seed: int | None = None
