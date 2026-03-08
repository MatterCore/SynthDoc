"""PDF rendering — WeasyPrint HTML-to-PDF and PDF-to-PNG conversion."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


class PDFRenderer:
    """Renders HTML to PDF using WeasyPrint, and converts PDF pages to PNG images."""

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi

    def html_to_pdf(self, html: str, output_path: str | Path) -> Path:
        """Render an HTML document to PDF.

        Args:
            html: Complete HTML string.
            output_path: Path for the output PDF file.

        Returns:
            Path to the generated PDF.
        """
        from weasyprint import HTML

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html_doc = HTML(string=html)
        html_doc.write_pdf(str(output_path))

        return output_path

    def html_to_png(self, html: str, output_path: str | Path) -> Path:
        """Render an HTML document directly to PNG via WeasyPrint.

        Args:
            html: Complete HTML string.
            output_path: Path for the output PNG file.

        Returns:
            Path to the generated PNG.
        """
        from weasyprint import HTML

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html_doc = HTML(string=html)
        png_bytes = html_doc.write_png(resolution=self.dpi)

        with open(output_path, "wb") as f:
            f.write(png_bytes)

        return output_path

    def pdf_to_png(self, pdf_path: str | Path, output_path: str | Path) -> Path:
        """Convert a PDF file to PNG using PIL.

        Note: This uses a simple approach — for production use with multi-page
        PDFs, consider using pdf2image or poppler. For single-page SynthDoc
        output, we re-render from HTML directly.

        Args:
            pdf_path: Path to input PDF.
            output_path: Path for output PNG.

        Returns:
            Path to the generated PNG.
        """
        from weasyprint import HTML

        pdf_path = Path(pdf_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read PDF and re-render to PNG via WeasyPrint
        # WeasyPrint can write PNG directly from document
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # For single-page documents, we use the PIL approach
        # WeasyPrint doesn't re-read PDFs, so we store the HTML reference
        # In practice, the engine uses html_to_png or saves images directly
        raise NotImplementedError(
            "Direct PDF-to-PNG conversion requires pdf2image/poppler. "
            "Use html_to_png() instead, or save the composite image directly."
        )

    def save_image_as_pdf(self, image: Image.Image, output_path: str | Path) -> Path:
        """Save a PIL Image as a PDF file.

        Args:
            image: PIL Image to save.
            output_path: Path for the output PDF.

        Returns:
            Path to the generated PDF.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        image.save(str(output_path), "PDF", resolution=self.dpi)
        return output_path
