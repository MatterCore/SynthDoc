"""HTML/CSS document builder — composes page images from layout slots and content."""

from __future__ import annotations

import base64
import io

from jinja2 import Template
from PIL import Image

from synthdoc.layout.grid import generate_grid_css, generate_grid_html_divs
from synthdoc.layout.templates import RegionSlot

PAGE_TEMPLATE = Template("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: white; }
{{ css }}
.region img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
</style>
</head>
<body>
<div class="page-container">
{% for region in regions %}
{{ region }}
{% endfor %}
</div>
</body>
</html>
""")


def _image_to_data_uri(img: Image.Image) -> str:
    """Convert a PIL Image to a base64 data URI for embedding in HTML."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class HTMLDocumentBuilder:
    """Builds an HTML page with CSS-positioned content images."""

    def __init__(self, page_width: int, page_height: int) -> None:
        self.page_width = page_width
        self.page_height = page_height

    def build(
        self,
        slots: list[RegionSlot],
        content_images: dict[int, Image.Image],
    ) -> str:
        """Build an HTML document with content images placed at slot positions.

        Args:
            slots: Layout region slots.
            content_images: Mapping from slot index to rendered content image.

        Returns:
            Complete HTML string.
        """
        css = generate_grid_css(slots, self.page_width, self.page_height)
        div_templates = generate_grid_html_divs(slots)

        # Replace content placeholders with actual image tags
        rendered_divs = []
        for i, div_tpl in enumerate(div_templates):
            if i in content_images:
                data_uri = _image_to_data_uri(content_images[i])
                content_html = f'<img src="{data_uri}" alt="{slots[i].region_type}">'
            else:
                content_html = ""

            rendered = div_tpl.replace(f"{{{{ region_{i}_content }}}}", content_html)
            rendered_divs.append(rendered)

        return PAGE_TEMPLATE.render(css=css, regions=rendered_divs)

    def build_composite_image(
        self,
        slots: list[RegionSlot],
        content_images: dict[int, Image.Image],
    ) -> Image.Image:
        """Build a composite PIL Image by pasting content images directly.

        This is faster than going through HTML/PDF rendering and produces
        pixel-perfect output with exact bbox alignment.

        Args:
            slots: Layout region slots.
            content_images: Mapping from slot index to rendered content image.

        Returns:
            Composite page image.
        """
        page = Image.new("RGB", (self.page_width, self.page_height), "white")

        for i, slot in enumerate(slots):
            if i not in content_images:
                continue

            img = content_images[i]
            x1, y1, x2, y2 = slot.bbox
            target_w = x2 - x1
            target_h = y2 - y1

            # Resize content image to fit the slot exactly
            if img.size != (target_w, target_h):
                img = img.resize((target_w, target_h), Image.LANCZOS)

            page.paste(img, (x1, y1))

        return page
