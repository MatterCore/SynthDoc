"""CLI entry point for SynthDoc using Click."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click

from synthdoc import __version__
from synthdoc.annotation.formats import export_coco, export_voc, export_yolo
from synthdoc.config import (
    ContentConfig,
    DegradationConfig,
    GenerationConfig,
    LayoutConfig,
    PageConfig,
)


@click.group()
@click.version_option(version=__version__, prog_name="synthdoc")
def main() -> None:
    """SynthDoc — Synthetic document generation for training document AI models."""
    pass


@main.command()
@click.option("--template", "-t", default="academic", show_default=True,
              type=click.Choice(["academic", "legal", "notebook", "mixed", "form", "report"]),
              help="Layout template to use.")
@click.option("--count", "-n", default=10, show_default=True,
              help="Number of pages to generate.")
@click.option("--output", "-o", default="./output", show_default=True,
              help="Output directory.")
@click.option("--degradation", "-d", default="light", show_default=True,
              type=click.Choice(["none", "light", "medium", "heavy"]),
              help="Degradation level.")
@click.option("--content", "-c", default=None,
              help="Comma-separated content types (text,formula,table,figure).")
@click.option("--columns", default=None, type=int,
              help="Override column count.")
@click.option("--seed", "-s", default=None, type=int,
              help="Random seed for reproducibility.")
@click.option("--width", default=2550, type=int, show_default=True,
              help="Page width in pixels.")
@click.option("--height", default=3300, type=int, show_default=True,
              help="Page height in pixels.")
@click.option("--dpi", default=300, type=int, show_default=True,
              help="DPI for rendering.")
@click.option("--format", "export_format", default="coco", show_default=True,
              type=click.Choice(["coco", "yolo", "voc", "all"]),
              help="Annotation export format.")
def generate(
    template: str,
    count: int,
    output: str,
    degradation: str,
    content: str | None,
    columns: int | None,
    seed: int | None,
    width: int,
    height: int,
    dpi: int,
    export_format: str,
) -> None:
    """Generate synthetic document images with annotations."""
    from synthdoc.engine import SynthDocEngine

    content_types = content.split(",") if content else ["text", "formula", "table", "figure"]

    config = GenerationConfig(
        page=PageConfig(width_px=width, height_px=height, dpi=dpi),
        layout=LayoutConfig(template=template, columns=columns),
        content=ContentConfig(types=content_types),
        degradation=DegradationConfig(level=degradation),
        count=count,
        output_dir=output,
        seed=seed,
    )

    click.echo(f"SynthDoc v{__version__}")
    click.echo(f"  Template:    {template}")
    click.echo(f"  Count:       {count}")
    click.echo(f"  Output:      {output}")
    click.echo(f"  Degradation: {degradation}")
    click.echo(f"  Content:     {', '.join(content_types)}")
    click.echo(f"  Seed:        {seed or 'random'}")
    click.echo()

    engine = SynthDocEngine(config)

    start_time = time.time()
    pages = []

    with click.progressbar(range(count), label="Generating pages") as bar:
        output_dir = Path(output)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for page_num in bar:
            page = engine.generate_page(page_num)
            image_path = images_dir / f"page_{page_num:05d}.png"
            page.image.save(str(image_path))
            page.image_path = image_path
            pages.append(page)

    elapsed = time.time() - start_time
    click.echo(f"\nGenerated {count} pages in {elapsed:.1f}s ({count / max(elapsed, 0.01):.1f} pages/sec)")

    # Save annotations
    engine.annotator.save_annotations(output)
    engine.annotator.save_per_page(Path(output) / "annotations")

    # Export in requested format(s)
    all_pages = engine.annotator.pages
    if export_format in ("coco", "all"):
        coco_path = export_coco(all_pages, output)
        click.echo(f"  COCO annotations: {coco_path}")

    if export_format in ("yolo", "all"):
        yolo_paths = export_yolo(all_pages, output)
        click.echo(f"  YOLO labels: {len(yolo_paths)} files in {output}/labels/")

    if export_format in ("voc", "all"):
        voc_paths = export_voc(all_pages, output)
        click.echo(f"  VOC annotations: {len(voc_paths)} files in {output}/voc_annotations/")

    click.echo(f"\nDone. Output saved to {output}/")


@main.command()
@click.argument("dataset_dir", type=click.Path(exists=True))
def validate(dataset_dir: str) -> None:
    """Validate a generated dataset — check images, annotations, and consistency."""
    dataset_path = Path(dataset_dir)

    ann_file = dataset_path / "annotations.json"
    if not ann_file.exists():
        click.echo(f"ERROR: annotations.json not found in {dataset_dir}", err=True)
        sys.exit(1)

    with open(ann_file) as f:
        data = json.load(f)

    num_pages = data.get("num_pages", 0)
    pages = data.get("pages", [])
    click.echo(f"Dataset: {dataset_dir}")
    click.echo(f"  Pages declared: {num_pages}")
    click.echo(f"  Pages found: {len(pages)}")

    images_dir = dataset_path / "images"
    if not images_dir.exists():
        click.echo("  WARNING: images/ directory not found", err=True)

    errors = 0
    warnings = 0

    for page in pages:
        image_name = page.get("image", "")
        image_path = images_dir / image_name

        if not image_path.exists():
            click.echo(f"  ERROR: Missing image {image_name}", err=True)
            errors += 1
            continue

        regions = page.get("regions", [])
        page_w = page.get("width", 0)
        page_h = page.get("height", 0)

        for region in regions:
            bbox = region.get("bbox", [])
            if len(bbox) != 4:
                click.echo(f"  ERROR: Invalid bbox in {image_name}: {bbox}", err=True)
                errors += 1
                continue

            x1, y1, x2, y2 = bbox
            if x1 >= x2 or y1 >= y2:
                click.echo(f"  ERROR: Degenerate bbox in {image_name}: {bbox}", err=True)
                errors += 1
            if x2 > page_w or y2 > page_h:
                click.echo(f"  WARNING: Bbox exceeds page bounds in {image_name}: {bbox}", err=True)
                warnings += 1

            if not region.get("type"):
                click.echo(f"  WARNING: Missing type in region {region.get('id')} of {image_name}", err=True)
                warnings += 1

    click.echo(f"\nValidation complete: {errors} errors, {warnings} warnings")
    if errors > 0:
        sys.exit(1)
    click.echo("Dataset is valid.")


@main.command()
@click.argument("dataset_dir", type=click.Path(exists=True))
def stats(dataset_dir: str) -> None:
    """Show statistics for a generated dataset."""
    dataset_path = Path(dataset_dir)
    ann_file = dataset_path / "annotations.json"

    if not ann_file.exists():
        click.echo(f"ERROR: annotations.json not found in {dataset_dir}", err=True)
        sys.exit(1)

    with open(ann_file) as f:
        data = json.load(f)

    pages = data.get("pages", [])
    total_regions = 0
    type_counts: dict[str, int] = {}
    total_text_chars = 0

    for page in pages:
        for region in page.get("regions", []):
            total_regions += 1
            rtype = region.get("type", "unknown")
            type_counts[rtype] = type_counts.get(rtype, 0) + 1
            total_text_chars += len(region.get("text", ""))

    click.echo(f"Dataset: {dataset_dir}")
    click.echo(f"  Total pages: {len(pages)}")
    click.echo(f"  Total regions: {total_regions}")
    click.echo(f"  Avg regions/page: {total_regions / max(len(pages), 1):.1f}")
    click.echo(f"  Total text chars: {total_text_chars:,}")
    click.echo()
    click.echo("  Region type distribution:")
    for rtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / max(total_regions, 1) * 100
        click.echo(f"    {rtype:15s}: {count:5d} ({pct:5.1f}%)")

    images_dir = dataset_path / "images"
    if images_dir.exists():
        image_files = list(images_dir.glob("*.png"))
        if image_files:
            sizes = [f.stat().st_size for f in image_files]
            total_mb = sum(sizes) / (1024 * 1024)
            avg_kb = (sum(sizes) / len(sizes)) / 1024
            click.echo(f"\n  Image files: {len(image_files)}")
            click.echo(f"  Total size: {total_mb:.1f} MB")
            click.echo(f"  Avg size: {avg_kb:.0f} KB")


if __name__ == "__main__":
    main()
