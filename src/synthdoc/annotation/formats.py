"""Export annotations in COCO, YOLO, and Pascal VOC formats."""

from __future__ import annotations

import json
from pathlib import Path
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

from synthdoc.annotation.generator import PageAnnotation

# Canonical category mapping
CATEGORY_MAP: dict[str, int] = {
    "body": 1,
    "title": 2,
    "header": 3,
    "footer": 4,
    "formula": 5,
    "table": 6,
    "figure": 7,
    "handwriting": 8,
    "signature": 9,
}


def export_coco(pages: list[PageAnnotation], output_dir: str | Path) -> Path:
    """Export annotations in COCO object detection format.

    Creates a single JSON with images, annotations, and categories arrays.

    Args:
        pages: List of PageAnnotation objects.
        output_dir: Output directory.

    Returns:
        Path to the saved COCO JSON file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1

    for img_id, page in enumerate(pages, start=1):
        images.append({
            "id": img_id,
            "file_name": page.image_filename,
            "width": page.width,
            "height": page.height,
        })

        for region in page.regions:
            x1, y1, x2, y2 = region.bbox
            w = x2 - x1
            h = y2 - y1
            cat_id = CATEGORY_MAP.get(region.region_type, 1)

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],  # COCO uses [x, y, width, height]
                "area": w * h,
                "iscrowd": 0,
                "attributes": {
                    "reading_order": region.reading_order,
                    "text": region.text,
                },
            })
            ann_id += 1

    categories = [
        {"id": cat_id, "name": name}
        for name, cat_id in sorted(CATEGORY_MAP.items(), key=lambda x: x[1])
    ]

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_file = output_path / "coco_annotations.json"
    with open(out_file, "w") as f:
        json.dump(coco_data, f, indent=2)

    return out_file


def export_yolo(pages: list[PageAnnotation], output_dir: str | Path) -> list[Path]:
    """Export annotations in YOLO format (one .txt per image).

    Each line: class_id x_center y_center width height (all normalized 0-1).

    Args:
        pages: List of PageAnnotation objects.
        output_dir: Output directory for label files.

    Returns:
        List of paths to saved label files.
    """
    output_path = Path(output_dir) / "labels"
    output_path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for page in pages:
        stem = Path(page.image_filename).stem
        label_file = output_path / f"{stem}.txt"
        lines = []

        for region in page.regions:
            x1, y1, x2, y2 = region.bbox
            cat_id = CATEGORY_MAP.get(region.region_type, 0)
            # YOLO class IDs are 0-indexed
            class_id = cat_id - 1

            # Normalize coordinates
            x_center = ((x1 + x2) / 2) / page.width
            y_center = ((y1 + y2) / 2) / page.height
            w = (x2 - x1) / page.width
            h = (y2 - y1) / page.height

            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        with open(label_file, "w") as f:
            f.write("\n".join(lines))

        paths.append(label_file)

    # Write classes.txt
    classes_file = output_path / "classes.txt"
    with open(classes_file, "w") as f:
        for name in sorted(CATEGORY_MAP, key=CATEGORY_MAP.get):  # type: ignore[arg-type]
            f.write(f"{name}\n")

    return paths


def export_voc(pages: list[PageAnnotation], output_dir: str | Path) -> list[Path]:
    """Export annotations in Pascal VOC XML format.

    One XML file per image with <annotation> root and <object> elements.

    Args:
        pages: List of PageAnnotation objects.
        output_dir: Output directory for XML files.

    Returns:
        List of paths to saved XML files.
    """
    output_path = Path(output_dir) / "voc_annotations"
    output_path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for page in pages:
        stem = Path(page.image_filename).stem
        xml_file = output_path / f"{stem}.xml"

        root = Element("annotation")

        folder_el = SubElement(root, "folder")
        folder_el.text = "images"

        filename_el = SubElement(root, "filename")
        filename_el.text = page.image_filename

        size_el = SubElement(root, "size")
        width_el = SubElement(size_el, "width")
        width_el.text = str(page.width)
        height_el = SubElement(size_el, "height")
        height_el.text = str(page.height)
        depth_el = SubElement(size_el, "depth")
        depth_el.text = "3"

        for region in page.regions:
            obj = SubElement(root, "object")

            name_el = SubElement(obj, "name")
            name_el.text = region.region_type

            difficult_el = SubElement(obj, "difficult")
            difficult_el.text = "0"

            bndbox = SubElement(obj, "bndbox")
            x1, y1, x2, y2 = region.bbox
            SubElement(bndbox, "xmin").text = str(x1)
            SubElement(bndbox, "ymin").text = str(y1)
            SubElement(bndbox, "xmax").text = str(x2)
            SubElement(bndbox, "ymax").text = str(y2)

        raw_xml = tostring(root, encoding="unicode")
        pretty_xml = parseString(raw_xml).toprettyxml(indent="  ")
        with open(xml_file, "w") as f:
            f.write(pretty_xml)

        paths.append(xml_file)

    return paths
