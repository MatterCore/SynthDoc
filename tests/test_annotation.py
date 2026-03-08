"""Tests for annotation generation and export formats (COCO, YOLO, VOC)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from synthdoc.annotation.formats import export_coco, export_voc, export_yolo
from synthdoc.annotation.generator import AnnotationGenerator, PageAnnotation, RegionAnnotation


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def sample_page() -> PageAnnotation:
    """Create a sample page annotation for testing."""
    gen = AnnotationGenerator()
    page = gen.start_page("page_00000.png", 2550, 3300)
    gen.add_region(page, "title", (150, 150, 2400, 270), 0, text="Sample Title")
    gen.add_region(page, "body", (150, 300, 1200, 1500), 1, text="Body text here...")
    gen.add_region(page, "formula", (150, 1550, 1200, 1700), 2, text="E=mc^2", latex="E=mc^2")
    gen.add_region(page, "table", (1300, 300, 2400, 900), 3, text="Col1\tCol2\n1\t2")
    gen.add_region(page, "figure", (1300, 950, 2400, 1700), 4, text="[Figure: Chart]")
    gen.add_region(page, "signature", (1800, 2800, 2400, 3000), 5, text="[signature]")
    return page


@pytest.fixture()
def two_pages() -> list[PageAnnotation]:
    gen = AnnotationGenerator()
    p1 = gen.start_page("page_00000.png", 2550, 3300)
    gen.add_region(p1, "title", (150, 150, 2400, 270), 0, text="Title 1")
    gen.add_region(p1, "body", (150, 300, 2400, 3000), 1, text="Body 1")

    p2 = gen.start_page("page_00001.png", 2550, 3300)
    gen.add_region(p2, "title", (150, 150, 2400, 270), 0, text="Title 2")
    gen.add_region(p2, "table", (150, 300, 2400, 1500), 1, text="Table data")
    return gen.pages


class TestAnnotationGenerator:
    def test_start_page(self):
        gen = AnnotationGenerator()
        page = gen.start_page("test.png", 800, 600)
        assert isinstance(page, PageAnnotation)
        assert page.image_filename == "test.png"
        assert page.width == 800

    def test_add_region(self):
        gen = AnnotationGenerator()
        page = gen.start_page("test.png", 800, 600)
        ann = gen.add_region(page, "body", (10, 20, 200, 300), 0, text="hello")
        assert isinstance(ann, RegionAnnotation)
        assert ann.region_type == "body"
        assert ann.text == "hello"

    def test_region_ids_increment(self):
        gen = AnnotationGenerator()
        page = gen.start_page("test.png", 800, 600)
        a1 = gen.add_region(page, "body", (0, 0, 100, 100), 0)
        a2 = gen.add_region(page, "title", (0, 100, 100, 200), 1)
        assert a2.id == a1.id + 1

    def test_to_dict(self, sample_page: PageAnnotation):
        d = sample_page.to_dict()
        assert d["image"] == "page_00000.png"
        assert d["width"] == 2550
        assert d["height"] == 3300
        assert len(d["regions"]) == 6

    def test_save_annotations(self, sample_page: PageAnnotation, tmp_path: Path):
        gen = AnnotationGenerator()
        gen._pages = [sample_page]
        path = gen.save_annotations(tmp_path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["num_pages"] == 1
        assert len(data["pages"]) == 1

    def test_save_per_page(self, two_pages: list[PageAnnotation], tmp_path: Path):
        gen = AnnotationGenerator()
        gen._pages = two_pages
        paths = gen.save_per_page(tmp_path)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()


class TestCOCOFormat:
    def test_structure(self, two_pages: list[PageAnnotation], tmp_path: Path):
        path = export_coco(two_pages, tmp_path)
        assert path.exists()
        with open(path) as f:
            coco = json.load(f)
        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco

    def test_image_entries(self, two_pages: list[PageAnnotation], tmp_path: Path):
        path = export_coco(two_pages, tmp_path)
        with open(path) as f:
            coco = json.load(f)
        assert len(coco["images"]) == 2
        assert coco["images"][0]["file_name"] == "page_00000.png"

    def test_annotation_count(self, two_pages: list[PageAnnotation], tmp_path: Path):
        path = export_coco(two_pages, tmp_path)
        with open(path) as f:
            coco = json.load(f)
        total_regions = sum(len(p.regions) for p in two_pages)
        assert len(coco["annotations"]) == total_regions

    def test_bbox_format(self, sample_page: PageAnnotation, tmp_path: Path):
        path = export_coco([sample_page], tmp_path)
        with open(path) as f:
            coco = json.load(f)
        for ann in coco["annotations"]:
            bbox = ann["bbox"]
            assert len(bbox) == 4
            assert bbox[2] > 0
            assert bbox[3] > 0


class TestYOLOFormat:
    def test_creates_label_files(self, two_pages: list[PageAnnotation], tmp_path: Path):
        paths = export_yolo(two_pages, tmp_path)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_normalized_coordinates(self, sample_page: PageAnnotation, tmp_path: Path):
        paths = export_yolo([sample_page], tmp_path)
        with open(paths[0]) as f:
            for line in f:
                parts = line.strip().split()
                assert len(parts) == 5
                cx, cy, w, h = (float(v) for v in parts[1:])
                assert 0.0 <= cx <= 1.0
                assert 0.0 <= cy <= 1.0
                assert 0.0 < w <= 1.0
                assert 0.0 < h <= 1.0

    def test_line_count(self, sample_page: PageAnnotation, tmp_path: Path):
        paths = export_yolo([sample_page], tmp_path)
        with open(paths[0]) as f:
            lines = [line for line in f.readlines() if line.strip()]
        assert len(lines) == len(sample_page.regions)

    def test_classes_file(self, sample_page: PageAnnotation, tmp_path: Path):
        export_yolo([sample_page], tmp_path)
        classes_file = tmp_path / "labels" / "classes.txt"
        assert classes_file.exists()


class TestVOCFormat:
    def test_creates_xml_files(self, two_pages: list[PageAnnotation], tmp_path: Path):
        paths = export_voc(two_pages, tmp_path)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_valid_xml(self, sample_page: PageAnnotation, tmp_path: Path):
        paths = export_voc([sample_page], tmp_path)
        tree = ET.parse(paths[0])
        root = tree.getroot()
        assert root.tag == "annotation"

    def test_object_count(self, sample_page: PageAnnotation, tmp_path: Path):
        paths = export_voc([sample_page], tmp_path)
        tree = ET.parse(paths[0])
        root = tree.getroot()
        objects = root.findall("object")
        assert len(objects) == len(sample_page.regions)

    def test_bndbox_values(self, sample_page: PageAnnotation, tmp_path: Path):
        paths = export_voc([sample_page], tmp_path)
        tree = ET.parse(paths[0])
        root = tree.getroot()
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            assert bndbox is not None
            xmin = int(bndbox.findtext("xmin"))  # type: ignore[arg-type]
            ymin = int(bndbox.findtext("ymin"))  # type: ignore[arg-type]
            xmax = int(bndbox.findtext("xmax"))  # type: ignore[arg-type]
            ymax = int(bndbox.findtext("ymax"))  # type: ignore[arg-type]
            assert xmin < xmax
            assert ymin < ymax

    def test_region_types_match(self, sample_page: PageAnnotation, tmp_path: Path):
        paths = export_voc([sample_page], tmp_path)
        tree = ET.parse(paths[0])
        root = tree.getroot()
        xml_names = {obj.findtext("name") for obj in root.findall("object")}
        layout_types = {r.region_type for r in sample_page.regions}
        assert xml_names == layout_types
