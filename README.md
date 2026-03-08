# SynthDoc

Synthetic document generation for training document understanding models.

SynthDoc generates photorealistic document images with pixel-perfect ground truth annotations — bounding boxes, region types, reading order, OCR text, and layout structure. Use it to create unlimited training data for document AI models like Nemotron Parse, LayoutLM, and Donut.

## Why synthetic data?

Manual document annotation is slow (5-10 min/page), expensive, and error-prone. SynthDoc generates thousands of annotated pages per minute with perfect labels — because you created the document, you know exactly where everything is.

## What it generates

```
Input:  Configuration (layout type, content mix, degradation level)
Output: Document image (PNG/PDF) + COCO-format annotations (JSON)
```

| Content Type | Method |
|-------------|--------|
| **Body text** | LLM-generated or sampled from corpora |
| **Multi-column layouts** | Programmatic CSS grid → PDF rendering |
| **Mathematical formulas** | LaTeX sampler → rendered equations |
| **Tables** | Structured data → formatted tables with borders, merged cells |
| **Handwriting** | Font simulation with stroke variation, or GAN-generated |
| **Figures/charts** | Matplotlib/seaborn generated plots with captions |
| **Headers/footers** | Template-based with page numbers, dates, titles |
| **Margin notes** | Positioned annotations with handwriting fonts |
| **Signatures** | Bezier curve generation with natural variation |

## Degradation pipeline

Makes synthetic docs look like real scans:

```
Clean PDF → Render at DPI → Add noise → Skew/rotate → Blur →
Adjust contrast → Paper texture overlay → JPEG artifacts →
Salt-and-pepper noise → Output
```

Each degradation step is configurable and randomized within bounds.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  SynthDoc Engine                 │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐ │
│  │ Layout   │  │ Content  │  │ Degradation   │ │
│  │ Composer │  │ Fillers  │  │ Pipeline      │ │
│  │          │  │          │  │               │ │
│  │ Grid     │  │ Text     │  │ Noise         │ │
│  │ Column   │  │ LaTeX    │  │ Skew          │ │
│  │ Mixed    │  │ Table    │  │ Blur          │ │
│  │ Academic │  │ Figure   │  │ Texture       │ │
│  │ Legal    │  │ Handwrite│  │ JPEG          │ │
│  └──────────┘  └──────────┘  └───────────────┘ │
│                      │                           │
│              ┌───────▼───────┐                  │
│              │  Annotation   │                  │
│              │  Generator    │                  │
│              │  (COCO JSON)  │                  │
│              └───────────────┘                  │
└─────────────────────────────────────────────────┘
```

## Output format

```json
{
  "image": "page_00142.png",
  "width": 2550,
  "height": 3300,
  "regions": [
    {
      "id": 1,
      "type": "body",
      "bbox": [120, 340, 1200, 2800],
      "reading_order": 1,
      "text": "The experiment was conducted...",
      "confidence": 1.0
    },
    {
      "type": "formula",
      "bbox": [200, 1400, 1100, 1520],
      "reading_order": 3,
      "text": "E = mc^2",
      "latex": "E = mc^2"
    }
  ]
}
```

Compatible with COCO detection format, easily convertible to YOLO, VOC, or custom formats.

## Quick start

```bash
pip install synthdoc

# Generate 1000 academic paper pages
synthdoc generate --template academic --count 1000 --output ./dataset/

# Generate with heavy degradation (simulating old scans)
synthdoc generate --template mixed --count 500 --degradation heavy --output ./dataset/

# Generate specific content types
synthdoc generate --content "text,formula,table" --layout two-column --count 200

# Validate generated annotations
synthdoc validate ./dataset/
```

## Templates

| Template | Description |
|----------|-------------|
| `academic` | Two-column papers with formulas, figures, citations |
| `legal` | Dense single-column contracts with headers, clauses, signatures |
| `notebook` | Handwritten notes with diagrams and margin annotations |
| `mixed` | Random mix of all content types and layouts |
| `form` | Structured forms with fields, checkboxes, tables |
| `report` | Business reports with charts, tables, headers |

## Tech stack

- **Python 3.12+**
- **WeasyPrint** — HTML/CSS → PDF rendering
- **Pillow + OpenCV** — Image processing and degradation
- **LaTeX** — Formula rendering (via matplotlib's mathtext or actual LaTeX)
- **Faker** — Realistic text content generation
- **NumPy** — Random variation and noise generation

## Use cases

- Train document layout detection models (YOLO, Faster R-CNN)
- Train OCR models with known ground truth
- Benchmark document understanding systems
- Augment real training data with synthetic examples
- Test document processing pipelines

## License

MIT
