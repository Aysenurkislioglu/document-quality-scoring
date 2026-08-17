"""
Occlusion deneyi için sentetik belge + kontrollü kapanma enjeksiyonu.

"Belge No" alanı, sağdan sola doğru artan oranda opak bir dikdörtgenle
("parmak" benzetmesi) kapatılır. Yer gerçeği = kapatılan alanın, alan
genişliğine oranı (0.0 - 1.0).
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
from synthetic_documents import render_document, default_combinations  # noqa: E402

RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "occlusion"

COVERAGE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
OCCLUDER_FILL = 128  # "parmak" rengini temsil eden gri ton
TARGET_FIELD = "Belge No"


def apply_occlusion(image: Image.Image, bbox, coverage: float) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    occ_x0 = x1 - coverage * w
    if coverage > 0:
        draw.rectangle([occ_x0 - 2, y0 - 3, x1 + 3, y1 + 3], fill=OCCLUDER_FILL)
    return img


def generate_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    rows = []
    doc_id = 0
    for font_size, num_paragraphs, replica in default_combinations():
        doc_id += 1
        doc = render_document(font_size, num_paragraphs, rng)
        doc_name = f"doc_{doc_id:03d}"

        field = next(f for f in doc.field_boxes if f.label == TARGET_FIELD)

        for level, coverage in enumerate(COVERAGE_LEVELS):
            occluded_img = apply_occlusion(doc.image, field.bbox, coverage)
            out_name = f"{doc_name}_cov{level}.png"
            out_path = OUTPUT_DIR / out_name
            occluded_img.save(out_path)

            rows.append(
                {
                    "doc_id": doc_name,
                    "coverage_level": level,
                    "coverage_fraction": coverage,
                    "path": str(out_path),
                    "field_bbox": json.dumps(list(field.bbox)),
                    "field_value": field.value,
                    "expected_length": len(field.value),
                    "font_size": font_size,
                    "num_paragraphs": num_paragraphs,
                }
            )

    manifest_path = OUTPUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} occlusion-enjekte edilmiş görüntü üretildi -> {manifest_path}")


if __name__ == "__main__":
    generate_all()
