"""
Darkness deneyi için sentetik belge + iki ayrı bozulma senaryosu:

1. GLOBAL karanlık: tüm görüntü aynı oranda karartılır (örn. kötü genel
   pozlama / yetersiz ortam ışığı senaryosu).
2. LOKAL karanlık: yalnızca "Belge No" alanı karartılır (örn. belgenin bir
   köşesine düşen gölge / lokal aydınlatma sorunu senaryosu), görüntünün
   geri kalanı DEĞİŞMEDEN bırakılır.

Amaç: `research/literature_review.md`'de aktarılan "aynı global ortalama
parlaklığa sahip iki görüntü, kullanılabilirlik açısından çok farklı
olabilir" iddiasını sayısal olarak test etmek.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
from synthetic_documents import render_document, default_combinations  # noqa: E402

RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "darkness"

DARKENING_FACTORS = [1.0, 0.85, 0.70, 0.55, 0.40, 0.25]  # 1.0 = değişiklik yok
LOCAL_FIELD_LABEL = "Belge No"
LOCAL_PADDING = 8  # alan kutusunun etrafına eklenen piksel (gerçekçi "gölge" için)


def darken_global(img_array: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img_array.astype(np.float64) * factor, 0, 255).astype(np.uint8)


def darken_region(img_array: np.ndarray, bbox, factor: float) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    out = img_array.copy()
    region = out[y0:y1, x0:x1].astype(np.float64)
    out[y0:y1, x0:x1] = np.clip(region * factor, 0, 255).astype(np.uint8)
    return out


def generate_all():
    rng = random.Random(RANDOM_SEED)

    global_dir = BASE_DIR / "global"
    local_dir = BASE_DIR / "local"
    global_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    global_rows = []
    local_rows = []

    doc_id = 0
    for font_size, num_paragraphs, replica in default_combinations():
        doc_id += 1
        doc = render_document(font_size, num_paragraphs, rng)
        doc_name = f"doc_{doc_id:03d}"
        img_array = np.array(doc.image)

        field = next(f for f in doc.field_boxes if f.label == LOCAL_FIELD_LABEL)
        x0, y0, x1, y1 = field.bbox
        padded_bbox = (
            max(0, x0 - LOCAL_PADDING),
            max(0, y0 - LOCAL_PADDING),
            min(img_array.shape[1], x1 + LOCAL_PADDING),
            min(img_array.shape[0], y1 + LOCAL_PADDING),
        )

        for level, factor in enumerate(DARKENING_FACTORS):
            # --- Global senaryo ---
            g_img = darken_global(img_array, factor)
            g_path = global_dir / f"{doc_name}_sev{level}.png"
            Image.fromarray(g_img).save(g_path)
            global_rows.append(
                {
                    "doc_id": doc_name,
                    "severity_level": level,
                    "darkening_factor": factor,
                    "path": str(g_path),
                    "font_size": font_size,
                    "num_paragraphs": num_paragraphs,
                    "content_bbox": json.dumps(list(doc.content_bbox)),
                    "field_bbox": json.dumps(list(padded_bbox)),
                }
            )

            # --- Lokal senaryo ---
            l_img = darken_region(img_array, padded_bbox, factor)
            l_path = local_dir / f"{doc_name}_sev{level}.png"
            Image.fromarray(l_img).save(l_path)
            local_rows.append(
                {
                    "doc_id": doc_name,
                    "severity_level": level,
                    "darkening_factor": factor,
                    "path": str(l_path),
                    "font_size": font_size,
                    "num_paragraphs": num_paragraphs,
                    "content_bbox": json.dumps(list(doc.content_bbox)),
                    "field_bbox": json.dumps(list(padded_bbox)),
                }
            )

    for name, rows in [("global", global_rows), ("local", local_rows)]:
        manifest_path = BASE_DIR / name / "manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}: {len(rows)} görüntü -> {manifest_path}")


if __name__ == "__main__":
    generate_all()
