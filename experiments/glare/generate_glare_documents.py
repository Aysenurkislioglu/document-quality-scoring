"""
Glare deneyi için sentetik belge + kontrollü glare enjeksiyonu.

Yaklaşım (blur modülüyle tutarlı): önce temiz sentetik belgeler üretilir,
sonra her belgeye artan şiddette, YER GERÇEĞİ (ground truth) alanı bilinen
sentetik glare "lekesi" eklenir.

Glare lekesi, belge içeriğinin ortasına yerleştirilen yumuşak kenarlı
(Gaussian falloff) parlak bir daire olarak modellenir — gerçek kamera
glare'ının "merkezde en parlak, kenarlara doğru sönümlenen" görünümünü
taklit eder. Yer gerçeği glare alanı, alpha > 0.5 olan piksellerin sayısı
olarak tanımlanır (bkz. apply_glare_blob).
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
BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "glare"
ORIGINALS_DIR = BASE_DIR / "originals"
DEGRADED_DIR = BASE_DIR / "degraded"

# severity_level -> content-bbox'a oranla hedeflenen glare alanı
TARGET_AREA_FRACTIONS = [0.0, 0.03, 0.06, 0.10, 0.15, 0.22]
MAX_ALPHA = 0.95


def apply_glare_blob(img_array: np.ndarray, center, sigma: float, max_alpha: float = MAX_ALPHA):
    ys, xs = np.indices(img_array.shape)
    d2 = (xs - center[0]) ** 2 + (ys - center[1]) ** 2
    if sigma <= 0:
        alpha = np.zeros_like(img_array, dtype=np.float64)
    else:
        alpha = max_alpha * np.exp(-d2 / (2 * sigma ** 2))
    out = img_array.astype(np.float64) * (1 - alpha) + 255 * alpha
    return out.astype(np.uint8), alpha


def sigma_for_target_area(target_area_px: float, max_alpha: float = MAX_ALPHA) -> float:
    if target_area_px <= 0:
        return 0.0
    k = 2 * np.log(max_alpha / 0.5)
    return float(np.sqrt(target_area_px / (np.pi * k)))


def generate_all():
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
    DEGRADED_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    originals_manifest = []
    degraded_manifest = []

    doc_id = 0
    for font_size, num_paragraphs, replica in default_combinations():
        doc_id += 1
        doc = render_document(font_size, num_paragraphs, rng)
        doc_name = f"doc_{doc_id:03d}"

        orig_path = ORIGINALS_DIR / f"{doc_name}.png"
        doc.image.save(orig_path)

        originals_manifest.append(
            {
                "doc_id": doc_name,
                "path": str(orig_path),
                "font_size": font_size,
                "num_paragraphs": num_paragraphs,
                "content_bbox": list(doc.content_bbox),
            }
        )

        x0, y0, x1, y1 = doc.content_bbox
        content_w, content_h = x1 - x0, y1 - y0
        content_area_px = content_w * content_h
        center = (x0 + content_w // 2, y0 + content_h // 2)

        img_array = np.array(doc.image)

        for level, frac in enumerate(TARGET_AREA_FRACTIONS):
            target_area_px = frac * content_area_px
            sigma = sigma_for_target_area(target_area_px)
            degraded_array, alpha = apply_glare_blob(img_array, center, sigma)

            # Yer gerçeği: content bbox içinde alpha > 0.5 olan piksel oranı
            alpha_roi = alpha[y0:y1, x0:x1]
            ground_truth_fraction = float(np.count_nonzero(alpha_roi > 0.5)) / content_area_px

            out_name = f"{doc_name}_sev{level}.png"
            out_path = DEGRADED_DIR / out_name
            Image.fromarray(degraded_array).save(out_path)

            degraded_manifest.append(
                {
                    "doc_id": doc_name,
                    "severity_level": level,
                    "target_area_fraction": frac,
                    "ground_truth_glare_fraction": ground_truth_fraction,
                    "path": str(out_path),
                    "font_size": font_size,
                    "num_paragraphs": num_paragraphs,
                    "content_bbox": json.dumps(list(doc.content_bbox)),
                }
            )

    with open(ORIGINALS_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        rows = [{**r, "content_bbox": json.dumps(r["content_bbox"])} for r in originals_manifest]
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(DEGRADED_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(degraded_manifest[0].keys()))
        writer.writeheader()
        writer.writerows(degraded_manifest)

    print(f"{len(originals_manifest)} orijinal belge, {len(degraded_manifest)} glare-enjekte edilmiş görüntü üretildi.")
    print(f"Originals manifest -> {ORIGINALS_DIR / 'manifest.csv'}")
    print(f"Degraded manifest  -> {DEGRADED_DIR / 'manifest.csv'}")


if __name__ == "__main__":
    generate_all()
