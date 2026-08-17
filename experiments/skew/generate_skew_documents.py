"""
Skew deneyi için sentetik belge + bilinen açılarda kontrollü döndürme.

Not (işaret kuralı): `cv2.getRotationMatrix2D(center, angle, 1.0)` ile
uygulanan döndürme, src/skew/metrics.py'deki tahmincilerle KARŞIT işarette
ölçülüyor (deneysel olarak doğrulandı — bkz. src/skew/metrics.py docstring).
Bu yüzden yer gerçeği açı, `-applied_angle` olarak kaydedilir; böylece
tahmin edilen açı ile doğrudan karşılaştırılabilir.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
from synthetic_documents import render_document, default_combinations  # noqa: E402

RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "skew"

APPLIED_ANGLES = [-12, -8, -5, -2, -1, 0, 1, 2, 5, 8, 12]  # derece


def rotate(img_array: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img_array.shape
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img_array, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )


def generate_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    rows = []
    doc_id = 0
    for font_size, num_paragraphs, replica in default_combinations():
        doc_id += 1
        doc = render_document(font_size, num_paragraphs, rng)
        doc_name = f"doc_{doc_id:03d}"
        img_array = np.array(doc.image)

        for applied_angle in APPLIED_ANGLES:
            rotated = rotate(img_array, applied_angle)
            ground_truth_angle = -applied_angle  # bkz. modül docstring'i
            out_name = f"{doc_name}_angle{applied_angle:+03d}.png"
            out_path = OUTPUT_DIR / out_name
            cv2.imwrite(str(out_path), rotated)
            rows.append(
                {
                    "doc_id": doc_name,
                    "applied_angle": applied_angle,
                    "ground_truth_angle": ground_truth_angle,
                    "path": str(out_path),
                    "font_size": font_size,
                    "num_paragraphs": num_paragraphs,
                }
            )

    manifest_path = OUTPUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} döndürülmüş görüntü üretildi -> {manifest_path}")


if __name__ == "__main__":
    generate_all()
