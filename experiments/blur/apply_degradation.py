"""
Kontrollü blur bozulma pipeline'ı.

Her sentetik orijinal belgeye, artan şiddette Gaussian blur uygulanır. Amaç,
"bozulma şiddeti arttıkça ölçülen skorun monoton biçimde kötüleşip
kötüleşmediğini" test etmektir (bkz. initial_research.md, Bölüm 13/Aşama 2).

Gaussian blur'un sigma değeri, bilinen/kontrollü bir "yer gerçeği" (ground
truth) bozulma şiddeti olarak kullanılır: sigma büyüdükçe görüntü daha çok
bulanıklaşır. Bu, gerçek kamera blur'unu birebir taklit etmez (gerçek blur
genelde motion blur + defocus karışımıdır) ama kontrollü, tekrarlanabilir ve
literatürde de yaygın kullanılan bir yaklaşımdır.

Çıktı:
- data/synthetic/blur/degraded/<doc_id>_sev<level>.png
- data/synthetic/blur/degraded/manifest.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

ORIGINALS_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "blur" / "originals"
DEGRADED_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "blur" / "degraded"

# severity_level -> gaussian sigma. 0 = bozulma yok (orijinal).
SIGMA_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.copy()
    # ksize=(0,0) -> OpenCV, sigma'dan uygun (tek sayı) kernel boyutunu otomatik hesaplar.
    return cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)


def generate_all() -> Path:
    DEGRADED_DIR.mkdir(parents=True, exist_ok=True)

    manifest_in = ORIGINALS_DIR / "manifest.csv"
    with open(manifest_in, newline="", encoding="utf-8") as f:
        originals = list(csv.DictReader(f))

    rows = []
    for row in originals:
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(row["path"])

        for level, sigma in enumerate(SIGMA_LEVELS):
            degraded = apply_gaussian_blur(img, sigma)
            out_name = f"{row['doc_id']}_sev{level}.png"
            out_path = DEGRADED_DIR / out_name
            cv2.imwrite(str(out_path), degraded)

            rows.append(
                {
                    "doc_id": row["doc_id"],
                    "severity_level": level,
                    "sigma": sigma,
                    "path": str(out_path),
                    "font_size": row["font_size"],
                    "num_paragraphs": row["num_paragraphs"],
                }
            )

    manifest_out = DEGRADED_DIR / "manifest.csv"
    with open(manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} bozulmuş görüntü üretildi ({len(originals)} belge x {len(SIGMA_LEVELS)} şiddet).")
    print(f"Manifest -> {manifest_out}")
    return manifest_out


if __name__ == "__main__":
    generate_all()
