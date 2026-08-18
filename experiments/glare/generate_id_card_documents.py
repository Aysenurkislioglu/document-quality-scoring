"""
Kimlik kartı/pasaport benzeri (RENKLİ ZEMİNLİ) sentetik belge + kontrollü
glare enjeksiyonu.

GEREKÇE: Bütün glare denemeleri (bkz. project_notes.md, "Glare ML v1-v5")
DÜZ BEYAZ KAĞITTA yapıldı ve hepsi aynı temel soruna çarptı — glare ile
beyaz kağıt, HSV uzayında (ve öğrenilmiş özelliklerde) ayırt edilemiyordu,
çünkü ikisi de "yüksek parlaklık + düşük saturasyon". Kimlik kartı/pasaport
gibi RENKLİ zeminli belgelerde durum farklı: kartın kendi rengi (mavi, bej
vb.) saturasyonu olan bir renktir; glare (fiziksel olarak ışık kaynağının
rengi, genelde beyaza yakın) bu zeminden saturasyon açısından gerçekten
AYRIŞIR. Bu, dichromatic reflection model literatüründeki bilinen bir
prensiptir — 🔎 dış araştırmada bulundu: klasik specular highlight
tespit yöntemleri (dichromatic model) renkli/parlak yüzeylerde (meyve,
seramik, cilt) işe yarar; düz beyaz kağıt bu yüzden literatürde neredeyse
hiç bu yöntemlerle ele alınmamış — çünkü yüzeyin "gerçek rengi" zaten
beyaz, ayrışacak bir renk farkı yok.

Bu deney, mevcut (zaten var olan, DEĞİŞTİRİLMEMİŞ) src/glare/metrics.py
HSV+CC yönteminin, sadece zemin rengi renkli olduğunda, hiç ML gerekmeden
işe yarayıp yaramadığını test eder.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_glare_documents import sigma_for_target_area, MAX_ALPHA  # noqa: E402


def apply_glare_blob_rgb(img_array: np.ndarray, center, sigma: float, max_alpha: float = MAX_ALPHA):
    """`generate_glare_documents.apply_glare_blob`'un RGB (3 kanallı) sürümü —
    orijinali yalnızca gri-tonlamalı (2D) görüntüler için yazılmıştı."""
    h, w = img_array.shape[:2]
    ys, xs = np.indices((h, w))
    d2 = (xs - center[0]) ** 2 + (ys - center[1]) ** 2
    if sigma <= 0:
        alpha = np.zeros((h, w), dtype=np.float64)
    else:
        alpha = max_alpha * np.exp(-d2 / (2 * sigma ** 2))
    alpha_3ch = alpha[..., None]
    out = img_array.astype(np.float64) * (1 - alpha_3ch) + 255 * alpha_3ch
    return out.astype(np.uint8), alpha

RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "glare_id_card"

CARD_SIZE = (856, 540)  # ID-1 standardı oranına yakın (kredi kartı/kimlik boyutu)
MARGIN = 40

# Birden fazla kart renk şeması — TEK bir renge aşırı uyum riskini test etmek için.
CARD_COLOR_SCHEMES = {
    "mavi_gri": {"bg": (196, 214, 226), "photo": (150, 170, 190), "text": (40, 50, 90)},
    "bej": (lambda: {"bg": (232, 220, 196), "photo": (200, 185, 155), "text": (90, 60, 30)})(),
    "yesilimsi": (lambda: {"bg": (205, 224, 210), "photo": (160, 185, 165), "text": (30, 70, 45)})(),
}

TARGET_AREA_FRACTIONS = [0.0, 0.03, 0.06, 0.10, 0.15, 0.22]  # glare deneyiyle tutarlı


def render_id_card(scheme: dict, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", CARD_SIZE, color=scheme["bg"])
    draw = ImageDraw.Draw(img)
    draw.rectangle([MARGIN, MARGIN, MARGIN + 220, MARGIN + 300], fill=scheme["photo"])
    text_x0 = MARGIN + 260
    for i, ty in enumerate(range(MARGIN + 20, MARGIN + 280, 40)):
        width = rng.randint(250, 420)
        draw.rectangle([text_x0, ty, text_x0 + width, ty + 18], fill=scheme["text"])
    return img


def generate_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    rows = []
    card_id = 0
    for scheme_name, scheme in CARD_COLOR_SCHEMES.items():
        for replica in range(4):  # her renk şemasından 4 varyant
            card_id += 1
            card_img = render_id_card(scheme, rng)
            img_array = np.array(card_img)
            w, h = CARD_SIZE
            center = (w // 2, h // 2)
            content_area_px = w * h
            card_name = f"card_{card_id:03d}_{scheme_name}"

            for level, frac in enumerate(TARGET_AREA_FRACTIONS):
                target_area_px = frac * content_area_px
                sigma = sigma_for_target_area(target_area_px)
                degraded_array, alpha = apply_glare_blob_rgb(img_array, center, sigma)

                ground_truth_fraction = float(np.count_nonzero(alpha > 0.5)) / content_area_px

                out_name = f"{card_name}_sev{level}.png"
                out_path = OUTPUT_DIR / out_name
                Image.fromarray(degraded_array).save(out_path)

                rows.append({
                    "card_id": card_name, "color_scheme": scheme_name,
                    "severity_level": level, "target_area_fraction": frac,
                    "ground_truth_glare_fraction": ground_truth_fraction,
                    "path": str(out_path),
                })

    manifest_path = OUTPUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} kimlik-kartı-benzeri glare görüntüsü üretildi -> {manifest_path}")


if __name__ == "__main__":
    generate_all()
