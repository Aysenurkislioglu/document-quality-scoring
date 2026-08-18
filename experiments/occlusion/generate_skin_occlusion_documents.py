"""
Ten rengi (skin-color) tabanlı, KONUMDAN BAĞIMSIZ occlusion deneyi için
sentetik belge + kontrollü kapanma enjeksiyonu.

`generate_occlusion_documents.py`'den farkı: o deney, konumu ÖNCEDEN BİLİNEN
"Belge No" alanını gri bir dikdörtgenle kapatıyordu (OCR tabanlı yöntemi
doğrulamak için). Bu deney ise RASTGELE konumda, RENKLİ (ten tonu) bir
dikdörtgen ekleyip, konumu bilmeden yalnızca renk sinyaliyle occlusion'ın
yakalanıp yakalanamadığını test ediyor — bkz. project_notes.md,
"Occlusion Aşama 1".

Üç farklı ten tonu (açık/orta/koyu) ayrı ayrı test edilir, çünkü literatür
ten rengi tabanlı yöntemlerin farklı tonlarda güvenilirliğinin
değişebileceğini belirtiyor — bu deney bunu doğrudan ölçer.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
from synthetic_documents import render_document, default_combinations  # noqa: E402

RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "occlusion_skin"

COVERAGE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# (R, G, B) — açık/orta/koyu ten tonu yaklaşık değerleri.
SKIN_TONES = {
    "acik": (224, 172, 135),
    "orta": (198, 134, 92),
    "koyu": (110, 74, 51),
}


def apply_skin_occlusion(image: Image.Image, content_bbox, coverage: float, tone_rgb, rng: random.Random):
    """content_bbox içinde RASTGELE konumda, coverage oranına uyan alanlı bir
    ten-tonu dikdörtgeni çizer. Konum kasıtlı olarak rastgele — yöntemin
    "alan konumunu bilmeden" çalıştığını test etmek bunun amacı."""
    img = image.copy()
    x0, y0, x1, y1 = content_bbox
    w, h = x1 - x0, y1 - y0

    if coverage <= 0:
        return img, None

    # Alan oranı coverage olacak şekilde kare-benzeri bir yama (hem genişlik
    # hem yükseklik sqrt(coverage) ile ölçeklenir).
    patch_w = max(1, int(w * (coverage ** 0.5)))
    patch_h = max(1, int(h * (coverage ** 0.5)))
    px0 = rng.randint(x0, max(x0, x1 - patch_w))
    py0 = rng.randint(y0, max(y0, y1 - patch_h))
    patch_bbox = (px0, py0, px0 + patch_w, py0 + patch_h)

    draw = ImageDraw.Draw(img)
    draw.rectangle(list(patch_bbox), fill=tone_rgb)
    return img, patch_bbox


def generate_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    rows = []
    doc_id = 0
    for font_size, num_paragraphs, replica in default_combinations():
        doc_id += 1
        doc = render_document(font_size, num_paragraphs, rng)
        doc_name = f"doc_{doc_id:03d}"
        rgb_image = doc.image.convert("RGB")

        for tone_name, tone_rgb in SKIN_TONES.items():
            for level, coverage in enumerate(COVERAGE_LEVELS):
                occluded_img, patch_bbox = apply_skin_occlusion(
                    rgb_image, doc.content_bbox, coverage, tone_rgb, rng
                )
                out_name = f"{doc_name}_{tone_name}_cov{level}.png"
                out_path = OUTPUT_DIR / out_name
                occluded_img.save(out_path)

                rows.append(
                    {
                        "doc_id": doc_name,
                        "skin_tone": tone_name,
                        "coverage_level": level,
                        "coverage_fraction": coverage,
                        "path": str(out_path),
                        "content_bbox": list(doc.content_bbox),
                        "patch_bbox": list(patch_bbox) if patch_bbox else None,
                        "font_size": font_size,
                        "num_paragraphs": num_paragraphs,
                    }
                )

    manifest_path = OUTPUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} ten-rengi occlusion-enjekte edilmiş görüntü üretildi -> {manifest_path}")


if __name__ == "__main__":
    generate_all()
