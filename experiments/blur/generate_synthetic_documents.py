"""
Sentetik belge görüntüsü üreteci.

Gerçek bir dataset (örn. SmartDoc-QA) henüz indirilip projeye entegre
edilmediği için, ilk baseline deneyi kontrollü sentetik belge görüntüleri
üzerinde yapılır (bkz. project_notes.md, "Aldığımız kararlar").

Her sentetik belge:
- Beyaz arka plan üzerine siyah metin (başlık + paragraflar) içerir.
- Font boyutu ve paragraf sayısı (metin yoğunluğu) kasıtlı olarak
  DEĞİŞTİRİLİR — bu, ileride "Laplacian threshold metin yoğunluğundan
  ne kadar etkileniyor?" sorusunu aynı veri seti üzerinden analiz
  edebilmemizi sağlar (ekstra veri üretmeye gerek kalmadan).

Çıktı:
- data/synthetic/blur/originals/doc_XXX.png
- data/synthetic/blur/originals/manifest.csv
"""

from __future__ import annotations

import csv
import os
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RANDOM_SEED = 42
PAGE_SIZE = (850, 1100)  # yaklaşık A4 oranı, 100 DPI civarı
MARGIN = 70


def _find_dejavu_font(bold: bool) -> str:
    """DejaVuSans fontunu işletim sistemine göre bulur.

    Sabit bir Linux yolu (/usr/share/fonts/...) yerine önce yaygın sistem
    konumlarını dener, bulamazsa matplotlib'in kendi paketiyle taşıdığı
    DejaVuSans kopyasına düşer (matplotlib zaten requirements.txt'de var,
    bu yüzden macOS/Windows dahil her ortamda çalışır).
    """
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        f"/usr/share/fonts/truetype/dejavu/{name}",  # Linux (Debian/Ubuntu)
        f"/usr/share/fonts/dejavu/{name}",  # Linux (Fedora/RHEL)
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    import matplotlib

    mpl_path = os.path.join(
        os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", name
    )
    if os.path.exists(mpl_path):
        return mpl_path

    raise FileNotFoundError(
        f"{name} bulunamadı. matplotlib kurulu mu? (pip install -r requirements.txt)"
    )


FONT_PATH = _find_dejavu_font(bold=False)
FONT_PATH_BOLD = _find_dejavu_font(bold=True)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "blur" / "originals"

# Font boyutu x paragraf sayısı kombinasyonlarıyla 12 sentetik belge üretilir.
FONT_SIZES = [14, 20, 28]
PARAGRAPH_COUNTS = [2, 5]
REPLICAS_PER_COMBO = 2  # her kombinasyondan kaç farklı belge

WORD_POOL = (
    "belge kalite skorlama sistemi kimlik doğrulama görüntü işleme "
    "bulanıklık parlaklık aydınlatma eğiklik kapanma tespit algoritma "
    "ölçüm analiz literatür yöntem model değerlendirme doğruluk metin "
    "satır karakter tanıma optik test deney sonuç veri seti eğitim "
    "gradyan kenar varyans eşik istatistik ortalama standart sapma"
).split()


def _random_paragraph(rng: random.Random, min_words: int = 40, max_words: int = 90) -> str:
    n = rng.randint(min_words, max_words)
    words = [rng.choice(WORD_POOL) for _ in range(n)]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def _render_document(font_size: int, num_paragraphs: int, rng: random.Random) -> Image.Image:
    img = Image.new("L", PAGE_SIZE, color=255)
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.truetype(FONT_PATH_BOLD, size=font_size + 6)
    body_font = ImageFont.truetype(FONT_PATH, size=font_size)

    y = MARGIN
    draw.text((MARGIN, y), "SENTETİK BELGE ÖRNEĞİ", font=title_font, fill=0)
    y += font_size + 20

    # Kimlik-belgesi benzeri birkaç alan (occlusion/skew fazlarında da kullanılabilir)
    for label, value in [
        ("Ad Soyad", "AYŞENUR KIŞLIOĞLU"),
        ("Belge No", f"{rng.randint(10**9, 10**10 - 1)}"),
        ("Tarih", "17.08.2026"),
    ]:
        draw.text((MARGIN, y), f"{label}: {value}", font=body_font, fill=0)
        y += font_size + 10

    y += 10
    draw.line([(MARGIN, y), (PAGE_SIZE[0] - MARGIN, y)], fill=0, width=1)
    y += 20

    chars_per_line = max(20, int((PAGE_SIZE[0] - 2 * MARGIN) / (font_size * 0.55)))

    for _ in range(num_paragraphs):
        paragraph = _random_paragraph(rng)
        wrapped = textwrap.wrap(paragraph, width=chars_per_line)
        for line in wrapped:
            if y > PAGE_SIZE[1] - MARGIN:
                break
            draw.text((MARGIN, y), line, font=body_font, fill=0)
            y += int(font_size * 1.4)
        y += int(font_size * 0.8)

    return img


def generate_all() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    manifest_rows = []
    doc_id = 0
    for font_size in FONT_SIZES:
        for num_paragraphs in PARAGRAPH_COUNTS:
            for replica in range(REPLICAS_PER_COMBO):
                doc_id += 1
                img = _render_document(font_size, num_paragraphs, rng)
                filename = f"doc_{doc_id:03d}.png"
                out_path = OUTPUT_DIR / filename
                img.save(out_path)
                manifest_rows.append(
                    {
                        "doc_id": f"doc_{doc_id:03d}",
                        "path": str(out_path),
                        "font_size": font_size,
                        "num_paragraphs": num_paragraphs,
                        "replica": replica,
                    }
                )

    manifest_path = OUTPUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"{len(manifest_rows)} sentetik belge üretildi -> {OUTPUT_DIR}")
    print(f"Manifest -> {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    generate_all()
