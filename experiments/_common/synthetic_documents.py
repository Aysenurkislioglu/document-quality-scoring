"""
Ortak sentetik belge üretici.

Blur modülünde kullanılan üreticinin genişletilmiş hâlidir. Glare, Darkness ve
Occlusion deneyleri, kimlik alanlarının (Ad Soyad / Belge No / Tarih) tam
konumunu (bounding box) ve beklenen içerik desenini (regex) bilmesi gerektiği
için burada bu bilgiler de kaydedilir. Blur deneyi bu bilgilere ihtiyaç
duymadığı için kendi (daha basit) üreticisini korur; bu dosya sonraki
modüllerde tekrar tekrar aynı kodu yazmamak için oluşturulmuştur.

Not: Bu ortak modül, Glare modülü geliştirilirken (retroaktif olarak) DRY
prensibiyle eklenmiştir — bkz. project_notes.md, Glare bölümü / "Aldığımız
kararlar".
"""

from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass, field
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

PAGE_SIZE = (850, 1100)
MARGIN = 70
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

WORD_POOL = (
    "belge kalite skorlama sistemi kimlik doğrulama görüntü işleme "
    "bulanıklık parlaklık aydınlatma eğiklik kapanma tespit algoritma "
    "ölçüm analiz literatür yöntem model değerlendirme doğruluk metin "
    "satır karakter tanıma optik test deney sonuç veri seti eğitim "
    "gradyan kenar varyans eşik istatistik ortalama standart sapma"
).split()

BBox = Tuple[int, int, int, int]  # (x0, y0, x1, y1)


@dataclass
class FieldBox:
    label: str
    value: str
    expected_pattern: str  # regex — occlusion modülünde "beklenen içerik" kontrolü için
    bbox: BBox


@dataclass
class SyntheticDocument:
    image: Image.Image
    content_bbox: BBox
    field_boxes: List[FieldBox] = field(default_factory=list)
    font_size: int = 0
    num_paragraphs: int = 0


def _random_paragraph(rng: random.Random, min_words: int = 40, max_words: int = 90) -> str:
    n = rng.randint(min_words, max_words)
    words = [rng.choice(WORD_POOL) for _ in range(n)]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def render_document(font_size: int, num_paragraphs: int, rng: random.Random) -> SyntheticDocument:
    img = Image.new("L", PAGE_SIZE, color=255)
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.truetype(FONT_PATH_BOLD, size=font_size + 6)
    body_font = ImageFont.truetype(FONT_PATH, size=font_size)

    y = MARGIN
    draw.text((MARGIN, y), "SENTETİK BELGE ÖRNEĞİ", font=title_font, fill=0)
    y += font_size + 20

    belge_no = f"{rng.randint(10**9, 10**10 - 1)}"
    field_defs = [
        ("Ad Soyad", "AYŞENUR KIŞLIOĞLU", r"^[A-ZÇĞİÖŞÜ ]+$"),
        ("Belge No", belge_no, r"^\d{10}$"),
        ("Tarih", "17.08.2026", r"^\d{2}\.\d{2}\.\d{4}$"),
    ]

    field_boxes: List[FieldBox] = []
    for label, value, pattern in field_defs:
        text = f"{label}: {value}"
        y0 = y
        draw.text((MARGIN, y), text, font=body_font, fill=0)
        # Değerin başladığı x konumunu ölç (yalnızca "value" kısmının kutusunu
        # kaydediyoruz, çünkü occlusion/OCR karşılaştırması yalnızca değer
        # üzerinde yapılacak; label sabit ve her zaman biliniyor).
        label_prefix = f"{label}: "
        prefix_bbox = draw.textbbox((MARGIN, y), label_prefix, font=body_font)
        value_bbox = draw.textbbox((prefix_bbox[2], y), value, font=body_font)
        field_boxes.append(
            FieldBox(label=label, value=value, expected_pattern=pattern, bbox=value_bbox)
        )
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

    content_bbox = (MARGIN, MARGIN, PAGE_SIZE[0] - MARGIN, min(y, PAGE_SIZE[1] - MARGIN))

    return SyntheticDocument(
        image=img,
        content_bbox=content_bbox,
        field_boxes=field_boxes,
        font_size=font_size,
        num_paragraphs=num_paragraphs,
    )


def default_combinations():
    """Blur modülüyle aynı 12'li (font_size x num_paragraphs x replica) ızgara."""
    for font_size in (14, 20, 28):
        for num_paragraphs in (2, 5):
            for replica in range(2):
                yield font_size, num_paragraphs, replica
