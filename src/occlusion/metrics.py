"""
Occlusion tespiti: OCR + "beklenen alan deseni/uzunluğu" karşılaştırması.

Yöntemin nasıl çalıştığına ve sınırlamasına dair açıklama için bkz. README.md.

Bağımlılık: pytesseract + sistemde kurulu tesseract-ocr (+ dil paketi, örn.
tesseract-ocr-tur). Kurulum: requirements.txt.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import pytesseract

BBox = Tuple[int, int, int, int]


def _crop_and_upscale(image: np.ndarray, bbox: BBox, padding: int = 6, upscale: int = 3) -> Image.Image:
    x0, y0, x1, y1 = bbox
    h, w = image.shape[:2]
    x0p, y0p = max(0, x0 - padding), max(0, y0 - padding)
    x1p, y1p = min(w, x1 + padding), min(h, y1 + padding)
    crop = image[y0p:y1p, x0p:x1p]
    pil_img = Image.fromarray(crop)
    return pil_img.resize((pil_img.width * upscale, pil_img.height * upscale))


def ocr_field(
    image: np.ndarray,
    bbox: BBox,
    lang: str = "eng",
    char_whitelist: Optional[str] = None,
    padding: int = 6,
    upscale: int = 3,
) -> Dict[str, object]:
    """
    Belirtilen alan (bbox) üzerinde OCR çalıştırır.

    Returns:
        dict: {"text": str, "mean_confidence": float (0-100)}
    """
    crop = _crop_and_upscale(image, bbox, padding, upscale)

    config = "--psm 7"
    if char_whitelist:
        config += f" -c tessedit_char_whitelist={char_whitelist}"

    text = pytesseract.image_to_string(crop, lang=lang, config=config).strip()

    data = pytesseract.image_to_data(crop, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c in data.get("conf", []) if str(c) not in ("-1",) and float(c) >= 0]
    mean_conf = float(np.mean(confs)) if confs else 0.0

    return {"text": text, "mean_confidence": mean_conf}


def occlusion_suspicion_score(
    image: np.ndarray,
    bbox: BBox,
    expected_length: int,
    lang: str = "eng",
    char_whitelist: Optional[str] = None,
) -> Dict[str, float]:
    """
    Bir kimlik alanı için occlusion şüphe skorunu hesaplar.

    İki bağımsız sinyal birleştirilir:
    - length_ratio: OCR'ın okuduğu (yalnızca whitelist'teki) karakter sayısının,
      beklenen karakter sayısına oranı (0-1, 1 = tam okundu).
    - mean_confidence / 100: OCR'ın kendi güven skoru (0-1).

    suspicion_score = 100 * (1 - ortalama(length_ratio, confidence_ratio))
    (0 = occlusion yok gibi görünüyor / şüphe düşük, 100 = tamamen kapalı gibi
    görünüyor / şüphe yüksek)

    NOT (yön/convention uyarısı): Bu, projedeki DİĞER modüllerin (blur, glare,
    darkness) tersi bir yöndür — onlarda YÜKSEK skor = İYİ kalite anlamına
    gelirken, buradaki "suspicion_score" YÜKSEK = KÖTÜ (occlusion şüphesi
    yüksek) anlamına gelir. Bu kasıtlı bir tasarım tercihi değil, modülün adının
    ("şüphe skoru") doğal anlamından kaynaklanıyor. Nihai skor füzyonu
    aşamasında (Aşama 5) tüm alt-skorlar TEK bir yöne (örn. yüksek=iyi)
    normalize edilmeli — bkz. project_notes.md, Occlusion bölümü, "Aldığımız
    kararlar".
    """
    result = ocr_field(image, bbox, lang=lang, char_whitelist=char_whitelist)
    recognized_len = len(result["text"])
    length_ratio = min(recognized_len / expected_length, 1.0) if expected_length > 0 else 0.0
    confidence_ratio = result["mean_confidence"] / 100.0

    combined_ok_ratio = (length_ratio + confidence_ratio) / 2.0
    suspicion_score = 100.0 * (1.0 - combined_ok_ratio)

    return {
        "recognized_text": result["text"],
        "recognized_length": recognized_len,
        "length_ratio": length_ratio,
        "mean_confidence": result["mean_confidence"],
        "occlusion_suspicion_score": suspicion_score,
    }
