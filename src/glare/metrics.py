"""
Glare tespiti: HSV eşikleme + connected components.

Yöntemin nasıl çalıştığına ve bilinen sınırlamasına dair açıklama için
bkz. README.md.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # (x0, y0, x1, y1)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """HSV dönüşümü için 3 kanallı görüntü gerekir; gri ise BGR'ye genişletir."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def glare_mask(
    image: np.ndarray,
    v_threshold: int = 235,
    s_threshold: int = 35,
    min_component_area: int = 15,
) -> np.ndarray:
    """
    Glare adayı piksellerin ikili (binary) maskesini üretir.

    Args:
        image: BGR veya gri numpy array.
        v_threshold: HSV V (parlaklık) kanalında bu değerin ÜZERİ glare adayı (0-255).
        s_threshold: HSV S (saturasyon) kanalında bu değerin ALTI glare adayı (0-255).
        min_component_area: Bu piksel alanından KÜÇÜK bağlı bileşenler gürültü
            sayılıp maskeden temizlenir.

    Returns:
        np.ndarray: uint8, 0/255 değerli ikili maske (image ile aynı H x W).
    """
    bgr = _to_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    raw_mask = ((v_channel >= v_threshold) & (s_channel <= s_threshold)).astype(np.uint8) * 255

    if min_component_area <= 0:
        return raw_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    filtered = np.zeros_like(raw_mask)
    for label_id in range(1, num_labels):  # 0 = background
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            filtered[labels == label_id] = 255
    return filtered


def glare_ratio(
    image: np.ndarray,
    roi: Optional[BBox] = None,
    v_threshold: int = 235,
    s_threshold: int = 35,
    min_component_area: int = 15,
) -> float:
    """
    Glare alanının, ilgilenilen bölgeye (roi) oranını hesaplar.

    Args:
        roi: (x0, y0, x1, y1) — belirtilirse yalnızca bu bölge analiz edilir
            (örn. belgenin içerik kutusu, boş kenarlıklar hariç).
    """
    if roi is not None:
        x0, y0, x1, y1 = roi
        region = image[y0:y1, x0:x1]
    else:
        region = image

    mask = glare_mask(region, v_threshold, s_threshold, min_component_area)
    total_pixels = mask.shape[0] * mask.shape[1]
    if total_pixels == 0:
        return 0.0
    glare_pixels = int(np.count_nonzero(mask))
    return glare_pixels / total_pixels


def glare_score(ratio: float) -> float:
    """
    Glare oranını kaba bir 0-100 alt-skora çevirir (100 = glare yok).

    NOT: Bu doğrusal eşleme GEÇİCİ bir yer tutucudur — gerçek kalibrasyon
    ancak ML skor füzyonu aşamasında (etiketli veriyle) yapılabilir.
    """
    score = 100.0 * (1.0 - min(ratio, 1.0))
    return max(0.0, score)
