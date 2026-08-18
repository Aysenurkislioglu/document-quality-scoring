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


def has_colored_background(
    image: np.ndarray,
    saturated_pixel_threshold: int = 15,
    min_fraction: float = 0.05,
    value_ceiling: int = 245,
) -> bool:
    """
    Belgenin zemininin (kimlik kartı/pasaport gibi) RENKLİ mi, yoksa düz
    beyaz/gri kağıt mı olduğunu tahmin eder.

    GEREKÇE: `glare_ratio` (HSV+CC), düz beyaz kağıtta glare'i zeminden
    ayırt edemiyor — ikisi de "yüksek parlaklık + düşük saturasyon" (bkz.
    project_notes.md, "Glare Aşama 1" ve "Glare ML v1-v5", altı ayrı
    başarısız deneme). Ancak RENKLİ zeminlerde (kimlik kartı, pasaport)
    AYNI yöntem mükemmel çalışıyor (bkz. project_notes.md, "Glare — Kimlik
    Kartı Zemini Deneyi": rho=1.00, hatalı-pozitif=0, 3 renk şemasında,
    hem glare-yok hem bulanıklık+glare-yok durumunda) — çünkü zeminin
    "gerçek rengi" saturasyonlu olduğu için glare (fiziksel olarak beyaza
    yakın) ondan gerçekten ayrışabiliyor (dichromatic reflection model
    prensibi). Bu fonksiyon, hangi durumda olduğumuzu görüntüden tahmin
    eder — `src/scoring/fusion.py` bu bilgiyle glare skoruna ne kadar
    güveneceğine karar verir.

    v2 DÜZELTMESİ (kırpılmış piksel filtresi): Kullanıcı bildirdi — aşırı
    parlak (aşırı pozlanmış) gerçek fotoğraflarda, gerçekten renkli bir
    kimlik kartı bile "beyaz kağıt" sanılıp glare hiç hesaplanmıyor,
    darkness da daha az sağlam bir yönteme düşüyordu. Kök neden: genel bir
    parlaklık artışı (aşırı pozlama), rengi MATEMATİKSEL olarak yıkıyor
    (HSV'de S = (max-min)/max; kanallar 255'e yaklaşıp kırpıldıkça oran
    bozuluyor) — eski yöntem TÜM piksellere bakıyordu, bu yüzden aşırı
    pozlanmış (255'e kırpılmış) büyük alanlar saturasyonu yapay olarak
    düşürüyordu. Düzeltme: yalnızca KIRPILMAMIŞ piksellere (V < 245) bakılır
    — bu piksellerdeki renk hâlâ güvenilir bir sinyal taşıyor. Doğrulama:
    aynı kimlik kartı +150 parlaklık artışına kadar test edildi, hepsinde
    doğru "renkli" tespit edildi; gerçek beyaz kağıt belgelerde yanlışlıkla
    "renkli" denmedi (bkz. project_notes.md, "has_colored_background —
    aşırı pozlama hatası").

    Args:
        image: BGR veya gri numpy array.
        saturated_pixel_threshold: HSV S kanalında bu değerin ÜZERİ
            "renkli" piksel sayılır.
        min_fraction: KIRPILMAMIŞ piksellerin en az bu kadarı renkliyse,
            zemin "renkli" kabul edilir.
        value_ceiling: HSV V kanalında bu değerin ÜZERİ "kırpılmış/aşırı
            pozlanmış" sayılıp analiz dışı bırakılır.
    """
    bgr = _to_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    not_clipped = hsv[:, :, 2] < value_ceiling
    total_pixels = hsv.shape[0] * hsv.shape[1]

    if not_clipped.sum() < 0.02 * total_pixels:
        # Görüntünün neredeyse tamamı kırpılmışsa (belge tümüyle beyaza
        # yakınsa — ya gerçekten beyaz kağıt ya da tam glare/aşırı pozlama),
        # ayırt edici bir piksel havuzu kalmıyor; eski (tüm görüntü) yönteme
        # düşülür.
        fraction_saturated = float((hsv[:, :, 1] > saturated_pixel_threshold).mean())
        return fraction_saturated >= min_fraction

    saturated_and_visible = (hsv[:, :, 1] > saturated_pixel_threshold) & not_clipped
    fraction_saturated = float(saturated_and_visible.sum()) / float(not_clipped.sum())
    return fraction_saturated >= min_fraction
