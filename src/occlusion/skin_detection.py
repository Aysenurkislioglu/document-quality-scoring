"""
Ten rengi (skin-color) tabanlı occlusion tespiti — KONUMDAN BAĞIMSIZ.

`metrics.py`'deki OCR tabanlı yöntem yalnızca konumu ve beklenen formatı
ÖNCEDEN BİLİNEN alanlarda (örn. "Belge No") çalışır. Bu modül, belgenin
HERHANGİ bir yerinde parmak/el ile kapatılmış bir bölgeyi — konumunu
bilmeden — yakalamayı hedefler. Gerekçe için project_notes.md, "Occlusion
Aşama 1" bölümüne bakınız.

Yöntem: YCrCb renk uzayında klasik ten rengi eşiklemesi + bağlı bileşen
gürültü temizliği (glare modülündeki HSV eşiklemesiyle aynı aile).

ÖNEMLİ — DOĞRULAMA DURUMU: Bu yöntem, glare modülündeki başarısız "şekil
filtresi" denemesinden ders alınarak, ENTEGRE EDİLMEDEN ÖNCE sentetik
veriyle doğrulanmıştır (bkz. experiments/occlusion/run_skin_experiment.py
ve results/occlusion/skin_scores.csv). Bilinen sınırlama: literatürdeki
genel uyarıyla tutarlı olarak, farklı ten tonlarında ve aydınlatma
koşullarında güvenilirliği değişebilir — bkz. deney sonuçlarındaki ton
bazlı kırılım.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]

# Klasik YCrCb ten rengi aralığı (literatürde yaygın kullanılan yaklaşık
# sınırlar, örn. Chai & Ngan tarzı çalışmalar). Y (parlaklık) kasıtlı olarak
# sınırlanmıyor çünkü ten farklı aydınlatmalarda geniş bir parlaklık
# aralığına yayılabilir; asıl ayırt edici sinyal Cr/Cb (renk) kanallarında.
CR_MIN, CR_MAX = 133, 173
CB_MIN, CB_MAX = 77, 127


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def skin_mask(image: np.ndarray, min_component_area: int = 40) -> np.ndarray:
    """Ten rengi adayı piksellerin ikili maskesini üretir.

    Not: Gri tonlamalı (renksiz) bir görüntüde ten rengi TANIM GEREĞİ tespit
    edilemez (Cr/Cb kanalları renksiz görüntüde nötr/sabit kalır) — bu
    fonksiyon yalnızca RENKLİ (BGR) girdilerde anlamlıdır.
    """
    bgr = _to_bgr(image)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    raw_mask = (
        (cr >= CR_MIN) & (cr <= CR_MAX) & (cb >= CB_MIN) & (cb <= CB_MAX)
    ).astype(np.uint8) * 255

    if min_component_area <= 0:
        return raw_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    filtered = np.zeros_like(raw_mask)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_component_area:
            filtered[labels == label_id] = 255
    return filtered


def skin_occlusion_ratio(
    image: np.ndarray, roi: Optional[BBox] = None, min_component_area: int = 40
) -> float:
    """Ten rengi alanının, ilgilenilen bölgeye (roi) oranını hesaplar.

    Konumdan bağımsızdır — roi verilmezse tüm görüntü taranır. Yüksek oran =
    belgenin büyükçe bir kısmının parmak/el benzeri bir nesneyle kaplı
    olduğuna dair şüphe.
    """
    region = image
    if roi is not None:
        x0, y0, x1, y1 = roi
        region = image[y0:y1, x0:x1]

    mask = skin_mask(region, min_component_area=min_component_area)
    total_pixels = mask.shape[0] * mask.shape[1]
    if total_pixels == 0:
        return 0.0
    return int(np.count_nonzero(mask)) / total_pixels
