"""
Feature Fusion — basit, birleşik "Document Quality Score" (0-100).

ÖNEMLİ / DÜRÜSTLÜK NOTU:
Bu modül, Aşama 5'in (project_notes.md'de tanımlı "Feature Fusion / ML Scoring")
İLK, en basit hâlidir. Aşağıdaki normalizasyon eşikleri (bad/good sınırları)
GERÇEK ETİKETLİ VERİYLE ÖĞRENİLMEMİŞTİR — literatürden ve deneylerimizden
(results/*/*.csv) esinlenen KABA, GEÇİCİ sezgisel (heuristic) değerlerdir.
Yani ürettiği "0-100 skor", bir ML modelinin doğruluk oranı DEĞİLDİR; her
modülün kendi ham metriğini okunabilir bir ölçeğe taşıyan bir özettir.
Gerçek kalibrasyon, gerçek (sentetik olmayan) etiketli belge verisiyle
yapılmalıdır — bkz. project_notes.md, "Bir sonraki adım".

Kapsam dışı bırakılanlar:
- Glare: project_notes.md'de baseline'ın (HSV+CC) yetersiz bulunduğu
  belgelenmiştir (severity=0'da bile ~%85 false-positive glare oranı —
  bkz. results/glare/false_positive_baseline.csv). Bu yüzden varsayılan
  olarak nihai skora KATILMAZ; yalnızca bilgi amaçlı ayrıca hesaplanır.
- Occlusion: yöntem yalnızca ÖNCEDEN BİLİNEN, yapılandırılmış alanlar için
  çalışır (örn. "Belge No" alanının tam konumu/beklenen uzunluğu bilinmeli).
  Genel/rastgele yüklenen bir belge fotoğrafında bu bilgi mevcut olmadığından
  bu genel akışa dahil edilmemiştir.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.blur.metrics import laplacian_variance
from src.darkness.metrics import darkest_block_mean
from src.glare.metrics import glare_ratio, glare_score
from src.skew.metrics import estimate_skew_hough, estimate_skew_projection_profile

# Heuristik normalizasyon sınırları: (bu değerde ve ötesi = 0 puan, bu değerde
# ve ötesi = 100 puan). Kaynak: research/literature_review.md + results/ altındaki
# deney çıktıları (örn. blur/scores.csv'deki gözlemlenen değer aralıkları).
BLUR_BAD, BLUR_GOOD = 20.0, 300.0          # laplacian_variance
DARKNESS_BAD, DARKNESS_GOOD = 40.0, 180.0  # darkest_block_mean
SKEW_BAD, SKEW_GOOD = 20.0, 0.0            # |açı|, derece (ters yönlü: 0=iyi)


def _linear_score(value: float, bad: float, good: float) -> float:
    """`value`'yu [bad, good] aralığında 0-100 puana doğrusal eşler (clip'li).

    `bad` ve `good` herhangi bir sırada olabilir — hangisi 0 hangisi 100
    puana karşılık geliyorsa ona göre yön otomatik ayarlanır (örn. skew'de
    düşük açı iyi, blur'da yüksek varyans iyi).
    """
    if good == bad:
        return 100.0
    t = (value - bad) / (good - bad)
    t = max(0.0, min(1.0, t))
    return 100.0 * t


def score_blur(image: np.ndarray) -> Dict[str, object]:
    """Laplacian Variance tabanlı bulanıklık alt-skoru (yüksek=keskin=iyi)."""
    variance = laplacian_variance(image)
    return {
        "raw_value": variance,
        "raw_label": "laplacian_variance",
        "score": _linear_score(variance, BLUR_BAD, BLUR_GOOD),
    }


def score_darkness(image: np.ndarray) -> Dict[str, object]:
    """En karanlık blok ortalaması tabanlı karanlık alt-skoru (yüksek=aydınlık=iyi)."""
    dbm = darkest_block_mean(image)
    return {
        "raw_value": dbm,
        "raw_label": "darkest_block_mean",
        "score": _linear_score(dbm, DARKNESS_BAD, DARKNESS_GOOD),
    }


def score_skew(image: np.ndarray) -> Dict[str, object]:
    """Eğiklik açısı alt-skoru (0 derece = iyi). Önce Hough, olmazsa Projection Profile."""
    angle = estimate_skew_hough(image)
    method = "hough"
    if angle is None:
        angle = estimate_skew_projection_profile(image)
        method = "projection_profile"
    return {
        "raw_value": angle,
        "raw_label": f"skew_angle_degrees ({method})",
        "score": _linear_score(abs(angle), SKEW_BAD, SKEW_GOOD),
    }


def score_glare(image: np.ndarray) -> Dict[str, object]:
    """Glare alt-skoru — GÜVENİLMEZ, bkz. modül docstring'i. Bilgi amaçlıdır."""
    ratio = glare_ratio(image)
    return {
        "raw_value": ratio,
        "raw_label": "glare_ratio",
        "score": glare_score(ratio),
        "reliable": False,
    }


def compute_document_quality_score(
    image: np.ndarray, include_glare: bool = False
) -> Dict[str, object]:
    """
    Bir belge görüntüsü için birleşik Document Quality Score (0-100) hesaplar.

    Args:
        image: BGR (OpenCV formatında) numpy array — tek bir belge fotoğrafı.
        include_glare: True ise, güvenilmez olduğu bilinen glare alt-skoru da
            ortalamaya dahil edilir (varsayılan: hariç tutulur).

    Returns:
        dict: overall_score, components (her modül için ham değer + alt-skor),
        ve kapsam dışı bırakılanlara dair notlar.
    """
    components: Dict[str, Dict[str, object]] = {
        "blur": score_blur(image),
        "darkness": score_darkness(image),
        "skew": score_skew(image),
    }

    glare = score_glare(image)
    components["glare"] = glare

    fused = [components["blur"]["score"], components["darkness"]["score"], components["skew"]["score"]]
    if include_glare:
        fused.append(glare["score"])

    overall = float(np.mean(fused))

    return {
        "overall_score": overall,
        "components": components,
        "glare_included_in_overall": include_glare,
        "occlusion_note": (
            "Occlusion modülü yalnızca önceden bilinen, yapılandırılmış alanlar "
            "(örn. 'Belge No') için çalışır; bu genel yüklemede otomatik "
            "uygulanmadı."
        ),
        "calibration_note": (
            "Bu skor, gerçek etiketli veriyle kalibre edilmiş bir ML modelinin "
            "çıktısı değildir; literatür + sentetik deneylerden esinlenen "
            "geçici/sezgisel eşiklerle üretilmiştir (bkz. project_notes.md)."
        ),
    }
