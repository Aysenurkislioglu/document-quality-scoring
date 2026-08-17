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

from src.blur.metrics import compute_all_blur_metrics, laplacian_variance
from src.darkness.metrics import compute_all_darkness_metrics, darkest_block_mean
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


def compare_module_methods(image: np.ndarray) -> Dict[str, object]:
    """
    Her modülün BİRDEN FAZLA yöntemi varsa, hepsini aynı görüntü üzerinde
    hesaplayıp yan yana döndürür — "yöntem seçimi sonucu nasıl etkiliyor"
    sorusuna, tek bir fotoğraf üzerinden görsel/sayısal cevap vermek içindir.

    Nihai skora (compute_document_quality_score) etkisi YOKTUR; yalnızca
    karşılaştırma/gösterim amaçlıdır. Yöntemlerin genel (çok belge üzerinde,
    bilinen bozulma şiddetiyle) nasıl karşılaştırıldığına dair asıl kanıt
    results/<modül>/scores.csv ve results/<modül>/plots/ altındadır — bu
    fonksiyon yalnızca TEK bir yüklenen görüntü için hızlı bir özet sağlar.
    """
    blur_all = compute_all_blur_metrics(image)
    darkness_all = compute_all_darkness_metrics(image)

    hough_angle = estimate_skew_hough(image)
    projection_angle = estimate_skew_projection_profile(image)

    return {
        "blur": {
            "methods": {
                "Laplacian Variance": blur_all["laplacian_variance"],
                "Tenengrad": blur_all["tenengrad"],
                "Gradient Magnitude (ortalama)": blur_all["gradient_magnitude_mean"],
            },
            "used_in_overall": "Laplacian Variance",
            "note": "Üçü de aynı yönde okunur: yüksek değer = daha keskin/net.",
        },
        "darkness": {
            "methods": {
                "Global ortalama parlaklık": darkness_all["mean"],
                "Global medyan parlaklık": darkness_all["median"],
                "P5 (en karanlık %5)": darkness_all["p5"],
                "P25": darkness_all["p25"],
                "En karanlık blok ortalaması": darkness_all["darkest_block_mean"],
                "Ortalama yerel kontrast": darkness_all["mean_local_contrast"],
            },
            "used_in_overall": "En karanlık blok ortalaması",
            "note": (
                "Global ortalama, belgenin küçük bir bölgesindeki lokal "
                "karanlığı (örn. gölge düşmüş tek bir köşe) çoğu zaman "
                "gizler; en karanlık blok bunu yakalar — bu yüzden füzyonda "
                "o kullanılıyor. Aradaki farkı görmek için: bu iki değer "
                "birbirine yakınsa karanlık YAYGIN, uzaksa karanlık "
                "LOKALİZE demektir."
            ),
        },
        "skew": {
            "methods": {
                "Hough Transform": hough_angle,
                "Projection Profile": projection_angle,
            },
            "used_in_overall": "Hough Transform (bulamazsa Projection Profile'a düşer)",
            "note": (
                "results/skew/scores.csv deneyinde Hough daha düşük ortalama "
                "hata verdi (MAE≈0.9°) — Projection Profile az metinli "
                "belgelerde daha çok şaşıyor (MAE≈1.8°). İki değer birbirinden "
                "çok uzaksa, bu görüntüde yöntemlerden biri muhtemelen yanılıyor."
            ),
        },
        "glare": {
            "methods": {"HSV + Connected Components (glare_ratio)": glare_ratio(image)},
            "used_in_overall": "(varsayılan: hiçbiri, isteğe bağlı dahil edilir)",
            "note": (
                "Bu modülde şu an tek yöntem var, karşılaştırma yok. "
                "Kendisi de project_notes.md'de yetersiz bulunmuş durumda."
            ),
        },
    }
