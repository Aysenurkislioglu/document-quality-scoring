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
  bkz. results/glare/false_positive_baseline.csv). Ayrıca bir "şekil filtresi"
  düzeltmesi denenmiş, gerçek veriyle test edilmiş ve BAŞARISIZ olduğu
  görülmüştür (bkz. project_notes.md, "Glare Aşama 1 denemesi"). Bu yüzden
  varsayılan olarak nihai skora KATILMAZ; yalnızca bilgi amaçlı hesaplanır.

Kapsamı GENİŞLETİLENLER:
- Occlusion: OCR tabanlı yöntem (metrics.py) hâlâ yalnızca ÖNCEDEN BİLİNEN,
  yapılandırılmış alanlarda (örn. "Belge No") çalışır ve bu genel akışa dahil
  değildir. Ancak KONUMDAN BAĞIMSIZ bir ek sinyal eklendi: ten rengi (skin-
  color) tabanlı tespit (skin_detection.py) — parmak/el ile kapatılmış
  BİLİNMEYEN konumdaki alanları yakalar. Sentetik veriyle (3 ten tonu, 216
  görüntü) doğrulanmıştır: rho=1.00, hatalı-pozitif=0 (bkz.
  results/occlusion/skin_scores.csv). Bu yüzden varsayılan olarak nihai
  skora DAHİL edilir — ama yalnızca SENTETİK, düz renkli yamalarla test
  edildiğini unutmayın; gerçek el/parmak dokusu, farklı aydınlatma ve
  ten-rengi-benzeri arka plan nesneleri (örn. ahşap masa) henüz test
  edilmedi.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.blur.metrics import compute_all_blur_metrics, laplacian_variance
from src.darkness.metrics import compute_all_darkness_metrics, darkest_block_mean
from src.glare.metrics import glare_ratio, glare_score
from src.occlusion.skin_detection import skin_occlusion_ratio
from src.skew.metrics import estimate_skew_hough, estimate_skew_projection_profile

# Heuristik normalizasyon sınırları — bu değerde ve ötesi = 0 puan, bu değerde
# ve ötesi = 100 puan. results/*/scores.csv'deki GERÇEK ölçülmüş dağılımlara göre
# kalibre edilmiştir (bkz. project_notes.md, "Aşama 5: Feature Fusion — Kalibrasyon
# Düzeltmesi" — önceki sürümdeki keyfi sabitlerin sebep olduğu satürasyon hatası
# için). Yine de GERÇEK ETİKETLİ VERİYLE öğrenilmiş değildir — bkz. modül docstring'i.
BLUR_BAD, BLUR_GOOD = 1.0, 6000.0          # laplacian_variance (LOG ölçekte, aşağıya bkz.)
DARKNESS_BAD, DARKNESS_GOOD = 50.0, 110.0  # darkest_block_mean (block_size=16 ile)
DARKNESS_BLOCK_SIZE = 16  # experiments/darkness/run_experiment.py ile AYNI olmalı —
# küçük (~105x26px) kimlik alanlarını yakalamak için deneyde bilinçli olarak
# fonksiyonun varsayılanından (32) küçük seçilmiş. Bu proje bu değeri
# görmezden gelip varsayılanı kullanmıştı; sonuç, küçük lokal karanlık
# bölgelerin komşu aydınlık piksellerle "sulanıp" gizlenmesiydi (bkz.
# project_notes.md, kalibrasyon düzeltmesi notu).
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


def _log_linear_score(value: float, bad: float, good: float) -> float:
    """`_linear_score` ile aynı, ama önce log1p ile logaritmik ölçeğe taşır.

    Laplacian Variance gibi metrikler bozulma şiddeti arttıkça DOĞRUSAL değil,
    yaklaşık ÜSTEL biçimde küçülür (results/blur/scores.csv'de severity 0→8
    arası 8287 → 1.3 gibi, birkaç büyüklük mertebesi kat eden bir düşüş).
    Böyle bir metriği doğrudan doğrusal ölçeklemek, en hafif bozulmalarda bile
    skorun anında 0/100'e "satüre olmasına" (yapışmasına) yol açar — bu
    projede tam olarak yaşanmış, tespit edilip düzeltilmiş bir hatadır (bkz.
    project_notes.md). log1p, büyük değerleri sıkıştırıp küçük değerlere daha
    fazla "yer" açarak orta şiddetlerde de anlamlı ayrım sağlar.
    """
    return _linear_score(np.log1p(max(value, 0.0)), np.log1p(bad), np.log1p(good))


def score_blur(image: np.ndarray) -> Dict[str, object]:
    """Laplacian Variance tabanlı bulanıklık alt-skoru (yüksek=keskin=iyi).

    Log ölçekte normalize edilir — bkz. `_log_linear_score` docstring'i.
    """
    variance = laplacian_variance(image)
    return {
        "raw_value": variance,
        "raw_label": "laplacian_variance",
        "score": _log_linear_score(variance, BLUR_BAD, BLUR_GOOD),
    }


def score_darkness(image: np.ndarray) -> Dict[str, object]:
    """En karanlık blok ortalaması tabanlı karanlık alt-skoru (yüksek=aydınlık=iyi).

    BİLİNEN SINIRLAMA: darkest_block_mean, metin yoğunluğuyla karışır (koyu
    mürekkep pikselleri yoğun bir blok, gerçek bir aydınlatma sorunu olmasa
    bile düşük ortalama üretir) — results/darkness/scores_local.csv'de HİÇ
    karartma uygulanmamış (severity=0) belgelerde bile bu değerin 52-168
    arasında değiştiği gözlemlenmiştir. Aşağıdaki eşikler bu gerçek dağılıma
    göre kalibre edilmiştir, ama bu confound (karışma) tamamen çözülmüş
    değildir — font/metin yoğunluğu çok yüksek temiz bir belge yine de düşük
    puan alabilir. Kalıcı çözüm, project_notes.md'de planlanan ML regresyon
    katmanının bunu diğer özelliklerle (örn. metin yoğunluğu) birlikte
    öğrenmesidir.
    """
    dbm = darkest_block_mean(image, block_size=DARKNESS_BLOCK_SIZE)
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


def score_occlusion_skin(image: np.ndarray) -> Dict[str, object]:
    """Ten rengi tabanlı occlusion alt-skoru (yüksek=kapanma yok=iyi).

    Konumdan bağımsız çalışır (bkz. src/occlusion/skin_detection.py). Sentetik
    veriyle doğrulanmıştır (rho=1.00, 3 ten tonu) ama YALNIZCA gerçek fotoğraf
    değil, düz renkli sentetik yamalarla — bkz. modül docstring'indeki
    "Kapsamı GENİŞLETİLENLER" notu.
    """
    ratio = skin_occlusion_ratio(image)
    return {
        "raw_value": ratio,
        "raw_label": "skin_occlusion_ratio",
        "score": _linear_score(ratio, 1.0, 0.0),  # oran 0=iyi(100 puan), 1=kötü(0 puan)
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
        "occlusion_skin": score_occlusion_skin(image),
    }

    glare = score_glare(image)
    components["glare"] = glare

    fused = [
        components["blur"]["score"],
        components["darkness"]["score"],
        components["skew"]["score"],
        components["occlusion_skin"]["score"],
    ]
    if include_glare:
        fused.append(glare["score"])

    overall = float(np.mean(fused))

    return {
        "overall_score": overall,
        "components": components,
        "glare_included_in_overall": include_glare,
        "occlusion_note": (
            "Ten rengi tabanlı occlusion sinyali (occlusion_skin) dahil "
            "edildi — konumdan bağımsız çalışır, parmak/el benzeri kapanmayı "
            "yakalar. Ancak OCR tabanlı, alan-bazlı occlusion yöntemi "
            "(örn. 'Belge No' doğrulaması) hâlâ yalnızca önceden bilinen "
            "şablonlarla çalışır; bu genel yüklemede uygulanmadı."
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
    darkness_all = compute_all_darkness_metrics(image, block_size=DARKNESS_BLOCK_SIZE)

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
        "occlusion_skin": {
            "methods": {"YCrCb ten rengi eşiklemesi (skin_occlusion_ratio)": skin_occlusion_ratio(image)},
            "used_in_overall": "YCrCb ten rengi eşiklemesi (skin_occlusion_ratio)",
            "note": (
                "Bu modülde şu an tek yöntem var, karşılaştırma yok. Konumdan "
                "bağımsız çalışır — OCR tabanlı occlusion yönteminin (yalnızca "
                "bilinen alanlarda çalışan) tersine, belgenin HERHANGİ bir "
                "yerinde ten rengi arar."
            ),
        },
    }
