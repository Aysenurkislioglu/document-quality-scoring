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

Kapsamı GENİŞLETİLENLER:
- Glare: DÜZ BEYAZ KAĞITTA altı ayrı deneme (HSV+CC baseline, şekil
  filtresi, saf blok ML, bulanık-negatif eklenmiş ML, belge-geneli
  özellikli ML, eğitim/üretim uyumu düzeltmesi) yapıldı — HEPSİ ya
  yetersiz kaldı ya da bulanıklıkla karıştı (bkz. project_notes.md, "Glare
  Aşama 1" ve "Glare ML v1-v5"). Dış araştırma bunun NEDENİNİ açıkladı:
  klasik specular-highlight yöntemleri "dichromatic reflection model"e
  dayanır — parlamanın rengi (beyaza yakın) yüzeyin KENDİ rengiyle
  ayrışır; ama beyaz kağıtta yüzeyin "gerçek rengi" zaten beyaz, ayrışacak
  bir fark yok.

  KAPSAM KARARI: Bu projenin hedef kullanım alanı KİMLİK KARTI/PASAPORT
  benzeri RENKLİ zeminli belgeler olduğu için (kullanıcı kararı), DÜZ
  BEYAZ KAĞIT DESTEĞİ KAPSAM DIŞI BIRAKILDI — orada güvenilmez bir tahmin
  göstermek yerine glare "uygulanamaz" (score=None) olarak işaretlenir.
  RENKLİ zeminde ise aynı basit, DEĞİŞTİRİLMEMİŞ HSV+CC yöntemi (glare_ratio)
  test edildi ve MÜKEMMEL sonuç verdi: 3 renk şemasında, 72 görüntüde
  rho=1.00, hatalı-pozitif=0 (hem glare-yok hem bulanıklık+glare-yok
  durumunda) — bkz. results/glare/id_card_scores.csv. `has_colored_
  background()` (bkz. src/glare/metrics.py) her görüntü için hangi
  rejimde olduğumuzu tahmin eder.
- Occlusion: OCR tabanlı yöntem (metrics.py) hâlâ yalnızca ÖNCEDEN BİLİNEN,
  yapılandırılmış alanlarda (örn. "Belge No") çalışır ve bu genel akışa dahil
  değildir. Ancak KONUMDAN VE RENKTEN BAĞIMSIZ bir ek sinyal eklendi: blok-
  bazlı ML sınıflandırıcı (ml_detection.py, Random Forest) — parmak/el/
  sticker/kumaş gibi HERHANGİ bir yabancı nesneyle kapatılmış BİLİNMEYEN
  konumdaki alanları yakalar. Bu, önce denenen ve yalnızca ten rengine
  sınırlı olan skin_detection.py'nin genelleştirilmiş hâlidir (o modül hâlâ
  kodda duruyor ama fusion.py artık ML sürümünü kullanıyor).

  Doğrulama: 14 renk (ten tonu HARİÇ) ile eğitilip, 5 GÖRÜLMEMİŞ renkte
  (3 ten tonu + turkuaz + lacivert), hem DÜZ hem DOKULU/gürültülü
  varyantlarda test edildi (bkz. results/occlusion/ml_scores.csv). Sonuç:
  10 kombinasyonun HEPSİNDE rho=1.00, hatalı-pozitif=0. Model gerçekten
  renk+doku örüntüsünü öğrendi, belirli renkleri ezberlemedi. Bu yüzden
  varsayılan olarak nihai skora DAHİL edilir — ama yine de yalnızca
  SENTETİK yamalarla test edildiğini unutmayın; gerçek el/parmak dokusu,
  gölgeler, eklem kıvrımları ve gerçek fotoğraf koşulları henüz test
  edilmedi (bkz. project_notes.md, "Occlusion Aşama 2").
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.blur.metrics import compute_all_blur_metrics, laplacian_variance
from src.darkness.metrics import compute_all_darkness_metrics, darkest_block_mean
from src.glare.metrics import glare_ratio, has_colored_background
from src.occlusion.ml_detection import ml_occlusion_ratio
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
GLARE_CARD_BAD, GLARE_CARD_GOOD = 0.35, 0.0  # glare_ratio, YALNIZCA renkli zeminde
# (bkz. results/glare/id_card_scores.csv: severity=5 ortalama 0.33, max 0.48 —
# 0.35 bu aralığı makul şekilde kapsıyor).


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
    """Glare alt-skoru — YALNIZCA RENKLİ ZEMİNLİ belgeler (kimlik kartı/
    pasaport benzeri) için tasarlanmıştır.

    GEREKÇE / KAPSAM KARARI: Beyaz kağıtta altı ayrı deneme (HSV+CC, şekil
    filtresi, dört ML varyantı) hiçbiri güvenilir olmadı (bkz.
    project_notes.md, "Glare ML v1-v5"). Bu projenin hedef kullanım alanı
    KİMLİK/PASAPORT tipi belgeler olduğu için (kullanıcı kararı), beyaz
    kağıt desteği kapsam dışı bırakıldı — orada güvenilmez bir ML tahmini
    göstermek yerine net biçimde "uygulanamaz" denir. RENKLİ zeminde
    (`has_colored_background`) klasik HSV+CC (glare_ratio) kullanılır —
    72 görüntüde rho=1.00, hatalı-pozitif=0 ile doğrulandı (bkz.
    results/glare/id_card_scores.csv).
    """
    if not has_colored_background(image):
        return {
            "raw_value": None,
            "raw_label": "N/A",
            "score": None,
            "reliable": False,
            "applicable": False,
        }
    ratio = glare_ratio(image)
    return {
        "raw_value": ratio,
        "raw_label": "glare_ratio (renkli zemin)",
        "score": _linear_score(ratio, GLARE_CARD_BAD, GLARE_CARD_GOOD),
        "reliable": True,
        "applicable": True,
    }


def score_occlusion(image: np.ndarray) -> Dict[str, object]:
    """ML tabanlı occlusion alt-skoru (yüksek=kapanma yok=iyi).

    Konumdan VE renkten bağımsız çalışır (bkz. src/occlusion/ml_detection.py).
    Sentetik veriyle doğrulanmıştır: 5 GÖRÜLMEMİŞ renk/doku kombinasyonunun
    hepsinde rho=1.00, hatalı-pozitif=0 (bkz. results/occlusion/ml_scores.csv)
    — ama yalnızca sentetik yamalarla, bkz. modül docstring'indeki "Kapsamı
    GENİŞLETİLENLER" notu.
    """
    ratio = ml_occlusion_ratio(image)
    return {
        "raw_value": ratio,
        "raw_label": "ml_occlusion_ratio",
        "score": _linear_score(ratio, 1.0, 0.0),  # oran 0=iyi(100 puan), 1=kötü(0 puan)
    }


def compute_document_quality_score(image: np.ndarray) -> Dict[str, object]:
    """
    Bir belge görüntüsü için birleşik Document Quality Score (0-100) hesaplar.

    blur/darkness/skew/occlusion her zaman ortalamaya dahildir. Glare ise
    KOŞULLUDUR: yalnızca zemin RENKLİ ise (kimlik kartı/pasaport benzeri —
    bkz. `has_colored_background`) ortalamaya katılır. Düz beyaz kağıt bu
    projenin kapsamı dışıdır (bkz. modül docstring'i) — bu durumda glare
    "uygulanamaz" (score=None) olarak işaretlenir, tahmini bir sayı
    üretilmez.

    Args:
        image: BGR (OpenCV formatında) numpy array — tek bir belge fotoğrafı.

    Returns:
        dict: overall_score, components (her modül için ham değer + alt-skor),
        ve kapsam notları.
    """
    glare = score_glare(image)
    components: Dict[str, Dict[str, object]] = {
        "blur": score_blur(image),
        "darkness": score_darkness(image),
        "skew": score_skew(image),
        "occlusion": score_occlusion(image),
        "glare": glare,
    }

    fused = [c["score"] for key, c in components.items() if key != "glare"]
    if glare.get("reliable"):
        fused.append(glare["score"])
    overall = float(np.mean(fused))

    return {
        "overall_score": overall,
        "components": components,
        "occlusion_note": (
            "ML tabanlı occlusion sinyali (blok-bazlı Random Forest) dahil "
            "edildi — konumdan VE renkten bağımsız çalışır, parmak/el/sticker "
            "gibi herhangi bir yabancı nesneyi yakalar. Ancak OCR tabanlı, "
            "alan-bazlı occlusion yöntemi (örn. 'Belge No' doğrulaması) hâlâ "
            "yalnızca önceden bilinen şablonlarla çalışır; bu genel "
            "yüklemede uygulanmadı."
        ),
        "glare_note": (
            "Glare tespiti bu projede KİMLİK KARTI/PASAPORT benzeri RENKLİ "
            "zeminli belgeler için kapsandı (72 test görüntüsünde rho=1.00, "
            "hatalı-pozitif=0). Düz beyaz kağıt kapsam dışıdır — bu durumda "
            "glare 'uygulanamaz' olarak işaretlenir, tahmini skor üretilmez."
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
            "methods": {
                "HSV + Connected Components (glare_ratio)": glare_ratio(image),
            },
            "used_in_overall": "HSV + Connected Components (glare_ratio) — yalnızca renkli zeminde",
            "note": (
                "Bu proje glare tespitini KİMLİK KARTI/PASAPORT benzeri RENKLİ "
                "zeminli belgeler için kapsıyor (kullanıcı kararı) — düz beyaz "
                "kağıt kapsam dışı bırakıldı (bkz. project_notes.md, altı ayrı "
                "başarısız deneme). Renkli zeminde bu basit yöntem mükemmel "
                "çalışıyor (rho=1.00, hatalı-pozitif=0, bkz. "
                "results/glare/id_card_scores.csv) — glare'in rengi (beyaza "
                "yakın) zeminin KENDİ renginden ayrışabiliyor (dichromatic "
                "reflection model)."
            ),
        },
        "occlusion": {
            "methods": {
                "ML — blok-bazlı Random Forest (ml_occlusion_ratio)": ml_occlusion_ratio(image),
                "YCrCb ten rengi eşiklemesi (skin_occlusion_ratio)": skin_occlusion_ratio(image),
            },
            "used_in_overall": "ML — blok-bazlı Random Forest (ml_occlusion_ratio)",
            "note": (
                "İkisi de konumdan bağımsız çalışır (OCR tabanlı yöntemin "
                "aksine, belgenin HERHANGİ bir yerinde arar). ML sürümü, "
                "yalnızca ten rengine değil HERHANGİ bir renk/dokudaki "
                "kapanmaya genelleyebildiği için (5 görülmemiş renk/dokuda "
                "rho=1.00 — bkz. results/occlusion/ml_scores.csv) füzyonda "
                "kullanılan budur; ten rengi yöntemi daha basit/hızlı ama "
                "yalnızca ten rengiyle sınırlıdır."
            ),
        },
    }
