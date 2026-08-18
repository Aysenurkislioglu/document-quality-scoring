"""
ML tabanlı, BAĞLAM-FARKINDA (context-aware) glare tespiti.

`metrics.py`'deki HSV+CC baseline'ı ve onu düzeltmeye yönelik "şekil filtresi"
denemesi (bkz. project_notes.md, "Glare Aşama 1") başarısız oldu; ardından
saf blok-bazlı bir ML denemesi de (bkz. project_notes.md, "Glare ML
denemesi") aynı temel soruna çarptı: tam doymuş bir glare bölgesi ile boş
beyaz kağıt, TEK BİR BLOĞUN kendi pikselleri üzerinden AYIRT EDİLEMEZ —
ikisi de "düz ve parlak". Bu, mühendislik eksikliği değil, tek kare + tek
blok üzerinden bilgi kuramsal bir sınırdı.

Bu modül bu sınırı BAĞLAM (context) ekleyerek aşıyor: her bloğun kendi
özelliklerine (parlaklık, doku) ek olarak, AYNI SATIRDAKİ diğer bloklara
göre ne kadar "beklenmedik şekilde düzleştiğini" de bir özellik olarak
kullanır (bkz. `_row_context_features`). Sezgi: boş bir satırın TAMAMI zaten
düzdür (fark yok = normal); ama bir metin satırının ortasında TEK bir blok
aniden düzleşmişse, bu güçlü bir "burası silinmiş/parlamış" sinyalidir —
konumdan bağımsızdır (glare'in belgede NEREDE olacağını varsaymaz), bu
yüzden Rodin & Orlov (2019)'un "stroke histogram" fikrine benzer ama çok
daha basit bir yaklaşımdır.

DOĞRULAMA: Eğitimde HİÇ kullanılmayan, farklı random seed ile üretilmiş
belgelerde test edilmiştir — bkz. experiments/glare/run_ml_experiment.py ve
results/glare/ml_scores.csv (ortalama rho=0.99, severity=0 hatalı-pozitifi
%1.1).

BİLİNEN SINIRLAMA (v1, kasıtlı olarak ÇÖZÜLMEDEN bırakıldı): Bu model,
belgenin TAMAMI zaten ağır bulanıksa (blur) glare ile karıştırabiliyor —
çünkü hem ağır blur hem glare, blokları benzer şekilde "düz" hale getiriyor.
İki düzeltme denemesi (bulanık negatif örnekler eklemek, ve belge-geneli
bulanıklık seviyesini ayrı bir özellik olarak eklemek) yapıldı; ikisi de bu
sınırlamayı gidereyim derken ASIL glare tespit performansını BOZDU (ikinci
denemede model glare'i neredeyse hiç yakalayamaz hale geldi). Bu yüzden
BİLİNÇLİ OLARAK v1'e (yalnızca satır-bağlamlı, 6 özellikli) geri dönüldü —
bkz. project_notes.md, "Glare ML v2/v3 denemeleri (geri alındı)". Üretimde
kullanıcıya bu sınırlama açıkça bildirilir.

Eğitim scripti: experiments/glare/train_glare_classifier.py
Model dosyası: src/glare/models/glare_rf.joblib
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]

BLOCK_SIZE = 16
MODEL_PATH = Path(__file__).parent / "models" / "glare_rf.joblib"

FEATURE_NAMES = ["mean", "std", "laplacian_var", "row_deficit", "row_deficit_ratio", "row_median_lap"]

_model_cache = None


def _block_stats(gray_image: np.ndarray, y0: int, x0: int, size: int) -> Optional[Tuple[float, float, float]]:
    block = gray_image[y0 : y0 + size, x0 : x0 + size]
    if block.size == 0:
        return None
    lap_var = float(cv2.Laplacian(block, cv2.CV_64F).var())
    return float(block.mean()), float(block.std()), lap_var


def extract_context_features(gray_image: np.ndarray, roi: Optional[BBox] = None, block_size: int = BLOCK_SIZE):
    """Her blok için [mean, std, lap_var, row_deficit, row_deficit_ratio,
    row_median_lap] özelliklerini ve konumunu döndürür.

    row_deficit: bloğun Laplacian varyansının, AYNI SATIRDAKİ diğer blokların
    medyanından ne kadar DÜŞÜK olduğu (0 = satır ortalamasına uygun ya da
    üstü, yüksek = satırın geri kalanına göre şüpheli derecede düz).
    """
    if roi is not None:
        x0, y0, x1, y1 = roi
    else:
        x0, y0 = 0, 0
        y1, x1 = gray_image.shape[:2]

    features: List[List[float]] = []
    positions: List[Tuple[int, int]] = []

    for by in range(y0, max(y0, y1 - block_size), block_size):
        row_entries = []
        for bx in range(x0, max(x0, x1 - block_size), block_size):
            stats = _block_stats(gray_image, by, bx, block_size)
            if stats is not None:
                row_entries.append((bx, stats))
        if not row_entries:
            continue

        lap_vars = np.array([s[2] for _, s in row_entries])
        row_median_lap = float(np.median(lap_vars))

        for bx, (mean, std, lap) in row_entries:
            deficit = max(0.0, row_median_lap - lap)
            deficit_ratio = deficit / (row_median_lap + 1e-6)
            features.append([mean, std, lap, deficit, deficit_ratio, row_median_lap])
            positions.append((bx, by))

    return features, positions


def _load_model():
    global _model_cache
    if _model_cache is None:
        import joblib

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model bulunamadı: {MODEL_PATH}. Önce "
                "experiments/glare/train_glare_classifier.py çalıştırılmalı."
            )
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


# v5 düzeltmesi — FİZİKSEL MAKUL ÜST SINIR: Kendi sentetik üretimimizde
# (generate_glare_documents.py, TARGET_AREA_FRACTIONS) gerçek glare ASLA
# içerik alanının %22'sinden fazlasını kaplamıyor — glare fiziksel olarak
# yerel bir ışık yansımasıdır, sayfanın büyük kısmını kaplaması gerçekçi
# değildir. Model tahmini bu makul sınırı (biraz payla %25) aşıyorsa, bu
# muhtemelen GERÇEK GLARE DEĞİL, belgenin geneli zaten düz/bulanık olduğu
# için modelin yanıldığı anlamına gelir (bkz. project_notes.md, "Glare ML
# v5"). Bu durumda tahmin bu sınıra kırpılır. Test verisiyle doğrulandı:
# gerçek glare skorlarının %100'ü bu sınırın altında kalıyor (etkilenmiyor),
# bulanıklık kaynaklı hatalı-pozitiflerin ~%80'i bu sınırın üzerinde
# olduğu için bastırılıyor (bkz. results/glare/ml_scores.csv vs.
# ml_blur_false_positive_check.csv).
MAX_PLAUSIBLE_GLARE_RATIO = 0.25


def glare_ml_ratio(image: np.ndarray, roi: Optional[BBox] = None) -> float:
    """Görüntüdeki (veya roi içindeki) bloklardan kaçının glare olarak
    sınıflandırıldığının oranını döndürür (0-1). Bağlam-farkında — bkz.
    modül docstring'i. Fiziksel makul üst sınırla kırpılır — bkz.
    MAX_PLAUSIBLE_GLARE_RATIO.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    features, _ = extract_context_features(gray, roi)
    if not features:
        return 0.0
    model = _load_model()
    preds = model.predict(features)
    raw_ratio = float(np.mean(preds))
    return min(raw_ratio, MAX_PLAUSIBLE_GLARE_RATIO)
