"""
ML tabanlı, konumdan VE renkten bağımsız occlusion tespiti.

`skin_detection.py`'nin genelleştirilmiş hâli: o modül yalnızca SABİT bir
renk aralığını (ten rengi) arıyordu — sticker, kumaş, plastik gibi farklı
renk/dokudaki kapanmaları yakalayamazdı. Bu modül bunun yerine, görüntüyü
küçük bloklara ayırıp her blok için renk + doku özellikleri çıkarır ve
önceden eğitilmiş bir Random Forest sınıflandırıcıyla "bu blok normal
kağıt/metin yüzeyine mi, yoksa yabancı bir nesneye mi benziyor?" sorusunu
sorar.

DOĞRULAMA (v1): 10 farklı renk (ten tonu HARİÇ) ile eğitilip, 5 GÖRÜLMEMİŞ
renkte (3 ten tonu + turkuaz + lacivert, hem düz hem dokulu/gürültülü
varyantlarda) test edilmiştir. Sonuç: hepsinde rho=1.00, hatalı-pozitif≈0
(bkz. project_notes.md, "Occlusion Aşama 2"). Bu, sınıflandırıcının
gerçekten renk+doku örüntüsünü öğrendiğini, yalnızca ezberlemediğini
gösterir.

BİLİNEN HATA VE v2 DÜZELTMESİ ("belgenin kendi rengi" bağlamı): v1, yalnızca
DÜZ BEYAZ/GRİ zeminli belgelerle eğitildiği için, kullanıcı gerçek RENKLİ
kimlik kartı yükleyince kartın kendi tasarımının TAMAMINI (fotoğraf, renkli
zemin) "yabancı nesne" sanıp oranı ~%95-99'a çıkarıyordu (bkz.
project_notes.md, "Occlusion — renkli zemin hatası"). Kök neden glare'deki
ile aynı aile: model yalnızca MUTLAK renk/dokuya bakıyordu, belgenin
KENDİ tipik renginin ne olduğunu bilmiyordu. Çözüm: her bloğa, o bloğun
renginin BELGENİN KENDİ MEDYAN (baskın) rengine ne kadar UZAK olduğunu da
(`color_dist_from_doc_median`) bir özellik olarak eklemek — böylece "bu blok
kartın kendi renk şemasına uyuyor mu, yoksa gerçekten farklı bir nesne mi"
ayrımı yapılabiliyor.

Eğitim scripti: experiments/occlusion/train_occlusion_classifier.py
Model dosyası: src/occlusion/models/occlusion_rf.joblib
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]

BLOCK_SIZE = 16  # darkness modülüyle aynı granülerlik (küçük alanları yakalamak için)
MODEL_PATH = Path(__file__).parent / "models" / "occlusion_rf.joblib"

FEATURE_NAMES = [
    "b_mean", "g_mean", "r_mean", "b_std", "g_std", "r_std", "gray_std", "laplacian_var",
    "color_dist_from_doc_median",
]

_model_cache = None


def _block_stats(bgr_image: np.ndarray, y0: int, x0: int, size: int) -> Optional[Tuple]:
    block = bgr_image[y0 : y0 + size, x0 : x0 + size]
    if block.size == 0:
        return None
    b = block[:, :, 0].astype(np.float64)
    g = block[:, :, 1].astype(np.float64)
    r = block[:, :, 2].astype(np.float64)
    gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return (
        float(b.mean()), float(g.mean()), float(r.mean()),
        float(b.std()), float(g.std()), float(r.std()),
        float(gray.std()), laplacian_var,
    )


def block_features(bgr_image: np.ndarray, y0: int, x0: int, size: int = BLOCK_SIZE) -> Optional[List[float]]:
    """Tek bir bloğun 8 temel özelliğini döndürür (belge-bağlamı özelliği
    OLMADAN — bkz. `extract_block_grid`, doğru kullanım budur). Bu fonksiyon
    yalnızca geriye dönük uyumluluk/tekil blok testleri için tutuluyor."""
    stats = _block_stats(bgr_image, y0, x0, size)
    return list(stats) if stats is not None else None


def extract_block_grid(bgr_image: np.ndarray, roi: Optional[BBox] = None, block_size: int = BLOCK_SIZE):
    """roi (veya tüm görüntü) içindeki her bloğun özelliğini ve konumunu
    döndürür — BELGENİN KENDİ MEDYAN RENGİNE göre bağlamsal özellik dahil
    (bkz. modül docstring'i, "v2 düzeltmesi").

    Returns:
        (features: List[List[float]], positions: List[Tuple[x0,y0]])
    """
    if roi is not None:
        x0, y0, x1, y1 = roi
    else:
        x0, y0 = 0, 0
        y1, x1 = bgr_image.shape[:2]

    raw_stats, positions = [], []
    for by in range(y0, max(y0, y1 - block_size), block_size):
        for bx in range(x0, max(x0, x1 - block_size), block_size):
            stats = _block_stats(bgr_image, by, bx, block_size)
            if stats is not None:
                raw_stats.append(stats)
                positions.append((bx, by))

    if not raw_stats:
        return [], []

    colors = np.array([[s[0], s[1], s[2]] for s in raw_stats])
    doc_median_color = np.median(colors, axis=0)

    features = []
    for stats, color in zip(raw_stats, colors):
        dist = float(np.linalg.norm(color - doc_median_color))
        features.append(list(stats) + [dist])
    return features, positions


def _load_model():
    global _model_cache
    if _model_cache is None:
        import joblib

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model bulunamadı: {MODEL_PATH}. Önce "
                "experiments/occlusion/train_occlusion_classifier.py çalıştırılmalı."
            )
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def ml_occlusion_ratio(image: np.ndarray, roi: Optional[BBox] = None) -> float:
    """Görüntüdeki (veya roi içindeki) bloklardan kaçının "yabancı nesne
    (occluder)" olarak sınıflandırıldığının oranını döndürür (0-1).

    Konumdan VE renkten bağımsızdır — bkz. modül docstring'i.
    """
    features, _ = extract_block_grid(image, roi)
    if not features:
        return 0.0
    model = _load_model()
    preds = model.predict(features)
    return float(np.mean(preds))
