"""
Bağlam-farkında (context-aware) glare sınıflandırıcısını eğitir.

Standart 12 belgelik ızgara (`_common/synthetic_documents.py`, RANDOM_SEED=42)
ve `generate_glare_documents.py`'deki TARGET_AREA_FRACTIONS şiddet
seviyeleri kullanılır. Her görüntü için glare blob'unun merkezi/sigma'sı
DETERMİNİSTİK olarak yeniden hesaplanır (bkz. generate_glare_documents.py,
`sigma_for_target_area`) — böylece elle etiketleme yapmadan, sentetik
üretimin kendi zemin gerçeğinden (ground truth) doğru etiketli eğitim
verisi çıkarılır.

GEREKÇE (neden bağlam/satır özelliği): bkz. src/glare/ml_detection.py
docstring'i — saf blok-bazlı (bağlamsız) bir deneme, tam doymuş glare ile
boş beyaz kağıdı ayırt edemediği için başarısız olmuştu (bkz.
project_notes.md, "Glare ML denemesi").

NOT (v2/v3 denemeleri geri alındı): Bu v1 sürümünün bilinen bir sınırlaması
var — ağır bulanık (blur) belgelerde glare ile karışabiliyor. İki düzeltme
denemesi (bulanık negatif örnekler eklemek; belge-geneli bulanıklık
özelliği eklemek) yapıldı, ikisi de asıl glare tespit performansını
BOZDUĞU için geri alındı — bkz. src/glare/ml_detection.py docstring'i ve
project_notes.md, "Glare ML v2/v3 denemeleri (geri alındı)".

NOT (v4 düzeltmesi — eğitim/üretim uyumsuzluğu): Önceki sürümler eğitimi
`roi=doc.content_bbox` ile kısıtlıyordu, ama üretim kodu (fusion.py) gerçek
bir fotoğrafta content_bbox bilinmediği için `roi=None` (TÜM görüntü)
kullanıyor. Model, kenar boşluğu bloklarını HİÇ görmeden eğitilip onlarla
test ediliyordu — bu, doğrulama sonuçlarının yanıltıcı derecede iyimser
çıkmasına neden oldu. Bu sürüm eğitimi de `roi=None` ile yapar.

Çıktı: src/glare/models/glare_rf.joblib
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "_common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_documents import render_document, default_combinations  # noqa: E402
from generate_glare_documents import apply_glare_blob, sigma_for_target_area, TARGET_AREA_FRACTIONS  # noqa: E402
from glare.ml_detection import extract_context_features, MODEL_PATH, FEATURE_NAMES  # noqa: E402

RANDOM_SEED = 42


def alpha_at(cx, cy, center, sigma, max_alpha=0.95):
    if sigma <= 0:
        return 0.0
    d2 = (cx - center[0]) ** 2 + (cy - center[1]) ** 2
    return max_alpha * np.exp(-d2 / (2 * sigma ** 2))


def main():
    import random
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    rng = random.Random(RANDOM_SEED)
    combos = list(default_combinations())

    X_train, Y_train = [], []
    print(f"Eğitim verisi çıkarılıyor ({len(combos)} belge x {len(TARGET_AREA_FRACTIONS)} şiddet)...")

    for font_size, num_paragraphs, replica in combos:
        doc = render_document(font_size, num_paragraphs, rng)
        img_array = np.array(doc.image)
        x0, y0, x1, y1 = doc.content_bbox
        w, h = x1 - x0, y1 - y0
        center = (x0 + w // 2, y0 + h // 2)
        content_area_px = w * h

        for frac in TARGET_AREA_FRACTIONS:
            target_area_px = frac * content_area_px
            sigma = sigma_for_target_area(target_area_px)
            degraded_array, _ = apply_glare_blob(img_array, center, sigma)

            features, positions = extract_context_features(degraded_array, roi=None)
            for (bx, by), feat in zip(positions, features):
                cx, cy = bx + 8, by + 8  # BLOCK_SIZE//2
                a = alpha_at(cx, cy, center, sigma)
                label = 1 if a > 0.5 else 0
                X_train.append(feat)
                Y_train.append(label)

    X_train, Y_train = np.array(X_train), np.array(Y_train)
    print(f"Toplam blok: {len(X_train)}, glare oranı: {Y_train.mean():.3f}")

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    )
    clf.fit(X_train, Y_train)

    print(f"Eğitim seti doğruluğu (bilgi amaçlı): {clf.score(X_train, Y_train):.4f}")
    print("Özellik önemleri:", dict(zip(FEATURE_NAMES, clf.feature_importances_.round(3))))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model kaydedildi -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
