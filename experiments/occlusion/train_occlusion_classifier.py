"""
ML tabanlı, renkten bağımsız occlusion sınıflandırıcısını eğitir.

`generate_skin_occlusion_documents.py`'nin (yalnızca ten rengi) genelleştirilmiş
hâli: burada onlarca FARKLI renk (ten tonu HARİÇ — o, doğrulama için bilinçli
olarak dışarıda bırakılıyor, bkz. run_ml_experiment.py) hem DÜZ hem DOKULU
(gürültülü) varyantlarda kullanılarak sentetik eğitim verisi üretilir; bir
Random Forest sınıflandırıcı bu bloklardan "occluder mı, değil mi" ayrımını
öğrenir.

GEREKÇE (neden blok-bazlı ML, neden sabit renk eşiği değil): src/occlusion/
skin_detection.py yalnızca sabit bir YCrCb aralığını arıyordu — farklı renk/
dokudaki (sticker, kumaş, plastik) kapanmaları prensipte kaçırabilirdi. Bu
script, modelin ezber değil GERÇEKTEN renk+doku örüntüsünü öğrendiğini
kanıtlamak için, eğitimde HİÇ görülmeyen renklerle (ten tonları dahil) ayrı
bir deneyde (run_ml_experiment.py) test edilir.

v2 DÜZELTMESİ ("belgenin kendi rengi" bağlamı): v1 yalnızca DÜZ BEYAZ/GRİ
zeminli belgelerle eğitilmişti; gerçek kullanımda kullanıcı RENKLİ bir
kimlik kartı yükleyince kartın kendi tasarımının TAMAMINI "yabancı nesne"
sanıp oranı ~%95-99'a çıkarıyordu (bkz. src/occlusion/ml_detection.py
docstring'i, project_notes.md "Occlusion — renkli zemin hatası"). Bu
scriptin eğitim verisine artık RENKLİ KİMLİK KARTI ARKA PLANLARI da (hem
occlusion'sız hem occlusion'lı örnekler) eklendi — model, "bu blok
BELGENİN KENDİ renk şemasına mı uyuyor, yoksa gerçekten farklı bir nesne mi"
ayrımını (`color_dist_from_doc_median` özelliği sayesinde) öğreniyor.

Çıktı: src/occlusion/models/occlusion_rf.joblib
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "_common"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "glare"))

from synthetic_documents import render_document, default_combinations  # noqa: E402
from occlusion.ml_detection import extract_block_grid, BLOCK_SIZE, MODEL_PATH  # noqa: E402
from generate_id_card_documents import render_id_card, CARD_SIZE  # noqa: E402

RANDOM_SEED = 42

# Eğitimde kullanılan renkler — TEN TONU KASITLI OLARAK DIŞARIDA (bkz.
# run_ml_experiment.py, genelleme testinde kullanılacak). Geniş bir renk/ton
# yelpazesi: canlı renkler + nötrler + koyu/açık varyantlar.
TRAIN_COLORS = {
    "kirmizi": (200, 40, 40), "yesil": (40, 150, 60), "mavi": (40, 70, 180),
    "sari": (220, 200, 40), "mor": (120, 40, 150), "turuncu": (220, 120, 30),
    "gri_acik": (180, 180, 180), "gri_koyu": (70, 70, 70), "siyah": (25, 25, 25),
    "pembe": (230, 120, 160), "kahve": (90, 60, 30), "beyaz_krem": (245, 240, 225),
    "bordo": (110, 20, 40), "zeytin": (100, 110, 40),
}
COVERAGE_LEVELS = [0.0, 0.15, 0.3, 0.5, 0.75]
NOISE_STD = 25  # dokulu varyant için gürültü şiddeti

# Eğitimde kullanılan kimlik kartı renk şemaları — run_ml_experiment.py'de
# GÖRÜLMEMİŞ farklı şemalarla test edilecek (genelleme kanıtı).
CARD_TRAIN_SCHEMES = {
    "mavi_gri": {"bg": (196, 214, 226), "photo": (150, 170, 190), "text": (40, 50, 90)},
    "bej": {"bg": (232, 220, 196), "photo": (200, 185, 155), "text": (90, 60, 30)},
    "yesilimsi": {"bg": (205, 224, 210), "photo": (160, 185, 165), "text": (30, 70, 45)},
    "gri": {"bg": (210, 210, 214), "photo": (170, 170, 176), "text": (40, 40, 46)},
    "pembemsi": {"bg": (236, 210, 214), "photo": (200, 165, 172), "text": (90, 30, 40)},
}
CARD_OCCLUDER_COLORS = TRAIN_COLORS  # kartlarda da AYNI 14 renk kullanılır


def apply_patch(rgb_array: np.ndarray, content_bbox, coverage: float, color, textured: bool, rng: random.Random):
    x0, y0, x1, y1 = content_bbox
    w, h = x1 - x0, y1 - y0
    if coverage <= 0:
        return None
    pw, ph = max(1, int(w * (coverage ** 0.5))), max(1, int(h * (coverage ** 0.5)))
    px0 = rng.randint(x0, max(x0, x1 - pw))
    py0 = rng.randint(y0, max(y0, y1 - ph))
    if textured:
        noise = np.random.RandomState(rng.randint(0, 10_000)).randint(-NOISE_STD, NOISE_STD, size=(ph, pw, 3))
        patch = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)
        rgb_array[py0 : py0 + ph, px0 : px0 + pw] = patch
    else:
        rgb_array[py0 : py0 + ph, px0 : px0 + pw] = color
    return (px0, py0, px0 + pw, py0 + ph)


def extract_labeled_blocks(bgr_image, roi_bbox, patch_bbox):
    """extract_block_grid (belge-bağlamlı özelliklerle) kullanıp her bloğu
    patch_bbox'a göre etiketler."""
    features, positions = extract_block_grid(bgr_image, roi=roi_bbox, block_size=BLOCK_SIZE)
    X, Y = [], []
    for (bx, by), feat in zip(positions, features):
        label = 0
        if patch_bbox:
            px0, py0, px1, py1 = patch_bbox
            cx, cy = bx + BLOCK_SIZE // 2, by + BLOCK_SIZE // 2
            if px0 <= cx <= px1 and py0 <= cy <= py1:
                label = 1
        X.append(feat)
        Y.append(label)
    return X, Y


def main():
    import cv2
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    rng = random.Random(RANDOM_SEED)
    combos = list(default_combinations())

    X_train, Y_train = [], []

    # --- 1) Beyaz/gri belgeler + renkli yamalar (v1'den, DEĞİŞMEDİ) ---
    print(f"[1/2] Beyaz kağıt eğitim verisi ({len(TRAIN_COLORS)} renk x {len(combos)} belge x "
          f"{len(COVERAGE_LEVELS)} kapsama x 2 doku varyantı)...")
    for font_size, num_paragraphs, replica in combos:
        doc = render_document(font_size, num_paragraphs, rng)
        for color in TRAIN_COLORS.values():
            for coverage in COVERAGE_LEVELS:
                for textured in (False, True):
                    rgb_arr = np.array(doc.image.convert("RGB"))
                    patch_bbox = apply_patch(rgb_arr, doc.content_bbox, coverage, color, textured, rng)
                    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
                    X, Y = extract_labeled_blocks(bgr, doc.content_bbox, patch_bbox)
                    X_train.extend(X)
                    Y_train.extend(Y)

    # --- 2) RENKLİ kimlik kartları: occlusion'sız (label=0) + occlusion'lı (label=1) ---
    print(f"[2/2] Kimlik kartı eğitim verisi ({len(CARD_TRAIN_SCHEMES)} şema x "
          f"{len(CARD_OCCLUDER_COLORS)} yama rengi x {len(COVERAGE_LEVELS)} kapsama)...")
    card_bbox = (0, 0, CARD_SIZE[0], CARD_SIZE[1])
    for scheme_name, scheme in CARD_TRAIN_SCHEMES.items():
        for color in CARD_OCCLUDER_COLORS.values():
            for coverage in COVERAGE_LEVELS:
                for textured in (False, True):
                    card_img = render_id_card(scheme, rng)
                    rgb_arr = np.array(card_img)
                    patch_bbox = apply_patch(rgb_arr, card_bbox, coverage, color, textured, rng)
                    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
                    X, Y = extract_labeled_blocks(bgr, card_bbox, patch_bbox)
                    X_train.extend(X)
                    Y_train.extend(Y)

    X_train, Y_train = np.array(X_train), np.array(Y_train)
    print(f"\nToplam blok: {len(X_train)}, occluder oranı: {Y_train.mean():.3f}")

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    )
    clf.fit(X_train, Y_train)

    train_acc = clf.score(X_train, Y_train)
    print(f"Eğitim seti doğruluğu (yalnızca bilgi amaçlı, GERÇEK doğrulama "
          f"run_ml_experiment.py'de görülmemiş renklerle yapılır): {train_acc:.4f}")

    from occlusion.ml_detection import FEATURE_NAMES
    print("Özellik önemleri:", dict(zip(FEATURE_NAMES, clf.feature_importances_.round(3))))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model kaydedildi -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
