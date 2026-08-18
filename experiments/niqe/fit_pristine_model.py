"""
NIQE için "pristine" (temiz/referans) modeli, projenin kendi temiz sentetik
belgelerinden fit eder.

Standart NIQE, genel doğa fotoğraflarından (örn. Berkeley segmentation veri
seti) bir referans model kullanır. Bu proje belge görüntüleriyle çalıştığı
için, referans modelin de BELGE görüntülerinden fit edilmesi daha uygun
olmalı — bu varsayım run_niqe_experiment.py'de test edilir.

Girdi: data/synthetic/blur/originals/ (12 temiz, bozulmamış sentetik belge —
blur deneyinin severity=0 durumu).
Çıktı: src/scoring/models/niqe_pristine.npz (mu, cov)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoring.niqe import extract_patch_features, fit_mvg, MODEL_PATH  # noqa: E402

ORIGINALS_MANIFEST = PROJECT_ROOT / "data" / "synthetic" / "blur" / "originals" / "manifest.csv"


def main():
    manifest = pd.read_csv(ORIGINALS_MANIFEST)
    all_feats = []
    for _, row in manifest.iterrows():
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        feats = extract_patch_features(img)
        all_feats.append(feats)
        print(f"  {row['doc_id']}: {len(feats)} blok özelliği çıkarıldı")

    all_feats = np.concatenate(all_feats, axis=0)
    print(f"\nToplam blok özelliği: {len(all_feats)}")

    mu, cov = fit_mvg(all_feats)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_PATH, mu=mu, cov=cov)
    print(f"Pristine model kaydedildi -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
