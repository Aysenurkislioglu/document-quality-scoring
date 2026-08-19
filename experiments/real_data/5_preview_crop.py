"""
ADIM 5 (opsiyonel, tanılama) — Belge kırpma sonuçlarını GÖRSEL olarak
kendi gözünle kontrol etmek için.

GEREKÇE: `src/detection/document_crop.py` klasik CV (kenar+kontur) ile
belgeyi arkaplandan kırpmaya çalışıyor. Bu betiğin doğruluğunu (belgeyi
doğru mu kırpıyor, yoksa bir kısmını mı kesiyor) yalnızca GÖREREK
doğrulayabilirsin — Claude bu fotoğrafları hiç görmediği için sayısal
"tespit oranı" tek başına yeterli değil.

Bu araç, rastgele seçilmiş N fotoğrafın ORİJİNAL ve KIRPILMIŞ halini yan
yana gösterir (yalnızca senin ekranında — tamamen yerel, dışarı hiçbir
şey çıkmaz). Kapatmak için herhangi bir tuşa bas, sıradaki örneğe geçer.

Kullanım:
    python3 experiments/real_data/5_preview_crop.py [N]
    (N verilmezse varsayılan 15 örnek gösterilir)
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.detection.document_crop import detect_and_crop_document  # noqa: E402
from src.scoring.fusion import _normalize_scale  # noqa: E402

MAPPING_PATH = PROJECT_ROOT / "data" / "raw" / "anon_mapping.csv"
MAX_DISPLAY_HEIGHT = 700


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15

    if not MAPPING_PATH.exists():
        print(f"Önce 1_prepare_dataset.py çalıştırılmalı ({MAPPING_PATH} yok).")
        return

    with open(MAPPING_PATH, newline="", encoding="utf-8") as f:
        mapping = [(int(r["anon_id"]), r["absolute_path"]) for r in csv.DictReader(f)]

    rng = random.Random(7)
    sample = rng.sample(mapping, min(n, len(mapping)))

    print(f"{len(sample)} örnek gösterilecek. Pencerede herhangi bir tuşa bas -> sıradaki.")
    print("Çıkmak için 'q'.\n")

    cv2.namedWindow("Kirpma Onizleme (SOL orijinal, SAG kirpilmis)", cv2.WINDOW_AUTOSIZE)

    detected_count = 0
    for anon_id, path in sample:
        img = cv2.imread(path)
        if img is None:
            print(f"#{anon_id}: okunamadı, atlanıyor")
            continue

        original = _normalize_scale(img)
        cropped, detected = detect_and_crop_document(img)
        cropped = _normalize_scale(cropped)
        if detected:
            detected_count += 1

        # İkisini de aynı yüksekliğe getirip yan yana koy.
        def resize_h(im, target_h):
            h, w = im.shape[:2]
            scale = target_h / h
            return cv2.resize(im, (int(w * scale), target_h))

        target_h = min(MAX_DISPLAY_HEIGHT, original.shape[0])
        left = resize_h(original, target_h)
        right = resize_h(cropped, target_h)
        gap = 10
        # Not: pencere basligindaki/metnindeki Turkce karakterler (I, s, vb.)
        # bazi sistemlerde OpenCV'nin dahili yazi tipinde (Hershey) hic
        # gorunmeyebiliyor -- bu yuzden ekran-uzeri metni bilerek SADECE ASCII
        # tutuyoruz. Ayni bilgiyi terminale de yaziyoruz (garanti gorunur).
        status_ascii = "TESPIT EDILDI (kirpildi)" if detected else "TESPIT EDILEMEDI (orijinal kullaniliyor)"
        top_strip = np.full((50, left.shape[1] + right.shape[1] + gap, 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(top_strip, f"#{anon_id} -- {status_ascii}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        middle = cv2.hconcat([left, cv2.copyMakeBorder(right, 0, 0, gap, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))])
        canvas = cv2.vconcat([top_strip, middle])

        print(f"#{anon_id}: {status_ascii}")
        cv2.imshow("Kirpma Onizleme (SOL orijinal, SAG kirpilmis)", canvas)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    print(f"\nBu örneklemde tespit oranı: {detected_count}/{len(sample)}")
    print("Kendi gözlemin: kırpma çoğunlukla DOĞRU mu (belgeyi tam alıyor, kesmiyor) yoksa YANLIŞ mı?")


if __name__ == "__main__":
    main()
