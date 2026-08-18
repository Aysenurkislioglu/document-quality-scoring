"""
ADIM 2 — Yerel, klavye-tabanlı etiketleme aracı.

Her görüntüyü sırayla gösterir; sen tuşlarla genel kalite + gördüğün
spesifik sorunları işaretlersin. İlerleme her onaydan sonra diske
kaydedilir (yarıda bırakıp devam edebilirsin). Çıktı dosyası
(labels.csv) yalnızca `anon_id` içerir — gerçek dosya adı YOK.

Tuşlar:
  1 = genel kalite: KÖTÜ       (onaylar, sıradaki görüntüye geçer)
  2 = genel kalite: ORTA       (onaylar, sıradaki görüntüye geçer)
  3 = genel kalite: İYİ        (onaylar, sıradaki görüntüye geçer)
  b = bulanık (toggle)         p = parlama/glare (toggle)
  k = karanlık (toggle)        o = kapanma/örtülü (toggle)
  e = eğik (toggle)
  r = bu görüntü için işaretleri sıfırla
  u = bir önceki görüntüye geri dön (düzeltmek için)
  s = bu görüntüyü atla (etiketlemeden geç)
  q = kaydet ve çık (istediğin an, ilerleme kaybolmaz)

Kullanım:
    python3 experiments/real_data/2_label_tool.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = PROJECT_ROOT / "data" / "raw" / "anon_mapping.csv"
LABELS_PATH = PROJECT_ROOT / "results" / "real_data" / "labels.csv"

FLAG_KEYS = {ord("b"): "bulanik", ord("p"): "parlama", ord("k"): "karanlik", ord("o"): "kapanma", ord("e"): "egik"}
QUALITY_KEYS = {ord("1"): "kotu", ord("2"): "orta", ord("3"): "iyi"}
MAX_DISPLAY_HEIGHT = 850


def load_mapping():
    rows = []
    with open(MAPPING_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["anon_id"]), row["absolute_path"]))
    return sorted(rows)


def load_existing_labels():
    if not LABELS_PATH.exists():
        return {}
    labels = {}
    with open(LABELS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[int(row["anon_id"])] = row
    return labels


def save_labels(labels: dict):
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["anon_id", "genel_kalite", "bulanik", "parlama", "karanlik", "kapanma", "egik"]
    with open(LABELS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for anon_id in sorted(labels):
            writer.writerow(labels[anon_id])


def render(image, anon_id, index, total, flags):
    h, w = image.shape[:2]
    if h > MAX_DISPLAY_HEIGHT:
        scale = MAX_DISPLAY_HEIGHT / h
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    canvas = cv2.copyMakeBorder(image, 0, 70, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    flag_str = " ".join(f"[{name}]" for name in flags) if flags else "(işaret yok)"
    line1 = f"#{anon_id}  ({index}/{total})   Etiketler: {flag_str}"
    line2 = "1=kotu 2=orta 3=iyi | b=bulanik p=parlama k=karanlik o=kapanma e=egik | r=sifirla u=geri s=atla q=cikis"
    cv2.putText(canvas, line1, (10, image.shape[0] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (10, image.shape[0] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def main():
    if not MAPPING_PATH.exists():
        print(f"Önce 1_prepare_dataset.py çalıştırılmalı ({MAPPING_PATH} yok).")
        return

    mapping = load_mapping()
    labels = load_existing_labels()
    total = len(mapping)

    remaining_indices = [i for i, (anon_id, _) in enumerate(mapping) if anon_id not in labels]
    if not remaining_indices:
        print("Tüm görüntüler zaten etiketlenmiş! (results/real_data/labels.csv)")
        return

    print(f"Toplam {total} görüntü, {len(labels)} tanesi zaten etiketli, {len(remaining_indices)} kaldı.")
    print("Pencere açıldığında tuşlarla işaretle (talimatlar pencerenin altında).")

    cv2.namedWindow("Etiketleme", cv2.WINDOW_AUTOSIZE)

    pos = remaining_indices[0]
    flags = set()
    history = []  # (pos) — 'u' ile geri dönmek için

    while 0 <= pos < total:
        anon_id, path = mapping[pos]
        img = cv2.imread(path)
        if img is None:
            print(f"UYARI: #{anon_id} okunamadı, atlanıyor ({path})")
            pos += 1
            continue

        canvas = render(img, anon_id, pos + 1, total, flags)
        cv2.imshow("Etiketleme", canvas)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key in FLAG_KEYS:
            name = FLAG_KEYS[key]
            flags.symmetric_difference_update({name})
        elif key == ord("r"):
            flags = set()
        elif key == ord("s"):
            history.append(pos)
            pos += 1
            flags = set()
        elif key == ord("u"):
            if history:
                pos = history.pop()
                flags = set()
        elif key in QUALITY_KEYS:
            row = {"anon_id": anon_id, "genel_kalite": QUALITY_KEYS[key]}
            for flag_name in FLAG_KEYS.values():
                row[flag_name] = "1" if flag_name in flags else "0"
            labels[anon_id] = row
            save_labels(labels)  # her onayda diske yaz — ilerleme hiç kaybolmaz
            history.append(pos)
            pos += 1
            flags = set()

    cv2.destroyAllWindows()
    print(f"\nKaydedildi -> {LABELS_PATH} ({len(labels)}/{total} etiketlendi)")
    if len(labels) < total:
        print("Kaldığın yerden devam etmek için script'i tekrar çalıştır.")


if __name__ == "__main__":
    main()
