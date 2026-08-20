"""
ADIM 1 — Gerçek veri setini anonimleştirir.

GİZLİLİK: Bu script, girdi klasöründeki her görüntü dosyasına rastgele bir
`anon_id` atar. Gerçek dosya adı <-> anon_id eşlemesi YALNIZCA
`data/raw/anon_mapping.csv` dosyasına yazılır (bu dosya .gitignore'da,
ASLA commit edilmez, ASLA paylaşılmaz — yalnızca SENİN kendi
bilgisayarında, kendi referansın için). Bundan sonraki TÜM script'ler
(etiketleme, skorlama, analiz) yalnızca anon_id kullanır; hiçbir çıktı
dosyası (results/real_data/ altındakiler) gerçek dosya adı İÇERMEZ.

EKLEME (APPEND) MODU: `anon_mapping.csv` zaten varsa, bu script onu
SİLİP BAŞTAN YAZMAZ — mevcut eşlemeleri korur (eski anon_id'ler, eski
etiketler/skorlar geçerliliğini sürdürür), yalnızca YENİ (daha önce
eşlenmemiş) dosyalara yeni anon_id'ler atayıp EKLER. Aynı dosya
(absolute_path aynıysa) tekrar eklenmez, atlanır.

Kullanım:
    python3 experiments/real_data/1_prepare_dataset.py /path/to/kimlik/klasoru
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = PROJECT_ROOT / "data" / "raw" / "anon_mapping.csv"


def load_existing_mapping():
    if not MAPPING_PATH.exists():
        return {}
    with open(MAPPING_PATH, newline="", encoding="utf-8") as f:
        return {row["absolute_path"]: int(row["anon_id"]) for row in csv.DictReader(f)}


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python3 1_prepare_dataset.py /path/to/kimlik/klasoru")
        sys.exit(1)

    source_dir = Path(sys.argv[1]).expanduser().resolve()
    if not source_dir.is_dir():
        print(f"Klasör bulunamadı: {source_dir}")
        sys.exit(1)

    files = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
    if not files:
        print(f"'{source_dir}' içinde desteklenen bir görüntü dosyası bulunamadı.")
        sys.exit(1)

    existing = load_existing_mapping()  # absolute_path -> anon_id
    new_files = [p for p in files if str(p) not in existing]
    skipped = len(files) - len(new_files)

    if not new_files:
        print(f"'{source_dir}' içindeki tüm görüntüler zaten eşlenmiş (0 yeni). "
              f"{len(existing)} toplam eşleme değişmedi.")
        return

    rng = random.Random(2026)
    next_id = (max(existing.values()) + 1) if existing else 1
    new_ids = list(range(next_id, next_id + len(new_files)))
    rng.shuffle(new_ids)  # dosya adı sırasıyla anon_id arasında bariz bir ilişki olmasın

    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existing:
            writer.writerow(["anon_id", "absolute_path"])
        for anon_id, path in zip(new_ids, new_files):
            writer.writerow([anon_id, str(path)])

    total = len(existing) + len(new_files)
    print(f"{len(new_files)} YENİ görüntü eklendi (anon_id {min(new_ids)}-{max(new_ids)})"
          f"{f', {skipped} tanesi zaten eşliydi, atlandı' if skipped else ''}.")
    print(f"Toplam eşleme: {total} ({len(existing)} eski + {len(new_files)} yeni).")
    print(f"Eşleme dosyası -> {MAPPING_PATH}")
    print()
    print("⚠️  BU DOSYAYI (anon_mapping.csv) KİMSEYLE PAYLAŞMA — yalnızca kendi")
    print("   referansın için (örn. 'anon_id 42 hangi fotoğraftı?' diye bakmak için).")
    print("   Zaten .gitignore'da, ama başka bir yolla da (mail, mesaj vb.) paylaşma.")


if __name__ == "__main__":
    main()
