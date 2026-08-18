"""
ADIM 1 — Gerçek veri setini anonimleştirir.

GİZLİLİK: Bu script, girdi klasöründeki her görüntü dosyasına rastgele bir
`anon_id` atar. Gerçek dosya adı <-> anon_id eşlemesi YALNIZCA
`data/raw/anon_mapping.csv` dosyasına yazılır (bu dosya .gitignore'da,
ASLA commit edilmez, ASLA paylaşılmaz — yalnızca SENİN kendi
bilgisayarında, kendi referansın için). Bundan sonraki TÜM script'ler
(etiketleme, skorlama, analiz) yalnızca anon_id kullanır; hiçbir çıktı
dosyası (results/real_data/ altındakiler) gerçek dosya adı İÇERMEZ.

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

    rng = random.Random(2026)
    ids = list(range(1, len(files) + 1))
    rng.shuffle(ids)  # dosya adı sırasıyla anon_id arasında bariz bir ilişki olmasın

    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["anon_id", "absolute_path"])
        for anon_id, path in zip(ids, files):
            writer.writerow([anon_id, str(path)])

    print(f"{len(files)} görüntü bulundu, anonimleştirildi.")
    print(f"Eşleme dosyası -> {MAPPING_PATH}")
    print()
    print("⚠️  BU DOSYAYI (anon_mapping.csv) KİMSEYLE PAYLAŞMA — yalnızca kendi")
    print("   referansın için (örn. 'anon_id 42 hangi fotoğraftı?' diye bakmak için).")
    print("   Zaten .gitignore'da, ama başka bir yolla da (mail, mesaj vb.) paylaşma.")


if __name__ == "__main__":
    main()
