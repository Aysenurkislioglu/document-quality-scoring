"""
ADIM 3 — Tüm veri setini mevcut sistemle toplu skorlar.

Çıktı (results/real_data/scores.csv) yalnızca `anon_id` içerir — gerçek
dosya adı YOK, görüntünün kendisi YOK, yalnızca sayısal skorlar.

Kullanım:
    python3 experiments/real_data/3_batch_score.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.scoring.fusion import compute_document_quality_score  # noqa: E402

MAPPING_PATH = PROJECT_ROOT / "data" / "raw" / "anon_mapping.csv"
SCORES_PATH = PROJECT_ROOT / "results" / "real_data" / "scores.csv"

FIELDNAMES = [
    "anon_id", "overall_score", "verdict",
    "blur_score", "blur_raw",
    "darkness_score", "darkness_raw",
    "skew_score", "skew_raw",
    "occlusion_score", "occlusion_raw",
    "glare_applicable", "glare_score", "glare_raw",
    "worst_module",
]


def verdict_of(score: float) -> str:
    if score >= 70:
        return "iyi"
    if score >= 40:
        return "orta"
    return "kotu"


def main():
    if not MAPPING_PATH.exists():
        print(f"Önce 1_prepare_dataset.py çalıştırılmalı ({MAPPING_PATH} yok).")
        return

    with open(MAPPING_PATH, newline="", encoding="utf-8") as f:
        mapping = [(int(row["anon_id"]), row["absolute_path"]) for row in csv.DictReader(f)]

    rows = []
    errors = 0
    for i, (anon_id, path) in enumerate(sorted(mapping)):
        img = cv2.imread(path)
        if img is None:
            print(f"UYARI: #{anon_id} okunamadı, atlanıyor.")
            errors += 1
            continue

        result = compute_document_quality_score(img)
        comps = result["components"]

        # "en kötü modül" — fusion.py'nin nihai skoru hesaplarken kullandığı
        # aynı mantık (glare yalnızca uygulanabilirse dahil).
        applicable_scores = {
            k: v["score"] for k, v in comps.items()
            if v["score"] is not None
        }
        worst_module = min(applicable_scores, key=applicable_scores.get) if applicable_scores else None

        rows.append({
            "anon_id": anon_id,
            "overall_score": round(result["overall_score"], 1),
            "verdict": verdict_of(result["overall_score"]),
            "blur_score": round(comps["blur"]["score"], 1),
            "blur_raw": round(comps["blur"]["raw_value"], 3),
            "darkness_score": round(comps["darkness"]["score"], 1),
            "darkness_raw": round(comps["darkness"]["raw_value"], 3),
            "skew_score": round(comps["skew"]["score"], 1),
            "skew_raw": round(comps["skew"]["raw_value"], 3),
            "occlusion_score": round(comps["occlusion"]["score"], 1),
            "occlusion_raw": round(comps["occlusion"]["raw_value"], 3),
            "glare_applicable": comps["glare"]["score"] is not None,
            "glare_score": round(comps["glare"]["score"], 1) if comps["glare"]["score"] is not None else "",
            "glare_raw": round(comps["glare"]["raw_value"], 3) if comps["glare"]["raw_value"] is not None else "",
            "worst_module": worst_module,
        })

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(mapping)} işlendi")

    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} görüntü skorlandı ({errors} hata) -> {SCORES_PATH}")

    print("\n=== Hızlı özet ===")
    from collections import Counter
    verdict_counts = Counter(r["verdict"] for r in rows)
    print("Verdict dağılımı:", dict(verdict_counts))
    worst_counts = Counter(r["worst_module"] for r in rows)
    print("En sık 'en kötü modül':", dict(worst_counts))


if __name__ == "__main__":
    main()
