"""
Glare baseline deneyi — ana çalıştırma scripti.

Bu deney iki metrik hesaplar:

1. **naive_glare_ratio** — src/glare/metrics.py içindeki, literatürün önerdiği
   gerçek (no-reference, üretimde kullanılabilir) HSV + connected components
   yöntemi. ROI = belgenin içerik kutusu (content bounding box).

2. **oracle_text_washout_ratio** — YALNIZCA bu deneyi doğrulamak için
   eklenen, REFERANS GEREKTİREN bir "oracle" metrik: orijinal (bozulmamış)
   görüntüdeki metin piksellerinin (koyu pikseller) kaçının, bozulmuş
   görüntüde "yıkanmış/beyaza yakın" hale geldiğini ölçer. Bu metrik
   üretimde KULLANILAMAZ (temiz referans görüntüye ihtiyaç duyar) — yalnızca
   "enjekte ettiğimiz sentetik glare gerçekten ölçülebilir bir etki
   yaratıyor mu?" sorusunu doğrulamak için buradadır.

Neden iki metrik? Çünkü ilk çalıştırmada naive_glare_ratio'nun beklenmedik
şekilde davrandığı görüldü (bkz. project_notes.md, Glare bölümü). Oracle
metrik, sorunun "sentetik veride gerçek bir sinyal yok" mu yoksa "yöntem bu
sinyali yakalayamıyor" mu olduğunu ayırt etmek için eklendi.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glare.metrics import glare_ratio  # noqa: E402

ORIGINALS_MANIFEST = PROJECT_ROOT / "data" / "synthetic" / "glare" / "originals" / "manifest.csv"
DEGRADED_MANIFEST = PROJECT_ROOT / "data" / "synthetic" / "glare" / "degraded" / "manifest.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "glare"
PLOTS_DIR = RESULTS_DIR / "plots"

TEXT_PIXEL_THRESHOLD = 200  # orijinalde bundan koyu pikseller "metin" sayılır
WASHED_OUT_THRESHOLD = 235  # bu değerin üzeri "yıkanmış/beyaz" sayılır


def oracle_text_washout_ratio(original_region: np.ndarray, degraded_region: np.ndarray) -> float:
    text_mask = original_region < TEXT_PIXEL_THRESHOLD
    if text_mask.sum() == 0:
        return float("nan")
    washed_out = (degraded_region >= WASHED_OUT_THRESHOLD) & text_mask
    return float(washed_out.sum()) / float(text_mask.sum())


def compute_scores() -> pd.DataFrame:
    originals = pd.read_csv(ORIGINALS_MANIFEST)
    degraded = pd.read_csv(DEGRADED_MANIFEST)

    original_images = {}
    for _, row in originals.iterrows():
        bbox = json.loads(row["content_bbox"])
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        original_images[row["doc_id"]] = (img, tuple(bbox))

    records = []
    for _, row in degraded.iterrows():
        doc_id = row["doc_id"]
        orig_img, bbox = original_images[doc_id]
        x0, y0, x1, y1 = bbox
        deg_img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)

        naive_ratio = glare_ratio(deg_img, roi=bbox)
        oracle_ratio = oracle_text_washout_ratio(
            orig_img[y0:y1, x0:x1], deg_img[y0:y1, x0:x1]
        )

        records.append(
            {
                "doc_id": doc_id,
                "severity_level": int(row["severity_level"]),
                "target_area_fraction": float(row["target_area_fraction"]),
                "ground_truth_glare_fraction": float(row["ground_truth_glare_fraction"]),
                "naive_glare_ratio": naive_ratio,
                "oracle_text_washout_ratio": oracle_ratio,
                "font_size": int(row["font_size"]),
                "num_paragraphs": int(row["num_paragraphs"]),
            }
        )

    return pd.DataFrame.from_records(records)


def monotonicity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for doc_id, group in df.groupby("doc_id"):
        group = group.sort_values("severity_level")
        row = {"doc_id": doc_id}
        for metric in ["naive_glare_ratio", "oracle_text_washout_ratio"]:
            rho, pval = spearmanr(group["severity_level"], group[metric])
            row[f"{metric}_spearman_rho"] = rho
            row[f"{metric}_pvalue"] = pval
        rows.append(row)
    return pd.DataFrame(rows)


def false_positive_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """severity_level == 0 (enjekte glare yok) durumunda naive_glare_ratio ne kadar yüksek çıkıyor?"""
    sev0 = df[df["severity_level"] == 0]
    return pd.DataFrame(
        [
            {
                "metric": "naive_glare_ratio",
                "mean_false_positive_ratio": sev0["naive_glare_ratio"].mean(),
                "std": sev0["naive_glare_ratio"].std(),
                "min": sev0["naive_glare_ratio"].min(),
                "max": sev0["naive_glare_ratio"].max(),
            }
        ]
    )


def dynamic_range_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman korelasyonu yalnızca SIRALAMANIN doğru olup olmadığını ölçer;
    skorun pratikte ne kadar "ayırt edici" olduğunu (dynamic range) göstermez.
    Bu fonksiyon, severity=0 -> severity=max arasında skorun ne kadar
    değiştiğini, baseline'a göreceli olarak da ölçer.
    """
    max_level = df["severity_level"].max()
    rows = []
    for metric in ["naive_glare_ratio", "oracle_text_washout_ratio"]:
        by_level = df.groupby("severity_level")[metric].mean()
        start, end = by_level.loc[0], by_level.loc[max_level]
        rows.append(
            {
                "metric": metric,
                "mean_at_severity_0": start,
                "mean_at_max_severity": end,
                "absolute_range": end - start,
                "relative_range_pct": (
                    100.0 * (end - start) / start if start not in (0, None) and not pd.isna(start) and start != 0 else float("inf")
                ),
            }
        )
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ["naive_glare_ratio", "oracle_text_washout_ratio"]):
        for doc_id, group in df.groupby("doc_id"):
            group = group.sort_values("severity_level")
            ax.plot(group["severity_level"], group[metric], color="lightgray", linewidth=1)
        mean_curve = df.groupby("severity_level")[metric].mean()
        ax.plot(mean_curve.index, mean_curve.values, color="crimson", linewidth=2.5, label="Ortalama (12 belge)")
        ax.set_xlabel("Enjekte edilen glare şiddet seviyesi (0 = yok, 5 = en geniş)")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Naive (üretime uygun) vs. Oracle (yalnızca doğrulama) Glare Metrikleri")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "naive_vs_oracle.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(df["ground_truth_glare_fraction"], df["naive_glare_ratio"], alpha=0.5, label="naive_glare_ratio")
    ax.set_xlabel("Yer gerçeği glare alanı oranı (enjekte edilen)")
    ax.set_ylabel("naive_glare_ratio (içerik kutusu içindeki tespit oranı)")
    ax.set_title("Naive Yöntem: Yer Gerçeği Glare Alanına Göre Tepki")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "naive_vs_ground_truth_scatter.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Skorlar hesaplanıyor...")
    df = compute_scores()
    df.to_csv(RESULTS_DIR / "scores.csv", index=False)
    print(f"  -> {RESULTS_DIR / 'scores.csv'} ({len(df)} satır)")

    print("Monotonluk analizi...")
    mono_df = monotonicity_analysis(df)
    mono_df.to_csv(RESULTS_DIR / "monotonicity_summary.csv", index=False)

    print("Yanlış pozitif (false positive) baseline analizi...")
    fp_df = false_positive_baseline(df)
    fp_df.to_csv(RESULTS_DIR / "false_positive_baseline.csv", index=False)

    print("Dynamic range (etki büyüklüğü) analizi...")
    range_df = dynamic_range_analysis(df)
    range_df.to_csv(RESULTS_DIR / "dynamic_range.csv", index=False)

    print("Grafikler...")
    make_plots(df)

    print("\n=== ÖZET ===")
    for metric in ["naive_glare_ratio", "oracle_text_washout_ratio"]:
        col = f"{metric}_spearman_rho"
        print(f"{metric}: ortalama Spearman rho = {mono_df[col].mean():.4f} (std={mono_df[col].std():.4f})")

    print("\nFalse positive baseline (severity=0, naive_glare_ratio):")
    print(fp_df.to_string(index=False))

    print("\nDynamic range (severity=0 -> severity=5):")
    print(range_df.to_string(index=False))


if __name__ == "__main__":
    main()
