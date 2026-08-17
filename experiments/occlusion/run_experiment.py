"""
Occlusion baseline deneyi — ana çalıştırma scripti.

Her görüntüde "Belge No" alanı OCR edilir; kapanma (occlusion) oranı arttıkça
(a) uzunluk oranının (length_ratio) ve (b) OCR güveninin (confidence) nasıl
değiştiği ölçülür.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd
from scipy.stats import spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from occlusion.metrics import occlusion_suspicion_score  # noqa: E402

MANIFEST = PROJECT_ROOT / "data" / "synthetic" / "occlusion" / "manifest.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "occlusion"
PLOTS_DIR = RESULTS_DIR / "plots"

DIGIT_WHITELIST = "0123456789"


def compute_scores() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    records = []
    for i, row in manifest.iterrows():
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        bbox = eval(row["field_bbox"])  # json list saklandı, basit parse

        result = occlusion_suspicion_score(
            img, tuple(bbox), expected_length=int(row["expected_length"]),
            char_whitelist=DIGIT_WHITELIST,
        )

        records.append(
            {
                "doc_id": row["doc_id"],
                "coverage_level": int(row["coverage_level"]),
                "coverage_fraction": float(row["coverage_fraction"]),
                "recognized_text": result["recognized_text"],
                "recognized_length": result["recognized_length"],
                "length_ratio": result["length_ratio"],
                "mean_confidence": result["mean_confidence"],
                "occlusion_suspicion_score": result["occlusion_suspicion_score"],
                "font_size": row["font_size"],
                "num_paragraphs": row["num_paragraphs"],
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(manifest)} işlendi")

    return pd.DataFrame.from_records(records)


def monotonicity_analysis(df: pd.DataFrame, metrics) -> pd.DataFrame:
    rows = []
    for doc_id, group in df.groupby("doc_id"):
        group = group.sort_values("coverage_level")
        row = {"doc_id": doc_id}
        for metric in metrics:
            if group[metric].nunique() <= 1:
                rho = float("nan")
            else:
                rho, _ = spearmanr(group["coverage_level"], group[metric])
            row[f"{metric}_spearman_rho"] = rho
        rows.append(row)
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["length_ratio", "mean_confidence", "occlusion_suspicion_score"]
    for ax, metric in zip(axes, metrics):
        for doc_id, group in df.groupby("doc_id"):
            group = group.sort_values("coverage_fraction")
            ax.plot(group["coverage_fraction"], group[metric], color="lightgray", linewidth=1)
        mean_curve = df.groupby("coverage_fraction")[metric].mean()
        ax.plot(mean_curve.index, mean_curve.values, color="crimson", linewidth=2.5, label="Ortalama (12 belge)")
        ax.set_xlabel("Enjekte edilen kapanma (occlusion) oranı")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Occlusion Oranı Arttıkça OCR Tabanlı Metriklerin Değişimi")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "metrics_vs_coverage.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("OCR + occlusion skorları hesaplanıyor (bu biraz sürebilir)...")
    df = compute_scores()
    df.to_csv(RESULTS_DIR / "scores.csv", index=False)
    print(f"  -> {RESULTS_DIR / 'scores.csv'} ({len(df)} satır)")

    metrics = ["length_ratio", "mean_confidence", "occlusion_suspicion_score"]
    mono_df = monotonicity_analysis(df, metrics)
    mono_df.to_csv(RESULTS_DIR / "monotonicity_summary.csv", index=False)

    make_plots(df)

    print("\n=== ÖZET (ortalama Spearman rho, 12 belge) ===")
    for metric in metrics:
        col = f"{metric}_spearman_rho"
        print(f"{metric}: {mono_df[col].mean():.4f} (std={mono_df[col].std():.4f}, NaN sayısı={mono_df[col].isna().sum()})")

    print("\nCoverage=0 (kapanma yok) durumunda ortalama uzunluk oranı ve güven:")
    sev0 = df[df["coverage_level"] == 0]
    print(f"  length_ratio: {sev0['length_ratio'].mean():.3f}, mean_confidence: {sev0['mean_confidence'].mean():.1f}")
    print("\nCoverage=1.0 (tamamen kapalı) durumunda ortalama uzunluk oranı ve güven:")
    sev_max = df[df["coverage_level"] == df["coverage_level"].max()]
    print(f"  length_ratio: {sev_max['length_ratio'].mean():.3f}, mean_confidence: {sev_max['mean_confidence'].mean():.1f}")


if __name__ == "__main__":
    main()
