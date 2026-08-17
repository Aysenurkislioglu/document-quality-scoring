"""
Darkness baseline deneyi — ana çalıştırma scripti.

İki senaryo ayrı ayrı analiz edilir:
- GLOBAL: tüm görüntü karartılmış. Beklenti: global_mean dahil TÜM
  metrikler şiddetle birlikte düzgünce azalmalı.
- LOKAL: yalnızca küçük bir alan (Belge No) karartılmış, görüntünün geri
  kalanı aynı. Beklenti: global_mean neredeyse SABİT kalmalı (çünkü
  karartılan alan görüntünün <%1'i), ama darkest_block_mean (yerel analiz)
  belirgin şekilde düşmeli. Percentile'ların (özellikle P5) bu küçük alanı
  yakalayıp yakalayamadığı da ayrıca raporlanır.
"""

from __future__ import annotations

import json
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

from darkness.metrics import global_brightness, brightness_percentiles, darkest_block_mean  # noqa: E402

DARKNESS_DIR = PROJECT_ROOT / "data" / "synthetic" / "darkness"
RESULTS_DIR = PROJECT_ROOT / "results" / "darkness"
PLOTS_DIR = RESULTS_DIR / "plots"

BLOCK_SIZE = 16  # küçük kimlik alanını (~105x26 px) yakalayabilmek için küçük blok


def compute_scenario_scores(scenario: str) -> pd.DataFrame:
    manifest = pd.read_csv(DARKNESS_DIR / scenario / "manifest.csv")
    records = []
    for _, row in manifest.iterrows():
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        gb = global_brightness(img)
        pct = brightness_percentiles(img)
        dbm = darkest_block_mean(img, block_size=BLOCK_SIZE)

        record = {
            "doc_id": row["doc_id"],
            "severity_level": int(row["severity_level"]),
            "darkening_factor": float(row["darkening_factor"]),
            "global_mean": gb["mean"],
            "darkest_block_mean": dbm,
            **pct,
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def monotonicity_analysis(df: pd.DataFrame, metrics) -> pd.DataFrame:
    rows = []
    for doc_id, group in df.groupby("doc_id"):
        group = group.sort_values("severity_level")
        row = {"doc_id": doc_id}
        for metric in metrics:
            rho, pval = spearmanr(group["severity_level"], group[metric])
            row[f"{metric}_spearman_rho"] = rho
        rows.append(row)
    return pd.DataFrame(rows)


def dynamic_range(df: pd.DataFrame, metrics) -> pd.DataFrame:
    max_level = df["severity_level"].max()
    rows = []
    for metric in metrics:
        by_level = df.groupby("severity_level")[metric].mean()
        start, end = by_level.loc[0], by_level.loc[max_level]
        rel = 100.0 * (end - start) / start if start != 0 else float("inf")
        rows.append(
            {
                "metric": metric,
                "mean_at_severity_0": start,
                "mean_at_max_severity": end,
                "absolute_change": end - start,
                "relative_change_pct": rel,
            }
        )
    return pd.DataFrame(rows)


def make_comparison_plot(df_global: pd.DataFrame, df_local: pd.DataFrame, metrics, filename: str, title: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        g_mean = df_global.groupby("severity_level")[metric].mean()
        l_mean = df_local.groupby("severity_level")[metric].mean()
        ax.plot(g_mean.index, g_mean.values, marker="o", color="crimson", label="Global karanlık senaryosu")
        ax.plot(l_mean.index, l_mean.values, marker="s", color="steelblue", label="Lokal karanlık senaryosu (yalnızca Belge No)")
        ax.set_xlabel("Şiddet seviyesi (0 = orijinal, 5 = en karanlık)")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["global_mean", "p5", "p25", "p50", "darkest_block_mean"]

    print("GLOBAL senaryo skorları hesaplanıyor...")
    df_global = compute_scenario_scores("global")
    df_global.to_csv(RESULTS_DIR / "scores_global.csv", index=False)

    print("LOKAL senaryo skorları hesaplanıyor...")
    df_local = compute_scenario_scores("local")
    df_local.to_csv(RESULTS_DIR / "scores_local.csv", index=False)

    print("Monotonluk analizi...")
    mono_global = monotonicity_analysis(df_global, metrics)
    mono_local = monotonicity_analysis(df_local, metrics)
    mono_global.to_csv(RESULTS_DIR / "monotonicity_global.csv", index=False)
    mono_local.to_csv(RESULTS_DIR / "monotonicity_local.csv", index=False)

    print("Dynamic range analizi...")
    range_global = dynamic_range(df_global, metrics)
    range_local = dynamic_range(df_local, metrics)
    range_global.to_csv(RESULTS_DIR / "dynamic_range_global.csv", index=False)
    range_local.to_csv(RESULTS_DIR / "dynamic_range_local.csv", index=False)

    print("Grafikler...")
    make_comparison_plot(
        df_global, df_local,
        ["global_mean", "p5"],
        "global_vs_local_meanp5.png",
        "Global Ortalama ve P5: Global vs Lokal Karanlık Senaryosu",
    )
    make_comparison_plot(
        df_global, df_local,
        ["darkest_block_mean"],
        "global_vs_local_darkestblock.png",
        "Darkest Block Mean: Global vs Lokal Karanlık Senaryosu",
    )

    print("\n=== ÖZET: GLOBAL senaryo (ortalama Spearman rho) ===")
    for metric in metrics:
        print(f"  {metric}: {mono_global[f'{metric}_spearman_rho'].mean():.4f}")

    print("\n=== ÖZET: LOKAL senaryo (ortalama Spearman rho) ===")
    for metric in metrics:
        print(f"  {metric}: {mono_local[f'{metric}_spearman_rho'].mean():.4f}")

    print("\n=== LOKAL senaryo dynamic range (metrik ne kadar 'tepki veriyor'?) ===")
    print(range_local.to_string(index=False))


if __name__ == "__main__":
    main()
