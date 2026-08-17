"""
Blur baseline deneyi — ana çalıştırma scripti.

Bu script:
1. data/synthetic/blur/degraded/manifest.csv içindeki tüm görüntüler için
   Laplacian Variance ve Tenengrad skorlarını hesaplar.
2. Her belge için, bozulma şiddeti arttıkça skorun monoton olarak
   azalıp azalmadığını (Spearman korelasyonu ile) test eder.
3. Font boyutuna göre "sharp" (severity=0) skorların ne kadar değiştiğini
   ölçerek, literatürde belirtilen "tek eşik farklı belgelerde
   güvenilir değil" iddiasını sınar.
4. Sonuçları results/blur/ altına CSV + grafik olarak kaydeder.

Önemli: Bu deney SENTETİK, kontrollü Gaussian blur üzerinde yapılmıştır.
Gerçek kamera/telefon blur'unu birebir temsil etmez. Bkz. project_notes.md.
"""

from __future__ import annotations

import csv
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

from blur.metrics import laplacian_variance, tenengrad, gradient_magnitude_mean  # noqa: E402

DEGRADED_MANIFEST = PROJECT_ROOT / "data" / "synthetic" / "blur" / "degraded" / "manifest.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "blur"
PLOTS_DIR = RESULTS_DIR / "plots"

METRICS = {
    "laplacian_variance": laplacian_variance,
    "tenengrad": tenengrad,
    "gradient_magnitude_mean": gradient_magnitude_mean,
}


def compute_scores() -> pd.DataFrame:
    with open(DEGRADED_MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records = []
    for row in rows:
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(row["path"])

        record = {
            "doc_id": row["doc_id"],
            "severity_level": int(row["severity_level"]),
            "sigma": float(row["sigma"]),
            "font_size": int(row["font_size"]),
            "num_paragraphs": int(row["num_paragraphs"]),
        }
        for metric_name, fn in METRICS.items():
            record[metric_name] = fn(img)
        records.append(record)

    return pd.DataFrame.from_records(records)


def monotonicity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her belge (doc_id) için: severity_level arttıkça metrik skorunun
    ne yönde ve ne kadar güçlü değiştiğini Spearman korelasyonu ile ölçer.

    Beklenti: skor ile severity arasında GÜÇLÜ NEGATİF korelasyon
    (severity arttıkça skor azalmalı -> spearman rho ~ -1.0).
    """
    rows = []
    for doc_id, group in df.groupby("doc_id"):
        group = group.sort_values("severity_level")
        row = {"doc_id": doc_id, "font_size": group["font_size"].iloc[0],
               "num_paragraphs": group["num_paragraphs"].iloc[0]}
        for metric_name in METRICS:
            rho, pval = spearmanr(group["severity_level"], group[metric_name])
            row[f"{metric_name}_spearman_rho"] = rho
            row[f"{metric_name}_pvalue"] = pval
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_density_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    severity_level == 0 (bozulmamış) görüntülerde, yalnızca font_size /
    paragraf sayısı farkı yüzünden skorların ne kadar değiştiğini ölçer.
    Bu, "tek bir mutlak Laplacian eşiği her belgede güvenilir değildir"
    iddiasını sınamaya yönelik dolaylı bir kontroldür.
    """
    sharp = df[df["severity_level"] == 0].copy()
    summary = sharp.groupby("font_size")[list(METRICS.keys())].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    cv_rows = []
    for metric_name in METRICS:
        values = sharp[metric_name].values
        cv = float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else float("nan")
        cv_rows.append({"metric": metric_name, "coefficient_of_variation_across_docs": cv})
    cv_df = pd.DataFrame(cv_rows)

    return summary, cv_df


def make_plots(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric_name in zip(axes, ["laplacian_variance", "tenengrad"]):
        for doc_id, group in df.groupby("doc_id"):
            group = group.sort_values("severity_level")
            ax.plot(group["severity_level"], group[metric_name], color="lightgray", linewidth=1)

        mean_curve = df.groupby("severity_level")[metric_name].mean()
        ax.plot(mean_curve.index, mean_curve.values, color="crimson", linewidth=2.5, label="Ortalama (12 belge)")

        ax.set_xlabel("Bozulma şiddeti seviyesi (0 = orijinal, 8 = en ağır blur)")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} vs. blur şiddeti")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Blur Şiddeti Arttıkça Sharpness Skorlarının Değişimi (sentetik veri)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "score_vs_severity.png", dpi=150)
    plt.close(fig)

    # Aynı grafiğin log-ölçekli versiyonu: skorlar şiddetle birlikte çok hızlı
    # (üstel benzeri) küçüldüğü için, üst seviyelerdeki (5-8) davranış
    # doğrusal eksende neredeyse görünmez oluyor. Log ölçek bunu netleştirir.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric_name in zip(axes, ["laplacian_variance", "tenengrad"]):
        for doc_id, group in df.groupby("doc_id"):
            group = group.sort_values("severity_level")
            ax.plot(group["severity_level"], group[metric_name], color="lightgray", linewidth=1)

        mean_curve = df.groupby("severity_level")[metric_name].mean()
        ax.plot(mean_curve.index, mean_curve.values, color="crimson", linewidth=2.5, label="Ortalama (12 belge)")
        ax.set_yscale("log")
        ax.set_xlabel("Bozulma şiddeti seviyesi (0 = orijinal, 8 = en ağır blur)")
        ax.set_ylabel(f"{metric_name} (log ölçek)")
        ax.set_title(f"{metric_name} vs. blur şiddeti (log ölçek)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("Aynı Sonuç, Log Ölçekte — Yüksek Şiddetlerdeki Davranış Daha Görünür")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "score_vs_severity_logscale.png", dpi=150)
    plt.close(fig)

    # Font boyutuna göre baseline (severity=0) skor dağılımı
    sharp = df[df["severity_level"] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric_name in zip(axes, ["laplacian_variance", "tenengrad"]):
        groups = [sharp[sharp["font_size"] == fs][metric_name].values for fs in sorted(sharp["font_size"].unique())]
        ax.boxplot(groups, tick_labels=[str(fs) for fs in sorted(sharp["font_size"].unique())])
        ax.set_xlabel("Font boyutu (px)")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name}: font boyutuna göre baseline (blur yok)")
        ax.grid(alpha=0.3)

    fig.suptitle("Bozulma Olmadan Bile Skorların Font Boyutuna Duyarlılığı")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "baseline_by_fontsize.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Skorlar hesaplanıyor...")
    df = compute_scores()
    df.to_csv(RESULTS_DIR / "scores.csv", index=False)
    print(f"  -> {RESULTS_DIR / 'scores.csv'} ({len(df)} satır)")

    print("Monotonluk analizi yapılıyor...")
    mono_df = monotonicity_analysis(df)
    mono_df.to_csv(RESULTS_DIR / "monotonicity_summary.csv", index=False)
    print(f"  -> {RESULTS_DIR / 'monotonicity_summary.csv'}")

    print("Font boyutu duyarlılık analizi yapılıyor...")
    density_summary, cv_df = baseline_density_sensitivity(df)
    density_summary.to_csv(RESULTS_DIR / "baseline_by_fontsize.csv", index=False)
    cv_df.to_csv(RESULTS_DIR / "baseline_coefficient_of_variation.csv", index=False)
    print(f"  -> {RESULTS_DIR / 'baseline_by_fontsize.csv'}")
    print(f"  -> {RESULTS_DIR / 'baseline_coefficient_of_variation.csv'}")

    print("Grafikler oluşturuluyor...")
    make_plots(df)
    print(f"  -> {PLOTS_DIR}")

    print("\n=== ÖZET ===")
    for metric_name in METRICS:
        rho_col = f"{metric_name}_spearman_rho"
        print(f"{metric_name}: ortalama Spearman rho = {mono_df[rho_col].mean():.4f} "
              f"(std={mono_df[rho_col].std():.4f}, min={mono_df[rho_col].min():.4f}, max={mono_df[rho_col].max():.4f})")

    print("\nFont boyutuna göre baseline coefficient of variation:")
    print(cv_df.to_string(index=False))


if __name__ == "__main__":
    main()
