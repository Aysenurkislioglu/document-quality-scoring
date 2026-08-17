"""
Skew baseline deneyi — ana çalıştırma scripti.

Her görüntü için hem Hough hem Projection Profile ile açı tahmini yapılır,
bilinen (ground truth) açıyla karşılaştırılır (mutlak hata = |tahmin - gerçek|).
Ayrıca hatanın açı büyüklüğüne (küçük/orta/büyük skew) göre nasıl değiştiği
ayrıca incelenir — literatürün doğrudan test edilmesini istediği bir nokta
(bkz. research/ ilk analiz, "deneysel olarak test edilmeli" bölümü).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from skew.metrics import estimate_skew_hough, estimate_skew_projection_profile  # noqa: E402

MANIFEST = PROJECT_ROOT / "data" / "synthetic" / "skew" / "manifest.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "skew"
PLOTS_DIR = RESULTS_DIR / "plots"


def angle_bucket(angle: float) -> str:
    a = abs(angle)
    if a == 0:
        return "0 (referans)"
    if a <= 2:
        return "küçük (1-2°)"
    if a <= 8:
        return "orta (5-8°)"
    return "büyük (12°)"


def compute_scores() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    records = []
    t0 = time.time()
    for i, row in manifest.iterrows():
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        hough_est = estimate_skew_hough(img)
        proj_est = estimate_skew_projection_profile(img, angle_range=(-15, 15), angle_step=0.5)

        gt = row["ground_truth_angle"]
        record = {
            "doc_id": row["doc_id"],
            "ground_truth_angle": gt,
            "hough_estimate": hough_est,
            "hough_abs_error": abs(hough_est - gt) if hough_est is not None else np.nan,
            "hough_failed": hough_est is None,
            "projection_estimate": proj_est,
            "projection_abs_error": abs(proj_est - gt),
            "font_size": row["font_size"],
            "num_paragraphs": row["num_paragraphs"],
            "angle_bucket": angle_bucket(gt),
        }
        records.append(record)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(manifest)} işlendi ({time.time() - t0:.1f}s)")

    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    order = ["0 (referans)", "küçük (1-2°)", "orta (5-8°)", "büyük (12°)"]
    summary = (
        df.groupby("angle_bucket")[["hough_abs_error", "projection_abs_error"]]
        .agg(["mean", "std", "max"])
        .reindex(order)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()


def make_plots(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["ground_truth_angle"], df["hough_estimate"], alpha=0.5, label="Hough Transform", marker="o")
    ax.scatter(df["ground_truth_angle"], df["projection_estimate"], alpha=0.5, label="Projection Profile", marker="x")
    lims = [df["ground_truth_angle"].min() - 1, df["ground_truth_angle"].max() + 1]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="Mükemmel tahmin (y=x)")
    ax.set_xlabel("Yer gerçeği açı (derece)")
    ax.set_ylabel("Tahmin edilen açı (derece)")
    ax.set_title("Skew Açısı: Tahmin vs. Yer Gerçeği")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "prediction_vs_ground_truth.png", dpi=150)
    plt.close(fig)

    order = ["0 (referans)", "küçük (1-2°)", "orta (5-8°)", "büyük (12°)"]
    means = df.groupby("angle_bucket")[["hough_abs_error", "projection_abs_error"]].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(order))
    width = 0.35
    ax.bar(x - width / 2, means["hough_abs_error"], width, label="Hough Transform")
    ax.bar(x + width / 2, means["projection_abs_error"], width, label="Projection Profile")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Ortalama Mutlak Hata (derece)")
    ax.set_title("Açı Büyüklüğüne Göre Ortalama Hata")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "error_by_angle_bucket.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Skorlar hesaplanıyor (bu işlem biraz sürebilir)...")
    df = compute_scores()
    df.to_csv(RESULTS_DIR / "scores.csv", index=False)
    print(f"  -> {RESULTS_DIR / 'scores.csv'} ({len(df)} satır)")

    summary = summarize(df)
    summary.to_csv(RESULTS_DIR / "error_by_angle_bucket.csv", index=False)

    make_plots(df)

    print("\n=== ÖZET: Genel Ortalama Mutlak Hata (MAE) ===")
    print(f"Hough Transform:     {df['hough_abs_error'].mean():.4f}° (std={df['hough_abs_error'].std():.4f}, hough_failed={df['hough_failed'].sum()} görüntü)")
    print(f"Projection Profile:  {df['projection_abs_error'].mean():.4f}° (std={df['projection_abs_error'].std():.4f})")

    print("\n=== Açı büyüklüğüne göre MAE ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
