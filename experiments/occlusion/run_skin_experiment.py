"""
Ten rengi (skin-color) occlusion deneyi — ana çalıştırma scripti.

`generate_skin_occlusion_documents.py` ile üretilen, RASTGELE konumda
ten-tonu yaması eklenmiş görüntülerde `skin_occlusion_ratio` hesaplanır.
Üç eksende değerlendirilir:
1. Monotonluk — oran arttıkça skin_occlusion_ratio düzgün artıyor mu?
2. Hatalı-pozitif — kapanma YOKKEN (coverage=0) oran gerçekten ~0 mı?
3. Ten tonuna göre kırılım — üç ton da aynı güvenilirlikte mi?
   (Literatür, ten rengi yöntemlerinin tona duyarlı olabileceğini
   belirtiyor — bu deney bunu doğrudan ölçer.)
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

from occlusion.skin_detection import skin_occlusion_ratio  # noqa: E402

MANIFEST = PROJECT_ROOT / "data" / "synthetic" / "occlusion_skin" / "manifest.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "occlusion"
PLOTS_DIR = RESULTS_DIR / "plots"


def compute_scores() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    records = []
    for i, row in manifest.iterrows():
        img = cv2.imread(row["path"], cv2.IMREAD_COLOR)  # RENKLİ okunmalı — ten rengi için şart
        content_bbox = eval(row["content_bbox"])

        ratio = skin_occlusion_ratio(img, roi=tuple(content_bbox))

        records.append(
            {
                "doc_id": row["doc_id"],
                "skin_tone": row["skin_tone"],
                "coverage_level": int(row["coverage_level"]),
                "coverage_fraction": float(row["coverage_fraction"]),
                "skin_occlusion_ratio": ratio,
                "font_size": row["font_size"],
                "num_paragraphs": row["num_paragraphs"],
            }
        )
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(manifest)} işlendi")

    return pd.DataFrame.from_records(records)


def monotonicity_by_tone(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tone, doc_id), group in df.groupby(["skin_tone", "doc_id"]):
        group = group.sort_values("coverage_level")
        if group["skin_occlusion_ratio"].nunique() <= 1:
            rho = float("nan")
        else:
            rho, _ = spearmanr(group["coverage_level"], group["skin_occlusion_ratio"])
        rows.append({"skin_tone": tone, "doc_id": doc_id, "spearman_rho": rho})
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"acik": "#d9a066", "orta": "#a86a3d", "koyu": "#5c3a21"}
    for tone, group in df.groupby("skin_tone"):
        mean_curve = group.groupby("coverage_fraction")["skin_occlusion_ratio"].mean()
        ax.plot(mean_curve.index, mean_curve.values, marker="o", label=f"{tone} ten",
                 color=colors.get(tone, "gray"), linewidth=2)
    ax.set_xlabel("Enjekte edilen kapanma (occlusion) oranı")
    ax.set_ylabel("skin_occlusion_ratio")
    ax.set_title("Ten Rengi Tespiti — Kapanma Oranına Tepki (ton bazında)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "skin_detection_by_tone.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Ten rengi occlusion skorları hesaplanıyor...")
    df = compute_scores()
    df.to_csv(RESULTS_DIR / "skin_scores.csv", index=False)
    print(f"  -> {RESULTS_DIR / 'skin_scores.csv'} ({len(df)} satır)")

    mono_df = monotonicity_by_tone(df)
    mono_df.to_csv(RESULTS_DIR / "skin_monotonicity_by_tone.csv", index=False)

    make_plot(df)

    print("\n=== ÖZET: ton bazında ortalama Spearman rho ===")
    for tone, group in mono_df.groupby("skin_tone"):
        print(f"  {tone}: rho={group['spearman_rho'].mean():.4f} (std={group['spearman_rho'].std():.4f})")

    print("\n=== Hatalı-pozitif kontrolü (coverage=0, kapanma yok) ===")
    sev0 = df[df["coverage_level"] == 0]
    for tone, group in sev0.groupby("skin_tone"):
        print(f"  {tone}: ortalama skin_occlusion_ratio={group['skin_occlusion_ratio'].mean():.5f} "
              f"(max={group['skin_occlusion_ratio'].max():.5f})")

    print("\n=== Dynamic range (coverage=0 -> coverage=1.0) ===")
    max_cov = df["coverage_level"].max()
    for tone, group in df.groupby("skin_tone"):
        at0 = group[group.coverage_level == 0]["skin_occlusion_ratio"].mean()
        at_max = group[group.coverage_level == max_cov]["skin_occlusion_ratio"].mean()
        print(f"  {tone}: {at0:.4f} -> {at_max:.4f}")


if __name__ == "__main__":
    main()
