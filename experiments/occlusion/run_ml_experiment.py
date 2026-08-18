"""
ML tabanlı occlusion sınıflandırıcısının GENELLEME doğrulaması.

`train_occlusion_classifier.py`'de HİÇ görülmeyen renklerle (3 ten tonu +
turkuaz + lacivert) — hem düz hem dokulu/gürültülü varyantlarda — test
edilir. Amaç: modelin belirli renkleri ezberlemediğini, gerçekten
renk+doku örüntüsünü öğrendiğini kanıtlamak.
"""

from __future__ import annotations

import random
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
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "_common"))

from synthetic_documents import render_document, default_combinations  # noqa: E402
from occlusion.ml_detection import ml_occlusion_ratio  # noqa: E402
from train_occlusion_classifier import apply_patch  # noqa: E402

RANDOM_SEED = 123  # eğitimden FARKLI seed — belge içerikleri de görülmemiş olsun
RESULTS_DIR = PROJECT_ROOT / "results" / "occlusion"
PLOTS_DIR = RESULTS_DIR / "plots"

# EĞİTİMDE HİÇ GÖRÜLMEYEN renkler — asıl test bu.
HELD_OUT_COLORS = {
    "acik_ten": (224, 172, 135), "orta_ten": (198, 134, 92), "koyu_ten": (110, 74, 51),
    "turkuaz": (30, 180, 170), "lacivert": (20, 30, 90),
}
COVERAGE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    rng = random.Random(RANDOM_SEED)
    combos = list(default_combinations())[:6]  # 6 belge yeterli, hızlı olsun

    rows = []
    for font_size, num_paragraphs, replica in combos:
        doc = render_document(font_size, num_paragraphs, rng)
        doc_name = f"doc_{font_size}_{num_paragraphs}_{replica}"
        for tone_name, color in HELD_OUT_COLORS.items():
            for textured in (False, True):
                for level, coverage in enumerate(COVERAGE_LEVELS):
                    rgb_arr = np.array(doc.image.convert("RGB"))
                    patch_bbox = apply_patch(rgb_arr, doc.content_bbox, coverage, color, textured, rng)
                    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
                    ratio = ml_occlusion_ratio(bgr, roi=doc.content_bbox)
                    rows.append({
                        "doc_id": doc_name, "tone": tone_name, "textured": textured,
                        "coverage_level": level, "coverage_fraction": coverage,
                        "ml_occlusion_ratio": ratio,
                    })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "ml_scores.csv", index=False)
    print(f"-> {RESULTS_DIR / 'ml_scores.csv'} ({len(df)} satır)")

    print("\n=== ÖZET: (görülmemiş renk x doku) bazında ortalama Spearman rho ===")
    mono_rows = []
    for (tone, textured), group in df.groupby(["tone", "textured"]):
        rhos = []
        for doc_id, g in group.groupby("doc_id"):
            g = g.sort_values("coverage_level")
            if g["ml_occlusion_ratio"].nunique() <= 1:
                continue
            rho, _ = spearmanr(g["coverage_level"], g["ml_occlusion_ratio"])
            rhos.append(rho)
        mean_rho = np.nanmean(rhos) if rhos else float("nan")
        mono_rows.append({"tone": tone, "textured": textured, "mean_rho": mean_rho})
        print(f"  {tone} ({'dokulu' if textured else 'düz'}): rho={mean_rho:.4f}")

    pd.DataFrame(mono_rows).to_csv(RESULTS_DIR / "ml_monotonicity.csv", index=False)

    print("\n=== Hatalı-pozitif (coverage=0) ===")
    sev0 = df[df.coverage_level == 0]
    print(f"  ortalama={sev0['ml_occlusion_ratio'].mean():.5f}, max={sev0['ml_occlusion_ratio'].max():.5f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for (tone, textured), group in df.groupby(["tone", "textured"]):
        mean_curve = group.groupby("coverage_fraction")["ml_occlusion_ratio"].mean()
        style = "--" if textured else "-"
        ax.plot(mean_curve.index, mean_curve.values, style, marker="o", markersize=3,
                 label=f"{tone} ({'dokulu' if textured else 'düz'})", alpha=0.8)
    ax.set_xlabel("Enjekte edilen kapanma oranı")
    ax.set_ylabel("ml_occlusion_ratio")
    ax.set_title("ML Occlusion — GÖRÜLMEMİŞ Renk/Doku Genelleme Testi")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ml_generalization.png", dpi=150)
    plt.close(fig)
    print(f"-> {PLOTS_DIR / 'ml_generalization.png'}")


if __name__ == "__main__":
    main()
