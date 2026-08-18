"""
Bağlam-farkında glare sınıflandırıcısının GENELLEME doğrulaması.

`train_glare_classifier.py`'de kullanılan standart 12 belgelik ızgaradan
(RANDOM_SEED=42) TAMAMEN FARKLI bir random seed ile YENİ belgeler üretilip
test edilir — amaç, modelin belirli belge içeriklerine (kelimeler, paragraf
yerleşimi) aşırı uymadığını (overfit) doğrulamak.

Ayrıca BULANIKLIK+PARLAMASIZ hatalı-pozitif testi de içerir — v1'de
gerçek kullanımda (roi=None, üretimdeki gibi) keşfedilen bir hatanın
(bulanık belgeler "parlama var" olarak işaretleniyordu, bkz.
project_notes.md, "Glare ML v2") bir daha kaçırılmaması için.

NOT: Skorlar `roi=None` (tüm görüntü) ile hesaplanır — bu, üretimde
(src/scoring/fusion.py) gerçekte kullanılan çağrı biçimidir; gerçek bir
fotoğrafın content_bbox'ı bilinmez.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "_common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_documents import render_document  # noqa: E402
from generate_glare_documents import apply_glare_blob, sigma_for_target_area, TARGET_AREA_FRACTIONS  # noqa: E402
from glare.ml_detection import glare_ml_ratio  # noqa: E402

HELD_OUT_SEED = 999  # eğitimden (42) TAMAMEN FARKLI — belge içerikleri de görülmemiş
RESULTS_DIR = PROJECT_ROOT / "results" / "glare"
PLOTS_DIR = RESULTS_DIR / "plots"

# Eğitimde kullanılmayan font/paragraf kombinasyonları (varyasyon için)
HELD_OUT_COMBOS = [(16, 3, 0), (22, 4, 0), (12, 1, 0), (26, 6, 0), (18, 2, 0), (24, 5, 0)]


def main():
    rng = random.Random(HELD_OUT_SEED)
    rows = []

    for font_size, num_paragraphs, replica in HELD_OUT_COMBOS:
        doc = render_document(font_size, num_paragraphs, rng)
        img_array = np.array(doc.image)
        x0, y0, x1, y1 = doc.content_bbox
        w, h = x1 - x0, y1 - y0
        center = (x0 + w // 2, y0 + h // 2)
        content_area_px = w * h
        doc_name = f"held_out_{font_size}_{num_paragraphs}"

        for level, frac in enumerate(TARGET_AREA_FRACTIONS):
            target_area_px = frac * content_area_px
            sigma = sigma_for_target_area(target_area_px)
            degraded_array, alpha = apply_glare_blob(img_array, center, sigma)

            alpha_roi = alpha[y0:y1, x0:x1]
            ground_truth_fraction = float(np.count_nonzero(alpha_roi > 0.5)) / content_area_px

            ratio = glare_ml_ratio(degraded_array, roi=None)

            rows.append({
                "doc_id": doc_name, "severity_level": level,
                "target_area_fraction": frac,
                "ground_truth_glare_fraction": ground_truth_fraction,
                "glare_ml_ratio": ratio,
            })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "ml_scores.csv", index=False)
    print(f"-> {RESULTS_DIR / 'ml_scores.csv'} ({len(df)} satır)")

    print("\n=== ÖZET: GÖRÜLMEMİŞ belgelerde (yeni random seed) ===")
    mono_rows = []
    for doc_id, group in df.groupby("doc_id"):
        group = group.sort_values("severity_level")
        rho, _ = spearmanr(group["severity_level"], group["glare_ml_ratio"])
        mono_rows.append({"doc_id": doc_id, "rho": rho})
        ratios_str = ", ".join(f"{r:.3f}" for r in group["glare_ml_ratio"])
        print(f"  {doc_id}: [{ratios_str}]  rho={rho:.4f}")

    pd.DataFrame(mono_rows).to_csv(RESULTS_DIR / "ml_monotonicity.csv", index=False)
    print(f"\nOrtalama rho: {np.mean([r['rho'] for r in mono_rows]):.4f}")

    sev0 = df[df.severity_level == 0]
    sev_max = df[df.severity_level == df.severity_level.max()]
    print(f"\nSeverity=0 (glare yok) ortalama hatalı-pozitif: {sev0['glare_ml_ratio'].mean():.4f}")
    print(f"Severity=max (en ağır) ortalama gerçek-pozitif: {sev_max['glare_ml_ratio'].mean():.4f}")

    print("\n=== BULANIKLIK+PARLAMASIZ hatalı-pozitif testi (v1'de bulunan hata) ===")
    import cv2
    blur_rows = []
    rng2 = random.Random(HELD_OUT_SEED + 1)
    for font_size, num_paragraphs, replica in HELD_OUT_COMBOS:
        doc = render_document(font_size, num_paragraphs, rng2)
        img_array = np.array(doc.image)
        for sigma in [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]:
            blurred = img_array if sigma == 0 else cv2.GaussianBlur(
                img_array, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma
            )
            ratio = glare_ml_ratio(blurred, roi=None)
            blur_rows.append({"font_size": font_size, "blur_sigma": sigma, "glare_ml_ratio": ratio})
    blur_df = pd.DataFrame(blur_rows)
    blur_df.to_csv(RESULTS_DIR / "ml_blur_false_positive_check.csv", index=False)
    print(blur_df.groupby("blur_sigma")["glare_ml_ratio"].agg(["mean", "max"]).round(4))
    max_blur_fp = blur_df["glare_ml_ratio"].max()
    print(f"\nEn kötü durum (herhangi bir bulanıklık şiddetinde) hatalı-pozitif: {max_blur_fp:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for doc_id, group in df.groupby("doc_id"):
        group = group.sort_values("target_area_fraction")
        ax.plot(group["target_area_fraction"], group["glare_ml_ratio"], marker="o", label=doc_id, alpha=0.8)
    ax.set_xlabel("Enjekte edilen glare alanı oranı (yer gerçeği)")
    ax.set_ylabel("glare_ml_ratio")
    ax.set_title("Bağlam-farkında Glare ML — GÖRÜLMEMİŞ Belge Doğrulaması")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ml_generalization.png", dpi=150)
    plt.close(fig)
    print(f"-> {PLOTS_DIR / 'ml_generalization.png'}")


if __name__ == "__main__":
    main()
