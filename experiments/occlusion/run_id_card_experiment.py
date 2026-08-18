"""
ML occlusion sınıflandırıcısının RENKLİ KİMLİK KARTI genelleme doğrulaması.

`train_occlusion_classifier.py`'de eğitimde kullanılan 5 kart renk şemasından
(mavi_gri, bej, yesilimsi, gri, pembemsi) TAMAMEN FARKLI, HİÇ GÖRÜLMEYEN yeni
renk şemalarıyla test edilir. İki soru sorulur:

1. Hatalı-pozitif: Hiç kapanma olmayan renkli bir kartta oran ~0 mı?
   (Bu, kullanıcının bildirdiği orijinal hatanın — "kapanma yokken oran
   %95-99" — düzeltildiğinin asıl kanıtı.)
2. Gerçek-pozitif: Kartın kendi renginden FARKLI bir yama eklenince model
   bunu hâlâ yakalıyor mu? (Düzeltme, gerçek kapanmaları da kaçırmaya
   başlamadı mı?)
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
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "glare"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "occlusion"))

from generate_id_card_documents import render_id_card, CARD_SIZE  # noqa: E402
from occlusion.ml_detection import ml_occlusion_ratio  # noqa: E402
from train_occlusion_classifier import apply_patch  # noqa: E402

RANDOM_SEED = 777  # eğitimden (42) TAMAMEN FARKLI
RESULTS_DIR = PROJECT_ROOT / "results" / "occlusion"
PLOTS_DIR = RESULTS_DIR / "plots"

# EĞİTİMDE HİÇ GÖRÜLMEYEN kart renk şemaları (train'deki 5 şemadan hiçbiri
# değil: mavi_gri, bej, yesilimsi, gri, pembemsi).
HELD_OUT_CARD_SCHEMES = {
    "lila": {"bg": (222, 206, 232), "photo": (185, 165, 200), "text": (60, 35, 90)},
    "sari_krem": {"bg": (238, 228, 190), "photo": (205, 190, 145), "text": (95, 75, 20)},
    "turkuaz": {"bg": (198, 228, 226), "photo": (150, 195, 190), "text": (20, 75, 70)},
}
# Yama renkleri de eğitimdeki 14 renkten FARKLI seçildi.
HELD_OUT_PATCH_COLORS = {
    "acik_ten": (224, 172, 135), "turuncu_koyu": (180, 90, 20), "gri_mavi": (110, 130, 150),
}
COVERAGE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    rng = random.Random(RANDOM_SEED)
    card_bbox = (0, 0, CARD_SIZE[0], CARD_SIZE[1])
    rows = []

    for scheme_name, scheme in HELD_OUT_CARD_SCHEMES.items():
        for patch_name, color in HELD_OUT_PATCH_COLORS.items():
            for textured in (False, True):
                for level, coverage in enumerate(COVERAGE_LEVELS):
                    card_img = render_id_card(scheme, rng)
                    rgb_arr = np.array(card_img)
                    apply_patch(rgb_arr, card_bbox, coverage, color, textured, rng)
                    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
                    ratio = ml_occlusion_ratio(bgr)
                    rows.append({
                        "card_scheme": scheme_name, "patch_color": patch_name, "textured": textured,
                        "coverage_level": level, "coverage_fraction": coverage,
                        "ml_occlusion_ratio": ratio,
                    })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "id_card_scores.csv", index=False)
    print(f"-> {RESULTS_DIR / 'id_card_scores.csv'} ({len(df)} satır)")

    print("\n=== Hatalı-pozitif (coverage=0, HİÇ GÖRÜLMEMİŞ kart renginde) ===")
    sev0 = df[df.coverage_level == 0]
    print(f"  ortalama={sev0['ml_occlusion_ratio'].mean():.5f}, max={sev0['ml_occlusion_ratio'].max():.5f}")
    print(sev0.groupby("card_scheme")["ml_occlusion_ratio"].agg(["mean", "max"]).round(5))

    print("\n=== Gerçek-pozitif: (kart şeması x yama rengi x doku) bazında rho ===")
    mono_rows = []
    for (scheme, patch, textured), group in df.groupby(["card_scheme", "patch_color", "textured"]):
        group = group.sort_values("coverage_level")
        if group["ml_occlusion_ratio"].nunique() <= 1:
            rho = float("nan")
        else:
            rho, _ = spearmanr(group["coverage_level"], group["ml_occlusion_ratio"])
        mono_rows.append({"card_scheme": scheme, "patch_color": patch, "textured": textured, "rho": rho})
    mono_df = pd.DataFrame(mono_rows)
    mono_df.to_csv(RESULTS_DIR / "id_card_monotonicity.csv", index=False)
    print(f"Ortalama rho: {mono_df['rho'].mean():.4f} (std={mono_df['rho'].std():.4f})")
    print(mono_df.groupby("card_scheme")["rho"].mean().round(4))

    sev_max = df[df.coverage_level == df.coverage_level.max()]
    print(f"\nCoverage=1.0 ortalama oran: {sev_max['ml_occlusion_ratio'].mean():.4f} (yüksek olmalı)")

    fig, ax = plt.subplots(figsize=(8, 5))
    for (scheme, patch), group in df[~df.textured].groupby(["card_scheme", "patch_color"]):
        mean_curve = group.groupby("coverage_fraction")["ml_occlusion_ratio"].mean()
        ax.plot(mean_curve.index, mean_curve.values, marker="o", markersize=3,
                 label=f"{scheme} + {patch}", alpha=0.8)
    ax.set_xlabel("Enjekte edilen kapanma oranı")
    ax.set_ylabel("ml_occlusion_ratio")
    ax.set_title("Occlusion ML — GÖRÜLMEMİŞ Kimlik Kartı Şeması Genelleme Testi")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "id_card_generalization.png", dpi=150)
    plt.close(fig)
    print(f"-> {PLOTS_DIR / 'id_card_generalization.png'}")


if __name__ == "__main__":
    main()
