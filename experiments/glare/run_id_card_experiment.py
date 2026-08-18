"""
Kimlik kartı benzeri (renkli zeminli) belgelerde MEVCUT (değiştirilmemiş)
HSV+CC glare yönteminin (src/glare/metrics.py) doğrulaması.

Amaç: aynı basit, ML gerektirmeyen yöntemin, düz beyaz kağıtta neden
başarısız olduğunu (project_notes.md, "Glare Aşama 1" ve "Glare ML v1-v5")
renkli zeminde de başarısız olup olmadığını test etmek. Ayrıca bulanıklık
karışması (blur confound) de ayrıca test edilir — bu, beyaz kağıt
denemelerinde asıl çözülemeyen sorundu.
"""

from __future__ import annotations

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

MANIFEST = PROJECT_ROOT / "data" / "synthetic" / "glare_id_card" / "manifest.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "glare"
PLOTS_DIR = RESULTS_DIR / "plots"


def main():
    manifest = pd.read_csv(MANIFEST)
    rows = []
    for _, row in manifest.iterrows():
        img = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        ratio = glare_ratio(img)  # DEĞİŞTİRİLMEMİŞ, zaten var olan fonksiyon
        rows.append({
            "card_id": row["card_id"], "color_scheme": row["color_scheme"],
            "severity_level": row["severity_level"],
            "target_area_fraction": row["target_area_fraction"],
            "glare_ratio": ratio,
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "id_card_scores.csv", index=False)
    print(f"-> {RESULTS_DIR / 'id_card_scores.csv'} ({len(df)} satır)")

    print("\n=== ÖZET: renk şeması bazında ortalama Spearman rho ===")
    mono_rows = []
    for (scheme, card_id), group in df.groupby(["color_scheme", "card_id"]):
        group = group.sort_values("severity_level")
        rho, _ = spearmanr(group["severity_level"], group["glare_ratio"])
        mono_rows.append({"color_scheme": scheme, "card_id": card_id, "rho": rho})
    mono_df = pd.DataFrame(mono_rows)
    mono_df.to_csv(RESULTS_DIR / "id_card_monotonicity.csv", index=False)
    print(mono_df.groupby("color_scheme")["rho"].agg(["mean", "std"]).round(4))
    print(f"\nGenel ortalama rho: {mono_df['rho'].mean():.4f}")

    sev0 = df[df.severity_level == 0]
    sev_max = df[df.severity_level == df.severity_level.max()]
    print(f"\nSeverity=0 (glare yok) hatalı-pozitif: ortalama={sev0['glare_ratio'].mean():.4f}, max={sev0['glare_ratio'].max():.4f}")
    print(f"Severity=max gerçek-pozitif: ortalama={sev_max['glare_ratio'].mean():.4f}")

    print("\n=== BULANIKLIK+GLARE YOK hatalı-pozitif testi (renkli zeminde) ===")
    scheme_colors = {
        "mavi_gri": (196, 214, 226), "bej": (232, 220, 196), "yesilimsi": (205, 224, 210),
    }
    blur_rows = []
    from PIL import Image, ImageDraw
    for scheme_name, bg in scheme_colors.items():
        img = Image.new("RGB", (856, 540), color=bg)
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, 260, 340], fill=tuple(max(0, c - 40) for c in bg))
        for ty in range(60, 320, 40):
            draw.rectangle([300, ty, 650, ty + 18], fill=(40, 50, 90))
        arr = np.array(img)
        for sigma in [0, 1, 2, 4, 6, 8]:
            blurred = arr if sigma == 0 else cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
            ratio = glare_ratio(cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
            blur_rows.append({"color_scheme": scheme_name, "blur_sigma": sigma, "glare_ratio": ratio})
    blur_df = pd.DataFrame(blur_rows)
    blur_df.to_csv(RESULTS_DIR / "id_card_blur_false_positive_check.csv", index=False)
    print(blur_df.pivot(index="blur_sigma", columns="color_scheme", values="glare_ratio").round(4))
    print(f"\nEn kötü durum hatalı-pozitif: {blur_df['glare_ratio'].max():.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for (scheme, card_id), group in df.groupby(["color_scheme", "card_id"]):
        group = group.sort_values("target_area_fraction")
        ax.plot(group["target_area_fraction"], group["glare_ratio"], marker="o", alpha=0.5,
                 label=scheme if card_id.endswith("001") or "_001_" in card_id else None)
    ax.set_xlabel("Enjekte edilen glare alanı oranı")
    ax.set_ylabel("glare_ratio (mevcut HSV+CC yöntemi)")
    ax.set_title("Kimlik Kartı Zemininde Glare Tespiti (ML YOK, klasik yöntem)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "id_card_generalization.png", dpi=150)
    plt.close(fig)
    print(f"-> {PLOTS_DIR / 'id_card_generalization.png'}")


if __name__ == "__main__":
    main()
