"""
NIQE doğrulama deneyi — bilinen bozulma şiddetleriyle korelasyon testi.

Kendi yazdığımız NIQE'in (bkz. src/scoring/niqe.py), projedeki DÖRT ayrı
bozulma türünde (blur/darkness/glare/skew) bilinen şiddet seviyeleriyle
ne kadar örtüştüğünü test eder. Referans (pristine) model yalnızca BLUR
severity=0 belgelerinden fit edildiği için (fit_pristine_model.py), bu aynı
zamanda "belge-alanına-özgü referans, diğer bozulma türlerine de genelliyor
mu?" sorusunu da test eder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoring.niqe import niqe_score, _load_pristine_model  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "niqe"

DATASETS = {
    "blur": PROJECT_ROOT / "data" / "synthetic" / "blur" / "degraded" / "manifest.csv",
    "darkness": PROJECT_ROOT / "data" / "synthetic" / "darkness" / "local" / "manifest.csv",
    "glare": PROJECT_ROOT / "data" / "synthetic" / "glare" / "degraded" / "manifest.csv",
}


def evaluate_severity_dataset(name: str, manifest_path: Path, pristine_mu, pristine_cov) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows = []
    for _, row in manifest.iterrows():
        img = cv2.imread(row["path"])
        score = niqe_score(img, pristine_mu, pristine_cov)
        rows.append({"doc_id": row["doc_id"], "severity_level": row["severity_level"], "niqe": score})
        if len(rows) % 30 == 0:
            print(f"  [{name}] {len(rows)}/{len(manifest)} işlendi")
    return pd.DataFrame(rows)


def evaluate_skew_dataset(pristine_mu, pristine_cov) -> pd.DataFrame:
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "synthetic" / "skew" / "manifest.csv")
    rows = []
    for _, row in manifest.iterrows():
        img = cv2.imread(row["path"])
        score = niqe_score(img, pristine_mu, pristine_cov)
        rows.append({"doc_id": row["doc_id"], "abs_angle": abs(row["ground_truth_angle"]), "niqe": score})
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pristine_mu, pristine_cov = _load_pristine_model()

    print("=== NIQE Doğrulama Deneyi ===\n")

    all_rhos = {}
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"[{name}] atlanıyor (manifest yok: {path})")
            continue
        print(f"[{name}] işleniyor...")
        df = evaluate_severity_dataset(name, path, pristine_mu, pristine_cov)
        df.to_csv(RESULTS_DIR / f"niqe_{name}.csv", index=False)

        rhos = []
        for doc_id, group in df.groupby("doc_id"):
            group = group.sort_values("severity_level")
            if group["niqe"].isna().any() or group["niqe"].nunique() <= 1:
                continue
            rho, _ = spearmanr(group["severity_level"], group["niqe"])
            rhos.append(rho)
        mean_rho = np.nanmean(rhos) if rhos else float("nan")
        all_rhos[name] = mean_rho
        print(f"  -> ortalama Spearman rho (şiddet vs NIQE): {mean_rho:.4f}\n")

    print("[skew] işleniyor...")
    skew_df = evaluate_skew_dataset(pristine_mu, pristine_cov)
    skew_df.to_csv(RESULTS_DIR / "niqe_skew.csv", index=False)
    rho, _ = spearmanr(skew_df["abs_angle"], skew_df["niqe"])
    all_rhos["skew"] = rho
    print(f"  -> Spearman rho (|açı| vs NIQE): {rho:.4f}\n")

    print("=== ÖZET ===")
    for name, rho in all_rhos.items():
        yorum = "güçlü sinyal" if abs(rho) > 0.6 else ("orta sinyal" if abs(rho) > 0.3 else "zayıf/yok")
        print(f"  {name}: rho={rho:.4f} ({yorum})")

    pd.DataFrame([{"dataset": k, "rho": v} for k, v in all_rhos.items()]).to_csv(
        RESULTS_DIR / "niqe_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
