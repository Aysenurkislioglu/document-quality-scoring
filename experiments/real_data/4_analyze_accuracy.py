"""
ADIM 4 — Etiketlerin (senin gözünle) ve sistemin skorlarının GERÇEK
karşılaştırması. Bu, projede ilk kez üretilen GERÇEK doğruluk ölçümüdür
(bkz. project_notes.md — o ana kadarki her şey yalnızca sentetik veriyle
doğrulanmıştı).

Çıktı yalnızca anon_id kullanır, gerçek dosya adı hiçbir yerde yok.

Kullanım:
    python3 experiments/real_data/4_analyze_accuracy.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = PROJECT_ROOT / "results" / "real_data" / "labels.csv"
SCORES_PATH = PROJECT_ROOT / "results" / "real_data" / "scores.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "real_data" / "accuracy_report.txt"

QUALITY_ORDINAL = {"kotu": 0, "orta": 1, "iyi": 2}
# Her defekt etiketi -> ilgili sistem modülü (yüksek skor = iyi, bu yüzden
# etiket=1 [defekt VAR] iken modül skorunun DÜŞÜK olmasını bekliyoruz).
FLAG_TO_MODULE = {
    "bulanik": "blur_score",
    "parlama": "glare_score",
    "karanlik": "darkness_score",
    "kapanma": "occlusion_score",
    "egik": "skew_score",
}


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row["anon_id"]): row for row in csv.DictReader(f)}


def main():
    if not LABELS_PATH.exists() or not SCORES_PATH.exists():
        print("Önce 2_label_tool.py VE 3_batch_score.py çalıştırılmalı.")
        return

    labels = load_csv(LABELS_PATH)
    scores = load_csv(SCORES_PATH)
    common_ids = sorted(set(labels) & set(scores))

    if len(common_ids) < 5:
        print(f"Yalnızca {len(common_ids)} ortak (hem etiketli hem skorlanmış) görüntü var — "
              f"anlamlı bir analiz için daha fazla etiketleme gerekiyor.")
        return

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit(f"=== GERÇEK VERİ DOĞRULUK RAPORU ({len(common_ids)} görüntü) ===\n")

    # --- 1) Genel kalite: insan yargısı vs. sistem overall_score ---
    human_ord = [QUALITY_ORDINAL[labels[i]["genel_kalite"]] for i in common_ids]
    system_score = [float(scores[i]["overall_score"]) for i in common_ids]
    rho, _ = spearmanr(human_ord, system_score)
    emit(f"1) GENEL KALİTE: insan yargısı vs. sistem overall_score")
    emit(f"   Spearman rho = {rho:.4f}  (1.0 = mükemmel örtüşme, 0 = ilişki yok, negatif = ters yönlü)")

    exact_match = sum(
        1 for i in common_ids
        if labels[i]["genel_kalite"] == scores[i]["verdict"].replace("iyi", "iyi").replace("orta", "orta").replace("kotu", "kotu")
    )
    emit(f"   Tam eşleşme (insan kategorisi == sistem verdict'i): {exact_match}/{len(common_ids)} "
         f"(%{100*exact_match/len(common_ids):.1f})")
    emit("")

    # --- 2) Her defekt etiketi için: modül gerçekten ayırt edebiliyor mu? ---
    emit("2) DEFEKT BAZINDA: etiket VARKEN vs. YOKKEN ilgili modülün skoru")
    emit("   (Sağlıklı bir sistemde: defekt VARSA modül skoru BELİRGİN DÜŞÜK olmalı)\n")
    for flag, module_col in FLAG_TO_MODULE.items():
        with_defect = [float(scores[i][module_col]) for i in common_ids
                       if labels[i][flag] == "1" and scores[i][module_col] != ""]
        without_defect = [float(scores[i][module_col]) for i in common_ids
                           if labels[i][flag] == "0" and scores[i][module_col] != ""]
        if not with_defect or not without_defect:
            emit(f"   {flag:10s}: yetersiz veri (defektli={len(with_defect)}, defektsiz={len(without_defect)})")
            continue
        gap = np.mean(without_defect) - np.mean(with_defect)
        emit(f"   {flag:10s}: defekt VAR ortalama={np.mean(with_defect):6.1f}  "
             f"defekt YOK ortalama={np.mean(without_defect):6.1f}  fark={gap:+6.1f}  "
             f"(n_var={len(with_defect)}, n_yok={len(without_defect)})")
    emit("")

    # --- 3) "En kötü modül" gerçekten işaretlenen defektle örtüşüyor mu? ---
    emit("3) 'EN KÖTÜ MODÜL' TESPİTİ: sistemin işaret ettiği modül, senin işaretlediğin "
         "defektle örtüşüyor mu?")
    module_to_flag = {v: k for k, v in FLAG_TO_MODULE.items()}
    hits, total_flagged = 0, 0
    for i in common_ids:
        flagged = [f for f in FLAG_TO_MODULE if labels[i][f] == "1"]
        if not flagged:
            continue
        total_flagged += 1
        worst = scores[i]["worst_module"] + "_score"
        predicted_flag = module_to_flag.get(worst)
        if predicted_flag in flagged:
            hits += 1
    if total_flagged:
        emit(f"   En az bir defekt işaretlediğin {total_flagged} görüntüden {hits} tanesinde "
             f"(%{100*hits/total_flagged:.1f}) sistemin 'en kötü modülü', senin işaretlediğin "
             f"defektlerden biriyle eşleşti.")
    emit("")

    # --- 4) Yanlış alarm: hiç defekt işaretlemediğin ama sistemin "kötü" dediği görüntüler ---
    clean_labeled = [i for i in common_ids if not any(labels[i][f] == "1" for f in FLAG_TO_MODULE)]
    false_alarms = [i for i in clean_labeled if scores[i]["verdict"] == "kotu"]
    emit(f"4) YANLIŞ ALARM: hiç defekt işaretlemediğin {len(clean_labeled)} görüntüden "
         f"{len(false_alarms)} tanesine sistem 'kötü' dedi (%"
         f"{100*len(false_alarms)/len(clean_labeled) if clean_labeled else 0:.1f})")
    if false_alarms:
        emit(f"   anon_id'ler: {false_alarms}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapor kaydedildi -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
