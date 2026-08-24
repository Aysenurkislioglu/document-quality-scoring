"""
MIDV-2019 akışlı (streaming) doğrulama — blur, darkness VE glare'i GERÇEK
kamera fotoğraflarıyla, BİLİNEN şiddette sentetik bozulma enjekte ederek
test eder.

GEREKÇE: Kullanıcının 368 gerçek fotoğrafında darkness/blur'un gerçek
şiddetle ilişkisini ölçtük ama örneklem küçüktü (özellikle darkness için
yalnızca 2-368 karanlık örnek). MIDV-2019 (kamuya açık, kimlik/pasaport
fotoğrafları, düşük ışık koşulu dahil) çok daha büyük çeşitlilik sağlıyor
— ama ONA da gerçek şiddet etiketi yok. Çözüm: SENTETİK enjeksiyon —
biz kaç birim bulanıklık/karartma eklediğimizi TAM olarak biliyoruz, bu
yüzden ground-truth kesin. Gerçek kamera fotoğrafı (MIDV-2019) + kesin
ground-truth (bizim enjeksiyonumuz) = sentetik mockup'ların (rho=1.00 ama
gerçekte işe yaramayan) VE gerçek etiketlerin (doğru ama az sayıda)
ikisinin de eksik yönünü kapatıyor.

DİSK STRATEJİSİ: Her belge türü zip'i (~700-800MB) indirilir, işlenir,
YALNIZCA SAYISAL SONUÇLAR results/external_validation/midv2019_results.csv'ye
eklenir, sonra zip+çıkarılan dosyalar SİLİNİR. Disk kullanımı hiçbir zaman
~1.5GB'ı geçmez, kaç belge türü işlenirse işlensin.

Kullanım:
    python3 experiments/external_validation/run_midv2019_stream.py [doc_type1 doc_type2 ...]
    (argüman verilmezse DEFAULT_TYPES kullanılır)
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.blur.metrics import laplacian_variance  # noqa: E402
from src.darkness.metrics import local_brightness_blocks  # noqa: E402
from src.glare.metrics import glare_ratio  # noqa: E402

WORK_DIR = PROJECT_ROOT / "data" / "external" / "midv2019_stream"
RESULTS_PATH = PROJECT_ROOT / "results" / "external_validation" / "midv2019_results.csv"

FTP_BASE = "ftp://smartengines.com/midv-500/extra/midv-2019/dataset"

DEFAULT_TYPES = [
    "01_alb_id", "04_aut_id", "09_chn_id", "20_esp_id_new",
    "24_fin_id", "40_srb_id",
]

BLUR_KSIZES = [0, 3, 7, 13, 21]  # 0 = bozulmasız (kontrol)
DARK_FACTORS = [1.0, 0.8, 0.6, 0.4, 0.25]  # 1.0 = bozulmasız (kontrol)
GLARE_SEVERITIES = [0, 1, 2, 3, 4, 5]  # 0 = bozulmasız (kontrol)


def apply_synthetic_glare(bgr_image: np.ndarray, severity: int, rng: np.random.RandomState) -> np.ndarray:
    """Rastgele konumda, şiddet arttıkça büyüyen/opaklaşan beyaza yakın
    eliptik bir "highlight" ekler — dielektrik/specular yansımayı (lens
    parlaması, laminasyon parlaması) simüle eder. severity=0 -> değişiklik
    yok (kontrol)."""
    if severity <= 0:
        return bgr_image
    img = bgr_image.astype(np.float64)
    h, w = img.shape[:2]
    cx = rng.randint(int(w * 0.2), int(w * 0.8))
    cy = rng.randint(int(h * 0.2), int(h * 0.8))
    radius = max(5, int(min(h, w) * 0.07 * severity))
    alpha = min(0.15 * severity, 0.92)

    mask = np.zeros((h, w), dtype=np.uint8)
    angle = int(rng.randint(0, 180))
    cv2.ellipse(mask, (cx, cy), (radius, int(radius * 0.7)), angle, 0, 360, 255, -1)
    k = max(3, (radius // 2) | 1)  # tek sayı kernel
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask_f = (mask.astype(np.float64) / 255.0)[..., None] * alpha

    white = np.full_like(img, 250.0)
    blended = img * (1 - mask_f) + white * mask_f
    return np.clip(blended, 0, 255).astype(np.uint8)


def download_and_extract(doc_type: str) -> Path:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = WORK_DIR / f"{doc_type}.zip"
    extract_dir = WORK_DIR / f"extracted_{doc_type}"

    print(f"  İndiriliyor: {doc_type} ...")
    subprocess.run(
        ["curl", "-s", "-C", "-", "--max-time", "1800", "-o", str(zip_path),
         f"{FTP_BASE}/{doc_type}.zip"],
        check=True,
    )
    print(f"  Açılıyor: {doc_type} ...")
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(extract_dir)], check=True)
    return extract_dir


def cleanup(doc_type: str):
    zip_path = WORK_DIR / f"{doc_type}.zip"
    extract_dir = WORK_DIR / f"extracted_{doc_type}"
    if zip_path.exists():
        zip_path.unlink()
    if extract_dir.exists():
        shutil.rmtree(extract_dir)


def collect_sample_images(extract_dir: Path, max_images: int = 15) -> list:
    """images/ altındaki alt klasörlerden (koşul kategorileri) karışık
    şekilde birkaç .tif topluyor."""
    images = sorted((extract_dir / "images").rglob("*.tif"))
    # ana belge şablonu (images/<type>.tif) hariç, alt klasörlerdeki gerçek kareler
    images = [p for p in images if p.parent.name != "images"]
    if len(images) > max_images:
        step = len(images) // max_images
        images = images[::step][:max_images]
    return images


def evaluate_doc_type(doc_type: str, writer):
    extract_dir = download_and_extract(doc_type)
    rng = np.random.RandomState(hash(doc_type) % (2**31))
    try:
        samples = collect_sample_images(extract_dir)
        print(f"  {len(samples)} örnek görüntü işleniyor...")

        for img_path in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = 1100 / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            for ksize in BLUR_KSIZES:
                blurred = img if ksize == 0 else cv2.GaussianBlur(img, (ksize, ksize), 0)
                gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
                lap_var = laplacian_variance(gray)
                writer.writerow({
                    "doc_type": doc_type, "metric": "blur", "condition": ksize,
                    "raw_value": lap_var,
                })

            for factor in DARK_FACTORS:
                darkened = img if factor == 1.0 else np.clip(img.astype(np.float64) * factor, 0, 255).astype(np.uint8)
                blocks = local_brightness_blocks(darkened, block_size=16)
                p10 = float(np.percentile(blocks, 10))
                writer.writerow({
                    "doc_type": doc_type, "metric": "darkness", "condition": factor,
                    "raw_value": p10,
                })

            for severity in GLARE_SEVERITIES:
                glared = apply_synthetic_glare(img, severity, rng)
                ratio = glare_ratio(glared)
                writer.writerow({
                    "doc_type": doc_type, "metric": "glare", "condition": severity,
                    "raw_value": ratio,
                })
    finally:
        cleanup(doc_type)


def analyze():
    with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("\n=== SONUÇ: Sentetik şiddet vs. ham metrik (Spearman rho) ===")
    for metric in ["blur", "darkness", "glare"]:
        sub = [r for r in rows if r["metric"] == metric]
        conditions = [float(r["condition"]) for r in sub]
        values = [float(r["raw_value"]) for r in sub]
        rho, _ = spearmanr(conditions, values)
        n_types = len(set(r["doc_type"] for r in sub))
        print(f"  {metric:10s}: rho={rho:.4f}  (n_ölçüm={len(sub)}, {n_types} belge türü)")


def main():
    doc_types = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TYPES
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_type", "metric", "condition", "raw_value"])
        if not file_exists:
            writer.writeheader()
        for doc_type in doc_types:
            print(f"\n=== {doc_type} ===")
            try:
                evaluate_doc_type(doc_type, writer)
                f.flush()
            except Exception as e:
                print(f"  HATA ({doc_type}): {e} — atlanıyor, temizleniyor.")
                cleanup(doc_type)

    analyze()


if __name__ == "__main__":
    main()
