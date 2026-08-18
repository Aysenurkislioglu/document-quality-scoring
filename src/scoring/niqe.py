"""
NIQE (Natural Image Quality Evaluator) — sıfırdan, bu projeye özel uygulama.

GEREKÇE: Kullanıcı, "AI olmayan ama modern/genel amaçlı" bir kalite metriği
istedi. PyPI'daki hazır paketler (`brisque`, `image-quality`, `niqe`) denendi
— hepsi modern numpy/scikit-image sürümleriyle KIRIK çıktı (eski API'lere
bağımlı, bakımsız). Bu yüzden NIQE, Mittal, Soundararajan & Bovik (2013)
"Making a 'Completely Blind' Image Quality Analyzer" makalesindeki tarife
göre BASİTLEŞTİRİLMİŞ (tek ölçekli, orijinali 2 ölçekli) bir sürümü olarak
sıfırdan yazıldı. NIQE "AI" değildir — referans/eğitim etiketi (insan puanı)
gerektirmez, yalnızca "doğal/bozulmamış görüntülerin istatistiksel
düzenliliği" fikrine dayanan klasik bir istatistiksel modeldir (çok
değişkenli Gauss dağılımına uzaklık ölçümü).

ÖZGÜN KATKI: NIQE'in "pristine" (temiz/referans) modeli normalde genel doğa
fotoğraflarından fit edilir. Burada bunun yerine PROJEMİZİN KENDİ temiz
sentetik belgelerinden (`data/synthetic/blur/originals/`) fit ediliyor —
belge-alanına özgü bir referans, genel doğa istatistiklerinden daha uygun
olmalı (bu varsayım, aşağıdaki doğrulama deneyinde test edilir).

Yöntem özeti:
1. Görüntü, yerel ortalama/varyansa göre normalize edilir (MSCN katsayıları).
2. Örtüşmeyen bloklara bölünür; her blok için MSCN dağılımının şekli
   (Genelleştirilmiş Gauss Dağılımı parametreleri) + 4 komşuluk yönündeki
   (yatay/dikey/iki çapraz) çarpım dağılımlarının şekli (18 sayı/blok)
   çıkarılır.
3. "Temiz" görüntülerin blok özellikleri havuzundan bir çok-değişkenli
   Gauss modeli (ortalama vektör + kovaryans) fit edilir — bu, ONCE, tek
   seferlik `fit_pristine_model.py` ile yapılır ve diske kaydedilir.
4. Test edilecek görüntünün kendi blok özelliklerinden de bir Gauss modeli
   çıkarılır; NIQE skoru, bu iki modelin Mahalanobis mesafesidir. Yüksek
   mesafe = "doğal/temiz görüntülerden istatistiksel olarak uzak" = kötü
   kalite.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from scipy.special import gamma

BLOCK_SIZE = 64  # NIQE standardı 96 kullanır (doğa fotoğrafları için); bizim
# belge görüntülerimiz göreceli küçük olduğu için 64 kullanıldı.
MIN_BLOCK_VARIANCE = 4.0  # neredeyse düz/boş bloklar (örn. büyük beyaz
# kenarlık) GGD fit'ini bozduğu için elenir — standart NIQE pratiği.
MODEL_PATH = Path(__file__).parent / "models" / "niqe_pristine.npz"

_GAM_RANGE = np.arange(0.2, 10.0, 0.01)
_R_GAM_GGD = (gamma(1.0 / _GAM_RANGE) * gamma(3.0 / _GAM_RANGE)) / (gamma(2.0 / _GAM_RANGE) ** 2)
_R_GAM_AGGD = (gamma(2.0 / _GAM_RANGE) ** 2) / (gamma(1.0 / _GAM_RANGE) * gamma(3.0 / _GAM_RANGE))


def _gaussian_window(size: int = 7, sigma: float = 7 / 6) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float64)


_WINDOW = _gaussian_window()


def compute_mscn(gray: np.ndarray) -> np.ndarray:
    """Mean-Subtracted Contrast-Normalized katsayıları hesaplar.

    Yerel (Gauss ağırlıklı) ortalama çıkarılır, yerel standart sapmaya
    bölünür — bu, "doğal" görüntülerde neredeyse Gauss dağılımına yakın bir
    dağılım üretirken, bozulmalar bu dağılımı belirgin şekilde saptırır
    (NIQE'in temel dayanağı).
    """
    gray = gray.astype(np.float64)
    mu = cv2.filter2D(gray, -1, _WINDOW, borderType=cv2.BORDER_REPLICATE)
    mu_sq = cv2.filter2D(gray * gray, -1, _WINDOW, borderType=cv2.BORDER_REPLICATE)
    sigma = np.sqrt(np.maximum(mu_sq - mu * mu, 0))
    return (gray - mu) / (sigma + 1.0)


def _estimate_ggd(vec: np.ndarray) -> tuple:
    """Genelleştirilmiş Gauss Dağılımı (GGD) parametrelerini (şekil, sigma)
    moment eşleştirme yöntemiyle tahmin eder."""
    vec = vec.flatten()
    sigma_sq = float(np.mean(vec ** 2))
    sigma = float(np.sqrt(sigma_sq))
    mean_abs = float(np.mean(np.abs(vec)))
    rho = sigma_sq / (mean_abs ** 2 + 1e-12)
    idx = int(np.argmin(np.abs(_R_GAM_GGD - rho)))
    return _GAM_RANGE[idx], sigma


def _estimate_aggd(vec: np.ndarray) -> tuple:
    """Asimetrik GGD parametrelerini (şekil, ortalama, sol/sağ sigma)
    tahmin eder — komşu piksel çarpımları genelde asimetrik dağılır."""
    vec = vec.flatten()
    left = vec[vec < 0]
    right = vec[vec >= 0]
    left_std = float(np.sqrt(np.mean(left ** 2))) if left.size else 1e-6
    right_std = float(np.sqrt(np.mean(right ** 2))) if right.size else 1e-6
    gamma_hat = left_std / (right_std + 1e-12)
    r_hat = (float(np.mean(np.abs(vec))) ** 2) / (float(np.mean(vec ** 2)) + 1e-12)
    big_r = r_hat * (gamma_hat ** 3 + 1) * (gamma_hat + 1) / ((gamma_hat ** 2 + 1) ** 2 + 1e-12)
    idx = int(np.argmin(np.abs(_R_GAM_AGGD - big_r)))
    alpha = _GAM_RANGE[idx]
    ratio = np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_left = left_std * ratio
    beta_right = right_std * ratio
    mean = (beta_right - beta_left) * (gamma(2 / alpha) / gamma(1 / alpha))
    return alpha, mean, beta_left, beta_right


def block_features(mscn_block: np.ndarray) -> np.ndarray:
    """Bir bloğun 18 sayılık özellik vektörünü çıkarır: MSCN'nin kendi GGD
    parametreleri (2) + 4 komşuluk yönündeki çarpımların AGGD parametreleri
    (4x4=16)."""
    alpha, sigma = _estimate_ggd(mscn_block)
    feats = [alpha, sigma ** 2]
    h = mscn_block[:, :-1] * mscn_block[:, 1:]
    v = mscn_block[:-1, :] * mscn_block[1:, :]
    d1 = mscn_block[:-1, :-1] * mscn_block[1:, 1:]
    d2 = mscn_block[:-1, 1:] * mscn_block[1:, :-1]
    for prod in (h, v, d1, d2):
        a, m, bl, br = _estimate_aggd(prod)
        feats.extend([a, m, bl, br])
    return np.array(feats, dtype=np.float64)


def extract_patch_features(gray: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """Görüntüyü örtüşmeyen bloklara böler, her blok için 18'lik özellik
    vektörü çıkarır (neredeyse düz bloklar elenir). Returns: (N, 18) array."""
    mscn = compute_mscn(gray)
    h, w = mscn.shape
    feats = []
    for by in range(0, h - block_size + 1, block_size):
        for bx in range(0, w - block_size + 1, block_size):
            block = mscn[by : by + block_size, bx : bx + block_size]
            if block.var() < MIN_BLOCK_VARIANCE / (255.0 ** 2) * 255.0:
                # MSCN zaten normalize edilmiş küçük ölçekli; basit bir
                # varyans eşiği yeterli (deneyle ayarlandı).
                if block.var() < 1e-4:
                    continue
            feats.append(block_features(block))
    if not feats:
        return np.empty((0, 18))
    return np.array(feats)


def fit_mvg(features: np.ndarray) -> tuple:
    """Özellik matrisinden (N, 18) çok-değişkenli Gauss modeli (ortalama,
    kovaryans) fit eder."""
    mu = features.mean(axis=0)
    cov = np.cov(features, rowvar=False)
    return mu, cov


def _load_pristine_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Referans (pristine) model bulunamadı: {MODEL_PATH}. Önce "
            "experiments/niqe/fit_pristine_model.py çalıştırılmalı."
        )
    data = np.load(MODEL_PATH)
    return data["mu"], data["cov"]


def niqe_score(image: np.ndarray, pristine_mu: Optional[np.ndarray] = None, pristine_cov: Optional[np.ndarray] = None) -> float:
    """Bir görüntünün NIQE skorunu hesaplar (YÜKSEK = kalite DÜŞÜK/doğal
    olmayan; DÜŞÜK = kaliteli/temiz görüntülere istatistiksel olarak yakın).

    `pristine_mu`/`pristine_cov` verilmezse, önceden fit edilmiş ve diske
    kaydedilmiş model (`MODEL_PATH`) yüklenir.
    """
    if pristine_mu is None or pristine_cov is None:
        pristine_mu, pristine_cov = _load_pristine_model()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    feats = extract_patch_features(gray)
    if len(feats) < 2:
        return float("nan")

    test_mu, test_cov = fit_mvg(feats)
    cov_avg = (pristine_cov + test_cov) / 2.0
    try:
        inv_cov = np.linalg.pinv(cov_avg)
    except np.linalg.LinAlgError:
        return float("nan")
    diff = (pristine_mu - test_mu).reshape(1, -1)
    dist = np.sqrt(diff @ inv_cov @ diff.T)
    return float(dist[0, 0])
