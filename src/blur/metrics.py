"""
Blur / sharpness metrikleri.

Bu modül, belge görüntüleri için iki klasik, eğitim gerektirmeyen sharpness ölçütü
sağlar: Laplacian Variance ve Tenengrad (gradient magnitude tabanlı).

Yöntemlerin nasıl çalıştığına dair kavramsal açıklama için bkz. README.md.

Notlar:
- Her iki fonksiyon da girdiyi gri tonlamaya çevirir (renk kanalı varsa).
- Skorlar mutlak "iyi/kötü" eşiği olarak değil, GÖRECELİ karşılaştırma
  (aynı görüntünün farklı bozulma seviyeleri arasında) için tasarlanmıştır.
  Bkz. research/literature_review.md, Bölüm 2.1 — Laplacian threshold'unun
  belgeden belgeye taşınabilir olmadığı bilinen bir sınırlamadır.
"""

from __future__ import annotations

import numpy as np
import cv2


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Girdi görüntüyü tek kanallı gri tonlamaya çevirir (zaten gri ise dokunmaz)."""
    if image is None:
        raise ValueError("image None olamaz.")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Desteklenmeyen görüntü şekli: {image.shape}")


def laplacian_variance(image: np.ndarray, ksize: int = 1) -> float:
    """
    Laplacian Variance sharpness skoru.

    Görüntüye Laplacian (ikinci türev) filtresi uygulanır; sonuçtaki piksel
    değerlerinin varyansı hesaplanır. Yüksek varyans = güçlü/çok kenar = keskin
    görüntü. Düşük varyans = zayıf/az kenar = bulanık görüntü.

    Args:
        image: BGR, gri veya BGRA numpy array.
        ksize: Laplacian çekirdek boyutu (varsayılan 1, OpenCV standardı).

    Returns:
        float: Laplacian varyans skoru (birimsiz, göreceli karşılaştırma içindir).
    """
    gray = to_grayscale(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return float(laplacian.var())


def tenengrad(image: np.ndarray, ksize: int = 3, threshold: float = 0.0) -> float:
    """
    Tenengrad sharpness skoru (Sobel gradyan büyüklüğü tabanlı).

    Sobel operatörü ile x ve y yönündeki gradyanlar hesaplanır, gradyan
    büyüklüğünün karesi (Gx^2 + Gy^2) her piksel için bulunur ve ortalaması
    alınır. Yüksek skor = güçlü, yaygın kenarlar = keskin görüntü.

    Args:
        image: BGR, gri veya BGRA numpy array.
        ksize: Sobel çekirdek boyutu (varsayılan 3).
        threshold: Belirtilirse, yalnızca gradyan büyüklüğü bu eşiği aşan
            pikseller skora dahil edilir (zayıf/gürültü kaynaklı gradyanları
            filtrelemek için). 0.0 ise tüm pikseller kullanılır.

    Returns:
        float: Tenengrad skoru (birimsiz, göreceli karşılaştırma içindir).
    """
    gray = to_grayscale(image).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    gradient_sq = gx ** 2 + gy ** 2

    if threshold > 0:
        mask = gradient_sq > (threshold ** 2)
        if not np.any(mask):
            return 0.0
        return float(gradient_sq[mask].mean())

    return float(gradient_sq.mean())


def gradient_magnitude_mean(image: np.ndarray, ksize: int = 3) -> float:
    """
    Yardımcı metrik: ortalama Sobel gradyan büyüklüğü (kare almadan, |G|).

    Tenengrad'a benzer ama karesi alınmamış büyüklüğün ortalamasını kullanır;
    farklı bir ölçeklendirme sağladığı için deneylerde çapraz kontrol amaçlı
    tutulmuştur.
    """
    gray = to_grayscale(image).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return float(magnitude.mean())


def compute_all_blur_metrics(image: np.ndarray) -> dict:
    """Bir görüntü için tüm blur metriklerini tek seferde hesaplar."""
    return {
        "laplacian_variance": laplacian_variance(image),
        "tenengrad": tenengrad(image),
        "gradient_magnitude_mean": gradient_magnitude_mean(image),
    }
