"""
Darkness / illumination metrikleri: global istatistikler, percentile analizi
ve blok-bazlı yerel (local) parlaklık/kontrast.

Yöntemlerin nasıl çalıştığına dair açıklama için bkz. README.md.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import cv2

BBox = Tuple[int, int, int, int]

DEFAULT_PERCENTILES = (5, 25, 50, 75, 95)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def global_brightness(image: np.ndarray) -> Dict[str, float]:
    """Global ortalama ve medyan parlaklık."""
    gray = _to_grayscale(image)
    return {"mean": float(gray.mean()), "median": float(np.median(gray))}


def brightness_percentiles(image: np.ndarray, percentiles=DEFAULT_PERCENTILES) -> Dict[str, float]:
    """Belirtilen percentile'larda parlaklık değerleri (örn. P5 = en karanlık %5'in eşiği)."""
    gray = _to_grayscale(image)
    values = np.percentile(gray, percentiles)
    return {f"p{p}": float(v) for p, v in zip(percentiles, values)}


def local_brightness_blocks(image: np.ndarray, block_size: int = 32) -> np.ndarray:
    """
    Görüntüyü block_size x block_size bloklara ayırıp her bloğun ortalama
    parlaklığını içeren 2D bir harita döndürür.
    """
    gray = _to_grayscale(image).astype(np.float64)
    h, w = gray.shape
    rows = h // block_size
    cols = w // block_size
    if rows == 0 or cols == 0:
        return np.array([[gray.mean()]])

    cropped = gray[: rows * block_size, : cols * block_size]
    blocks = cropped.reshape(rows, block_size, cols, block_size)
    return blocks.mean(axis=(1, 3))


def local_contrast_blocks(image: np.ndarray, block_size: int = 32) -> np.ndarray:
    """Her bloğun standart sapmasını (yerel kontrast) içeren 2D harita."""
    gray = _to_grayscale(image).astype(np.float64)
    h, w = gray.shape
    rows = h // block_size
    cols = w // block_size
    if rows == 0 or cols == 0:
        return np.array([[gray.std()]])

    cropped = gray[: rows * block_size, : cols * block_size]
    blocks = cropped.reshape(rows, block_size, cols, block_size)
    return blocks.std(axis=(1, 3))


def darkest_block_mean(image: np.ndarray, block_size: int = 32, roi: Optional[BBox] = None) -> float:
    """
    En karanlık bloğun ortalama parlaklığı — küçük, lokalize karanlık
    bölgeleri (örn. tek bir kimlik alanı) yakalamak için tasarlanmıştır.
    """
    region = image
    if roi is not None:
        x0, y0, x1, y1 = roi
        region = image[y0:y1, x0:x1]
    blocks = local_brightness_blocks(region, block_size)
    return float(blocks.min())


def compute_all_darkness_metrics(image: np.ndarray, block_size: int = 32) -> Dict[str, float]:
    result = {}
    result.update(global_brightness(image))
    result.update(brightness_percentiles(image))
    result["darkest_block_mean"] = darkest_block_mean(image, block_size)
    contrast_blocks = local_contrast_blocks(image, block_size)
    result["mean_local_contrast"] = float(contrast_blocks.mean())
    return result
