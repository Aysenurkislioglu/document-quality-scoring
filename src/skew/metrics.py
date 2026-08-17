"""
Skew açısı tahmini: Hough Transform ve Projection Profile yöntemleri.

Açı kuralı (convention): pozitif açı = belge SAAT YÖNÜNÜN TERSİNE (counter-
clockwise) döndürülmüş anlamına gelir (görüntü koordinat sisteminde, satır
başının biraz "yukarı" kalkması gibi düşünülebilir). Bu, bu modüldeki
sentetik test döndürmeleriyle (deney scriptinde PIL/cv2 rotate) aynı işareti
kullanacak şekilde deneysel olarak doğrulanmıştır (bkz. experiments/skew/).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def estimate_skew_hough(
    image: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 150,
    max_angle_deviation: float = 30.0,
) -> Optional[float]:
    """
    Hough Transform ile skew açısı tahmini (derece cinsinden).

    Returns:
        float: tahmini açı (derece). Yeterli sayıda doğru bulunamazsa None.
    """
    gray = _to_grayscale(image)
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 360, hough_threshold)

    if lines is None or len(lines) == 0:
        return None

    angles = []
    for line in lines:
        rho, theta = line[0]
        # theta: doğrunun normalinin x eksenine göre açısı (radyan).
        # Yatay bir çizginin normali dikeydir -> theta ~ pi/2.
        deviation_deg = (theta * 180.0 / np.pi) - 90.0
        if abs(deviation_deg) <= max_angle_deviation:
            angles.append(deviation_deg)

    if not angles:
        return None

    return float(np.median(angles))


def _rotate_image(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = gray.shape
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        gray, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )


def estimate_skew_projection_profile(
    image: np.ndarray,
    angle_range: tuple = (-15.0, 15.0),
    angle_step: float = 0.5,
    dark_threshold: int = 180,
) -> float:
    """
    Projection Profile ile skew açısı tahmini (derece cinsinden).

    Aday açı aralığında görüntü döndürülür, her adayda satır bazlı koyu
    piksel sayısı profili çıkarılır, varyansı en yüksek olan açı seçilir.
    """
    gray = _to_grayscale(image)
    binary = (gray < dark_threshold).astype(np.float64)  # metin=1, arka plan=0

    best_angle = 0.0
    best_score = -1.0
    for angle in np.arange(angle_range[0], angle_range[1] + angle_step, angle_step):
        rotated = _rotate_image((binary * 255).astype(np.uint8), angle)
        rotated_binary = rotated > 127
        profile = rotated_binary.sum(axis=1)
        score = float(profile.var())
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    return best_angle
