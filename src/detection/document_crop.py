"""
Belge (kimlik kartı/pasaport) tespiti ve arkaplandan kırpma.

GEREKÇE (gerçek veri doğrulamasında bulundu, bkz. project_notes.md):
Bu projenin bugüne kadarki tüm sentetik doğrulaması, PIL ile üretilmiş
görüntüler üzerinde yapıldı — o görüntülerin TAMAMI zaten sadece belgeydi,
hiç arkaplan/masa/el yoktu. Bu varsayım hiçbir yerde açıkça yazılmadı,
örtük kaldı. Kullanıcının 368 gerçek kimlik fotoğrafıyla ilk kez test
edilince kırıldı: `compute_document_quality_score`, fotoğrafın TAMAMINI
(arkaplandaki masa, sabit kamera kurulumu, bazı fotoğraflarda üstüne
konan nesneler dahil) doğrudan her metriğe veriyordu.

Somut kanıt: Kullanıcı çoğu fotoğrafı SABİT bir kamera konumundan çektiğini
belirtti. 368 fotoğrafın 59 tanesinde darkness_raw (P10 persentil) BİREBİR
AYNI çıktı (104.613, 3 ondalık basamağa kadar) — ayrı ayrı gerçek fotoğraf
için istatistiksel olarak imkansıza yakın bir durum. Kök neden analizi:
bu görüntülerin en karanlık VE en parlak blokları (block-grid konumu
[21,17] ve [0,67]) hem konum hem DEĞER olarak birebir eşleşiyordu — yani
metrik, kartın kendisini değil, her çekimde aynı kalan SABİT ARKAPLANI
ölçüyordu.

ÇÖZÜM: Skorlama öncesinde klasik CV (kenar tespiti + kontur + dörtgen
yaklaşımı — telefon tarayıcı uygulamalarının kullandığı standart yöntem,
AI/derin öğrenme DEĞİL, projenin geri kalanıyla tutarlı) ile belgenin
4 köşesini bulup perspektif düzeltmesiyle kırpıyoruz. Böylece darkness/
blur/occlusion gibi blok-bazlı metrikler yalnızca belgenin kendisini
görür, arkaplanı değil.

BİLİNEN SINIRLAMA: Tespit başarısız olursa (belge net bir dörtgen kontur
olarak bulunamazsa — örn. çok düşük kontrast, belge kenarları
arkaplanla aynı renkte, ya da belge kadrajın çoğunu zaten dolduruyorsa)
orijinal (kırpılmamış) görüntüye SESSİZCE geri dönülmez — `detected=False`
ile birlikte döner, böylece çağıran taraf (fusion.py) bu durumu şeffaf
şekilde işaretleyebilir (bkz. `document_detected` alanı).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
    """4 köşeyi (top-left, top-right, bottom-right, bottom-left) sırasına koyar."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """4 köşe noktasına göre perspektif düzeltmesi yapıp üstten-görünüm
    (top-down) kırpılmış görüntü döndürür."""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 20 or max_height < 20:
        raise ValueError("Tespit edilen belge alanı çok küçük — güvenilmez.")

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def detect_document_quad(image: np.ndarray, min_area_fraction: float = 0.15) -> Optional[np.ndarray]:
    """
    Görüntüde belgeyi temsil eden en büyük 4-köşeli konturu bulur.

    Yöntem: gri tonlama -> Gaussian blur -> Canny kenar tespiti -> dilate
    (kopuk kenarları birleştirmek için) -> dış konturlar -> alana göre
    sırala -> ilk 5 aday arasında 4 köşeye (dörtgene) yaklaşan ve yeterince
    büyük (min_area_fraction) olanı seç.

    BİLİNÇLİ TASARIM KARARI: Yalnızca NET bir 4-köşe eşleşmesi kabul
    edilir — "en büyük konturun sınırlayıcı kutusu" gibi gevşek bir yedek
    YOK. Denendi (bkz. project_notes.md) ama gerçek fotoğraflarda neredeyse
    her büyük konturu "belge" sayıp %97 gibi yapay yüksek bir "tespit
    oranı" üretti — görsel doğrulama (bu kodun göremediği gerçek
    fotoğraflar) olmadan bu oranın gerçekten doğru kırpma mı yoksa yanlış
    kırpma mı ürettiği bilinemez. YANLIŞ bir kırpma (belgenin bir kısmını
    kesmek/yanlış bölgeyi almak), kırpmamaktan (mevcut, bilinen davranış)
    DAHA KÖTÜDÜR — bu yüzden emin olamadığımızda tespit BAŞARISIZ sayılır.

    Bulamazsa None döner — bu NORMAL bir durum (örn. belge kadrajı zaten
    dolduruyorsa kenar konturu dış çerçeveyle çakışabilir), çağıran taraf
    bu durumda orijinal görüntüyü kullanmaya devam eder.
    """
    h, w = image.shape[:2]
    total_area = h * w
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in candidates:
        area = cv2.contourArea(c)
        if area < total_area * min_area_fraction:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
    return None


def detect_and_crop_document(
    image: np.ndarray, min_area_fraction: float = 0.15
) -> Tuple[np.ndarray, bool]:
    """
    Belgeyi tespit edip arkaplandan kırpar.

    Returns:
        (görüntü, detected) — detected=True ise görüntü kırpılmış/perspektif
        düzeltilmiş belgedir. detected=False ise tespit başarısız olmuştur
        ve görüntü DEĞİŞTİRİLMEDEN (orijinal haliyle) döner — bu durumda
        arkaplan hâlâ metriklere karışabilir, çağıran taraf bunu
        kullanıcıya bildirmelidir.
    """
    quad = detect_document_quad(image, min_area_fraction=min_area_fraction)
    if quad is None:
        return image, False
    try:
        warped = _four_point_warp(image, quad)
    except (ValueError, cv2.error):
        return image, False
    if warped.size == 0 or min(warped.shape[:2]) < 20:
        return image, False
    return warped, True
