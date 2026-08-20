"""
Occlusion — renk sapması (color anomaly) oranı, GERÇEK VERİYLE kalibre
edilmiş klasik (ML olmayan) yöntem.

GEREKÇE / GEÇMİŞ (bkz. project_notes.md, "Gerçek Veri Doğrulaması"):
Bu projedeki önceki iki occlusion yöntemi de gerçek 368 kimlik fotoğrafında
başarısız oldu:
  1. skin_detection.py (ten rengi/YCrCb) — kimlik kartının KENDİ ÜZERİNDEKİ
     gerçek yüz fotoğrafını (gerçek ten rengi) "kapatan el" sanıyordu.
  2. ml_detection.py (Random Forest, blok-bazlı) — tamamen sentetik/
     vektörel görüntülerle eğitildiği için gerçek kameranın doğal dokusunu
     (sensör gürültüsü, JPEG sıkıştırma) hiç görmemişti (domain gap);
     gerçek kapanma şiddetiyle korelasyonu yalnızca rho≈0.14 (çok zayıf).

Kullanıcı, 368 fotoğrafın TAMAMI için ek olarak kapanma ŞİDDETİNİ (az/orta/
çok) etiketledi — bu, projede İLK KEZ occlusion için gerçek, dereceli bir
ground-truth sağladı (önceki tüm doğrulamalar yalnızca sentetik veriyle
yapılmıştı). Bu yöntem DOĞRUDAN bu gerçek etiketlerle geliştirildi ve
kalibre edildi.

YÖNTEM: Görüntü Lab renk uzayına çevrilir (CIE Lab — algısal renk
farkını Öklid mesafesiyle iyi yaklaştırır, HSV'den daha güvenilir).
Görüntünün KENDİ medyan rengi hesaplanır (belgenin baskın rengi/tonu —
konumdan ve kartın kendi renk şemasından bağımsız). Her pikselin bu
medyan renkten Lab mesafesi hesaplanır; sabit bir eşiğin (COLOR_DISTANCE_
THRESHOLD) ÜZERİNDEKİ piksellerin oranı döndürülür — bu, "belgenin kendi
renk şemasına uymayan" (yabancı nesne/el/obje) piksellerin kabaca
oranıdır.

NEDEN DAHA ÖNCEKİ YÖNTEMLERDEN İYİ ÇALIŞIYOR:
- Ten rengi yönteminden farklı olarak, HERHANGİ bir renk sapmasını yakalar
  (yalnızca ten rengini değil) — kartın kendi üzerindeki yüz fotoğrafı,
  medyan renkten çok sapmadığı sürece (genelde sapmaz, çünkü kart
  tasarımının küçük bir parçasıdır) yanlış alarm üretmez.
- ML modelinden farklı olarak, EĞİTİM VERİSİ YOK — hiçbir domain gap riski
  yok, doğrudan görüntünün kendi istatistiğini kullanır.

DOĞRULAMA (368 gerçek fotoğraf, gerçek kapanma şiddeti etiketleriyle):
    color_anomaly_ratio (eşik=50) vs. gerçek kapanma şiddeti: rho=0.56
  Karşılaştırma:
    Eski ML occlusion_score vs. gerçek kapanma şiddeti: rho=-0.14 (zayıf)
    Ten rengi yöntemi: yön bile YANLIŞ (bkz. skin_detection.py docstring'i)
  Şiddet gruplarına göre medyan oran: az=0.150, orta=0.180, çok=0.260
  (bkz. results/occlusion/ — bu deneyin ham verisi yalnızca anonim
  istatistiklerden üretildi, hiçbir gerçek görüntü saklanmadı).

BİLİNEN SINIRLAMA: rho=0.56 mükemmel değil (bu projedeki sentetik
doğrulamaların rho≈1.00'ından çok daha düşük) — GERÇEK dünya verisiyle
ulaşılan en iyi sonuç bu, ve önceki yöntemlerden BELİRGİN ÖLÇÜDE iyi,
ama occlusion hâlâ bu projenin en az kesin modülü. Eşik (50) ve BAD/GOOD
kalibrasyonu (bkz. fusion.py) 368 fotoğrafın gerçek dağılımından
(persentiller) türetildi, keyfi değil — ama başka bir kullanıcı
popülasyonunda yeniden doğrulanması gerekebilir.
"""

from __future__ import annotations

import cv2
import numpy as np

COLOR_DISTANCE_THRESHOLD = 50.0  # Lab uzayında, 368 fotoğrafla kalibre edildi


def color_anomaly_ratio(image: np.ndarray, distance_threshold: float = COLOR_DISTANCE_THRESHOLD) -> float:
    """Görüntünün kendi medyan renginden (Lab uzayında) belirgin şekilde
    sapan piksellerin oranını döndürür — yüksek oran = daha fazla
    kapanma şüphesi. Konumdan ve belgenin kendi renk şemasından bağımsız
    çalışır; bkz. modül docstring'i için gerekçe ve doğrulama."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    median_color = np.median(lab.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(lab - median_color, axis=2)
    return float(np.mean(dist > distance_threshold))
