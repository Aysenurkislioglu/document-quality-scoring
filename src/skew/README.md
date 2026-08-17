# Skew Detection — Yöntem Açıklaması

## Temel fikir

Skew, belgenin (veya metin satırlarının) yatay eksene göre açısal sapmasıdır — genelde
kamera belgeye tam paralel tutulmadığında oluşur. Amaç, tek bir sayı bulmaktır: **açı (θ)**.

```
Düz belge:                Eğik belge (skew):

────────────              ╱────────────
────────────             ╱────────────
────────────            ╱────────────
```

Bu modülde iki klasik, birbirinden tamamen farklı çalışan yöntem uygulanıp karşılaştırıldı.

## 1. Hough Transform

**Sezgi:** Görüntüdeki kenarları (özellikle metin satırlarının alt/üst çizgilerini,
tablo çizgilerini) bulup, bu kenarların hangi baskın açıda hizalandığını tespit eder.

```
Görüntü
   │
   ▼
Kenar tespiti (Canny)
   │
   ▼
Hough Transform: her kenar pikseli, olası tüm (açı, uzaklık)
kombinasyonlarına "oy verir"
   │
   ▼
En çok oy alan (baskın) doğrular bulunur
   │
   ▼
Bu doğruların açılarının medyanı = tahmini skew açısı
```

**Avantajı:** Training gerektirmez, açıklanabilir, güçlü/net kenarlar varsa (örn. tablo
çizgileri, alt çizgi) çok isabetli olabilir. **Dezavantajı:** Net, uzun doğrusal kenarlar
azsa (örn. yalnızca serbest metin, çizgi/tablo yoksa) güvenilirliği düşebilir.

## 2. Projection Profile

**Sezgi:** Metin satırları yatay olduğunda, görüntüyü satır satır topladığınızda
(her satırdaki koyu piksel sayısını toplama) net, keskin tepe noktaları (satırların
olduğu yerler) ve net çukurlar (satır araları) görürsünüz. Görüntü eğikse, bu tepe
noktaları birbirine karışır ve profil "bulanıklaşır".

```
Doğru açıda (0°):          Yanlış açıda (eğik):

Satır 1  ▓▓▓▓▓▓▓▓ (yüksek)  ▓▓▓▓▓▓ (düşük, bulanık)
boşluk   ░░░░░░░░ (düşük)   ▓▓▓▓▓░ (karışmış)
Satır 2  ▓▓▓▓▓▓▓▓ (yüksek)  ▓▓░░▓▓ (karışmış)
boşluk   ░░░░░░░░ (düşük)   ▓░▓▓▓░ (karışmış)
```

**Yöntem:** Görüntü, aday açı aralığında (örn. -15°'den +15°'ye) küçük adımlarla
döndürülür. Her aday açı için satır-toplamı profili çıkarılır ve bu profilin
**varyansı** hesaplanır. Profil ne kadar "keskin/net" ise (yüksek tepe - düşük çukur
farkı) varyans o kadar yüksektir. **En yüksek varyansı veren açı, tahmini skew açısı
olarak seçilir.**

```
Aday açılar: -15°, -14°, ..., 0°, ..., +14°, +15°
                        │
                        ▼
        Her açı için: döndür → satır profili çıkar → varyans hesapla
                        │
                        ▼
           En yüksek varyansa sahip açı = tahmini skew
```

**Avantajı:** Metin satırı düzenine dayandığı için, net çizgi/kenar olmasa bile (yalnızca
serbest metin varsa) çalışabilir. **Dezavantajı:** Aday açı aralığı ve adım boyutu
performansı ve hassasiyeti doğrudan etkiler (daha ince adım = daha hassas ama daha yavaş).
