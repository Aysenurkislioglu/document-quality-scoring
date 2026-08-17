# Document Quality Scoring

## Belge Görüntülerindeki Kalite Problemlerinin Tespiti ve Kalite Skorlaması

> **Çalışma amacı:** Belge görüntülerindeki **blur, glare, darkness,
> skew ve occlusion** problemlerini ayrı ayrı ölçmek; ardından bu
> ölçümleri birleştirerek açıklanabilir bir **Document Quality Score
> (0--100)** üretmek.

------------------------------------------------------------------------

## 1. Problem Tanımı

Akıllı telefonlarla çekilen belge görüntülerinde kamera açısı, odaklama,
hareket, aydınlatma ve fiziksel engeller nedeniyle çeşitli bozulmalar
oluşabilir. Bu bozulmalar OCR, bilgi çıkarımı, belge doğrulama ve
sınıflandırma gibi sonraki işlemlerin başarısını düşürebilir.

Literatürde bu problem **Document Image Quality Assessment (DIQA)**
olarak ele alınmaktadır. 2023 tarihli kapsamlı bir DIQA survey'i, kalite
değerlendirme çalışmalarını genel olarak **subjective** (insan
değerlendirmesi) ve **objective** (ölçülebilir özelliklere dayalı)
yaklaşımlar şeklinde inceler; ayrıca OCR tabanlı kalite ölçümünü önemli
bir değerlendirme biçimi olarak ele alır.

**Ana kaynak:** Alaei, Bui, Doermann & Pal, *Document Image Quality
Assessment: A Survey*, ACM Computing Surveys, 2023.

------------------------------------------------------------------------

## 2. Literatürde Kullanılan Temel Benchmark: SmartDoc-QA

**SmartDoc-QA**, smartphone ile yakalanmış belge görüntülerinin kalite
değerlendirilmesi için oluşturulmuş önemli benchmark veri setlerinden
biridir.

Veri setinin amacı, görüntü kalitesini yalnızca insan algısıyla değil,
**OCR doğruluğu** üzerinden de değerlendirebilmektir.

### Veri setinden gerçek bilgiler

  -----------------------------------------------------------------------
  Özellik                             Bilgi
  ----------------------------------- -----------------------------------
  Yayın                               CBDAR 2015

  Doküman türleri                     Modern belgeler, eski idari
                                      mektuplar, fişler

  Bozulmalar                          Blur, ışık değişimleri,
                                      perspektif/geometrik bozulmalar vb.

  Ground truth                        Belge metin transkripsiyonları, OCR
                                      çıktıları ve capture parametreleri

  Kullanım                            DIQA ve OCR performansı
                                      değerlendirmesi

  Arşiv boyutu                        Yaklaşık 13.7 GB
  -----------------------------------------------------------------------

SmartDoc-QA'nın önemli fikri şudur:

``` text
Document Image
      ↓
     OCR
      ↓
OCR Result vs Ground Truth
      ↓
OCR Accuracy
      ↓
Image Quality Indicator
```

Bu yaklaşım bizim proje açısından önemlidir çünkü nihai amaç yalnızca
görüntünün "güzel" görünmesi değil, belgenin **işlenebilir ve okunabilir
olmasıdır**.

**Kaynak:** Nayef et al., *SmartDoc-QA: A dataset for quality assessment
of smartphone captured document images - single and multiple
distortions*, CBDAR, 2015.

------------------------------------------------------------------------

# 3. Kalite Problemlerinin Genel Sınıflandırması

  -----------------------------------------------------------------------
  Problem                 Temel soru              En doğal ölçüm türü
  ----------------------- ----------------------- -----------------------
  **Blur**                Görüntü ne kadar        Sharpness / edge /
                          bulanık?                frequency

  **Glare**               Hangi bölgeler aşırı    Highlight mask /
                          yansıma nedeniyle       segmentation
                          kullanılamıyor?         

  **Darkness**            Görüntü veya hangi      Brightness /
                          bölgeler yetersiz       illumination
                          aydınlatılmış?          

  **Skew**                Belge kaç derece eğik?  Geometric angle

  **Occlusion**           Belgenin hangi bölgesi  Mask / overlap / layout
                          ne kadar kapalı?        
  -----------------------------------------------------------------------

Bu farklılık nedeniyle bütün problemlerde aynı algoritmayı kullanmak
yerine **problem bazlı yöntem seçimi** daha uygundur.

------------------------------------------------------------------------

# 4. BLUR

## 4.1. Temel fikir

Blur oluştuğunda görüntünün yüksek frekanslı detayları ve kenarları
zayıflar.

``` text
Sharp image
    ↓
Strong edges
    ↓
High-frequency details


Blurred image
    ↓
Weak edges
    ↓
Low-frequency dominant image
```

Bu nedenle blur ölçümünün temelinde **sharpness ölçümü** bulunur.

------------------------------------------------------------------------

## 4.2. Klasik yöntemler

Literatürde kullanılan başlıca yöntem aileleri:

-   Laplacian variance
-   Gradient magnitude
-   Sobel / Scharr
-   Tenengrad
-   Edge density
-   Frequency-domain / FFT
-   Local sharpness measures

### Laplacian Variance

En yaygın başlangıç yöntemlerinden biridir.

``` text
Image
  ↓
Laplacian
  ↓
Pixel intensity variation
  ↓
Variance
  ↓
Sharpness score
```

Genel olarak yüksek varyans daha fazla kenar/detay, düşük varyans ise
daha fazla blur ile ilişkilidir.

### Önemli sınırlama

Tek bir Laplacian threshold'u her belge ve kamera koşulunda güvenilir
değildir.

Örneğin:

-   text yoğunluğu,
-   font büyüklüğü,
-   çözünürlük,
-   noise,
-   JPEG compression

sonucu etkileyebilir.

Bu nedenle Laplacian iyi bir **baseline**, fakat nihai çözüm olarak
değerlendirilmemelidir.

------------------------------------------------------------------------

## 4.3. Daha gelişmiş yöntemler

Literatürde handcrafted özellikler ile öğrenme tabanlı yöntemler de
kullanılmıştır.

Örneğin spatial ve frequency-domain özellikleri birleştiren
yaklaşımlarda:

-   LBP
-   local variation
-   Log-Gabor
-   gradient
-   entropy

gibi özellikler çıkarılıp SVR gibi regression modelleriyle kalite tahmin
edilmektedir.

Bu yaklaşımın genel yapısı:

``` text
Image
 │
 ├── Spatial Features
 │
 └── Frequency Features
          ↓
    Feature Fusion
          ↓
         SVR
          ↓
     Quality Score
```

### Proje için öneri

İlk baseline:

**Laplacian Variance + Gradient/Tenengrad**

Sonraki karşılaştırmalar:

**FFT/Frequency + ML regression + CNN**

------------------------------------------------------------------------

# 5. GLARE

## 5.1. Temel fikir

Glare, belge yüzeyinden gelen güçlü ışık yansımasının bazı bölgelerde
bilgi kaybına neden olmasıdır.

Özellikle plastik kaplı veya parlak yüzeyli belgelerde kritik olabilir.

``` text
Normal text
████████████████

Glare region
██████████░░░░░░
           ↑
      Information
        lost
```

------------------------------------------------------------------------

## 5.2. Klasik yöntemler

Başlangıçta şu özellikler kullanılabilir:

-   luminance / brightness
-   HSV
-   saturation
-   thresholding
-   connected components
-   local intensity

Örneğin yüksek parlaklık + düşük saturation bir glare adayı
oluşturabilir.

Ancak:

> **Beyaz belge alanı da yüksek parlaklığa sahip olabilir.**

Dolayısıyla yalnızca threshold kullanmak yanlış pozitiflere yol
açabilir.

------------------------------------------------------------------------

## 5.3. Literatürde CNN tabanlı glare detection

Rodin ve Orlov'un 2019 tarihli *Fast Glare Detection in Document Images*
çalışması doğrudan belge görüntülerindeki glare problemine odaklanır.

Çalışmada:

1.  Belge görüntüsü bloklara ayrılır.
2.  Luminance özellikleri çıkarılır.
3.  Binarize görüntüden black-white stroke histogramları elde edilir.
4.  Bu özellikler CNN'e verilir.
5.  Sonuç olarak glare heatmap oluşturulur.

``` text
Document
    ↓
Blocks
    ↓
Luminance + Stroke Histograms
    ↓
CNN
    ↓
Glare Heatmap
```

Bu çalışma glare'ı yalnızca "var/yok" şeklinde değil, **görüntü üzerinde
konumlandırılabilir bir problem** olarak ele alması açısından önemlidir.

**Kaynak:** Rodin & Orlov, *Fast Glare Detection in Document Images*,
2019.

------------------------------------------------------------------------

## 5.4. Proje için öneri

İlk aşama:

**HSV + luminance + connected components**

İleri aşama:

**CNN segmentation / glare heatmap**

Üretilecek temel feature:

``` text
glare_ratio =
glare_area / document_area
```

Ancak sadece alan oranı yeterli değildir. Glare'ın **hangi bilgi
alanını** kapattığı da önemlidir.

------------------------------------------------------------------------

# 6. DARKNESS / ILLUMINATION

## 6.1. Temel fikir

Darkness yalnızca görüntünün ortalama olarak koyu olması değildir.

Örneğin:

``` text
Image A:
Her yer biraz karanlık

Image B:
Genel olarak aydınlık,
ama ID number bölgesi çok karanlık
```

İki görüntünün ortalama brightness değeri benzer olabilir fakat kullanım
açısından kalite aynı değildir.

------------------------------------------------------------------------

## 6.2. Basit ölçümler

Başlangıç için:

-   Mean brightness
-   Median brightness
-   Histogram
-   Percentiles
-   Local brightness
-   Local contrast

kullanılabilir.

Özellikle **percentile** analizi, görüntünün en karanlık bölgelerinin
dağılımını anlamaya yardımcı olabilir.

Örneğin:

``` text
P5  → çok karanlık uç
P25 → karanlık bölge eğilimi
P50 → median
P75 → aydınlık bölge
P95 → parlak uç
```

Bu nedenle darkness için tek bir mean değeri yerine histogram +
percentile + local analysis daha açıklayıcıdır.

------------------------------------------------------------------------

## 6.3. Illumination estimation

Literatürde document darkness/shadow problemi çoğunlukla **illumination
estimation/correction** veya **shadow removal** başlıkları altında
incelenmektedir.

2025 tarihli kapsamlı bir document shadow removal survey'i, klasik
yöntemleri iki ana gruba ayırmaktadır:

1.  **Shadow-map based methods**
2.  **Illumination-based methods**

Aynı survey, neural network tabanlı yöntemleri de ayrıca
sınıflandırmaktadır.

``` text
Document
   ↓
Illumination Estimation
   ↓
Illumination Map
   ↓
Dark / Shadow Regions
```

Bu yaklaşım global brightness'tan daha anlamlıdır çünkü aydınlatmanın
belge üzerinde **nerede ve ne kadar değiştiğini** modelleyebilir.

**Kaynak:** Wang et al., *A comprehensive survey on shadow removal from
document images: datasets, methods, and opportunities*, 2025.

------------------------------------------------------------------------

## 6.4. Proje için öneri

Baseline:

``` text
Global brightness
+
Brightness percentiles
+
Local brightness
+
Local contrast
```

İleri seviye:

``` text
Illumination estimation
+
Shadow segmentation
```

------------------------------------------------------------------------

# 7. SKEW

## 7.1. Temel fikir

Skew, belgenin veya text satırlarının yatay/dikey eksene göre açısal
olarak sapmasıdır.

Amaç:

``` text
Input
  ↓
Skew detection
  ↓
Angle = θ
```

şeklinde **skew angle** değerini bulmaktır.

------------------------------------------------------------------------

## 7.2. Literatürdeki yöntemler

Skew detection literatürü oldukça eski ve geniştir.

Yöntem aileleri:

-   Projection Profile
-   Hough Transform
-   PCA
-   Connected Component Analysis
-   Nearest Neighbor
-   Cross Correlation
-   Radon Transform
-   CNN

1998 tarihli Hull survey'i yöntemleri projection profile, feature
distribution, Hough transform ve yön-duyarlı local mask yaklaşımları
altında sınıflandırmaktadır.

2023 tarihli survey ise Hough, PCA, projection profile, nearest-neighbor
clustering, connected component analysis, cross-correlation, Radon
transform ve CNN gibi yöntemleri birlikte incelemektedir.

------------------------------------------------------------------------

## 7.3. Hough Transform

``` text
Document
   ↓
Edge Detection
   ↓
Hough Transform
   ↓
Dominant Lines
   ↓
Angle
```

Avantajları:

-   training gerektirmez,
-   açıklanabilirdir,
-   geometrik olarak doğrudan yorumlanabilir.

------------------------------------------------------------------------

## 7.4. Projection Profile

Farklı açılardaki text projection profilleri incelenir.

``` text
-5°  → profile
-4°  → profile
...
 0°  → strongest alignment
...
+5°  → profile
```

En uygun projection yapısının bulunduğu açı skew angle için adaydır.

Literatürde projection profile ve Hough transform en sık kullanılan
klasik yöntem aileleri arasındadır.

------------------------------------------------------------------------

## 7.5. Proje için öneri

İlk baseline:

**Hough Transform + Projection Profile**

Deep learning:

**CNN angle regression**

Ancak skew için CNN kullanmak başlangıç aşamasında zorunlu değildir.

------------------------------------------------------------------------

# 8. OCCLUSION

## 8.1. Temel fikir

Occlusion, belge üzerindeki bilginin başka bir nesne veya engel
tarafından kapatılmasıdır.

Örnek:

-   parmak
-   el
-   başka bir belge
-   sticker
-   nesne
-   fiziksel hasar

------------------------------------------------------------------------

## 8.2. Diğer problemlerden farkı

Blur:

> "Ne kadar bulanık?"

Skew:

> "Kaç derece eğik?"

Darkness:

> "Ne kadar karanlık?"

Occlusion:

> **"Neresi kapalı ve kapalı bölgenin bilgi açısından önemi ne?"**

Bu nedenle occlusion için yalnızca global bir skor yeterli olmayabilir.

------------------------------------------------------------------------

## 8.3. OCR tabanlı yaklaşım

Örneğin beklenen text:

``` text
AYŞENUR KIŞLIOĞLU
```

OCR sonucu:

``` text
AYŞENUR K______
```

ise occlusion şüphesi oluşabilir.

Ancak:

> Düşük OCR confidence = kesin occlusion

denemez.

Çünkü blur, glare, darkness ve skew de OCR confidence'ı düşürür.

Bu nedenle OCR confidence **yardımcı feature** olarak kullanılmalıdır.

------------------------------------------------------------------------

## 8.4. Object Detection

Belgeyi kapatan nesneler YOLO gibi object detection modelleriyle
bulunabilir.

``` text
Document
   ↓
Object Detection
   ↓
Finger / Hand / Object
   ↓
Overlap
   ↓
Occlusion Ratio
```

Temel ölçüm:

``` text
Occlusion Ratio =
Occluded Area / Document Area
```

------------------------------------------------------------------------

## 8.5. Segmentation

Daha gelişmiş yaklaşım:

``` text
Document
   ↓
Segmentation Model
   ↓
Occlusion Mask
   ↓
Area + Location
```

Burada yalnızca occlusion oranı değil, **hangi alanın kapatıldığı** da
belirlenebilir.

Örneğin:

  Bölge             Öncelik
  ------------ ------------
  ID Number      Çok yüksek
  Name               Yüksek
  Surname            Yüksek
  Date               Yüksek
  Photo                Orta
  Background          Düşük

Bu nedenle:

> **Occlusion severity = Area + Location + Region Importance**

yaklaşımı daha anlamlıdır.

------------------------------------------------------------------------

# 9. Text-Line Based Quality Assessment

Belgenin tamamını tek bir görüntü olarak değerlendirmek her zaman en iyi
yaklaşım olmayabilir.

Li, Zhu ve Qiu'nun 2019 çalışmasında **text-line based document image
quality assessment** önerilmiştir.

Pipeline:

``` text
Document
   ↓
Text Line Detection
   ↓
Line 1 → CNN → Quality
Line 2 → CNN → Quality
Line 3 → CNN → Quality
   ↓
Ensemble
   ↓
Overall Document Quality
```

Çalışmanın önemli noktalarından biri, **52,094 sentetik text-line
image** içeren bir dataset oluşturulmuş olmasıdır.

Bu yaklaşım özellikle kimlik belgeleri açısından değerlidir. Çünkü
belgenin tamamı kaliteli görünürken kritik bir text alanı kötü olabilir.

Örneğin:

``` text
Name       → 0.95
Surname    → 0.94
Birth Date → 0.91
ID Number  → 0.43
```

Bu durumda overall score'un yalnızca tüm görüntünün ortalama
kalitesinden oluşması yeterli olmayabilir.

**Kaynak:** Li, Zhu & Qiu, *Towards Document Image Quality Assessment: A
Text Line Based Framework and A Synthetic Text Line Image Dataset*,
2019.

------------------------------------------------------------------------

# 10. Üç Olası Mimari

## 10.1. Architecture 1 --- Classical CV

``` text
Document
   │
   ├── Blur → Laplacian
   ├── Glare → HSV
   ├── Darkness → Brightness
   ├── Skew → Hough
   └── Occlusion → OCR/Layout
              ↓
        Feature Vector
              ↓
        Weighted Score
              ↓
       Quality Score
```

### Avantaj

-   Az veri
-   Hızlı
-   Açıklanabilir
-   Kolay debug

### Dezavantaj

Karmaşık görüntülerde sınırlı kalabilir.

------------------------------------------------------------------------

# 11. Architecture 2 --- Hybrid

## **Önerilen yaklaşım**

``` text
                     DOCUMENT
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
   Classical CV      OCR / Layout    Deep Learning
          │              │              │
          └──────────────┼──────────────┘
                         ↓
                  Feature Vector
                         ↓
              ML / Regression Model
                         ↓
                  QUALITY SCORE
```

Örnek feature vector:

``` text
blur_score
glare_ratio
darkness_score
skew_angle
occlusion_ratio
ocr_confidence
text_visibility
region_quality
...
```

Sonrasında:

``` text
Features
   ↓
Random Forest / XGBoost / SVR
   ↓
Quality = 0–100
```

### Neden en mantıklı?

Her kalite problemi için en uygun yöntemi ayrı seçmeye izin verir.

Örneğin:

``` text
Blur       → Classical CV
Skew       → Classical CV
Darkness   → Classical + Illumination
Glare      → Classical → Segmentation
Occlusion  → OCR/Layout → Detection/Segmentation
```

Böylece her problem için gereksiz yere CNN eğitilmez.

------------------------------------------------------------------------

# 12. Architecture 3 --- End-to-End Deep Learning

``` text
Document Image
      ↓
   CNN / ViT
      ↓
 ┌────┼────┬────┬────┐
 ↓    ↓    ↓    ↓    ↓
Blur Glare Dark Skew Occ.
 └────┴────┴────┴────┘
          ↓
    Quality Score
```

### Avantaj

Yeterli veri olduğunda karmaşık ilişkileri otomatik öğrenebilir.

### Dezavantaj

-   Çok miktarda etiketli veri gerekir.
-   Açıklanabilirlik düşüktür.
-   Hangi problemin skoru düşürdüğünü anlamak zorlaşır.
-   Her degradation için yeterli örnek gerektirir.

Bu nedenle ilk aşama için önerilmemektedir.

------------------------------------------------------------------------

# 13. Önerilen Sistem

## Aşama 1 --- Classical Baseline

  Problem     İlk yöntem
  ----------- ------------------------------------------------------
  Blur        Laplacian + Gradient/Tenengrad
  Glare       HSV + luminance + connected components
  Darkness    Brightness + histogram + percentile + local analysis
  Skew        Hough + Projection Profile
  Occlusion   OCR + layout analysis

------------------------------------------------------------------------

## Aşama 2 --- Kontrollü Bozulma Testleri

Aynı orijinal görüntüden farklı şiddetlerde bozulmalar oluşturulmalıdır:

``` text
Original
 ├── Blur 1
 ├── Blur 2
 ├── Blur 3
 ├── Glare 1
 ├── Glare 2
 ├── Darkness 1
 ├── Darkness 2
 ├── Skew 1°
 ├── Skew 3°
 └── Occlusion 10%
```

Amaç:

> Bozulma şiddeti arttığında ölçülen quality score'un **monoton biçimde
> kötüleşip kötüleşmediğini** test etmektir.

Bu aşama, seçilen ölçüm yöntemlerinin gerçekten doğru davranıp
davranmadığını anlamak açısından kritiktir.

------------------------------------------------------------------------

# 14. Aşama 3 --- Ground Truth

Quality score için güvenilir ground truth oluşturulmalıdır.

### İnsan değerlendirmesi

``` text
0 → Unusable
1 → Poor
2 → Acceptable
3 → Good
4 → Excellent
```

### OCR performansı

``` text
Ground Truth Text
       ↓
OCR
       ↓
OCR Result
       ↓
Character / Word Accuracy
```

İnsan değerlendirmesi ile OCR performansı birlikte kullanılarak daha
anlamlı bir quality label oluşturulabilir.

------------------------------------------------------------------------

# 15. Aşama 4 --- ML Scoring

İlk prototipte:

``` text
Quality =
w₁ × Blur
+ w₂ × Glare
+ w₃ × Darkness
+ w₄ × Skew
+ w₅ × Occlusion
```

şeklinde basit bir weighted score kullanılabilir.

Daha sonra gerçek etiketli veri ile:

``` text
Feature Vector
      ↓
Random Forest
      ↓
XGBoost
      ↓
SVR
      ↓
Predicted Quality Score
```

modelleri karşılaştırılabilir.

------------------------------------------------------------------------

# 16. Aşama 5 --- Deep Learning Entegrasyonu

Klasik yöntemlerin başarısız olduğu alanlar belirlendikten sonra deep
learning eklenmelidir.

Örneğin:

``` text
Glare
→ CNN segmentation

Occlusion
→ Object detection / segmentation

Complex illumination
→ Illumination-aware deep model
```

Bu yaklaşımda deep learning, tüm sistemi değiştiren bir teknoloji olarak
değil, **klasik yöntemlerin yetersiz kaldığı noktalarda kullanılan bir
araç** olarak konumlandırılır.

------------------------------------------------------------------------

# 17. Nihai Hedef Mimari

``` text
                         DOCUMENT IMAGE
                                │
              ┌─────────────────┼─────────────────┐
              ↓                 ↓                 ↓
            BLUR              GLARE            DARKNESS
              │                 │                 │
         Sharpness          Glare Mask       Illumination
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ↓                       ↓
                  SKEW                 OCCLUSION
                    │                       │
                 Angle                Mask + Location
                    │                       │
                    └───────────┬───────────┘
                                ↓
                         OCR / LAYOUT
                                ↓
                         Feature Fusion
                                ↓
                     ML Regression Model
                                ↓
                    ┌────────────────────┐
                    │ QUALITY SCORE      │
                    │       0–100        │
                    └────────────────────┘
                                │
                                ↓
                    Explainable Report
```

Örnek çıktı:

``` text
Document Quality: 63 / 100

Blur:       82 / 100
Glare:      51 / 100
Darkness:   91 / 100
Skew:       94 / 100
Occlusion:  38 / 100

OCR Confidence: 0.71

Main degradation:
→ Occlusion on ID-number region
```

Bu şekilde sistem yalnızca bir sayı üretmez; **kalitenin neden düşük
olduğunu da açıklar.**

------------------------------------------------------------------------

# 18. Yöntem Seçim Tablosu

  ------------------------------------------------------------------------------
  Problem           Baseline          İleri yöntem             Önerilen
                                                               başlangıç
  ----------------- ----------------- ------------------------ -----------------
  **Blur**          Laplacian         CNN / frequency + ML     **Laplacian +
                                                               Gradient**

  **Glare**         HSV / luminance   CNN segmentation         **HSV +
                                                               luminance**

  **Darkness**      Brightness        Illumination estimation  **Global + local
                                                               brightness**

  **Skew**          Hough /           CNN angle regression     **Hough +
                    Projection                                 Projection**

  **Occlusion**     OCR/Layout        Detection/Segmentation   **OCR + Layout**
  ------------------------------------------------------------------------------

------------------------------------------------------------------------

# 19. Sonuç

Literatür taraması sonucunda document quality scoring probleminin tek
bir görüntü sınıflandırma problemi olmadığı görülmektedir.

Farklı bozulmalar farklı özellikler üzerinden tanımlanmaktadır:

-   **Blur** → sharpness ve high-frequency information
-   **Glare** → local high-intensity reflection
-   **Darkness** → brightness ve illumination distribution
-   **Skew** → geometric angle
-   **Occlusion** → missing/covered information

Bu nedenle en uygun başlangıç mimarisinin **Hybrid Architecture** olduğu
değerlendirilmiştir.

Önerilen yaklaşım:

> **Classical Computer Vision + OCR/Layout Analysis + gerekli noktalarda
> Deep Learning + ML-based score fusion**

şeklindedir.

Bu yaklaşımın temel avantajı, sistemin önce basit ve açıklanabilir bir
baseline ile kurulmasına, ardından deney sonuçlarına göre daha gelişmiş
modellerin eklenmesine olanak sağlamasıdır.

------------------------------------------------------------------------

# 20. Önerilen Araştırma Sırası

``` text
1. Literature Review
        ↓
2. Dataset Investigation
        ↓
3. Classical Baseline
        ↓
4. Synthetic Degradation Tests
        ↓
5. Real-world Validation
        ↓
6. Ground Truth Creation
        ↓
7. ML Score Fusion
        ↓
8. Deep Learning for Weak Components
        ↓
9. Final Evaluation
```

İlk detaylı araştırma konusu olarak **Blur** seçilmelidir.

Blur için:

``` text
Laplacian Variance
        ↓
Gradient / Tenengrad
        ↓
FFT / Frequency Domain
        ↓
SVM / SVR
        ↓
CNN
        ↓
Document-specific methods
        ↓
Experimental comparison
        ↓
Final baseline selection
```

şeklinde ilerlenmesi önerilmektedir.

------------------------------------------------------------------------

# Kaynaklar

1.  Alaei, A., Bui, V., Doermann, D., & Pal, U. (2023). **Document Image
    Quality Assessment: A Survey.** ACM Computing Surveys. DOI:
    10.1145/3606692.
2.  Nayef, N., Luqman, M. M., Prum, S., Eskenazi, S., Chazalon, J., &
    Ogier, J.-M. (2015). **SmartDoc-QA: A Dataset for Quality Assessment
    of Smartphone Captured Document Images - Single and Multiple
    Distortions.** CBDAR.
3.  Rodin, D., & Orlov, N. (2019). **Fast Glare Detection in Document
    Images.**
4.  Hull, J. J. (1998). **Document Image Skew Detection: Survey and
    Annotated Bibliography.**
5.  Biswas, B., Bhattacharya, U., & Chaudhuri, B. B. (2023). **An
    Overview of Existing Literature on Document Skew Detection.**
    Malaysian Journal of Computer Science.
6.  Wang, B., Li, C., Zou, W., et al. (2025). **A Comprehensive Survey
    on Shadow Removal from Document Images: Datasets, Methods, and
    Opportunities.**
7.  Li, H., Zhu, F., & Qiu, J. (2019). **Towards Document Image Quality
    Assessment: A Text Line Based Framework and A Synthetic Text Line
    Image Dataset.**
8.  **Optical Character Recognition Based Document Image Quality
    Assessment** (2026), Frontiers in Signal Processing.

------------------------------------------------------------------------

## Kaynaklara erişim

-   SmartDoc-QA: Zenodo dataset arşivi
-   DIQA Survey: ACM Computing Surveys
-   Glare Detection: arXiv
-   Skew Detection Survey: Malaysian Journal of Computer Science
-   Document Shadow Removal Survey: Springer Nature
-   Text-Line DIQA: arXiv
