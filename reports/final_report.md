# Document Quality Scoring — Nihai Teknik Rapor

**Tarih:** 17 Ağustos 2026
**Kapsam:** Literatür taraması + Blur, Glare, Darkness, Skew, Occlusion modüllerinin
baseline geliştirmesi ve deneysel değerlendirmesi.

> **Okuma notu — üç bilgi türü:** Bu rapor boyunca üç farklı bilgi türü açıkça
> ayrıştırılmıştır:
> - **Literatür** — daha önce başkaları tarafından yapılmış, yayınlanmış çalışmalar.
> - **Bizim Yöntemimiz** — bu projede seçip uyguladığımız yöntem.
> - **Bizim Sonuçlarımız** — bu projede yaptığımız deneylerden elde edilen, ölçülmüş
>   sonuçlar.
>
> Deneysel olarak ölçülmemiş hiçbir değer kesin bir sonuç gibi sunulmamıştır; bu
> durumlarda **"Not evaluated"** veya **"N/A"** ibaresi kullanılmıştır. Rapor, yalnızca
> `project_notes.md`, `research/` klasörü, gerçekleştirilen deneyler ve bunların
> `results/` klasöründeki çıktılarına dayanmaktadır.

---

## 1. Abstract

Bu proje, akıllı telefonla çekilmiş belge/kimlik görüntülerindeki beş temel kalite
problemini — **blur (bulanıklık), glare (parlama), darkness (yetersiz aydınlatma), skew
(eğiklik) ve occlusion (kapanma)** — ayrı ayrı ölçüp, ileride açıklanabilir tek bir
**Document Quality Score (0-100)**'a birleştirmeyi hedefleyen bir sistemin ilk (baseline)
aşamasını kapsar. Beş modülün her biri için literatürde önerilen klasik (training-free)
yöntemler uygulanmış, sentetik ama kontrollü (yer gerçeği bilinen) veriler üzerinde
deneysel olarak test edilmiştir. Sonuçlar karışıktır ve bu raporun en değerli
bulgularından biri tam olarak budur: **Blur, Darkness ve Occlusion modüllerinde
seçilen baseline yöntemler beklendiği gibi güçlü, ölçülebilir sinyaller üretirken, Glare
modülünde seçilen baseline yöntem sentetik (gri tonlamalı) veri üzerinde etkisiz
bulunmuş, Skew modülünde ise iki yöntemden biri (Projection Profile) belirli koşullarda
(az metin yoğunluğu) ciddi biçimde başarısız olmuştur.** Bu proje, henüz feature-fusion /
ML skorlama aşamasına (Aşama 5) geçmemiştir; nihai 0-100 skor bu raporun kapsamı dışındadır
ve "Not evaluated" olarak işaretlenmiştir. Rapor ayrıca, projenin asıl motivasyonu olan
**kimlik belgesi sahteciliği tespiti** için, bu beş kalite modülünün ötesinde,
literatürden derlenen ve daha az bilinen adli (forensic) yöntemler hakkında gerekçeli
öneriler sunmaktadır (bkz. Bölüm 20).

## 2. Introduction

Kimlik doğrulama, belge işleme ve OCR sistemlerinin güvenilirliği, girdi olarak aldıkları
görüntünün kalitesine doğrudan bağımlıdır. Düşük kaliteli bir görüntü (bulanık, parlamalı,
karanlık, eğik veya kısmen kapalı), aşağı akış (downstream) sistemlerinde hem yanlış
kabul hem yanlış red hatalarına yol açabilir. Bu proje, böyle bir kalite değerlendirme
sisteminin ilk aşamasını — beş temel bozulma türünün ayrı ayrı, açıklanabilir biçimde
ölçülmesini — ele almaktadır.

Projenin nihai motivasyonu, yalnızca genel bir görüntü kalitesi ölçer inşa etmek değil,
**kimlik belgesi sahteciliğinde kullanılan/gizlenen bozulmaları**, doğrudan "kara kutu"
bir CNN yerine, daha yüksek doğruluk sağlayan, uygulanabilir ve gerekçelendirilebilir
yöntemlerle tespit edebilmektir. Bu rapor, bu hedefe giden yolun ilk adımını —
capture-quality (çekim kalitesi) değerlendirmesini — belgelemektedir; asıl sahtecilik
tespiti (tamper detection) için önerilen ek yöntemler Bölüm 20'de ayrıca ele alınmıştır.

## 3. Problem Definition

Akıllı telefonla çekilen belge görüntülerinde kamera açısı, odaklama, hareket,
aydınlatma ve fiziksel engeller nedeniyle beş temel bozulma oluşabilir:

| Problem | Temel soru |
|---|---|
| Blur | Görüntü ne kadar bulanık? |
| Glare | Hangi bölgeler aşırı yansıma nedeniyle kullanılamıyor? |
| Darkness | Görüntü/bölgeler ne kadar yetersiz aydınlatılmış? |
| Skew | Belge kaç derece eğik? |
| Occlusion | Belgenin hangi bölgesi, ne önemde kapalı? |

*(Kaynak: initial_research.md — proje başlangıç dokümanı.)*

Bu beş problem birbirinden farklı fiziksel nedenlere ve farklı en-doğal ölçüm türlerine
sahip olduğu için, tek bir algoritma yerine **problem-bazlı yöntem seçimi** tercih
edilmiştir.

## 4. Related Work / Literature Review

Tam literatür taraması `research/literature_review.md` ve `research/references.md`
dosyalarındadır; burada özet sunulmaktadır.

**Literatür:** Bu alan **Document Image Quality Assessment (DIQA)** olarak adlandırılır
(Alaei, Bui, Doermann & Pal, 2023 — ACM Computing Surveys survey'i). **SmartDoc-QA**
(Nayef et al., 2015, CBDAR), belge kalitesini OCR doğruluğu üzerinden değerlendiren
önemli bir benchmark'tır. Her problem için literatürde önerilen klasik ve ileri
yöntemler:

| Problem | Klasik yöntemler (literatür) | Öğrenme tabanlı yöntemler (literatür) |
|---|---|---|
| Blur | Laplacian, Gradient/Tenengrad, FFT | CNN, SVR + el yapımı özellikler |
| Glare | HSV, luminance, connected components | CNN segmentation (Rodin & Orlov, 2019) |
| Darkness | Brightness, histogram, percentile | Illumination estimation (Wang et al., 2025 survey) |
| Skew | Hough, Projection Profile | CNN angle regression |
| Occlusion | OCR/Layout | Detection/Segmentation |

*(Bu tablo, initial_research.md'de zaten sunulmuş ve bu oturumda literatürle çapraz
kontrol edilmiş bir özettir — bkz. `research/references.md` doğrulama durumları.)*

2025-2026 döneminde alanın MLLM (multimodal büyük dil modeli) tabanlı yaklaşımlara
(DeQA-Doc, Q-Doc — bkz. `research/literature_review.md` Bölüm 3) doğru genişlediği de
bu oturumda ayrıca tespit edilmiştir; bu yaklaşımlar bu projede uygulanmamıştır.

## 5. Dataset

**Literatür:** SmartDoc-QA (~13.7 GB, CBDAR 2015), gerçek smartphone ile çekilmiş belge
görüntüleri içerir ve OCR tabanlı ground truth sağlar. Kimlik belgesi odaklı ek veri
setleri (MIDV-500, MIDV-2020, IDNet) `research/literature_review.md` Bölüm 4.2'de
listelenmiştir.

**Bizim Yöntemimiz:** Bu projede **hiçbir gerçek/dış veri seti henüz indirilip
kullanılmamıştır.** Bunun yerine, tüm deneyler **sentetik olarak üretilen belge
görüntüleri** üzerinde yapılmıştır (`experiments/_common/synthetic_documents.py`):

- 850×1100 piksel, beyaz zemin, siyah metin.
- Başlık + 3 mock kimlik alanı (Ad Soyad, Belge No — 10 haneli, Tarih) + gövde
  paragrafları (rastgele Türkçe kelime havuzundan).
- 12 belgelik sabit bir ızgara: 3 font boyutu (14/20/28px) × 2 paragraf yoğunluğu (2/5) ×
  2 tekrar.
- Her modül için, bu 12 belgeye bilinen (yer gerçeği) şiddette kontrollü bozulmalar
  (Gaussian blur, sentetik glare lekesi, çarpımsal karartma, döndürme, opak
  kapanma dikdörtgeni) uygulanmıştır.

**Neden sentetik veri?** Gerçek veri setlerinin (SmartDoc-QA vb.) lisans/boyut
değerlendirmesi henüz yapılmadığı için (bkz. `project_notes.md`, Genel Proje Planı).
Sentetik veri, yer gerçeğinin (blur şiddeti, glare alanı, karartma faktörü, açı, kapanma
oranı) tam olarak bilinmesini sağladığı için yöntemlerin **temel davranışını** (monotonluk,
duyarlılık) doğrulamak için uygundur — ama **gerçek kamera koşullarını (defokus+hareket
karışımı, gerçek kağıt dokusu, JPEG artefaktları, renkli kağıt tonu) temsil etmez.** Bu,
raporun tamamında tekrarlanan temel bir sınırlamadır (bkz. Bölüm 18).

| Özellik | Değer |
|---|---|
| Toplam sentetik görüntü sayısı | Blur: 108, Glare: 84 (12 orijinal + 72 degrade), Darkness: 144 (72 global + 72 lokal), Skew: 132, Occlusion: 72 |
| Format | PNG, gri tonlamalı (8-bit) |
| Train/validation/test ayrımı | Yok — bu aşamada model eğitimi yapılmadı, yalnızca deterministik/analitik yöntemler test edildi |
| Ground truth | Her modülde algoritmik olarak (üretim parametrelerinden) kesin biçimde biliniyor |

## 6. Proposed Architecture

**Literatür + Bizim Yöntemimiz:** initial_research.md, üç mimari alternatifi karşılaştırıp
**Hibrit Mimari**yi önermiştir; bu proje bu öneriyi takip etmektedir.

| Özellik | Classical CV | Hybrid (seçilen) | End-to-End DL |
|---|---|---|---|
| Veri ihtiyacı | Düşük | Orta | Yüksek |
| Açıklanabilirlik | Yüksek | Yüksek/Orta | Düşük |
| Eğitim gereksinimi | Yok | Kısmi (füzyon aşamasında) | Yüksek |
| Debug kolaylığı | Yüksek | Orta-Yüksek | Düşük |
| Bu projedeki durumu | 5 modülün tamamı bu katmanda | Henüz uygulanmadı (Aşama 5) | Uygulanmadı |

```
                     DOCUMENT
                         │
          ┌──────────────┼──────────────┬─────────────┬─────────────┐
          ↓              ↓              ↓             ↓             ↓
        Blur           Glare        Darkness         Skew       Occlusion
   (Laplacian+       (HSV+CC)    (global/percentile  (Hough+    (OCR+beklenen
    Tenengrad)                    /blok-bazlı)      Projection)     desen)
          │              │              │             │             │
          └──────────────┴──────────────┴─────────────┴─────────────┘
                                        │
                          Feature Vector — ⏳ Not evaluated
                                        │
                        ML Regression Model — ⏳ Not evaluated
                                        │
                        QUALITY SCORE (0-100) — ⏳ Not evaluated
```

Beş kalite modülü bu raporda geliştirilip test edilmiştir; **feature fusion ve ML
regresyon aşaması bu raporun kapsamında DEĞİLDİR** (bkz. Bölüm 13).

## 7. Methodology

Her modül için aynı deneysel desen izlenmiştir:

1. **Literatür seçimi** — o problem için literatürde önerilen baseline yöntem(ler)
   belirlendi (`research/literature_review.md`).
2. **Uygulama** — yöntem `src/<modül>/metrics.py` içinde, bağımsız test edilebilir
   fonksiyonlar olarak yazıldı.
3. **Kontrollü sentetik deney** — bilinen şiddette/yer gerçeğinde bozulma enjekte edilen
   veri `experiments/<modül>/` altında üretildi.
4. **Monotonluk testi** — bozulma şiddeti arttıkça skorun tutarlı yönde değişip
   değişmediği **Spearman rank korelasyonu** ile ölçüldü.
5. **Dynamic range / etki büyüklüğü testi** — (Glare modülünde fark edilen bir yanılgı
   sonrası tüm modüllere eklendi) yalnızca korelasyona değil, skorun severity=0'dan
   severity=max'a ne kadar **pratik olarak değiştiğine** de bakıldı; yüksek korelasyon +
   düşük dynamic range kombinasyonu yanıltıcı olabilir (bkz. Bölüm 9, 16).
6. **Sonuçların `project_notes.md`'ye kaydı** — problem, yöntem, gerekçe, parametreler,
   deney, sonuç, karşılaşılan problemler, kararlar, sonraki adım.

Tüm deney kodları çalıştırılabilir script'ler olarak `experiments/` altında saklanmıştır;
hiçbir sonuç elle hesaplanmamış, tamamı kod çalıştırılarak üretilmiştir.

---

## 8. Blur Detection

**Literatür:** Laplacian variance en yaygın başlangıç yöntemi; tek başına threshold
olarak güvenilmez (text yoğunluğu, font, çözünürlük, noise, JPEG'den etkilenir).
Tenengrad/gradient magnitude ve FFT/frekans-alanı yöntemleri de literatürde yer alıyor.

**Bizim Yöntemimiz:** `src/blur/metrics.py` — **Laplacian Variance** ve **Tenengrad**
(Sobel gradyan büyüklüğü karesi ortalaması), artı yardımcı bir **Gradient Magnitude
Mean**. Yöntemlerin çalışma prensibi: `src/blur/README.md`.

**Bizim Sonuçlarımız:** 12 sentetik belgeye, 9 kademeli Gaussian blur şiddeti (σ = 0 →
8) uygulandı (108 görüntü).

| Metrik | Ortalama Spearman rho (12 belge) |
|---|---|
| Laplacian Variance | **−0.9986** (std 0.0048) |
| Tenengrad | **−1.0000** (std 0.0000) |

Her iki yöntem de blur şiddetiyle neredeyse mükemmel monoton azalma gösterdi.

**Ek bulgu — font boyutu duyarlılığı:** Blur hiç yokken bile (severity=0), yalnızca font
boyutu farkı skorlarda büyük değişkenlik yarattı — **coefficient of variation:**
Laplacian %27.2, Tenengrad %32.1. Bu, literatürün "tek eşik güvenilmez" iddiasını bu
sentetik veri özelinde doğrulamaktadır.

*(Detay: `project_notes.md` "Modül: BLUR"; ham veri: `results/blur/`.)*

## 9. Glare Detection

**Literatür:** HSV (yüksek Value + düşük Saturation) eşikleme + connected components
baseline; Rodin & Orlov (2019) CNN tabanlı ileri yöntem. Literatür açıkça uyarıyor: "beyaz
belge alanı da yüksek parlaklığa sahip olabilir."

**Bizim Yöntemimiz:** `src/glare/metrics.py` — HSV eşikleme (V≥235, S≤35) + connected
components filtreleme (min. 15px), yalnızca belgenin içerik kutusu içinde uygulandı.

**Bizim Sonuçlarımız:** 12 belgeye, içerik alanının %0-%22'sini kaplayan, yumuşak
kenarlı sentetik glare lekeleri eklendi (72 görüntü). **Sonuç: yöntem, mevcut haliyle
sentetik veride pratik olarak işe yaramamaktadır:**

| Metrik | Spearman rho | Severity=0 ort. | Severity=max ort. | Göreceli değişim |
|---|---|---|---|---|
| naive_glare_ratio (bizim yöntemimiz) | 1.0000 (yanıltıcı — bkz. altta) | 0.849 | 0.860 | **%1.3** |
| oracle_text_washout_ratio (yalnızca doğrulama amaçlı, referans gerektirir) | 1.0000 | 0.000 | 0.034 | anlamlı |

Spearman korelasyonu her iki metrik için de mükemmel (1.0) görünse de, **naive_glare_ratio
pratikte neredeyse hiçbir ayırt edici güce sahip değildir** — glare hiç yokken bile içerik
kutusunun %84.9'unu "glare" olarak işaretlemekte, gerçek glare eklendiğinde bu oran yalnızca
%86.0'a çıkmaktadır. Kök neden: sentetik belgeler gri tonlamalı olduğu için HSV'nin
Saturasyon kanalı **her zaman sıfır** çıkmaktadır — yöntemin ayırt edici gücünün yarısı
(S eşiği) matematiksel olarak devre dışı kalmaktadır. Bu, literatürün uyarısının en uç
biçimde doğrulanmasıdır.

**Sonuç:** Bu modül "tamamlandı" değil, **"baseline denendi ve mevcut haliyle yetersiz
bulundu"** statüsündedir.

*(Detay: `project_notes.md` "Modül: GLARE"; ham veri: `results/glare/`.)*

## 10. Darkness Detection

**Literatür:** Global/median brightness, histogram + percentile (P5-P95), local
brightness/contrast (klasik); illumination estimation / shadow removal (ileri, Wang et
al. 2025 survey). Literatürün temel iddiası: aynı global ortalamaya sahip iki görüntü
farklı kalitede olabilir (biri homojen karanlık, diğeri genel aydınlık ama kritik bir
alanı karanlık).

**Bizim Yöntemimiz:** `src/darkness/metrics.py` — global mean/median, percentile'lar
(P5/P25/P50/P75/P95), ve **16×16 piksellik blok-bazlı yerel analiz** (en karanlık blok
ortalaması + yerel kontrast).

**Bizim Sonuçlarımız:** İki senaryo test edildi (her biri 72 görüntü, 6 şiddet seviyesi):

**Global karartma senaryosu:** Tüm metrikler mükemmel monoton (rho = −1.0000, tümü).

**Lokal karartma senaryosu (yalnızca "Belge No" alanı, görüntünün ~%0.29'u karartıldı):**

| Metrik | Severity=0→max göreceli değişim | Spearman rho |
|---|---|---|
| global_mean | **−%0.35** (pratikte kör) | −1.0000 (yanıltıcı) |
| p25, p50 | **%0** (sabit, hiç tepki yok) | tanımsız (varyans yok) |
| p5 | −%4.84 (kısmi) | −0.874 |
| darkest_block_mean (bizim önerdiğimiz) | **−%55.58** (güçlü, net) | −0.828 |

**Sonuç:** Bu deney, literatürün "global ortalama yetersiz" iddiasını doğrudan
doğrulamaktadır. Global mean ve medyan/P25/P50, görüntünün <%1'ini kaplayan lokal bir
karanlık bölgeyi neredeyse hiç yakalayamazken, blok-bazlı yerel analiz güçlü bir tepki
vermiştir. **Bu, Glare modülünün aksine, baseline yöntemin (blok-bazlı bileşeni) beklendiği
gibi çalıştığı bir modüldür.**

*(Detay: `project_notes.md` "Modül: DARKNESS"; ham veri: `results/darkness/`.)*

## 11. Skew Detection

**Literatür:** Hough Transform ve Projection Profile, en sık kullanılan klasik yöntem
aileleri (Hull, 1998; Biswas et al., 2023).

**Bizim Yöntemimiz:** `src/skew/metrics.py` — Hough Transform (Canny + HoughLines, yakın-
yatay doğruların medyan açısı) ve Projection Profile (−15°…+15°, 0.5° adımlarla, satır
profili varyansını maksimize eden açı).

**Bizim Sonuçlarımız:** 12 belge, 11 bilinen açıyla (−12°…+12°) döndürüldü (132 görüntü).

| Yöntem | Genel MAE | Std |
|---|---|---|
| Hough Transform | **0.82°** | 2.01 |
| Projection Profile | 1.82° | 5.10 |

Genel ortalama Hough'u daha iyi gösterse de, dağılım çok farklı bir hikâye anlatıyor:
**Projection Profile, 12 belgenin 10'unda MAE = 0.0000° (mükemmel)**, ama **2 belgede
(en küçük font + en az paragraf, yani en az metin satırı) MAE ≈ 9.5-12.3°** ile
başarısız oldu — başarısız tahminlerin tamamı arama aralığının sınırına (−15°) kilitlendi.
Hough ise daha istikrarlı ama açı büyüdükçe kademeli olarak kötüleşti (0°'de MAE=0.21°,
12°'de MAE=1.79°).

**Sonuç:** İki yöntem farklı, tamamlayıcı hata modlarına sahip; ikisi arasındaki büyük
anlaşmazlık ileride bir güvenilirlik sinyali olarak kullanılabilir.

*(Detay: `project_notes.md` "Modül: SKEW"; ham veri: `results/skew/`.)*

## 12. Occlusion Detection

**Literatür:** OCR confidence (yardımcı sinyal, tek başına yeterli değil — diğer
bozulmalar da confidence'ı düşürür); object detection/segmentation (ileri yöntem, bu
projede uygulanmadı). Belgeye özel, geniş kabul görmüş bir occlusion veri seti/yöntemi bu
oturumda bulunamadı.

**Bizim Yöntemimiz:** `src/occlusion/metrics.py` — **OCR + beklenen alan deseni**: "Belge
No" alanı (bilinen konum, bilinen format = 10 haneli sayı) kırpılıp Tesseract ile OCR
edildi; iki sinyal hesaplandı: **length_ratio** (okunan karakter sayısı / beklenen 10) ve
**mean_confidence** (Tesseract'ın güven skoru), birleştirilerek **occlusion_suspicion_score**
(0-100, yüksek=şüpheli) üretildi.

**Bizim Sonuçlarımız:** 12 belgede, "Belge No" alanı %0-%100 arasında kademeli kapatıldı
(72 görüntü).

| Metrik | Spearman rho | Coverage=0 | Coverage=%100 |
|---|---|---|---|
| length_ratio | −0.976 | 1.000 | 0.000 |
| mean_confidence | −0.853 | 91.7 | 0.0 |
| occlusion_suspicion_score | +0.945 (yön ters — "şüphe" skoru) | 4.2 | 100.0 |

**Sonuç:** Üç metrik de güçlü, tutarlı monoton sinyal verdi. Bu modül, **Darkness gibi,
baseline yöntemin beklendiği gibi çalıştığı** bir modüldür — ama yalnızca **konumu ve
formatı bilinen, yapılandırılmış alanlar** için (serbest metin için uygulanmadı).

*(Detay: `project_notes.md` "Modül: OCCLUSION"; ham veri: `results/occlusion/`.)*

## 13. Feature Fusion and Quality Scoring

**⏳ Not evaluated.** initial_research.md, beş alt-skorun bir ML regresyon modeliyle
(Random Forest / XGBoost / SVR) tek bir 0-100 skora birleştirilmesini önermektedir
(Aşama 5). Bu proje bu aşamaya **henüz ulaşmamıştır**; `src/scoring/` klasörü boş bir
placeholder olarak durmaktadır. Bu aşamaya geçilmeden önce çözülmesi gereken, deneylerde
tespit edilen açık noktalar Bölüm 17 ve 19'da listelenmiştir (örn. skor yönü/convention
tutarsızlığı, Glare modülünün düzeltilmesi gerekliliği).

## 14. Experimental Setup

| Özellik | Değer |
|---|---|
| Python sürümü | 3.11 |
| Ana kütüphaneler | OpenCV (headless) 4.13, NumPy 2.4, pandas 3.0, SciPy, Matplotlib, Pillow, pytesseract + Tesseract 5.3 (+ `tur` dil paketi) |
| Veri | Tamamen sentetik (bkz. Bölüm 5) |
| Donanım | CPU (GPU kullanılmadı — hiçbir yöntem GPU gerektirmiyor) |
| Rastgelelik kontrolü | Tüm sentetik veri üretiminde sabit seed (42) kullanıldı — sonuçlar tekrarlanabilir |
| Değerlendirme metrikleri | Spearman rank korelasyonu (monotonluk), Mutlak Hata/MAE (skew), dynamic range / göreceli değişim (etki büyüklüğü) |

## 15. Results

Tüm modüllerin özet karşılaştırması (birincil monotonluk metriği):

| Modül | Ana metrik | Spearman rho | Pratik duyarlılık (dynamic range) | Durum |
|---|---|---|---|---|
| Blur | Tenengrad | −1.0000 | Yüksek (birkaç kat büyüklük değişimi) | ✅ Çalışıyor |
| Glare | naive_glare_ratio | 1.0000 (yanıltıcı) | **Çok düşük (%1.3)** | ⚠️ Yetersiz |
| Darkness (lokal) | darkest_block_mean | −0.828 | Yüksek (%55.6) | ✅ Çalışıyor |
| Skew | Hough | MAE 0.82° | — | ✅ Çalışıyor (istikrarlı) |
| Skew | Projection Profile | MAE 1.82° (ortalama, aykırı değerlerden etkilenmiş) | — | ⚠️ Koşullu (az metinde başarısız) |
| Occlusion | length_ratio | −0.976 | Yüksek (1.0→0.0) | ✅ Çalışıyor |

**En önemli genel bulgu:** Yalnızca Spearman korelasyonuna bakmak yanıltıcı olabilir
(Glare ve kısmen Darkness/global_mean bunun kanıtı). Bir yöntemin gerçekten kullanışlı
olup olmadığını anlamak için korelasyon YANINDA mutlaka **dynamic range / etki
büyüklüğü** de ölçülmelidir — bu proje boyunca öğrenilen ve metodolojiye sonradan
eklenen bir derstir.

## 16. Method Comparisons

Yalnızca gerçekten anlamlı karşılaştırmalar için tablolar:

**Blur — iki yöntemin karşılaştırması:**

| Yöntem | Type | Training | Explainability | Hız | Sonuç (bu projede) |
|---|---|---|---|---|---|
| Laplacian Variance | Classical (2. türev) | Hayır | Yüksek | Yüksek | rho=−0.9986, font boyutuna duyarlı (CV=%27.2) |
| Tenengrad | Classical (1. türev) | Hayır | Yüksek | Yüksek | rho=−1.0000, font boyutuna duyarlı (CV=%32.1) |
| FFT/Frequency, CNN | Literatür (uygulanmadı) | — | — | — | Not evaluated |

**Skew — iki yöntemin karşılaştırması:**

| Yöntem | MAE (genel) | En iyi senaryo | En kötü senaryo |
|---|---|---|---|
| Hough Transform | 0.82° | Küçük açılar (0.21-0.36°) | Büyük açılar (1.79° @ 12°) |
| Projection Profile | 1.82°* | Yeterli metin varsa (10/12 belgede MAE=0°) | Az metinli belgelerde (2/12 belgede MAE=9.5-12.3°) |

(*Ortalama, az-metin aykırı değerlerinden güçlü etkilenmiştir — bkz. Bölüm 11.)

**Occlusion — iki sinyalin karşılaştırması:**

| Sinyal | Spearman rho | Gürültü düzeyi |
|---|---|---|
| length_ratio | −0.976 | Düşük, düzgün eğri |
| mean_confidence | −0.853 | Orta, özellikle orta-kapanma seviyelerinde dalgalı |

**Mimari karşılaştırması (initial_research.md'den, bu projede yalnızca Classical CV
katmanı test edildi):**

| Özellik | Classical CV (bu projede test edildi) | Hybrid (planlandı, uygulanmadı) | End-to-End DL (uygulanmadı) |
|---|---|---|---|
| Veri ihtiyacı | Düşük — doğrulandı (sentetik/az veriyle çalıştı) | Orta — Not evaluated | Yüksek — Not evaluated |
| Açıklanabilirlik | Yüksek — doğrulandı (her skorun nedeni izlenebilir) | Yüksek/Orta — Not evaluated | Düşük — Not evaluated |
| Karmaşık bozulmalarda başarı | **Karışık** — 3/5 modülde iyi, 1/5 yetersiz (Glare), 1/5 koşullu (Skew/Projection) | Not evaluated | Not evaluated |

---

## 17. Discussion

Bu projenin en değerli çıktısı, beklendiği gibi çalışan yöntemler kadar, **beklendiği
gibi çalışmayan yöntemlerin neden çalışmadığının anlaşılmasıdır**:

- **Glare modülü**, literatürün "beyaz kağıt da parlak olabilir" uyarısını salt teorik
  bir dipnot olmaktan çıkarıp, sentetik veride %1.3'lük bir dynamic range olarak somut
  biçimde göstermiştir. Kök neden (gri tonlamalı sentetik veride S kanalının işlevsiz
  kalması) yalnızca kod yazılıp çalıştırıldığında ortaya çıkmıştır — bu, "önce basit
  baseline dene, sonuçları ölç" yaklaşımının değerini doğrulamaktadır.
- **Skew modülünde Projection Profile'ın az-metinli belgelerde arama sınırına
  kilitlenmesi**, yöntemin literatürde iyi bilinen bir avantajının (metin satırı
  düzenine dayanması) aynı zamanda bir zayıflık kaynağı (yetersiz metin = zayıf sinyal)
  olabileceğini göstermiştir.
- **Darkness ve Occlusion modüllerinde**, literatürün önerdiği "tek bir global ölçüm
  yerine yerel/yapılandırılmış analiz" yaklaşımı beklendiği gibi çalışmış ve büyük fark
  yaratmıştır (darkness'ta %55.6 vs %0.35 dynamic range farkı).

Genel çıkarım: **beş modülün hiçbiri "tak-çalıştır" düzeyinde production'a hazır
değildir** — her biri, bu raporda tespit edilen en az bir açık nokta (Bölüm 18) ile
birlikte değerlendirilmelidir.

## 18. Limitations

1. **Tüm veri sentetiktir.** Gerçek kamera görüntülerinin (defokus+motion blur karışımı,
   gerçek kağıt dokusu/rengi, JPEG sıkıştırma artefaktları, gerçek gölge/aydınlatma
   fiziği) hiçbiri bu deneylerde temsil edilmemiştir. Tüm sayısal sonuçlar yalnızca bu
   sentetik veri bağlamında geçerlidir.
2. **Glare modülü mevcut haliyle kullanılamaz** (bkz. Bölüm 9) — sentetik verinin
   grayscale olması bunu abartmış olabilir, ama ROI'nin (içerik kutusu) satır-arası
   boşlukları içermesi sorunu gerçek renkli veride de (daha az şiddetli olsa da) devam
   edecektir.
3. **Occlusion modülü yalnızca yapılandırılmış (format bilinen) alanlar için çalışır** —
   serbest metin occlusion tespiti bu projede ele alınmamıştır.
4. **Feature fusion / ML skorlama yapılmamıştır** — nihai 0-100 Quality Score bu raporda
   YOKTUR.
5. **Skor yönü (convention) tutarsızlığı** fark edilmiş ama düzeltilmemiştir: Blur/
   Darkness/Glare'de yüksek=iyi, Occlusion'ın `occlusion_suspicion_score`'unda
   yüksek=kötü. Füzyon öncesi normalize edilmelidir.
6. **Font-boyutu × blok-analizi etkileşimi** (Darkness modülü) ve **font-boyutu
   duyarlılığı** (Blur modülü) kesin olarak açıklanmamış, yalnızca gözlemlenmiştir.
7. **Değerlendirme yalnızca sentetik/algoritmik ground truth ile yapılmıştır** — hiçbir
   aşamada insan değerlendirmesi (initial_research.md'nin önerdiği 0-4 skala) veya
   gerçek OCR doğruluğu (paragraf metinleri için) kullanılmamıştır.

## 19. Future Work

`project_notes.md`'de her modülün kendi "Bir sonraki adım" notu vardır; genel öncelik
sırası:

1. Gerçek veri setiyle (SmartDoc-QA veya benzeri) tüm modüllerin yeniden doğrulanması.
2. Glare modülünün düzeltilmesi (renkli veri, ve/veya ROI'nin metin satırı sıkı
   kutularına daraltılması, ve/veya Rodin & Orlov'un CNN yaklaşımının denenmesi).
3. Skor yönü normalizasyonu (tüm alt-skorlar "yüksek=iyi" yönüne çevrilmeli).
4. Ground truth stratejisinin somutlaştırılması (insan değerlendirmesi + gerçek OCR
   doğruluğu birleşimi).
5. Feature fusion / ML skorlama (Aşama 5) — ağırlıklı toplam ile RF/XGBoost/SVR
   karşılaştırması.
6. Bölüm 20'de önerilen, kimlik sahteciliğine özel ek yöntemlerin değerlendirilmesi.

## 20. Kimlik Sahteciliğine Özel Önerilen Yöntemler

> **Önemli çerçeveleme notu:** Bu bölümdeki hiçbir yöntem bu projede **uygulanmamış veya
> test edilmemiştir** — tamamı 🔎 dış araştırmaya dayanan, gerekçeli **önerilerdir**.
> Deneysel bir sonuç DEĞİLDİR.
>
> Ayrıca kavramsal bir ayrım öne çıkıyor: Bölüm 8-12'deki beş modül **"bu bir çekim
> kalitesi sorunu mu?"** sorusuna cevap veriyor (yani: fotoğraf iyi mi çekilmiş?).
> Sahtecilik ise farklı bir soru: **"bu belge/görüntü değiştirilmiş mi?"** İyi çekilmiş
> (bulanık olmayan, iyi aydınlatılmış) bir görüntü, mükemmel biçimde sahte olabilir; kötü
> çekilmiş bir görüntü tamamen gerçek olabilir. Bu yüzden kimlik sahteciliği tespiti için
> Bölüm 8-12'deki modüller **gerekli ama yeterli değildir** — aşağıdaki yöntemler bu
> boşluğu kapatmaya yöneliktir.

### 20.1. Neden "herkesin aklına gelmeyecek" yöntemler önemli?

Çoğu sahtekar, sahte bir kimliği **gözle görülür şekilde ikna edici** hale getirmeye
odaklanır (net, doğru renkli, doğru fontlu görünmesi). Bu tür sahtecilikler genellikle
görsel kalite kontrolünü (blur/glare/darkness/skew) kolayca geçer — çünkü sahtekarın
amacı zaten "iyi görünmesini sağlamak"tır. Aşağıdaki yöntemler, sahtekarın **muhtemelen
hiç düşünmediği veya düzeltmeyi bilmediği** dijital/matematiksel izlere odaklanır.

### 20.2. Öncelik sırasıyla önerilen yöntemler

**1) Checksum / kontrol hanesi (check-digit) doğrulaması — en yüksek öncelik**

Birçok resmi kimlik numarası formatı (örn. MRZ — Machine Readable Zone, ICAO Doc 9303
standardı), belge numarası, doğum tarihi gibi alanlarda **matematiksel bir kontrol
hanesi** içerir: her karakter bir ağırlıkla (7-3-1 tekrar eden dizi) çarpılır, toplanır,
mod 10 alınır ve sonuç son haneyle karşılaştırılır. Bir sahtekar tek bir haneyi
değiştirdiğinde, kontrol hanesini de doğru biçimde yeniden hesaplaması gerekir — çoğu
sahtekar bu algoritmanın **varlığından bile haberdar değildir**. 🔎

- **Neden düşük maliyetli/yüksek etkili:** Görüntü işleme gerektirmez, saf aritmetik bir
  doğrulamadır — milisaniyeler sürer, false-positive riski neredeyse sıfırdır (format
  doğruysa kesin doğrulanır).
- **Neden "herkesin aklına gelmeyecek" bir yöntem:** Görsel olarak kusursuz sahte bir
  belge bile, kontrol hanesi yanlışsa anında yakalanır — sahtekarın bunu bilmesi ve doğru
  hesaplaması gerekir.
- Kaynak: [MRZ check digits explained](https://trustdochub.com/en/mrz-check-digits/),
  [ICAO 9303 MRZ check digits](https://idcheck.dev/icao-9303-check-digits/).

**2) Copy-move (klon) tespiti**

Sahtekarların en sık kullandığı tekniklerden biri, bir belgenin bir bölümünü (örn. bir
karakter, bir arka plan deseni, bir güvenlik hologramının parçası) kopyalayıp başka bir
yere yapıştırmaktır. **SIFT/ORB anahtar nokta eşleştirmesi** veya blok-tabanlı benzerlik
analizi, görüntü içinde **aynı desenin birden fazla kez** (farklı konumlarda) göründüğü
bölgeleri tespit edebilir. 🔎

- **Neden non-obvious:** Kopyalanan bölge gözle fark edilmeyecek kadar iyi
  harmanlanmış olsa bile, piksel-düzeyinde "aynılık" istatistiksel olarak tespit
  edilebilir.
- Kaynak: [Copy-move forgery detection: Survey, challenges and future directions](https://www.sciencedirect.com/science/article/abs/pii/S1084804516302144),
  [EURASIP — superpixel segmentation ile copy-move tespiti](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-019-0469-9).

**3) Error Level Analysis (ELA) / çift JPEG sıkıştırma analizi**

Bir görüntünün yalnızca bir bölümü düzenlenip tekrar kaydedildiğinde, o bölge geri kalan
görüntüden **farklı bir sıkıştırma "hata seviyesi"** taşır — bu farkı ELA veya çift JPEG
sıkıştırma analizi (çoğu güncel yaklaşım artık CNN tabanlı) ortaya çıkarabilir. 🔎

- **Neden non-obvious:** İnsan gözüyle görünmez; yalnızca sıkıştırma matematiğini analiz
  ederek ortaya çıkar.
- **Sınırlama:** Görüntü birden fazla kez yeniden sıkıştırılırsa (örn. WhatsApp/ekran
  görüntüsü üzerinden iletilirse) sinyal zayıflayabilir — bu yüzden tek başına değil,
  diğer yöntemlerle birlikte kullanılmalıdır.
- Kaynak: [Error level analysis (ELA) — genel açıklama](https://trustdochub.com/en/error-level-analysis-image/),
  [ID kartı sahteciliği için ELA+RPA uygulaması](https://arvindn-iitkgp.medium.com/on-document-forensics-for-id-card-fraud-detection-using-ela-and-rpa-23c68dd8f3e0),
  [Multi-branch network for double JPEG detection and localization (2025)](https://www.nature.com/articles/s41598-025-04203-0).

**4) Font/glyph tutarlılık analizi**

Kimlik şablonlarında değişken alanlar (isim, numara, tarih) genelde **sabit bir fontla**
basılır. Bir alan dijital olarak değiştirildiğinde (örn. farklı bir yazı tipi editörüyle),
harflerin kenar yumuşatması (anti-aliasing), karakter aralığı (kerning) veya kalınlığı
(stroke width) belgenin geri kalanından **istatistiksel olarak farklı** çıkabilir. Bu
proje kapsamında geliştirdiğimiz `src/blur` (kenar/sharpness ölçümü) ve `src/darkness`
(yerel kontrast) modüllerindeki teknikler, bu analiz için **doğrudan yeniden
kullanılabilir bir başlangıç noktası** sunuyor — aynı sharpness/local-contrast
ölçümlerini "kalite" yerine "belge-içi tutarlılık" sorusuna uygulamak. 🔎

- **Neden non-obvious:** Sahtekar tek bir alanı "iyi görünecek" şekilde değiştirebilir,
  ama o alanın font-istatistiklerini belgenin GERİ KALANIYLA piksel-düzeyinde birebir
  eşleştirmesi çok daha zordur.
- Kaynak: [DocForge-Bench: document forgery detection benchmark (2026)](https://arxiv.org/html/2603.01433v1).

**5) Ekran yeniden-çekim (screen recapture) / moiré deseni tespiti**

Artan bir saldırı vektörü: sahtekar, fiziksel bir sahte belge yerine, **bir ekranda
gösterilen sahte belgenin fotoğrafını** çeker (eKYC sistemlerini atlatmak için). Bu,
ekranın piksel ızgarasıyla kameranın sensör ızgarası arasındaki etkileşimden kaynaklanan
karakteristik **moiré (girişim) desenleri** ve renk-doku tutarsızlıkları bırakır. 🔎

- **Neden non-obvious:** Çoğu kalite/sahtecilik kontrolü yalnızca belgenin İÇERİĞİNE
  bakar; bu yöntem belgenin "gerçekten fiziksel bir yüzeyden mi yoksa bir ekrandan mı
  çekildiğine" bakar — tamamen farklı bir soru.
- Kaynak: [Screenshots, Printouts, and Recapture Attacks in eKYC](https://www.faceplusplus.com/blog/screenshots-printouts-and-recapture-attacks-common-document-fraud-risks-in-ekyc/),
  [Screen recapture detection based on color-texture analysis of document boundary regions](https://www.researchgate.net/publication/372007857_Screen_recapture_detection_based_on_color-texture_analysis_of_document_boundary_regions).

**6) PRNU sensör gürültüsü analizi — ileri seviye, opsiyonel**

Her kamera sensörü, üretim kusurlarından kaynaklanan benzersiz bir gürültü deseni
("parmak izi") üretir. Bir görüntünün bir bölümü başka bir kaynaktan (örn. başka bir
fotoğraf, bir ekran görüntüsü) geliyorsa, o bölgenin PRNU deseni geri kalanla
uyuşmayabilir. 🔎

- **Neden opsiyonel/ileri seviye:** Güvenilir çalışması için referans kamera gürültü
  profili veritabanı ve nispeten yüksek çözünürlüklü, az sıkıştırılmış görüntüler
  gerekir — mobil/production ortamında maliyetli ve kırılgan olabilir.
- Kaynak: [Digital Image Forensics Using Sensor Noise (Fridrich)](http://ws.binghamton.edu/fridrich/Research/full_paper_02.pdf),
  [Combining PRNU and noiseprint for robust device source identification](https://jis-eurasipjournals.springeropen.com/articles/10.1186/s13635-020-0101-7).

### 20.3. Özet öneri tablosu

| Yöntem | Uygulama maliyeti | Sahtekarın atlatma zorluğu | Bu projeyle ilişkisi |
|---|---|---|---|
| Checksum/check-digit doğrulama | Çok düşük (saf aritmetik) | Çok yüksek | Bağımsız, yeni bir katman |
| Copy-move tespiti | Orta | Yüksek | Bağımsız, yeni bir katman |
| ELA / çift JPEG analizi | Orta | Orta-Yüksek | Bağımsız, yeni bir katman |
| Font/glyph tutarlılık analizi | Orta | Yüksek | **Blur/Darkness modüllerinin tekniklerinden yeniden yararlanabilir** |
| Screen recapture / moiré tespiti | Orta-Yüksek | Yüksek (özellikle eKYC bypass'ına karşı) | Bağımsız, yeni bir katman |
| PRNU sensör analizi | Yüksek | Çok yüksek (ama kırılgan) | Bağımsız, ileri seviye/opsiyonel |

**Önerilen uygulama sırası (maliyet/etki oranına göre):** (1) Checksum doğrulama →
(2) Copy-move + ELA → (3) Font tutarlılığı → (4) Screen recapture/moiré → (5) PRNU
(yalnızca yüksek güvenlik gereksinimi varsa).

---

## 21. Conclusion

Bu proje, beş temel belge kalitesi probleminin (blur, glare, darkness, skew, occlusion)
literatürde önerilen klasik baseline yöntemlerle nasıl ölçülebileceğini, sentetik ama
kontrollü deneylerle test etmiştir. Üç modül (Blur, Darkness, Occlusion) beklenen,
kullanışlı sonuçlar üretmiş; bir modül (Glare) mevcut haliyle yetersiz bulunmuş; bir
modül (Skew) koşullu başarı göstermiştir. Bu karışık ama **dürüst** sonuç kümesi, tek bir
"her şey mükemmel çalıştı" anlatısından çok daha değerlidir — çünkü hangi yöntemlerin
hangi koşullarda güvenilir olduğunu somut sayılarla ortaya koymaktadır.

Projenin nihai hedefi olan kimlik sahteciliği tespiti için, bu beş modülün **çekim
kalitesi kontrolü** katmanını oluşturduğu, ama asıl sahtecilik (tamper) tespitinin
Bölüm 20'de önerilen, farklı bir kategori olan adli/forensic yöntemlerle
tamamlanması gerektiği sonucuna varılmıştır.

## 22. Appendix A — Projeyi Çalıştırma ve Kullanma Kılavuzu (Basit Anlatım)

Bu bölüm, teknik bilgisi az olan biri için de anlaşılır olacak şekilde yazılmıştır.

### Proje nedir, ne işe yarar?

Bu proje, bir belge/kimlik fotoğrafının **ne kadar "iyi çekilmiş"** olduğunu otomatik
olarak değerlendirmeye çalışan bir araç seti. Şu an 5 ayrı "dedektör" var: biri
bulanıklığa (blur), biri parlamaya (glare), biri karanlığa (darkness), biri eğikliğe
(skew), biri de bir şeyin belgenin üzerini kapatıp kapatmadığına (occlusion) bakıyor.
Her dedektör kendi başına çalışıyor ve bir sayı üretiyor; henüz hepsini tek bir "genel not"
(0-100 skor) haline getiren son adım yapılmadı.

### Projeyi çalıştırmak için ne gerekiyor?

1. **Python 3.11** kurulu bir bilgisayar (Mac, Windows veya Linux fark etmez).
2. Terminal/komut satırı açıp proje klasörüne girmek:
   ```bash
   cd document-quality-scoring
   ```
3. Gerekli kütüphaneleri kurmak (bir kere yapılır):
   ```bash
   pip install -r requirements.txt
   ```
4. Occlusion (kapanma) dedektörü için ayrıca bilgisayara "Tesseract OCR" adlı bir program
   kurulmalı (metin okuma motoru). Mac'te: `brew install tesseract tesseract-lang`.
   Windows'ta: Tesseract'ın resmi kurulum dosyası indirilip kurulur (`UB Mannheim
   tesseract installer` diye aratılabilir). Linux'ta: `sudo apt-get install tesseract-ocr
   tesseract-ocr-tur`.

### Bir dedektörü nasıl çalıştırırım? (örnek: Blur)

Her dedektörün 2-3 adımı var: önce test için sahte/örnek belgeler üretilir, sonra bu
belgelere yapay olarak bozulma eklenir, sonra da dedektör bu bozuk belgeleri "okuyup"
sonuç üretir.

```bash
python3 experiments/blur/generate_synthetic_documents.py   # sahte belgeler oluştur
python3 experiments/blur/apply_degradation.py               # belgeleri bulanıklaştır
python3 experiments/blur/run_experiment.py                  # ölç ve sonucu kaydet
```

Bu üç komutu çalıştırdıktan sonra `results/blur/` klasörüne bakabilirsin — orada hem
sayısal tablolar (Excel'de de açılabilen `.csv` dosyaları) hem de grafikler (`.png`
resim dosyaları) olacak.

Diğer dedektörler için de aynı mantık geçerli, sadece klasör adı değişiyor:
`experiments/glare/`, `experiments/darkness/`, `experiments/skew/`,
`experiments/occlusion/`. Her klasörün içinde hangi script'in hangi sırayla
çalıştırılacağı bellidir (`generate_*.py` → varsa `apply_*.py` → `run_experiment.py`).

### Kendi fotoğrafımı test edebilir miyim?

Şu anki haliyle proje, kendi ürettiği **sahte/örnek** belgeler üzerinde çalışacak şekilde
kuruldu — henüz "bana bir fotoğraf ver, sana skor vereyim" şeklinde hazır bir komut satırı
aracı (tool) yok. Ama alt yapı (`src/blur/metrics.py` vb. içindeki fonksiyonlar) gerçek
bir görüntü dosyası üzerinde de çalışacak şekilde yazıldı. Örneğin blur skorunu kendi bir
fotoğrafın için görmek istersen, üç satırlık bir Python kodu yeterli:

```python
import sys, cv2
sys.path.insert(0, "src")
from blur.metrics import laplacian_variance, tenengrad

img = cv2.imread("kendi_fotografim.jpg", cv2.IMREAD_GRAYSCALE)
print("Laplacian:", laplacian_variance(img))
print("Tenengrad:", tenengrad(img))
```

Bu sana ham bir sayı verir (örn. "3421.5") — ama bu sayının "iyi mi kötü mü" olduğunu
söyleyen bir eşik/skala henüz yok (bkz. Bölüm 13, 18) — bu, projenin bir sonraki
aşamasında (feature fusion) çözülecek.

### Sonuçları nerede görebilirim?

- `project_notes.md` — her modül için "ne yaptık, ne bulduk, neden bu kararı aldık"
  günlüğü. En kolay okunan, en anlaşılır dosya budur.
- `results/<modül>/` — sayısal tablolar ve grafikler.
- `reports/final_report.md` — (bu dosya) tüm projenin özet/resmi raporu.
- `README.md` — hızlı referans, komut listesi.

### Bir sonraki adımda ne olacak?

Bölüm 19 (Future Work) ve Bölüm 20'de (kimlik sahteciliğine özel öneriler) detaylandırıldığı
gibi: (1) gerçek fotoğraflarla test, (2) Glare dedektörünün düzeltilmesi, (3) beş
dedektörün tek bir "genel not"a birleştirilmesi, (4) sahtecilik tespiti için ek (checksum,
copy-move, ELA gibi) katmanların eklenmesi.

## 23. References

Tam kaynakça `research/references.md` dosyasındadır. Burada, bu raporda doğrudan atıfta
bulunulan kaynaklar listelenmiştir.

**Ana literatür (initial_research.md'den, bu oturumda doğrulandı):**

1. Alaei, A., Bui, V., Doermann, D., & Pal, U. (2023). *Document Image Quality
   Assessment: A Survey.* ACM Computing Surveys. DOI: 10.1145/3606692.
2. Nayef, N. et al. (2015). *SmartDoc-QA: A Dataset for Quality Assessment of Smartphone
   Captured Document Images.* CBDAR.
3. Rodin, D., & Orlov, N. (2019). *Fast Glare Detection in Document Images.*
   arXiv:1911.05189.
4. Wang, B., Li, C., Zou, W., et al. (2025). *A Comprehensive Survey on Shadow Removal
   from Document Images.*
5. Li, H., Zhu, F., & Qiu, J. (2019). *Towards Document Image Quality Assessment: A Text
   Line Based Framework.* arXiv:1906.01907.
6. Hull, J. J. (1998). *Document Image Skew Detection: Survey and Annotated
   Bibliography.*
7. Biswas, B., Bhattacharya, U., & Chaudhuri, B. B. (2023). *An Overview of Existing
   Literature on Document Skew Detection.* Malaysian Journal of Computer Science.

**Bölüm 20 için 🔎 dış araştırma kaynakları:**

8. [MRZ check digits explained](https://trustdochub.com/en/mrz-check-digits/) — TrustDocHub.
9. [ICAO 9303 MRZ check digits — how they work](https://idcheck.dev/icao-9303-check-digits/).
10. [Copy-move forgery detection: Survey, challenges and future directions](https://www.sciencedirect.com/science/article/abs/pii/S1084804516302144) — ScienceDirect.
11. [Copy-move forgery detection using superpixel segmentation and Helmert transformation](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-019-0469-9) — EURASIP.
12. [Error level analysis (ELA) — genel açıklama](https://trustdochub.com/en/error-level-analysis-image/) — TrustDocHub.
13. [ID card fraud detection using ELA and RPA](https://arvindn-iitkgp.medium.com/on-document-forensics-for-id-card-fraud-detection-using-ela-and-rpa-23c68dd8f3e0) — Medium.
14. [Multi-branch network for double JPEG detection and localization (2025)](https://www.nature.com/articles/s41598-025-04203-0) — Scientific Reports.
15. [DocForge-Bench: A Comprehensive Benchmark for Document Forgery Detection (2026)](https://arxiv.org/html/2603.01433v1) — arXiv.
16. [Screenshots, Printouts, and Recapture Attacks in eKYC](https://www.faceplusplus.com/blog/screenshots-printouts-and-recapture-attacks-common-document-fraud-risks-in-ekyc/) — Face++.
17. [Screen recapture detection based on color-texture analysis of document boundary regions](https://www.researchgate.net/publication/372007857_Screen_recapture_detection_based_on_color-texture_analysis_of_document_boundary_regions).
18. Fridrich, J. et al. [Digital Image Forensics Using Sensor Noise](http://ws.binghamton.edu/fridrich/Research/full_paper_02.pdf).
19. [Combining PRNU and noiseprint for robust and efficient device source identification](https://jis-eurasipjournals.springeropen.com/articles/10.1186/s13635-020-0101-7) — EURASIP.

Bu raporda atıfta bulunulmayan, ama projeye girdi sağlayan tüm diğer kaynaklar için:
`research/references.md`.
