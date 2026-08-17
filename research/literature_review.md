# Literature Review — Document Quality Scoring

**Durum:** Güncellenmiş sürüm (v2)
**Temel alınan doküman:** `initial_research.md`
**Bu belgenin amacı:** "Literatürde ne yapılmış?" sorusuna cevap vermek. Proje kararları ve
mimari tartışmaları `project_notes.md` içinde tutulur.

> **Okuma notu:** Bu belgede iki tür bilgi var. `initial_research.md`'den gelen ve bu
> oturumda çapraz kontrol edilen bilgiler **"Kaynak: initial_research.md"** olarak
> işaretlidir. Bu oturumda yapılan bağımsız web araştırmasından gelen her şey
> **🔎 Dış araştırma** etiketiyle işaretlidir. Hiçbir iddia, birincil kaynağın tam metni
> satır satır okunarak doğrulanmamıştır; doğrulamalar arama motoru sonuçları ve makale
> özetleri üzerinden yapılmıştır.

---

## 1. Problemin tanımı (Kaynak: initial_research.md)

Akıllı telefonla çekilen belge görüntülerinde kamera açısı, odaklama, hareket, aydınlatma
ve fiziksel engeller yüzünden beş ana bozulma türü oluşabilir: **blur, glare, darkness,
skew, occlusion**. Bu bozulmalar OCR, bilgi çıkarımı ve belge doğrulama gibi sonraki
işlemlerin başarısını düşürür. Literatürde bu alan **Document Image Quality Assessment
(DIQA)** olarak adlandırılıyor ve genel olarak *subjective* (insan değerlendirmesi) ve
*objective* (ölçülebilir özellik tabanlı) yaklaşımlara ayrılıyor; OCR tabanlı ölçüm de
önemli bir değerlendirme biçimi olarak kabul ediliyor.

**Ana kaynak:** Alaei, Bui, Doermann & Pal, *Document Image Quality Assessment: A Survey*,
ACM Computing Surveys, 2023. DOI: 10.1145/3606692. ✅ Bu oturumda doğrulandı (dl.acm.org,
digitalcommons.isical.ac.in üzerinden erişilebilir).

---

## 2. Problem bazlı literatür özeti

### 2.1. Blur

**Kaynak: initial_research.md** — Blur, görüntünün yüksek frekanslı detaylarının ve
kenarlarının zayıflamasıdır. Klasik yöntem aileleri: Laplacian variance, gradient magnitude
(Sobel/Scharr/Tenengrad), edge density, frekans-alanı (FFT) yöntemleri, local sharpness
ölçümleri. Laplacian variance en yaygın başlangıç yöntemi olsa da tek bir threshold'un
text yoğunluğu, font büyüklüğü, çözünürlük, noise ve JPEG sıkıştırmasından etkilenmesi
nedeniyle güvenilir olmadığı belirtiliyor — bu yüzden "iyi bir baseline, nihai çözüm değil"
olarak konumlandırılıyor. Daha gelişmiş yaklaşımlarda LBP, local variation, Log-Gabor,
gradient, entropy gibi özellikler çıkarılıp SVR ile kalite tahmin ediliyor.

**🔎 Dış araştırma (2025-2026 güncellemesi):**
- *Document Image Quality Assessment via Explicit Blur and Text Size Estimation*
  (Springer, ICDAR ailesi) — blur'u doğrudan text boyutuyla birlikte modelliyor; bu,
  initial_research.md'nin işaret ettiği "font büyüklüğü sonucu etkiler" sınırlamasına
  doğrudan bir çözüm önerisi.
- *A hybrid spatial blur detection and restoration algorithm for smartphone captured
  document images* (Scientific Reports, 2026) — tespit + restorasyonu birlikte ele alıyor.
- *A Method of Image Quality Assessment for Text Recognition on Camera-Captured and
  Projectively Distorted Documents* (MDPI Mathematics, 2021) — blur'u projektif bozulmayla
  birlikte, OCR başarımı üzerinden değerlendiriyor; SmartDoc-QA'nın OCR-merkezli
  felsefesiyle örtüşüyor.

### 2.2. Glare

**Kaynak: initial_research.md** — Glare, belge yüzeyinden gelen güçlü ışık yansımasının
bilgi kaybına yol açmasıdır. Klasik yaklaşım: luminance/brightness + HSV + saturation +
thresholding + connected components; ancak "beyaz belge alanı da yüksek parlaklığa sahip
olabilir" uyarısı nedeniyle tek başına threshold yanlış pozitiflere yol açabilir. Rodin &
Orlov (2019) CNN tabanlı bir glare heatmap yaklaşımı öneriyor: belge bloklara ayrılıyor,
luminance ve binarize stroke histogramları çıkarılıyor, CNN'e veriliyor.

**Kaynak (doğrulanmış):** Rodin, D., & Orlov, N. (2019). *Fast Glare Detection in Document
Images.* arXiv:1911.05189; ayrıca ICDAR 2019 Workshops'ta (IEEE Xplore doc. 8892889)
yayınlanmış. ✅

**🔎 Dış araştırma:** Belgeye özel, Rodin & Orlov (2019) sonrası doğrudan "document glare
detection" odaklı yeni bir çalışmaya bu oturumda rastlanmadı. Genel (belge dışı) specular
highlight removal literatüründe GlareNet (AIAA SciTech 2023) ve çeşitli GAN/attention
tabanlı highlight removal çalışmaları (2023-2024) var, ancak belgeye özel değiller. **Bu,
glare'ın hâlâ en az araştırılmış belge-kalite problemlerinden biri olduğunu gösteriyor** —
projenin özgün katkı yapabileceği bir alan.

### 2.3. Darkness / Illumination

**Kaynak: initial_research.md** — Darkness yalnızca ortalama koyuluk değil; aynı ortalama
brightness'a sahip iki görüntü, karanlık bölgelerin konumuna göre farklı kullanılabilirlikte
olabilir. Başlangıç ölçümleri: mean/median brightness, histogram, percentile (P5-P95),
local brightness/contrast. Daha ileri düzeyde, literatür darkness/shadow problemini
*illumination estimation/correction* veya *shadow removal* başlığı altında ele alıyor. Wang
et al. (2025) survey'i klasik yöntemleri **shadow-map based** ve **illumination-based**
olmak üzere ikiye ayırıyor, neural network tabanlı yöntemleri ayrıca sınıflandırıyor.

**Kaynak (doğrulanmış, küçük tarih notuyla):** Wang, B., Li, C., Zou, W. et al. *A
comprehensive survey on shadow removal from document images: datasets, methods, and
opportunities.* ✅ ResearchGate ve CoLab kayıtlarında mevcut; CoLab DOI kaydı 2024 yılını
gösteriyor (online-first 2024, resmi baskı 2025 olabilir) — bu küçük tutarsızlık not
edilmelidir.

**🔎 Dış araştırma:**
- *Synthetic Document Images with Diverse Shadows for Deep Shadow Removal Networks*
  (PMC/PubMed, 2024) — sentetik gölge veri seti üretim metodolojisi; projenin kontrollü
  bozulma testleri için doğrudan örnek teşkil ediyor.
- *DDSR-Net: Direct Document Shadow Removal Leveraging Multi-scale Attention* (Machine
  Intelligence Research, 2024) — multi-scale attention ile gölge kaldırma; Wang et al.
  (2025)'in "illumination-based methods" kategorisine güncel bir örnek.

### 2.4. Skew

**Kaynak: initial_research.md** — Skew, belgenin/text satırlarının eksene göre açısal
sapmasıdır. Yöntem aileleri: Projection Profile, Hough Transform, PCA, Connected Component
Analysis, Nearest Neighbor, Cross Correlation, Radon Transform, CNN. Hull (1998) survey'i
projection profile, feature distribution, Hough transform ve yön-duyarlı local mask
yaklaşımlarını sınıflandırıyor; Biswas et al. (2023) ise Hough, PCA, projection profile,
nearest-neighbor clustering, connected component analysis, cross-correlation, Radon
transform ve CNN'i birlikte inceliyor.

**Kaynak durumu:** Hull (1998) ⚠️ ve Biswas et al. (2023) ⚠️ — literatürde varlıkları
tutarlı görünüyor, ancak bu oturumda birincil kaynaklara doğrudan erişilip tam metin
kontrolü yapılamadı; ayrı bir doğrulama turu önerilir.

**🔎 Dış araştırma:** Bu oturumdaki aramalarda 2023-2025 arasında belgeye özel, skew'e
adanmış yeni bir deep learning yöntemine doğrudan rastlanmadı; skew düzeltmesi çoğunlukla
genel OCR/doküman-analiz sistemlerinin bir ön-işleme adımı olarak ele alınıyor. Bu,
initial_research.md'nin "skew için CNN kullanmak başlangıçta zorunlu değil" önerisini
destekliyor — klasik yöntemler (Hough, Projection Profile) hâlâ pratikte baskın.

### 2.5. Occlusion

**Kaynak: initial_research.md** — Occlusion, belge üzerindeki bilginin parmak, el, sticker,
başka bir nesne veya fiziksel hasar tarafından kapatılmasıdır. Diğer problemlerden farkı:
yalnızca "ne kadar" değil "neresi" ve "ne önemde" sorusunu da içeriyor. OCR confidence
düşüklüğü occlusion şüphesi oluşturabilir ama kesin kanıt değildir (blur/glare/darkness/skew
de OCR confidence'ı düşürür), bu yüzden yardımcı feature olarak kullanılmalı. Object
detection (YOLO vb.) veya segmentation ile occlusion ratio ve konum belirlenebilir; önerilen
formül: *Occlusion severity = Area + Location + Region Importance*.

**🔎 Dış araştırma:** Bu oturumda **belgeye özel occlusion tespiti** için tasarlanmış,
geniş kabul görmüş bir açık veri seti veya yöntem bulunamadı. En yakın literatür genel el
segmentasyonu/el pozu tahmini alanından geliyor (örn. HandSeg, occlusion-robust el pozu
çalışmaları) ve doğrudan belge bağlamında değil. **Bu, initial_research.md'nin işaret
ettiği "occlusion en az araştırılmış problem" gözlemini teyit ediyor** ve projenin en
özgün katkı potansiyeli taşıyan alanlarından biri olarak öne çıkıyor.

### 2.6. Text-Line Based Assessment (Kaynak: initial_research.md)

Li, Zhu & Qiu (2019), belgenin tamamını tek bir görüntü olarak değil, text satırı bazında
değerlendirmeyi öneriyor: her satır CNN ile ayrı skorlanıp ensemble ile birleştiriliyor.
52.094 sentetik text-line görüntüsü içeren bir dataset oluşturmuşlar. Bu yaklaşım kimlik
belgeleri için özellikle değerli, çünkü belgenin geneli kaliteli görünürken kritik bir alan
(örn. ID number) kötü kalitede olabilir.

**Kaynak (doğrulanmış):** Li, H., Zhu, F., & Qiu, J. (2019). arXiv:1906.01907. ✅

---

## 3. Genel DIQA yaklaşımları ve 2025-2026 gelişmeleri (🔎 Tamamı dış araştırma)

initial_research.md, DIQA'yı esas olarak klasik CV + hibrit ML çerçevesinde ele alıyordu.
Bu oturumda yapılan taramada, 2025-2026 döneminde alanın **multimodal büyük dil modelleri
(MLLM)** tabanlı yaklaşımlara doğru genişlediği görüldü:

| Çalışma | Yıl | Katkı |
|---|---|---|
| 🔎 DeQA-Doc (arXiv:2507.12796) | 2025 | Genel görüntü kalitesi MLLM skorlayıcısı DeQA-Score'u belgelere uyarlıyor; yüksek çözünürlük desteği, soft-label regresyon. |
| 🔎 Q-Doc (arXiv:2511.11410) | 2025 | MLLM'lerin DIQA yeteneğini 3 seviyede (kaba skor / bozulma tipi / şiddet) test eden benchmark. MLLM'ler temel yetenek gösteriyor ama tutarsız skorlama, bozulma yanlış tanımlama sorunları var; Chain-of-Thought prompting belirgin iyileşme sağlıyor. |
| 🔎 VQualA 2025 Document Image Quality Assessment Challenge | 2025 | ICCVW workshop yarışması — alanın topluluk düzeyinde aktif takip edildiğinin göstergesi. |
| 🔎 Kiruthika, Athanesious & Kiruthika (2026), Frontiers in Signal Processing | 2026 | 12 özellik (sharpness/focus/edge clarity/structural distortion) + 5 regresyon modeli (Lasso/Ridge/SVR/RF/XGBoost) ile OCR doğruluğunu tahmin ediyor; PaddleOCR/Keras OCR ile ground truth; en iyi PCC=0.9139 (XGBoost+Keras OCR). Bu, initial_research.md'nin Kaynak 8'i ile aynı makale — bu oturumda tam olarak doğrulandı ve tarihinin gerçek/geçerli olduğu teyit edildi (9 Nisan 2026 yayın tarihi, mevcut tarihten önce). |

**Proje açısından çıkarım:** MLLM tabanlı yöntemler şu an "hazır çözüm" değil (Q-Doc'un
kendi bulgularına göre), ama ileride özellikle *açıklanabilir rapor üretimi* aşamasında
tamamlayıcı bir bileşen olarak değerlendirilebilir. Bu proje için birincil yaklaşım olarak
seçilmiyor.

---

## 4. Veri setleri

### 4.1. SmartDoc-QA (Kaynak: initial_research.md, ✅ doğrulandı)

CBDAR 2015'te yayınlanmış, smartphone ile çekilmiş belge görüntülerinin kalitesini hem
insan algısı hem OCR doğruluğu üzerinden değerlendirmeyi amaçlayan bir benchmark. Modern
belgeler, eski idari mektuplar, fişler içeriyor; blur, ışık değişimleri, perspektif/geometrik
bozulmalar var. Ground truth: metin transkripsiyonları, OCR çıktıları, capture parametreleri.
Arşiv boyutu ~13.7 GB (bu spesifik sayı tam metinden ayrıca teyit edilmeli).

### 4.2. 🔎 Kimlik belgesi odaklı veri setleri (dış araştırma, initial_research.md'de yok)

| Dataset | Yıl | İçerik |
|---|---|---|
| MIDV-500 | 2018 | Mobil video akışıyla çekilmiş kimlik belgesi görüntüleri, farklı çekim koşulları |
| MIDV-2020 | 2021 | 1000 video klip + 2000 tarama + 1000 fotoğraf, 1000 mock kimlik, 72.409 etiketli görüntü |
| IDNet | 2024 | 837.060 sentetik kimlik belgesi, 20 tür, sahtecilik varyantları (morphing, portre değişimi, metin değişimi) |

Bu üç veri seti öncelikle belge tespiti/tanıma/sahtecilik odaklıdır, DIQA için özel
tasarlanmamıştır — ama mock/sentetik içerikleri nedeniyle projenin sentetik bozulma
pipeline'ı için ilham kaynağı olabilirler.

---

## 5. Üç mimari alternatif (Kaynak: initial_research.md)

1. **Classical CV** — her problem için elle tasarlanmış özellik + ağırlıklı skor. Az veri,
   hızlı, açıklanabilir; karmaşık durumlarda sınırlı.
2. **Hybrid (önerilen)** — Classical CV + OCR/Layout + gerekli noktalarda Deep Learning →
   feature vector → ML regresyon (RF/XGBoost/SVR) → 0-100 skor.
3. **End-to-End Deep Learning** — CNN/ViT ile tüm bozulmaların tek modelden öğrenilmesi;
   çok veri gerektirir, açıklanabilirliği düşük.

initial_research.md, Hybrid mimariyi önermektedir; bu proje planı da bu öneriyi takip
etmektedir (bkz. `project_notes.md`).

---

## 6. Bu incelemeden çıkan açık noktalar (özet)

- Ground truth stratejisi (insan + OCR birleşimi) somutlaştırılmamış.
- Ağırlıklandırma / model seçimi veriye dayalı değil, henüz sezgisel.
- Occlusion ve glare, belgeye özel literatürde en zayıf kalan alanlar.
- Problemler arası etkileşim (örn. skew + blur birlikte) hiçbir kaynakta ele alınmamış.
- Genel (document-agnostic) NR-IQA yöntemleriyle (BRISQUE/NIQE) karşılaştırma yapılmamış.

Bu noktalar `project_notes.md`'de deney planına dönüştürülmüştür.
