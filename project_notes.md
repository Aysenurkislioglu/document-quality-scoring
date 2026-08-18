# Project Notes — Document Quality Scoring

Bu dosya, projede yapılan çalışmaların kısa ve güncel bir kaydını tutar. Her modül
tamamlandığında buraya yeni bir bölüm eklenir. Detaylı literatür için `research/`
klasörüne, deney kodları için `experiments/` ve `src/` klasörlerine, ham sonuçlar için
`results/` klasörüne bakınız.

---

## Genel Proje Planı ve Mimari Kararlar

**Hedef:** Belge görüntülerindeki beş kalite problemini (blur, glare, darkness, skew,
occlusion) ayrı ayrı ölçüp, açıklanabilir bir 0-100 Document Quality Score üretmek.
Nihai motivasyon: kimlikte sahtecilikte kullanılan/gizlenen bozulmaları — özellikle CNN
gibi ağır, "kara kutu" ve sahtekârların atlatmasının nispeten kolay olduğu yöntemler yerine
**daha yüksek doğrulukta, uygulanabilir ve gerekçelendirilebilir** yöntemlerle — tespit
edebilmek.

**Seçilen mimari:** Hibrit (Classical CV + OCR/Layout + gerekli noktalarda Deep Learning →
feature vector → ML regresyon → 0-100 skor). Gerekçe: `research/literature_review.md`,
Bölüm 5.

**Yol haritası (yüksek seviye):**

| Aşama | İçerik | Durum |
|---|---|---|
| 1 | Literatür + dataset araştırması | ✅ Tamamlandı (`research/`) |
| 2 | Proje tasarımı, klasör yapısı, deney planı | ✅ Tamamlandı |
| 3 | Implementation: Blur → Glare → Darkness → Skew → Occlusion | 🔄 Blur tamamlandı, diğerleri bekliyor |
| 4 | Kontrollü sentetik + gerçek veri deneyleri | 🔄 Blur için tamamlandı |
| 5 | Feature fusion + ML skorlama | ⏳ Bekliyor |
| 6 | Karşılaştırmalı değerlendirme | ⏳ Bekliyor |
| 7 | Nihai rapor | ⏳ Bekliyor (tüm modüller bitince) |
| 8 | Final review | ⏳ Bekliyor |

**Önemli genel karar:** Şu ana kadar hiçbir gerçek belge görüntüsü veri seti (örn.
SmartDoc-QA) projeye indirilip entegre edilmedi. İlk baseline deneyleri **sentetik olarak
üretilen belge görüntüleri + kontrollü bozulma** üzerinde yapılıyor. Bunun nedeni ve
sınırlamaları ilgili modül notlarında (aşağıda) açıklanmıştır. Gerçek veri entegrasyonu
gelecek bir aşama olarak planlanmıştır.

---

## Modül: BLUR

**Tarih:** 17 Ağustos 2026

### Problem

Belge görüntülerinde odaklama hatası veya hareket nedeniyle oluşan bulanıklığı (blur)
ölçmek ve bunu 0-100 ölçeğinde yorumlanabilir bir alt-skora çevirmek.

### Araştırılan yöntemler

`research/literature_review.md` Bölüm 2.1'de özetlendiği gibi: Laplacian Variance,
Gradient magnitude (Sobel/Scharr), Tenengrad, edge density, frekans-alanı (FFT) yöntemleri,
local sharpness ölçümleri (LBP/Log-Gabor + SVR gibi öğrenme tabanlı yaklaşımlar dahil).
🔎 Dış araştırmada ayrıca blur+text-size'ı birlikte modelleyen yaklaşımlar ve blur
tespiti+restorasyonunu birleştiren güncel (2026) bir çalışma bulundu.

### Seçtiğimiz yöntem

Bu aşamada **Laplacian Variance** ve **Tenengrad (Sobel gradient magnitude)** uygulandı;
çapraz kontrol amacıyla ek olarak **Gradient Magnitude Mean** de hesaplandı.

### Neden bu yöntemi seçtik?

- İkisi de eğitim/veri gerektirmiyor → hızlı başlangıç (baseline) için uygun.
- Literatürün "önce klasik CV baseline, sonra gerekirse öğrenme tabanlı yöntem" önerisiyle
  tutarlı (`research/literature_review.md`, Bölüm 5).
- Laplacian (ikinci türev) ve Tenengrad (birinci türev) farklı matematiksel araçlar
  kullandığı için birbirini doğrulayan/çapraz kontrol eden iki bağımsız sinyal sağlıyor —
  proje hedefindeki "sahtekârların kolayca atlatamayacağı, açıklanabilir yöntem" isteğiyle
  örtüşüyor: ikisi çelişirse bu durum kendi başına bir sinyal (örn. gürültü/manipülasyon
  şüphesi) olarak kullanılabilir.
- Her iki yöntemin nasıl çalıştığına dair açıklama: `src/blur/README.md`.

### Kullanılan parametreler

| Parametre | Değer |
|---|---|
| Laplacian ksize | 1 (OpenCV varsayılanı) |
| Sobel ksize (Tenengrad) | 3 |
| Tenengrad threshold | 0 (tüm pikseller kullanıldı, gürültü filtresi uygulanmadı) |
| Gaussian blur sigma seviyeleri (bozulma şiddeti) | 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0 (9 seviye) |

### Yapılan deney

**Veri:** Gerçek veri seti henüz entegre edilmediği için, `experiments/blur/generate_synthetic_documents.py`
ile 12 sentetik belge görüntüsü üretildi (3 font boyutu: 14/20/28px × 2 paragraf yoğunluğu:
2/5 paragraf × 2 tekrar). Her belge başlık, mock kimlik alanları (Ad Soyad, Belge No,
Tarih) ve gövde metni içeriyor (`data/synthetic/blur/originals/`).

`experiments/blur/apply_degradation.py` ile her belgeye 9 farklı şiddette Gaussian blur
uygulanıp toplam 108 görüntü üretildi (`data/synthetic/blur/degraded/`).

`experiments/blur/run_experiment.py` ile:
1. Her görüntü için 3 metrik hesaplandı → `results/blur/scores.csv`
2. Her belge için, şiddet seviyesi ile skor arasındaki **Spearman korelasyonu**
   hesaplandı (monotonluk testi) → `results/blur/monotonicity_summary.csv`
3. Bozulma olmadan (severity=0), font boyutuna göre skorların ne kadar değiştiği
   (coefficient of variation) ölçüldü → `results/blur/baseline_by_fontsize.csv`,
   `results/blur/baseline_coefficient_of_variation.csv`
4. Grafikler üretildi → `results/blur/plots/`

### Deney sonucu

**1) Monotonluk (bozulma şiddeti arttıkça skor tutarlı biçimde düşüyor mu?)**

| Metrik | Ortalama Spearman rho (12 belge) | Min | Max |
|---|---|---|---|
| Laplacian Variance | −0.9986 | −1.0000 | −0.9833 |
| Tenengrad | −1.0000 | −1.0000 | −1.0000 |
| Gradient Magnitude Mean | −1.0000 | −1.0000 | −1.0000 |

Her iki ana yöntem de, test edilen sentetik veri üzerinde **neredeyse mükemmel monoton
azalma** gösterdi (rho ≈ −1). Tenengrad, 12 belgenin tamamında kusursuz monotonluk
gösterirken, Laplacian Variance 1 belgede (doc_001) çok küçük bir sapma gösterdi (rho =
−0.983, yine de çok güçlü). Grafikler: `results/blur/plots/score_vs_severity.png` ve
log-ölçekli versiyonu `score_vs_severity_logscale.png`.

**2) Font boyutu duyarlılığı (literatürün "tek eşik güvenilir değil" iddiasının testi)**

| Metrik | Coefficient of Variation (font boyutları arası, blur yokken) |
|---|---|
| Laplacian Variance | 0.272 |
| Tenengrad | 0.321 |
| Gradient Magnitude Mean | 0.276 |

Yani **hiç blur olmasa bile**, yalnızca font boyutu farkı yüzünden skorlar %27-32
oranında değişkenlik gösteriyor (`results/blur/plots/baseline_by_fontsize.png`). Bu,
`research/literature_review.md`'de aktarılan "tek bir sabit Laplacian threshold'u farklı
belge/font koşullarında güvenilir değildir" iddiasını **bu sentetik veri özelinde
doğruluyor.**

### Karşılaşılan problemler

- İlk grafik denemesinde (doğrusal y ekseni) skorlar şiddet seviyesi 3'ten sonra görsel
  olarak "sıfıra yapışmış" gibi görünüyordu; bu yanıltıcıydı çünkü Spearman korelasyonu
  yüksek seviyelerde de güçlü monotonluk gösteriyordu. **Çözüm:** log-ölçekli ek bir grafik
  eklendi (`score_vs_severity_logscale.png`), skorların hiçbir seviyede tam sıfıra
  inmediği (Bash ile min değerler kontrol edildi) doğrulandıktan sonra.
- Sentetik metin üretimi için gerçek bir "lorem ipsum" kütüphanesi yerine küçük bir
  Türkçe kelime havuzundan rastgele kelime seçimi kullanıldı; bu metinler anlamsız ama
  görsel olarak gerçek bir belgeye yeterince benziyor (kenar/doku istatistikleri
  açısından amaca uygun).

### Aldığımız kararlar

1. **Gerçek veri yerine sentetik veri ile başlama kararı:** SmartDoc-QA gibi gerçek
   datasetler henüz indirilip lisans/boyut değerlendirmesi yapılmadığı için, ilk baseline
   doğrulaması kontrollü sentetik veri üzerinde yapıldı. Bu, yöntemin *temel davranışını*
   (monotonluk) ucuza ve hızlıca doğrulamaya yetti, ama **gerçek kamera blur'unun
   (defocus + motion blur karışımı, JPEG artefaktları, gerçek kağıt dokusu) sentetik
   Gaussian blur'dan farklı davranabileceği** unutulmamalı. Gerçek veri ile tekrar test
   edilmesi gerekiyor (bkz. sonraki adım).
2. **Laplacian ve Tenengrad'ın birlikte kullanılmasına karar verildi** (tek yöntem değil)
   — ikisinin uyuşması, tek bir metriğe göre daha güvenilir bir sinyal.
3. **Mutlak bir "iyi/kötü" eşik değeri bu aşamada belirlenmedi.** Font boyutu deneyi,
   sabit bir eşiğin güvenilir olmayacağını gösterdiği için, ileride (Aşama 5 — ML skor
   füzyonu) mutlak skor yerine göreceli/öğrenilmiş bir skorlama tercih edilecek.

### Bir sonraki adım

Kullanıcı talimatına göre bu aşamada diğer kalite problemlerine geçilmiyor. Blur modülü
tamamlandı olarak işaretlendi. Sıradaki modül: **Glare**.

Ayrıca, ileride (tüm modüller bitince) ele alınacak açık noktalar:
- Blur ölçümünün gerçek (sentetik olmayan) belge görüntüleriyle tekrar test edilmesi.
- Gürültü (noise) etkisinin ayrıca test edilmesi (Laplacian'ın gürültüye Tenengrad'dan
  daha duyarlı olduğu literatür iddiası henüz sınanmadı).
- Font boyutu duyarlılığının, skor füzyon aşamasında nasıl telafi edileceğinin
  (normalizasyon mu, ek feature mı) belirlenmesi.

---

## Modül: GLARE

**Tarih:** 17 Ağustos 2026

### Problem

Belge yüzeyinden gelen güçlü ışık yansımasının hangi bölgelerde bilgi kaybına yol açtığını
tespit etmek ve bunu bir alt-skora çevirmek.

### Araştırılan yöntemler

`research/literature_review.md` Bölüm 2.2: luminance/brightness + HSV + saturation +
thresholding + connected components (klasik baseline); Rodin & Orlov (2019) CNN tabanlı
glare heatmap (ileri yöntem, bloklara ayırma + luminance + binarize stroke histogramı +
CNN). Literatür açıkça uyarıyor: "beyaz belge alanı da yüksek parlaklığa sahip olabilir",
bu yüzden tek başına threshold yanlış pozitiflere yol açabilir.

### Seçtiğimiz yöntem

Klasik baseline: **HSV eşikleme (yüksek V + düşük S) + Connected Components filtreleme**,
yalnızca belgenin içerik kutusu (content bounding box) içinde uygulandı.

### Neden bu yöntemi seçtik?

Literatürün önerdiği ilk aşama baseline bu olduğu için (`research/literature_review.md`,
Bölüm 5 — "önce klasik CV baseline"). Yöntemin nasıl çalıştığı: `src/glare/README.md`.

### Kullanılan parametreler

| Parametre | Değer |
|---|---|
| V (parlaklık) eşiği | ≥ 235 (0-255) |
| S (saturasyon) eşiği | ≤ 35 (0-255) |
| Min. bağlı bileşen alanı | 15 piksel |
| ROI | Belgenin içerik kutusu (kenar boşlukları hariç) |
| Enjekte edilen glare şiddet seviyeleri | içerik alanının %0 / %3 / %6 / %10 / %15 / %22'si (hedef alan, Gaussian yumuşak geçişli daire) |

### Yapılan deney

`experiments/glare/generate_glare_documents.py`: Blur modülüyle aynı 12 sentetik belge
ızgarası (font boyutu × paragraf sayısı × tekrar) yeniden üretildi (bu kez içerik
kutusu/kimlik alanı koordinatları da kaydedilerek — bkz. `experiments/_common/synthetic_documents.py`,
DRY amaçlı ortak üretici). Her belgeye, merkezi içerik kutusunun ortasında, yumuşak
(Gaussian) kenarlı, artan büyüklükte bir "glare lekesi" eklendi; yer gerçeği glare alanı
(alpha > 0.5 bölgesi) her seviye için ayrıca ölçülüp kaydedildi.

`experiments/glare/run_experiment.py`: her görüntü için iki metrik hesaplandı:
1. **naive_glare_ratio** — gerçek/üretime uygun yöntem (yukarıdaki HSV+CC baseline).
2. **oracle_text_washout_ratio** — YALNIZCA bu deneyi doğrulamak için eklenen,
   referans (temiz orijinal) gerektiren yardımcı metrik: orijinalde koyu (metin) olan
   piksellerin ne kadarının bozulmuş görüntüde "yıkanmış/beyaz" hale geldiğini ölçer.
   Üretimde kullanılamaz (temiz referans gerektirir), yalnızca "enjekte ettiğimiz sentetik
   glare gerçekten ölçülebilir bir etki yaratıyor mu?" sorusunu ayırt etmek için eklendi.

### Deney sonucu

**Önemli bulgu — sonuç ilk bakışta yanıltıcı olabilir:** Spearman korelasyonuna göre her
iki metrik de mükemmel monoton davranış gösterdi (rho = 1.0000, 12 belgenin tamamında).
Ancak bu, `naive_glare_ratio`'nun pratikte işe yaradığı anlamına GELMİYOR:

| Metrik | Severity=0 ortalama | Severity=5 ortalama | Mutlak değişim | Göreceli değişim |
|---|---|---|---|---|
| naive_glare_ratio | 0.849 | 0.860 | 0.011 | **%1.3** |
| oracle_text_washout_ratio | 0.000 | 0.034 | 0.034 | (0'dan başladığı için % tanımsız, ama mutlak değişim anlamlı) |

`naive_glare_ratio`, glare hiç yokken bile içerik kutusunun **%84.9'unu** "glare" olarak
işaretliyor (`results/glare/false_positive_baseline.csv`) — çünkü satır aralarındaki ve
paragraf boşluklarındaki sıradan beyaz alan da V yüksek + S düşük kriterini karşılıyor.
Glare şiddeti arttıkça oran yalnızca %84.9'dan %86.0'a çıkıyor: teknik olarak "monoton"
ama pratikte **neredeyse ayırt edici gücü yok** (bkz. `results/glare/plots/naive_vs_ground_truth_scatter.png`
— belgeler arası gürültü, gerçek sinyali tamamen gölgeliyor).

Buna karşılık `oracle_text_washout_ratio` (yalnızca doğrulama amaçlı, referans gerektiren
metrik), 0'dan 0.034'e net ve tutarlı bir artış gösterdi — yani **enjekte ettiğimiz
sentetik glare'in gerçek, ölçülebilir bir etkisi var**; sorun veride değil, `naive_glare_ratio`
yönteminin bu sinyali yakalayamamasında.

Grafikler: `results/glare/plots/naive_vs_oracle.png`, `naive_vs_ground_truth_scatter.png`.

### Karşılaşılan problemler

- **Kritik teknik bulgu:** Sentetik belgelerimiz gri tonlamalı (PIL "L" modu) üretildiği
  için, HSV'ye çevrildiğinde **Saturasyon (S) kanalı her zaman 0** çıkıyor (R=G=B olduğunda
  S matematiksel olarak sıfırdır). Bu, yöntemin "S düşük" kriterinin **hiçbir ayırt edici
  bilgi taşımadığı** anlamına geliyor — tespit tamamen V (parlaklık) eşiğine indirgeniyor,
  ki bu da sıradan beyaz kağıdı gerçek glare'den ayıramıyor. Bu, literatürün "beyaz belge
  alanı da yüksek parlaklığa sahip olabilir" uyarısının **en uç, en net biçimde
  doğrulanmasıdır** — ama aynı zamanda deney tasarımımızın bir sınırlamasını da ortaya
  koyuyor.
- İlk grafik yalnızca Spearman rho'ya bakılarak yorumlanmaya çalışıldığında sonuç
  yanıltıcı biçimde "başarılı" görünüyordu; `dynamic_range_analysis` eklenerek bu
  yanılgı düzeltildi (bkz. "Aldığımız kararlar").

### Aldığımız kararlar

1. **Yalnızca Spearman korelasyonuna güvenmemeye karar verildi.** Bundan sonraki tüm
   modüllerde monotonluk testinin yanına mutlaka bir **dynamic range / etki büyüklüğü**
   analizi eklenecek (severity=0 ile severity=max arasındaki mutlak/göreceli fark).
2. **Naive HSV+CC yöntemi mevcut haliyle bu projede kullanılmaya hazır değil** olarak
   işaretlendi. Production'a geçmeden önce ya (a) renkli/gerçekçi kağıt tonu içeren veri
   ile yeniden test edilmeli, ya da (b) ROI, tüm içerik kutusu yerine yalnızca metin
   satırlarının sıkı sınırlayıcı kutularına daraltılmalı (böylece satır arası boşluklar
   dışarıda kalır), ya da (c) literatürün önerdiği ileri yöntem (Rodin & Orlov'un stroke
   histogram + CNN yaklaşımı) değerlendirilmeli. Bu, bir sonraki adıma not olarak
   bırakıldı, bu pass'te düzeltilmedi (kapsamı dar tutma kararı).
3. **Oracle metrik yalnızca deney doğrulaması için tutuldu, `src/glare/metrics.py`'ye
   eklenmedi** — çünkü üretimde temiz referans görüntü olmayacak. `src/` klasörü yalnızca
   gerçekten kullanılabilir/üretime uygun yöntemleri içermeli.

### Bir sonraki adım

Kullanıcı talimatına göre bu aşamada bir sonraki modüle (**Darkness**) geçiliyor. Glare
yöntemi "tamamlandı" değil, "baseline olarak denendi ve mevcut haliyle yetersiz bulundu"
statüsünde bırakıldı — bu, ileride (Aşama 6/7) tekrar ele alınacak açık bir konu olarak
işaretlendi.

---

## Modül: DARKNESS

**Tarih:** 17 Ağustos 2026

### Problem

Görüntü veya görüntünün belirli bölgelerinin yetersiz aydınlatılmış olmasını tespit etmek
— ÖZELLİKLE global ortalamanın gizleyebileceği, küçük ama kritik bölgelerdeki (örn. kimlik
numarası alanı) lokal karanlığı yakalayabilmek.

### Araştırılan yöntemler

`research/literature_review.md` Bölüm 2.3: mean/median brightness, histogram, percentile
(P5-P95), local brightness/contrast (klasik baseline); illumination estimation / shadow
segmentation (ileri yöntem, Wang et al. 2025 survey). Literatürün verdiği örnek doğrudan
bu modülün test hipotezini oluşturdu: "aynı ortalama parlaklığa sahip iki görüntü farklı
kalitede olabilir."

### Seçtiğimiz yöntem

Üç tamamlayıcı ölçüm birlikte uygulandı: **global mean/median**, **percentile analizi
(P5/P25/P50/P75/P95)**, **blok-bazlı yerel (local) en-karanlık-blok ortalaması**
(block_size=16px) + yerel kontrast. Detaylı açıklama: `src/darkness/README.md`.

### Neden bu yöntemi seçtik?

Literatürün kendisi tek bir global ölçümün yetersiz olabileceğini işaret ediyor; üç
yaklaşımı birlikte test etmek, hangisinin gerçekten "lokal karanlık" problemini
yakaladığını (varsayımla değil) veriyle göstermeyi sağladı.

### Kullanılan parametreler

| Parametre | Değer |
|---|---|
| Percentile'lar | 5, 25, 50, 75, 95 |
| Blok boyutu (darkest_block_mean) | 16×16 piksel (küçük kimlik alanını —yaklaşık 105×26 px— yakalayabilmek için özellikle küçük seçildi) |
| Global karartma şiddet seviyeleri | çarpan: 1.00 / 0.85 / 0.70 / 0.55 / 0.40 / 0.25 |
| Lokal karartma şiddet seviyeleri | aynı çarpanlar, yalnızca "Belge No" alanına (+8px padding) uygulandı |

### Yapılan deney

`experiments/darkness/generate_darkness_documents.py`: aynı 12 sentetik belge ızgarası
kullanılarak iki ayrı senaryo üretildi:
1. **Global senaryo:** tüm görüntü aynı çarpanla karartıldı (72 görüntü).
2. **Lokal senaryo:** yalnızca "Belge No" alanı (görüntünün ~%0.29'u) karartıldı, geri
   kalan HİÇ değiştirilmedi (72 görüntü).

`experiments/darkness/run_experiment.py`: her iki senaryo için `global_mean`, percentile'lar
ve `darkest_block_mean` hesaplandı; monotonluk (Spearman) ve dynamic range analizleri
yapıldı; iki senaryo karşılaştırmalı grafiklerle görselleştirildi.

### Deney sonucu

**Global senaryo:** Beklendiği gibi TÜM metrikler mükemmel monoton azalma gösterdi
(rho = −1.0000, tüm metrikler, tüm belgeler).

**Lokal senaryo — literatürün iddiasının doğrudan doğrulanması:**

| Metrik | Severity=0 ort. | Severity=5 ort. | Göreceli değişim | Spearman rho (ort.) |
|---|---|---|---|---|
| global_mean | 238.33 | 237.50 | **−%0.35** | −1.0000 (ama pratikte anlamsız — bkz. aşağı) |
| p25 / p50 | 255.00 | 255.00 | **%0** (sabit) | tanımsız (varyans yok) |
| p5 | 70.58 | 67.17 | −%4.84 | −0.87 (kısmi duyarlılık) |
| darkest_block_mean | 99.09 | 44.02 | **−%55.58** | −0.83 (güçlü, belirgin tepki) |

Bu tablo, `research/literature_review.md`'de aktarılan iddiayı sayısal olarak doğruluyor:

- **global_mean, teknik olarak "monoton" (rho=-1.0) olsa bile**, gerçek değişimi yalnızca
  **%0.35** — yani pratikte tamamen kör. (Aynı glare modülünde gördüğümüz "yüksek rho,
  düşük dynamic range" tuzağı burada da tekrarlandı — bkz. Glare bölümü, "Aldığımız
  kararlar 1".)
- **P25 ve P50, karartılan alan görüntünün %5'inden çok daha küçük olduğu için (%0.29),
  HİÇ tepki vermedi** (sabit 255, korelasyon tanımsız). Bu, literatürdeki percentile
  önerisinin bile küçük/lokalize bölgeler için tek başına yeterli olmayabileceğini
  gösteren, dokümanda öngörülmemiş yeni bir bulgu.
- **P5 kısmi bir sinyal yakaladı** (muhtemelen zaten var olan koyu metin piksellerinin P5
  eşiğine yakın olması sayesinde) ama zayıf.
- **darkest_block_mean (16×16 blok) en güçlü ve en anlamlı tepkiyi verdi** (%55.6 değişim,
  görsel olarak da net bir eğim — bkz. `results/darkness/plots/global_vs_local_darkestblock.png`).
  Bu, literatürün "blok-bazlı yerel analiz" önerisinin bu senaryoda gerçekten işe yaradığını
  gösteriyor — Glare modülünün aksine, burada baseline yöntem beklendiği gibi çalıştı.

Grafikler: `results/darkness/plots/global_vs_local_meanp5.png`,
`global_vs_local_darkestblock.png`.

### Karşılaşılan problemler

- Font boyutu 28px olan belgelerde (doc_009-012), `darkest_block_mean`'in monotonluğu
  diğerlerine göre biraz daha zayıf çıktı (rho ≈ −0.65 vs. diğerlerinde ≈ −0.85/−0.99) ve
  `p5` bu belgelerde sabit (NaN korelasyon) kaldı. Olası açıklama: büyük fontta harf
  gövdeleri daha kalın olduğu için "en karanlık blok" zaten (karartma öncesinde) mevcut
  siyah metin pikselleri tarafından domine ediliyor olabilir; bu, karartmanın etkisini
  kısmen maskeliyor olabilir. Bu, kesin olarak doğrulanmış bir açıklama DEĞİL, bir
  hipotezdir — ileride ayrıca incelenmesi gerekiyor.

### Aldığımız kararlar

1. **Blok boyutu küçük tutuldu (16px).** İlk denemede daha büyük bir blok boyutu
   düşünülmüştü, ama hedef alanın (~105×26 px) büyük bloklarla (örn. 32px) yeterince
   "saf" biçimde örneklenemeyeceği öngörüldü; 16px seçimi, sonuçlarda görüldüğü gibi
   isabetli oldu.
2. **Nihai darkness skoru için, yalnızca global_mean/percentile YETERLİ DEĞİL** —
   blok-bazlı yerel analiz mutlaka dahil edilmeli. Bu karar, skor füzyonu aşamasına
   (Aşama 5) taşınacak.
3. **Font-boyutu/metin-yoğunluğu etkisiyle blok-analizi arasındaki etkileşim** (yukarıdaki
   "karşılaşılan problem") bu pass'te çözülmedi, sonraki adıma not düşüldü.

### Bir sonraki adım

Kullanıcı talimatına göre bir sonraki modüle (**Skew**) geçiliyor.

---

## Modül: SKEW

**Tarih:** 17 Ağustos 2026

### Problem

Belgenin (kamera açısından kaynaklanan) yatay eksene göre açısal sapmasını (θ) tahmin
etmek.

### Araştırılan yöntemler

`research/literature_review.md` Bölüm 2.4: Projection Profile, Hough Transform, PCA,
Connected Component Analysis, Nearest Neighbor, Cross Correlation, Radon Transform, CNN.
Hough ve Projection Profile literatürde en sık kullanılan klasik yöntem aileleri olarak
öne çıkıyor.

### Seçtiğimiz yöntem

**Hough Transform** ve **Projection Profile**, birbirinden bağımsız iki yöntem olarak
uygulanıp karşılaştırıldı. Yöntemlerin nasıl çalıştığı: `src/skew/README.md`.

### Neden bu yöntemi seçtik?

initial_research.md'nin doğrudan önerdiği baseline bu ikisiydi ("Skew için CNN kullanmak
başlangıçta zorunlu değil"). İki farklı prensip (kenar/doğru tespiti vs. satır profili)
kullanan yöntemleri karşılaştırmak, hangisinin bu proje bağlamında (sentetik kimlik
belgesi benzeri görüntüler) daha güvenilir olduğunu görmemizi sağladı.

### Kullanılan parametreler

| Parametre | Değer |
|---|---|
| Hough — Canny eşikleri | 50 / 150 |
| Hough — oy eşiği (threshold) | 150 |
| Hough — kabul edilen max açı sapması | ±30° (bu aralık dışındaki doğrular gürültü sayılıp elendi) |
| Projection Profile — aday açı aralığı | −15° … +15°, 0.5° adımlarla (61 aday) |
| Projection Profile — koyu piksel eşiği | gri değer < 180 |
| Enjekte edilen açılar (yer gerçeği) | −12, −8, −5, −2, −1, 0, 1, 2, 5, 8, 12 derece |

### Yapılan deney

`experiments/skew/generate_skew_documents.py`: 12 sentetik belge, 11 farklı bilinen açıyla
döndürüldü (132 görüntü). Döndürme `cv2.getRotationMatrix2D` ile yapıldı; ön testte iki
yöntemin de bu döndürmeyi **karşıt işaretle** ölçtüğü tespit edildi (deneysel doğrulama),
bu yüzden yer gerçeği açı `-applied_angle` olarak tanımlandı (bkz. kod docstring'leri).

`experiments/skew/run_experiment.py`: her görüntü için her iki yöntemle açı tahmini
yapıldı, gerçek açıyla karşılaştırılıp mutlak hata (MAE) hesaplandı; hata ayrıca açı
büyüklüğüne (0° / küçük 1-2° / orta 5-8° / büyük 12°) göre kırılımlandı.

### Deney sonucu

**Genel MAE:**

| Yöntem | Ortalama Mutlak Hata | Std |
|---|---|---|
| Hough Transform | 0.82° | 2.01 |
| Projection Profile | 1.82° | 5.10 |

Genel ortalamaya bakıldığında Hough daha iyi görünüyor, ama bu ortalama **yanıltıcı** —
altta yatan dağılım çok farklı bir hikâye anlatıyor:

**Projection Profile — çarpıcı bulgu:** 12 belgenin **10'unda MAE = 0.0000°** (yani
mükemmel tahmin), ama **2 belgede (doc_001, doc_002) MAE ≈ 9.5-12.3°** ile ciddi biçimde
başarısız oldu. Bu iki belgenin ortak özelliği: **en küçük font (14px) + en az paragraf
sayısı (2)** — yani ızgaradaki en az metin satırına sahip belgeler. Başarısız
tahminlerin **tamamı**, aday açı aralığının sınırına (**−15°**) kilitlenmiş durumda
(`results/skew/plots/prediction_vs_ground_truth.png`'de y=−15 çizgisinde dizilen turuncu
X'ler). Yorum: yeterince az satır olduğunda, satır-profili varyans kriterinin net bir
tepe noktası oluşturamadığı, bunun yerine arama aralığının kenarına kaçtığı görülüyor.
Bu KESİN olarak kanıtlanmış bir neden-sonuç açıklaması değil, veriyle tutarlı bir
gözlem/hipotezdir.

**Hough Transform — daha istikrarlı ama açı büyüdükçe kademeli olarak kötüleşiyor:**

| Açı grubu | Hough MAE | Projection MAE |
|---|---|---|
| 0° (referans) | 0.21° | 2.50° * |
| Küçük (1-2°) | 0.36° | 2.50° * |
| Orta (5-8°) | 0.95° | 1.88° * |
| Büyük (12°) | 1.79° | 0.00° |

(*Projection Profile'daki yüksek değerler yukarıda açıklanan 2 belgenin aykırı
değerlerinden kaynaklanıyor — bu satırlar tüm belgeler için "tipik" performansı temsil
etmiyor.)

Hough'un hatası açı büyüdükçe düzgün biçimde artıyor (0.21° → 1.79°) — bu, `max_angle_deviation`
filtresi ve doğru tespitinin büyük açılarda zorlaşmasıyla tutarlı, beklenen bir davranış.

Grafikler: `results/skew/plots/prediction_vs_ground_truth.png`,
`error_by_angle_bucket.png`.

### Karşılaşılan problemler

- **İşaret (sign) uyuşmazlığı:** İlk testte, uygulanan döndürme açısı ile her iki
  yöntemin ölçtüğü açının ZIT işaretli olduğu görüldü. Kısa bir doğrulama testiyle
  (`true_angle` bilinen küçük bir döndürme uygulanıp tahminlerle karşılaştırılarak)
  bu netleştirildi ve yer gerçeği tanımı buna göre düzeltildi (bkz. yukarıdaki not).
- **Projection Profile'ın az-metinli belgelerde arama sınırına kilitlenmesi** (yukarıda
  detaylandırıldı) — bu deneyin en önemli, beklenmedik bulgusu.
- Deney, 132 görüntü × 61 aday açı (Projection Profile için) nedeniyle ~60 saniye sürdü;
  büyük ölçekli veri setlerinde bu yöntemin performans/hız optimizasyonu (örn. kaba
  arama + ince arama iki aşamalı strateji) gerekebilir — bu pass'te yapılmadı.

### Aldığımız kararlar

1. **Tek bir yöntem yerine ikisinin birlikte tutulmasına karar verildi**, ama farklı
   gerekçeyle: Hough daha istikrarlı/güvenilir bir "varsayılan", Projection Profile ise
   yeterli metin yoğunluğu olduğunda daha hassas — iki yöntem arasındaki büyük anlaşmazlık
   (örn. biri 0°, diğeri 15° derse) ileride bir "güvenilirlik/şüphe" sinyali olarak
   kullanılabilir (bu, Blur modülündeki "iki yöntem birbirini doğrular" yaklaşımıyla
   tutarlı bir tasarım deseni).
2. **Projection Profile'ın metin yoğunluğuna duyarlılığı, ham haliyle production'a hazır
   olmadığını gösteriyor** — en azından "yeterli metin satırı var mı?" kontrolü (örn.
   minimum satır sayısı eşiği) olmadan güvenilmemeli. Bu, bir sonraki adıma not düşüldü.
3. **Arama sınırına kilitlenme (boundary lock-in) durumunu ayrıca bir "başarısızlık
   bayrağı" olarak işaretlemeye karar verildi** (örn. tahmin edilen açı, arama aralığının
   sınırına çok yakınsa güvenilirlik düşük sayılmalı) — henüz kodlanmadı, gelecek adım.

### Bir sonraki adım

Kullanıcı talimatına göre bir sonraki modüle (**Occlusion**) geçiliyor.

---

## Modül: OCCLUSION

**Tarih:** 17 Ağustos 2026

### Problem

Belgenin bir kısmının (parmak, el, sticker, nesne) kapatılmasını tespit etmek — özellikle
kimlik belgelerinde kritik alanların (ID number gibi) kapanma durumunu.

### Araştırılan yöntemler

`research/literature_review.md` Bölüm 2.5: OCR confidence (yardımcı sinyal, tek başına
yeterli değil — blur/glare/darkness/skew de confidence'ı düşürebilir), object detection
(YOLO), segmentation (occlusion mask + area + location + region importance). Belgeye özel,
geniş kabul görmüş bir occlusion veri seti/yöntemi bu oturumda bulunamadı (bkz.
`research/literature_review.md`, Bölüm 2.5 — "en az araştırılmış problem").

### Seçtiğimiz yöntem

**OCR + "beklenen alan deseni/uzunluğu" karşılaştırması.** "Belge No" alanı kırpılıp OCR
edildi; iki tamamlayıcı sinyal hesaplandı: (1) **length_ratio** — OCR'ın okuduğu karakter
sayısının beklenen uzunluğa (10 hane) oranı, (2) **mean_confidence** — Tesseract'ın kendi
güven skoru. İkisi birleştirilerek bir **occlusion_suspicion_score** (0-100) üretildi.
Detaylı açıklama: `src/occlusion/README.md`.

### Neden bu yöntemi seçtik?

Literatürün önerdiği baseline (OCR + layout) buydu; ayrıca initial_research.md'nin
"AYŞENUR K______" örneğini doğrudan somutlaştırıyor. "Sadece OCR confidence'a güvenme"
uyarısını ciddiye alarak, occlusion'a daha özgü olan **uzunluk/format sapması** sinyalini
de ayrıca hesaplayıp iki sinyali karşılaştırdık.

### Kullanılan parametreler

| Parametre | Değer |
|---|---|
| Hedef alan | "Belge No" (10 haneli sayı) |
| OCR motoru | Tesseract 5.3 (+ `tesseract-ocr-tur` dil paketi kuruldu, bu alan için `eng` + rakam whitelist kullanıldı) |
| Tesseract PSM modu | 7 (tek satır metin) |
| Karakter whitelist | yalnızca `0123456789` |
| Crop padding / upscale | 6px padding, 3x büyütme (küçük metin OCR doğruluğunu artırmak için) |
| Enjekte edilen kapanma oranları | %0 / %20 / %40 / %60 / %80 / %100 (alanın sağından sola kapatılarak) |

### Yapılan deney

`experiments/occlusion/generate_occlusion_documents.py`: aynı 12 belge ızgarasında,
"Belge No" alanı sağdan sola artan oranda gri bir dikdörtgenle ("parmak" benzetmesi)
kapatıldı (72 görüntü).

`experiments/occlusion/run_experiment.py`: her görüntüde alan OCR edildi,
`length_ratio`, `mean_confidence` ve birleşik `occlusion_suspicion_score` hesaplandı;
monotonluk (Spearman) analiz edildi.

### Deney sonucu

| Metrik | Ortalama Spearman rho (12 belge) | Coverage=0 ort. | Coverage=1.0 ort. |
|---|---|---|---|
| length_ratio | −0.9757 | 1.000 | 0.000 |
| mean_confidence | −0.8525 | 91.7 | 0.0 |
| occlusion_suspicion_score (birleşik) | **+0.9445** * | 3.8 (düşük şüphe) | 100.0 (tam şüphe) |

(*İşaret farkı kasıtlı ve doğru: `occlusion_suspicion_score` tanım gereği kapanma arttıkça
ARTAR — diğer metrikler "iyilik" ölçüp azalırken, bu metrik "kötülük/şüphe" ölçüyor. Bkz.
"Karşılaşılan problemler".)

Üç metrik de güçlü ve tutarlı monoton davranış gösterdi — bu modül, **Glare modülünün
aksine, baseline yöntemin beklendiği gibi çalıştığı** bir modül oldu. `length_ratio` en
temiz/düzgün eğriyi verdi (`results/occlusion/plots/metrics_vs_coverage.png`, sol panel);
`mean_confidence` daha gürültülü ama yine de güçlü bir sinyal verdi (özellikle %0→%20
kapanma arasında çok keskin bir düşüş var — OCR, kısmi kapanmaya karşı oldukça "kırılgan"
görünüyor). Birleşik `occlusion_suspicion_score`, ikisini ortalayarak daha yumuşak/dengeli
bir eğri üretti.

### Karşılaşılan problemler

- **Yön (convention) tutarsızlığı fark edildi:** `occlusion_suspicion_score`, projedeki
  diğer modüllerin (blur, glare, darkness) tersine, YÜKSEK değer = KÖTÜ kalite anlamına
  geliyor (bir "şüphe skoru" olduğu için doğal bu yönde). Bu, kod yazılırken fark edilip
  `src/occlusion/metrics.py` docstring'ine açıkça not düşüldü — ilk yazımda docstring
  yanlışlıkla ters yönü belirtiyordu, bu düzeltildi.
- `mean_confidence` metriği, %20-%60 kapanma aralığında beklenenden daha gürültülü/az
  monoton davrandı (bazı belgelerde ara seviyelerde confidence geçici olarak yükseliyor) —
  muhtemelen Tesseract'ın kısmi/bozuk rakamları bazen yanlış ama "kendinden emin" biçimde
  tanıması nedeniyle (örn. kapatılmış bir "8"i "3" olarak yüksek güvenle okuması).

### Aldığımız kararlar

1. **Bu yöntem yalnızca yapılandırılmış, formatı bilinen alanlar için uygulanabilir**
   olarak sınırlandırıldı (örn. "Belge No"); serbest metin (paragraflar) için
   uygulanmadı — bkz. `src/occlusion/README.md`, "Sınırlama". Serbest metin occlusion
   tespiti, ileride nesne tespiti/segmentasyon gerektiren ayrı bir alt problem olarak
   bırakıldı.
2. **Skor yönü (convention) tutarsızlığı, Aşama 5'e (ML skor füzyonu) not olarak
   bırakıldı** — o aşamada tüm alt-skorlar (blur, glare, darkness, skew, occlusion) TEK
   bir ortak yöne (örn. hepsi "yüksek = iyi kalite") normalize edilmeli.
3. **length_ratio ve mean_confidence'ın HER İKİSİNİN de tutulmasına karar verildi**
   (yalnızca birini seçmek yerine) — ikisi farklı hata modlarını yakalıyor (biri
   "karakter kayboldu mu", diğeri "OCR ne kadar emin") ve literatürün "yalnızca OCR
   confidence'a güvenme" uyarısıyla tutarlı.

### Bir sonraki adım

Beş modülün tamamı (Blur, Glare, Darkness, Skew, Occlusion) baseline seviyesinde
tamamlandı. Kullanıcı talimatına göre nihai rapor bu aşamada YAZILMAYACAK (yalnızca tüm
modüller ve `project_notes.md` bittiğinde, ayrı bir adımda istenecek). Bir sonraki mantıklı
adım — henüz başlanmadı — **Aşama 5: Feature Fusion / ML Skorlama** olacaktır; bu aşamada
ele alınması gereken, bu modüllerden çıkan açık noktalar:
- Tüm alt-skorların ortak bir yöne (yüksek=iyi) normalize edilmesi (bkz. Occlusion kararı 2).
- Glare modülünün mevcut haliyle kullanılamaz olması — ya düzeltilmeli ya da geçici
  olarak füzyon dışı bırakılmalı.
- Darkness modülünün blok-bazlı analizinin font boyutuyla etkileşimi.
- Skew modülünün Projection Profile bileşeninin az-metinli belgelerdeki başarısızlığı.
- Blur modülünün font boyutu duyarlılığı.
- Tüm bu modüllerin GERÇEK (sentetik olmayan) veri üzerinde yeniden doğrulanması.

---

## Aşama 5: Feature Fusion — Yöntem ve Mimari Kararı

`src/scoring/fusion.py` ile Aşama 5'in İLK sürümü yazıldı (basit doğrusal normalizasyon +
ortalama, `app.py` üzerinden web arayüzünden erişilebiliyor). Bu bölüm, her modül için
hangi yöntemde kalınacağına ve birleştirme (fusion) mimarisinin nasıl olgunlaştırılacağına
dair alınan kararları kaydeder.

### Modül bazlı yöntem kararları

| Modül | Karar | Gerekçe |
|---|---|---|
| Blur | Tenengrad birincil, Laplacian Variance çapraz kontrol — **değişiklik yok** | Tenengrad rho=−1.00, Laplacian rho≈−0.997; ikisi birlikte tutulursa aralarındaki fark (biri düşük biri değilse) gürültü/artefakt sinyali de verir. |
| Darkness | En karanlık blok ortalaması birincil, global ortalama yardımcı — **değişiklik yok** | Global/percentile lokal karanlığı kaçırıyor (deneyde P25/P50 tepkisiz kaldı); blok-bazlı analiz yakalıyor (rho≈−0.83). |
| Skew | Hough birincil, bulamazsa Projection Profile'a düşme — **değişiklik yok** | Hough MAE≈0.91° vs Projection MAE≈1.82°; literatürde de bu alana özel yeni bir DL yöntemi yok, klasik yöntemler hâlâ pratikte baskın. |
| **Glare** | **Değişmeli.** Mevcut HSV+CC yerine, literatürün önerdiği CNN tabanlı glare heatmap (Rodin & Orlov, 2019 — arXiv:1911.05189): belge bloklara ayrılır, her blok için luminance + binarize stroke histogramı çıkarılır, küçük bir CNN'e verilir. | HSV+CC, beyaz kağıt ile glare'i ayırt edemiyor (severity=0'da ~%85 hatalı-pozitif — bkz. `results/glare/false_positive_baseline.csv`). Bu, eşik ayarıyla düzelecek bir sorun değil, yöntemin kendisinin sınırlaması. |
| **Occlusion** | Yapılandırılmış alanlarda (OCR + beklenen uzunluk) **değişiklik yok** (rho≈−0.97, çok güçlü). Serbest/rastgele kapanma için **yeni bir bileşen eklenmeli**: el/parmak/nesne tespiti (object detection / segmentation, örn. YOLO ailesi). | Mevcut yöntem yalnızca konumu ÖNCEDEN BİLİNEN alanlarda çalışır; serbest metinde "beklenen uzunluk" tanımsız olduğu için hiç sinyal üretmiyor. |

**Özet:** Blur, Darkness, Skew modüllerinde mevcut klasik yöntemler hem kendi deney
sonuçlarımızla hem literatürle uyumlu — bu üçünde yöntem değişikliği planlanmıyor. Glare ve
Occlusion (serbest metin), literatürün de "belgeye özel en az araştırılmış" dediği iki alan;
ikisi de dar kapsamlı birer görüntü/nesne tanıma bileşeni (CNN / object detection)
gerektiriyor — uçtan uca kara kutu bir model değil, yalnızca o iki alt-problem için.

### Fusion mimarisi: doğrusal ortalamadan ML regresyona geçiş kararı

**Mevcut durum (v1, `fusion.py`):** Her alt-skor elle belirlenmiş bir aralıkla (`bad`/`good`
sabitleri) 0-100'e normalize edilip basit ortalaması alınıyor. Bu, docstring'de de açıkça
belirtildiği gibi GEÇİCİ bir yer tutucudur.

**Hedeflenen v2 mimarisi** (`research/literature_review.md`, Bölüm 5'teki "Hibrit" seçimiyle
uyumlu):

```
Katman 1 — Feature Extraction (mevcut src/* modülleri, glare/occlusion güncellemesiyle)
Katman 2 — Feature Vector (ham metriklerin birleştirilmesi + bağlamsal özellikler,
           örn. font boyutu / metin yoğunluğu — blur'un font duyarlılığını çözmek için)
Katman 3 — ML Regresyon (RF / XGBoost / SVR), gerçek etiketli veriyle eğitilir → 0-100 skor
Katman 4 — Açıklama (feature importance / SHAP; MLLM yalnızca doğal dile çevirme için,
           skorlamanın kendisi için DEĞİL — bkz. aşağıdaki Q-Doc bulgusu)
```

**Bunun basit doğrusal ortalamaya göre üç somut avantajı:**
1. **Kalibrasyon:** Şu anki eşikler gerçek veriyle öğrenilmedi (bkz. `fusion.py`
   docstring'i). Kiruthika, Athanesious & Kiruthika (2026) aynı yaklaşımı (12 özellik +
   XGBoost, OCR doğruluğu ground-truth) deneyip PCC=0.9139 almış — yöntemin işe yaradığına
   dair somut kanıt.
2. **Modüller arası etkileşim:** `literature_review.md`'nin kendi açık nokta listesinde var
   — "skew + blur birlikte" gibi durumlar hiçbir kaynakta ele alınmamış, doğrusal ortalama
   bunu yakalayamaz; ağaç-tabanlı bir model (RF/XGBoost) etkileşimi doğal öğrenir.
3. **Açıklanabilirlik korunur:** Regresyonun girdisi hâlâ bizim yorumlanabilir ham
   metriklerimiz; feature importance ile "skor düşük çünkü darkest_block_mean çok düşüktü"
   gibi somut açıklamalar üretilebilir — projenin en baştaki "kara kutu CNN'e alternatif"
   motivasyonu korunuyor.

**MLLM tabanlı skorlama neden seçilmedi:** `literature_review.md`, Bölüm 3'te incelenen
Q-Doc (arXiv:2511.11410, 2025) bulgusu: MLLM'ler DIQA'da temel yetenek gösteriyor ama
tutarsız skorlama ve bozulma tipini yanlış tanımlama sorunları var. Bu yüzden MLLM birincil
skorlayıcı olarak değil, ileride yalnızca açıklama/rapor metni üretmek için tamamlayıcı bir
bileşen olarak değerlendirilebilir.

**Pratik ilk adım (etiketli veri sorunu için):** Gerçek insan etiketlemesi maliyetli;
SmartDoc-QA ve Kiruthika et al.'ın kullandığı gibi **OCR doğruluğunu proxy ground-truth**
olarak kullanmak (Tesseract altyapısı Occlusion modülünde zaten hazır) hızlı bir başlangıç
noktası sağlar — manuel etiketlemeye gerek kalmadan modeli eğitmeye başlanabilir.

**Karar durumu:** Bu, bir mimari YÖN kararıdır; ML regresyon katmanının implementasyonu ve
gerçek veri toplama/etiketleme henüz YAPILMADI. Glare'in CNN tabanlı yeniden yazımı da henüz
başlanmadı — bu notta yalnızca hangi yöne gidileceği kayıt altına alındı.

### Kalibrasyon düzeltmesi (v1 → v1.1)

v1'in canlıya alınmasından kısa süre sonra, `fusion.py`'nin ürettiği skorlar `results/`
altındaki gerçek deney verisiyle karşılaştırılarak denetlendi. Üç somut hata bulundu ve
düzeltildi:

1. **Blur skoru çok erken satüre oluyordu.** `BLUR_GOOD=300` sabiti veriye bakılmadan
   tahmin edilmişti; oysa `results/blur/scores.csv`'de severity=2 (belirgin bulanık)
   belgelerin ortalama Laplacian Variance'ı bile 521 — yani "belirgin bulanık" bir belge
   dahi 100/100 alıyordu. **Kök neden:** Laplacian Variance, blur şiddeti arttıkça
   DOĞRUSAL değil ÜSTEL azalıyor (severity 0→8 arası 8287→1.3, birkaç büyüklük
   mertebesi). Çözüm: doğrusal normalizasyon yerine log1p-ölçekli normalizasyon
   (`_log_linear_score`) — artık severity 0→8 arası skor 100→2 şeklinde düzgün,
   kademeli düşüyor.
2. **Darkness skoru temiz belgeleri haksız yere düşük gösterebiliyordu.**
   `DARKNESS_GOOD=180` tahminiydi; gerçekte HİÇ karartma uygulanmamış (severity=0)
   belgelerin darkest_block_mean'i ortalama yalnızca 101.6 (metin yoğunluğu/mürekkep
   piksellerinden kaynaklanan doğal bir alt sınır). Eşikler gerçek severity=0..5
   dağılımına göre (50/110) yeniden kalibre edildi.
3. **(En ince olanı) `fusion.py`, deneyde doğrulanmış `block_size=16` yerine
   fonksiyonun varsayılanı `block_size=32`'yi kullanıyordu.** `experiments/darkness/
   run_experiment.py`, küçük (~105x26px) kimlik alanlarını yakalamak için bilinçli
   olarak 16 kullanmıştı (bkz. Darkness modül notları); `fusion.py` bunu miras almamıştı.
   Sonuç: küçük, lokalize karanlık bölgeler komşu aydınlık piksellerle "sulanıp"
   gizleniyordu. Somut örnek: `ornek_gorseller/4_karanlik.png` (yoğun lokal karartma
   içeren bir test görüntüsü) düzeltmeden önce genel skor **97** (neredeyse mükemmel)
   alıyordu; düzeltmeden sonra **64**'e düştü ve darkness alt-skoru artık deneydeki
   (`results/darkness/scores_local.csv`) ham değerle (51.08) birebir eşleşiyor.

**Çıkarım:** Bu üç hata da "eşiği elle tahmin etmenin" somut riskini gösteriyor — tam da
`fusion.py` docstring'inin baştan beri uyardığı nokta. Kalıcı çözüm hâlâ yukarıdaki ML
regresyon planı; bu düzeltme yalnızca v1'in kendi içinde daha az yanıltıcı olmasını
sağlıyor, "gerçek kalibrasyon" iddiası taşımıyor.

### Glare Aşama 1 denemesi — DENENDİ, ÇALIŞMADI (negatif sonuç, kayıt altında)

Yukarıdaki "hangi yöntemi kullanmalıyız" tartışmasında önerilen "Aşama 1: şekil filtresi"
(gerçek parlamanın yuvarlak/kompakt, satır-arası boşlukların ince/uzun şerit olduğu
varsayımıyla `connected components` çıktısına compactness + kenara-değme filtresi ekleme)
uygulanıp gerçek sentetik veriyle test edildi.

**Sonuç:** `results/glare/degraded/manifest.csv` üzerinde severity=0 (glare yok) hatalı-
pozitif oranını %84.87'den %0.97'ye düşürdü — ilk bakışta büyük başarı gibi göründü. Ama
severity=5 (en güçlü glare, hedef alan %22) üzerinde test edilince gerçek pozitifin de
neredeyse tamamen silindiği görüldü: eski maske content_bbox içinde 193.998 piksel
işaretliyordu (hedefin ~4 katı, aşırı işaretleme), filtreli maske yalnızca 777 piksel
işaretledi (hedefin ~%1.6'sı — aşırı silme).

**Kök neden:** Gerçek glare lekesi ile arka plan boşluğu, bu eşiklerde (V≥235, S≤35) AYNI
bağlı bileşene (connected component) düşüyor — çünkü content_bbox içindeki beyaz alan
zaten çoğunlukla bu eşiği geçiyor, glare lekesi bu "denizin" içinde ayrı bir ada değil,
onunla kaynaşmış durumda. Şekil filtresi bu TEK dev bileşene bakıyor, onu ya tamamen kabul
ediyor ya tamamen reddediyor — ikisini ayıramıyor. Morfolojik açma (opening, kernel 3-15
arası denendi) da neredeyse hiçbir şeyi değiştirmedi çünkü sorun "ince şeritler" değil,
sayfanın büyük kısmının zaten HSV eşiğinde beyaz görünmesi.

**Bu, `src/glare/README.md`'nin baştan beri belgelediği sınırlamayı somut veriyle
doğruluyor:** gerçek glare ile beyaz kağıt, bu renk uzayında hiçbir ayırt edici ÖZELLİK
(ne renk ne şekil) taşımıyor. Klasik post-processing (şekil, morfoloji) ile çözülebilecek
bir problem değil — literatürün (Rodin & Orlov, 2019) neden doğrudan öğrenilmiş/CNN tabanlı
bir yaklaşıma gittiği bu denemeyle bir kez daha teyit edildi.

**Karar:** Bu denemenin kodu geri alındı (repo'da yarım/yanıltıcı bir "düzeltme" olarak
kalmasın diye) — `src/glare/metrics.py` denemeden önceki haline döndürüldü. Glare için
gerçekçi bir sonraki adım doğrudan Aşama 3'e (CNN tabanlı blok sınıflandırma) geçmek; ara
bir "ucuz" çözüm bulunamadı.

### Occlusion Aşama 1 — Ten Rengi Tespiti: DENENDİ, ÇALIŞTI (pozitif sonuç)

Glare'in aksine, occlusion için önerilen "Aşama 1: ten rengi (skin-color) tespiti" gerçek
sentetik veriyle test edildi ve doğrulandı.

**Uygulama:** `src/occlusion/skin_detection.py` — YCrCb renk uzayında ten rengi eşiklemesi
+ bağlı bileşen gürültü temizliği. Mevcut OCR tabanlı occlusion yönteminden (`metrics.py`)
FARKI: konumu önceden bilmeye ihtiyaç duymaz, belgenin herhangi bir yerinde çalışır.

**Doğrulama deneyi:** `experiments/occlusion/generate_skin_occlusion_documents.py` (12
belge × 3 ten tonu × 6 kapanma seviyesi = 216 görüntü, yama konumu KASITLI RASTGELE) +
`run_skin_experiment.py`. Sonuç:

| Ten tonu | Spearman rho | Hatalı-pozitif (coverage=0) | Dynamic range |
|---|---|---|---|
| Açık | 1.0000 | 0.0000 | 0.00 → 1.00 |
| Orta | 1.0000 | 0.0000 | 0.00 → 1.00 |
| Koyu | 1.0000 | 0.0000 | 0.00 → 1.00 |

Üç ton için de mükemmel monoton, sıfır hatalı-pozitif — bkz. `results/occlusion/
skin_scores.csv`, `skin_monotonicity_by_tone.csv`, `plots/skin_detection_by_tone.png`.

**Neden glare'den farklı sonuç verdi?** Glare'de sorun, gerçek sinyal (parlama) ile
hatalı-pozitif kaynağının (beyaz kağıt) AYNI HSV bölgesinde, birbirinden AYRILAMAZ şekilde
iç içe geçmiş olmasıydı. Burada durum farklı: belge arka planı (beyaz/gri, düşük
saturasyon) ile ten rengi (YCrCb'de belirgin, dar bir Cr/Cb aralığı) renk uzayında GERÇEKTEN
AYRIK — üst üste binmiyor. Yani aynı "renk eşikleme" tekniği, ayırt edici bir özellik
gerçekten var olduğunda işe yarıyor; yoktuğunda (glare) yaramıyor. Bu iki deney birlikte,
"yöntem seçiminin veriye bakmadan güvenilemeyeceği" dersini iki yönde de (başarı VE
başarısızlık) somutlaştırıyor.

**Entegrasyon:** `src/scoring/fusion.py`'ye `score_occlusion_skin` olarak eklendi ve
varsayılan olarak nihai skora DAHİL edildi (glare'in aksine — çünkü bu, glare gibi
kanıtlanmış şekilde bozuk değil, tersine kanıtlanmış şekilde çalışıyor). Web arayüzüne
("Kapanma (el/parmak)" modülü) de eklendi.

**Kalan sınırlama (dürüstlük notu):** Doğrulama yalnızca düz renkli sentetik yamalarla
yapıldı — gerçek el/parmak dokusu, gölge, farklı aydınlatma ve ten-rengi-benzeri arka plan
nesneleri (ahşap masa vb.) HENÜZ test edilmedi. Bu, diğer tüm modüllerin de paylaştığı
"yalnızca sentetik veride doğrulandı" sınırlamasıyla aynı kategoride — gerçek fotoğraflarla
yeniden doğrulama hâlâ genel açık nokta listesinde.

### Occlusion Aşama 2 — Klasik ML (Random Forest) ile renkten bağımsız genelleme

Kullanıcı sorusu: "occlusion farklı renk ve dokularda da çalışmalı, glare hâlâ sıkıntılı —
ML/DL uygulanırsa ikisi de gelişir mi?" Ortamda `sklearn`/`torch`/`tensorflow` kurulu
değildi; karar: önce hafif `scikit-learn` ile klasik ML denenip, işe yaramazsa ağır bir
derin öğrenme yatırımına (PyTorch) geçilecek.

**Occlusion için sonuç: DENENDİ, ÇALIŞTI, ENTEGRE EDİLDİ.**

Aşama 1'in (ten rengi/YCrCb) sınırlaması: yalnızca SABİT bir renk aralığı arıyordu —
sticker, kumaş, plastik gibi farklı renk/dokudaki kapanmaları tanım gereği kaçırırdı.

**Uygulama:** `src/occlusion/ml_detection.py` — görüntü 16x16 bloklara ayrılır, her blok
için renk (kanal ortalaması) + doku (kanal std'si, gri std, Laplacian varyansı) özellikleri
çıkarılır; bir Random Forest (200 ağaç) her bloğu "normal kağıt/metin yüzeyi" veya "yabancı
nesne" olarak sınıflandırır.

**Eğitim** (`experiments/occlusion/train_occlusion_classifier.py`): 14 renk (kırmızı, yeşil,
mavi, sarı, mor, turuncu, gri tonları, siyah, pembe, kahve, bordo, zeytin, krem — TEN RENGİ
KASITLI OLARAK DIŞARIDA) x 12 belge x 5 kapsama seviyesi x 2 doku varyantı (düz/gürültülü)
= 1680 görüntü, ~3.5 milyon etiketli blok. Model: `src/occlusion/models/occlusion_rf.joblib`.

**Doğrulama** (`experiments/occlusion/run_ml_experiment.py`) — asıl kritik test: modelin
EĞİTİMDE HİÇ GÖRMEDİĞİ 5 renkte (açık/orta/koyu ten tonu + turkuaz + lacivert), hem düz hem
dokulu varyantlarda, farklı belgelerde (farklı random seed) test edildi:

| Görülmemiş renk | Düz — rho | Dokulu — rho |
|---|---|---|
| Açık ten | 1.0000 | 1.0000 |
| Orta ten | 1.0000 | 1.0000 |
| Koyu ten | 1.0000 | 1.0000 |
| Turkuaz | 1.0000 | 1.0000 |
| Lacivert | 1.0000 | 1.0000 |

10 kombinasyonun HEPSİNDE mükemmel monotonluk, hatalı-pozitif (coverage=0) = 0.0000 (bkz.
`results/occlusion/ml_scores.csv`, `ml_monotonicity.csv`, `plots/ml_generalization.png`).
Model gerçekten renk+doku örüntüsünü öğrendi, belirli renkleri ezberlemedi.

**Entegrasyon:** `src/scoring/fusion.py`'de `score_occlusion_skin` → `score_occlusion`
olarak değiştirildi, `ml_occlusion_ratio` kullanıyor (eskiden `skin_occlusion_ratio`).
`skin_detection.py` kod tabanında bırakıldı (daha basit/hızlı bir alternatif olarak,
`compare_module_methods` üzerinden karşılaştırmalı gösteriliyor) ama artık füzyonda
KULLANILMIYOR. Web arayüzündeki modül adı "Kapanma (el/parmak)"ten "Kapanma"ya
sadeleştirildi (artık yalnızca el/parmak değil, herhangi bir yabancı nesneyi hedefliyor).

**Kalan sınırlama (değişmeyen dürüstlük notu):** Doğrulama yine yalnızca sentetik, PIL ile
çizilmiş düz/gürültülü dikdörtgen yamalarla yapıldı. Gerçek el/parmak fotoğrafında doku çok
daha karmaşıktır (deri kıvrımları, tırnak, gölge, kısmi saydamlık yok ama gerçek 3D şekil
var) — bu still test edilmedi. Model + deney kodu tamamen tekrar üretilebilir olduğu için,
gerçek etiketli veri geldiğinde aynı `train_occlusion_classifier.py` iskeleti üzerine
kolayca yeniden eğitilebilir.

### Glare için ML denemesi — SONRAKI ADIM (bu oturumda başlanmadı)

Occlusion'ın aksine glare'de daha önce iki klasik CV denemesi (HSV+CC baseline VE şekil
filtresi) başarısız olmuştu — kök sebep, gerçek parlama ile beyaz kağıdın renkte HİÇBİR
ayırt edici özellik taşımaması, aynı bağlı bileşende birleşmesiydi. Occlusion'daki başarı,
"renk+doku özellikleriyle klasik ML" tarifinin PRENSİPTE işe yarayabileceğini gösteriyor,
ama glare'de aynı tarifin işe yarayıp yaramayacağı HENÜZ TEST EDİLMEDİ — bu, sırada bekleyen
bir sonraki deneme.

### Glare ML denemesi (Random Forest, blok-bazlı) — KISMİ SONUÇ: temel sorunu çözemedi

Occlusion'daki tarifle (16x16 blok, renk+doku özellikleri, Random Forest) aynı yöntem
glare'e uygulandı. Farkı: glare'in sentetik üretiminde blob merkezi/sigma tam
deterministik olduğu için (bkz. `experiments/glare/generate_glare_documents.py`,
`sigma_for_target_area`), her görüntü için GERÇEK alpha (glare şiddeti) haritası yeniden
hesaplanıp doğru etiketli eğitim verisi çıkarıldı — occlusion'daki gibi elle bir yama
çizmeye gerek kalmadı, sentetik üretimin kendi zemin gerçeği kullanıldı.

**Eğitim:** 8 belge, tüm severity seviyeleri, blok özellikleri = [ortalama parlaklık, std,
Laplacian varyansı] (glare görüntüleri gri tonlamalı olduğu için occlusion'daki BGR renk
özellikleri anlamsız, yalnızca doku/parlaklık kullanıldı). Etiket: bloğun merkezindeki
gerçek alpha değeri > 0.5 ise "glare".

**Sonuç — 4 görülmemiş belgede:**

| Severity | 0 (glare yok) | 1 | 2 | 3 | 4 | 5 (en ağır) |
|---|---|---|---|---|---|---|
| Model "glare" oranı | **~0.36-0.38** | 0.044 | 0.077 | 0.128 | 0.187 | 0.276 |

- **Severity 1→5 arası: rho = 1.0000 (mükemmel)** — model, parlama şiddetini gayet iyi
  takip ediyor, "zaten bir miktar parlama var" varsayımı altında.
- **Severity 0 (hiç parlama yok): hatalı-pozitif ~%36-38 — severity=5'in gerçek pozitif
  oranından (%27.6) bile YÜKSEK.** Bu, üretime alınamayacak kadar ciddi bir kusur —
  tertemiz bir belge, en ağır parlamalı belgeden daha "şüpheli" görünüyor.

**Kök neden (blok özellik ortalamalarına bakılarak doğrulandı):**

| | Ortalama parlaklık | Std (doku) | Laplacian varyansı |
|---|---|---|---|
| Gerçek glare blokları | 245.1 | 19.4 | 2357 |
| Glare olmayan bloklar | 231.0 | 47.5 | 14726 |

Model "düz ve parlak = glare" örüntüsünü öğrendi — ama bu tanım, belgenin DOĞAL beyaz
kenar boşluklarına da birebir uyuyor (onlar da düz ve parlak). Tam doymuş bir glare
bölgesi (alpha≈1) ile boş beyaz kağıt, piksel istatistiği açısından GERÇEKTEN özdeş —
occlusion'da renk+doku özellik mühendisliği işe yaradı çünkü occluder ile arka plan
FARKLI şeylerdi; glare'de "glare" ve "arka plan" aynı fiziksel görünüme (düz beyaz)
yakınsıyor. Bu, bir mühendislik eksikliği değil, TEK KARE üzerinden bilgi kuramsal bir
sınır — hiçbir özellik mühendisliği (klasik ya da ML) bunu çözemez.

**Değerli çıkarım:** Bu, Rodin & Orlov (2019)'un neden yalnızca parlaklık/doku değil,
**"stroke histogram"** (o bölgede normalde metin OLMASI beklenip beklenmediği — yani
belge DÜZENİ bağlamı) da kullandığını somut biçimde açıklıyor. Bizim blok
sınıflandırıcımız yalnızca kendi bloğuna bakıyor, "burada normalde metin olur muydu"
sorusunun cevabını bilmiyor — bu, TEK bir bloğa izole bakan hiçbir yöntemin (klasik ya da
ML) çözemeyeceği, belge düzeyinde bağlam (whole-image context) gerektiren bir soru.

**Karar:** Bu model üretime ALINMADI (severity=0 hatalı-pozitifi kabul edilemez düzeyde).
Glare için gerçekçi sıradaki adım hâlâ Aşama 3 (tam CNN, stroke-histogram/layout bağlamı
dahil) — ama artık NEDEN yalnızca blok-bazlı/context'siz yöntemlerin (ne klasik ne ML)
yeterli olmadığı üç ayrı denemeyle (HSV+CC, şekil filtresi, blok-bazlı RF) somut biçimde
kanıtlanmış durumda.

### Glare ML v2-v5 (bağlam-farkında versiyonlar) — dört ek deneme, hepsi yetersiz

Kullanıcı, occlusion'daki ML başarısından sonra glare için de ML denenmesini istedi.
Satır-bağlamlı bir Random Forest (`src/glare/ml_detection.py`) geliştirildi: her bloğun
kendi özelliklerine ek olarak AYNI SATIRDAKİ diğer bloklara göre ne kadar "beklenmedik
şekilde düz" olduğu da özellik olarak kullanıldı. İlk doğrulamada (rho=0.99) büyük başarı
gibi göründü — ama bu doğrulama **YANLIŞ** bir `roi=content_bbox` varsayımıyla yapılmıştı;
üretim kodu (`fusion.py`) gerçek bir fotoğrafta content_bbox bilmediği için `roi=None`
kullanıyor. Bu yanlışlık düzeltilip dört varyant `roi=None` (gerçekçi) koşulda test edildi:

| Versiyon | Değişiklik | Görülmemiş belge rho | Bulanıklıkta en kötü hatalı-pozitif |
|---|---|---|---|
| v1 | Satır bağlamı (temel) | 0.66 | %92 |
| v2 | + bulanık negatif eğitim örnekleri | 0.69 | %20 (ama tespit performansı da düştü) |
| v3 | + belge-geneli doku özelliği | ~0 (bozuldu) | %32 (tespit tamamen öldü) |
| v4 | v1 + eğitim/üretim roi uyumu | 0.76 | %88 |
| v5 | v4 + fiziksel makul üst sınır (cap=0.25) | 0.76 | %25 (ama gerçek şiddetli glare bile "orta" bantta kalıyor, tutarsız) |

**Hiçbiri üretime alınmadı.** Her düzeltme bir sorunu iyileştirirken başka birini kötüleştirdi
— klasik "whack-a-mole" paterni. v1 kodu (en basit, tarihsel referans olarak) korundu;
v2-v5'in kodu geri alındı.

### 🔎 Dış araştırma: modern (2024-2025) glare/specular highlight literatürü

Kullanıcı "daha güncel yöntemleri araştır" dedi. Bulgular:
- ICDAR 2024 "Document Specular Highlight Removal with Coarse-to-Fine Strategy",
  SHDocs (NeurIPS 2024), HighlightRemover (ACM MM 2024), Dual-Hybrid Attention Network
  (ACM MM 2024), TSHRNet, UnReflectAnything (2025) — HEPSİ derin öğrenme modelleri,
  EŞLEŞTİRİLMİŞ gerçek fotoğraf veri setleri (glare'li/glare'siz aynı sayfa) gerektiriyor.
  Bu projede ne böyle bir veri seti ne GPU eğitim altyapısı var — kapsam dışı.
- Klasik (eğitimsiz) yöntemlerin çoğu **dichromatic reflection model**e dayanıyor:
  parlamanın rengi (ışık kaynağının rengi, genelde beyaza yakın) yüzeyin KENDİ renginden
  ayrışır. Bu, meyve/seramik/cilt gibi RENKLİ/parlak yüzeylerde işe yarar. **Düz beyaz
  kağıtta işe yaramaz çünkü yüzeyin "gerçek rengi" zaten beyaz — ayrışacak bir fark yok.**
  Bu, altı ayrı denemenin (HSV+CC, şekil filtresi, 4 ML varyantı) neden hepsinin aynı
  duvara çarptığını bağımsız olarak, literatürden doğruluyor.

### Glare — Kimlik Kartı/Pasaport Zemini Deneyi: BAŞARILI (kullanıcı önerisi)

Kullanıcının önerisi: "hep beyaz kağıtta denedik, kimlik/pasaport gibi başka veri de
deneyelim." Dichromatic model prensibine göre bu RENKLİ zeminlerde işe yaramalıydı —
test edildi ve **doğrulandı**.

**Deney:** `experiments/glare/generate_id_card_documents.py` — 3 farklı kart renk şeması
(mavi-gri, bej, yeşilimsi) × 4 varyant × 6 glare şiddeti = 72 görüntü, ID-1 kart oranında
(856×540), fotoğraf placeholder + metin satırı blokları ile. `run_id_card_experiment.py`
— **hiçbir değişiklik yapılmadan**, zaten var olan `src/glare/metrics.py`'deki `glare_ratio`
(HSV+CC) fonksiyonu test edildi.

**Sonuç:**
| Metrik | Değer |
|---|---|
| Ortalama rho (3 renk şemasında) | **1.0000** (std=0.0) |
| Severity=0 hatalı-pozitif | **0.0000** (max de 0.0000) |
| Bulanıklık+glare-yok hatalı-pozitif (3 renk × 6 blur seviyesi) | **0.0000** (hepsinde) |

Beyaz kağıtta çözülemeyen İKİ sorun da (temel hatalı-pozitif VE bulanıklık karışması)
kimlik kartı zemininde TAMAMEN kayboldu — hiç ML gerekmeden. Bkz.
`results/glare/id_card_scores.csv`, `id_card_blur_false_positive_check.csv`,
`plots/id_card_generalization.png`.

**Entegrasyon:** `src/glare/metrics.py`'ye `has_colored_background()` eklendi — bir
görüntünün zemininin renkli mi (S>15 piksellerin oranı ≥%15) yoksa düz beyaz/gri mi
olduğunu tahmin eder (sentetik testte beyaz kağıt %0, kimlik kartı %100 çıktı — temiz bir
ayrım). `src/scoring/fusion.py`'deki `score_glare` artık KOŞULLU: zemin renkliyse
`glare_ratio` (klasik, güvenilir) kullanılır ve nihai ortalamaya dahil edilir; düz beyaz
kağıtsa `glare_ml_ratio` (v1, bilgi amaçlı) hesaplanır ama `reliable=False` ile ortalama
DIŞINDA tutulur. Diğer hiçbir modüle (blur/darkness/skew/occlusion) dokunulmadı.

**Bilinen sınırlama:** Blur/darkness/skew'in normalizasyon eşikleri, düz metin belgeleri
için kalibre edildi (bkz. "Kalibrasyon düzeltmesi" bölümü) — kimlik kartı gibi farklı bir
belge türünde (solid renk blokları, farklı doku) bu eşikler henüz ayrıca doğrulanmadı;
yalnızca glare'in kendisi bu deneyde test edildi.

### Kapsam kararı: Glare artık YALNIZCA renkli zeminli belgeler için

Kullanıcı: "bu deneme özellikle kimlikler üzerindeki glare oranını hesaplayacak, bu
sebeple beyaz kağıt fikrini bırakalım." Bu, projenin glare hedefini netleştirdi — beyaz
kağıt "henüz çözülemedi" değil, artık **kapsam dışı** (bilinçli bir tasarım kararı).

**Değişiklik:** `src/scoring/fusion.py`'deki `score_glare`, artık beyaz kağıtta
`glare_ml_ratio` (v1, güvenilmez tahmin) hesaplamıyor — `has_colored_background()` False
dönerse doğrudan `{"score": None, "applicable": False}` döndürüyor. Web arayüzü bu durumda
net bir "Bu belge renkli zeminli değil — glare tespiti bu belge türü için tasarlanmadı,
uygulanamaz" mesajı gösteriyor (eskiden güvenilmez bir sayı gösterip "dahil edilmedi"
diyordu — artık hiç sayı üretmiyor). `compare_module_methods`'taki glare bölümü de
sadeleştirildi, artık yalnızca `glare_ratio`'yu (renkli zeminde kullanılan yöntem) gösteriyor.

Bu, `glare_ml_ratio`/`ml_detection.py`'nin (v1-v5 denemeleri) kod olarak SİLİNMESİ değil —
hâlâ `src/glare/ml_detection.py`'de duruyor, tarihsel/gelecekte referans için — sadece
ana skorlama akışından çıkarıldı.
