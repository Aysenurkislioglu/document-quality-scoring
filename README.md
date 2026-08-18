# Document Quality Scoring

**Belge görüntülerindeki kalite bozulmalarını (blur, glare, darkness, skew, occlusion)
ayrı ayrı ölçen ve açıklanabilir, birleşik bir kalite skoru (0-100) üreten bir sistem.**

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Status](https://img.shields.io/badge/status-active--development-yellow)
![Stage](https://img.shields.io/badge/stage-5%20%2F%206-orange)

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Motivasyon](#motivasyon)
- [Mimari](#mimari)
- [Modüller](#modüller)
- [Başlarken](#başlarken)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Metodoloji ve Sonuçlar](#metodoloji-ve-sonuçlar)
- [Durum ve Yol Haritası](#durum-ve-yol-haritası)
- [Bilinen Sınırlamalar](#bilinen-sınırlamalar)
- [Dokümantasyon](#dokümantasyon)
- [Lisans](#lisans)

---

## Genel Bakış

Bu proje, bir belge fotoğrafının **neden düşük kaliteli olduğunu** açıklayabilen bir
kalite skorlama sistemi geliştirir. Her kalite problemi (bulanıklık, parlama, karanlık,
eğiklik, kapanma) klasik görüntü işleme yöntemleriyle **ayrı ayrı, sayısal olarak**
ölçülür; ardından bu alt-skorlar tek bir **Document Quality Score (0-100)** altında
birleştirilir.

Bir web arayüzü üzerinden ([Kullanım](#kullanım) bölümüne bakınız) herhangi bir belge
fotoğrafı yüklenip anında skorlanabilir.

## Motivasyon

Kimlik/belge doğrulama sistemlerinde kalite kontrolü genellikle uçtan uca eğitilmiş,
"kara kutu" bir CNN sınıflandırıcıyla yapılır. Bu proje bilinçli olarak farklı bir yol
izler: her bozulma türü için ayrı, **eğitim gerektirmeyen (training-free)** ve
**gerekçelendirilebilir** yöntemler kullanır. Bunun pratik gerekçeleri:

- **Açıklanabilirlik** — "belge neden reddedildi?" sorusuna "blur skoru 22/100, eşik
  50" gibi somut bir cevap verilebilir; bir CNN'in iç karar mekanizması bu netlikte
  değildir.
- **Etiketli veri gerektirmemesi** — mevcut modüllerin hiçbiri eğitim verisi
  istemez; yalnızca son birleştirme (fusion) aşaması ileride etiketli veriyle kalibre
  edilebilir.
- **Modülerlik** — her bozulma türü bağımsız geliştirilip test edilebilir; biri
  yetersiz kalırsa (bkz. Glare) sistemin geri kalanını bloklamaz.

## Mimari

```
                         ┌─────────────────────┐
                         │  Belge fotoğrafı      │
                         └──────────┬───────────┘
                                    │
        ┌────────────┬─────────────┼─────────────┬──────────────┐
        ▼             ▼             ▼             ▼              ▼
   ┌─────────┐  ┌───────────┐ ┌─────────┐  ┌───────────┐  ┌─────────────┐
   │  Blur   │  │ Darkness  │ │  Skew   │  │Occlusion  │  │   Glare     │
   │ (keskin-│  │(aydınlat- │ │(eğiklik │  │(ML: renk+ │  │ (koşullu:   │
   │  lik)   │  │  ma)      │ │ açısı)  │  │doku, konum│  │zemin renkli?)│
   └────┬────┘  └─────┬─────┘ └────┬────┘  └─────┬─────┘  └──────┬──────┘
        │             │            │             │               │
        └─────────────┴────────────┴─────────────┴───────────────┘
                                    │                    (Glare yalnızca zemin
                                    ▼                     RENKLİYSE dahil edilir
                  ┌───────────────────┐                   — bkz. Sınırlamalar)
                  │  Feature Fusion    │
                  │ (src/scoring/      │
                  │   fusion.py)       │
                  └─────────┬──────────┘
                            ▼
                ┌────────────────────────┐
                │ Document Quality Score  │
                │        (0–100)          │
                └────────────────────────┘
```

*(Not: Occlusion'ın OCR/alan-bazlı bileşeni — "Belge No" gibi bilinen şablon alanları için
— bu genel diyagrama dahil değildir; yalnızca şablonu bilinen belgelerde ayrıca çalıştırılır.)*

Her modül önce **bağımsız olarak** geliştirilip sentetik, kontrollü bozulma verisiyle
doğrulanır (`experiments/`); doğrulanan modüller ancak sonra `src/scoring/fusion.py`
üzerinden birleştirilir. Bu sıralama bilinçlidir — bkz. [Metodoloji](#metodoloji-ve-sonuçlar).

## Modüller

| Modül | Yöntem(ler) | Durum |
|---|---|---|
| **Blur** | Laplacian Variance, Tenengrad, Gradient Magnitude | ✅ Doğrulandı |
| **Darkness** | Global istatistik, percentile analizi, blok-bazlı yerel analiz | ✅ Doğrulandı |
| **Skew** | Hough Transform, Projection Profile | ✅ Doğrulandı |
| **Glare** | HSV eşikleme + Connected Components | ⚠️ Beyaz kağıtta yetersiz, ✅ renkli zeminde (kimlik kartı) doğrulandı |
| **Occlusion** | OCR + beklenen alan deseni (şablon gerekir) **+** ML (Random Forest) tespiti (konum/renk/dokudan bağımsız) | ✅ Doğrulandı |
| **Feature Fusion** | Ağırlıklı/doğrusal normalizasyon birleşimi | 🚧 İlk sürüm tamamlandı |

Her modülün yöntem seçimi, matematiksel gerekçesi ve bilinen sınırlamaları için ilgili
`src/<modül>/README.md` dosyasına bakınız.

## Başlarken

### Gereksinimler

- Python 3.11 veya üzeri (3.13 üzerinde de test edilmiştir)
- Occlusion modülü için sistemde Tesseract OCR kurulu olmalıdır:

  ```bash
  # macOS
  brew install tesseract tesseract-lang

  # Debian/Ubuntu
  apt-get install tesseract-ocr tesseract-ocr-tur
  ```

### Kurulum

```bash
git clone https://github.com/Aysenurkislioglu/document-quality-scoring.git
cd document-quality-scoring

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Kullanım

### Web arayüzü

En hızlı yol — bir belge fotoğrafı yükleyip anlık skor almak:

```bash
python3 app.py
# tarayıcıda http://127.0.0.1:5000 aç
```

Arayüz, `src/scoring/fusion.py` üzerinden blur/darkness/skew/occlusion(ML)
alt-skorlarını birleştirip tek bir 0-100 skor üretir; her modül kartının altında, o
modülün **birden fazla yöntemi** varsa (örn. blur için Laplacian Variance vs. Tenengrad)
bunların aynı görüntü üzerindeki karşılaştırması da gösterilir.

> **Not:** Bu skor, etiketli gerçek veriyle kalibre edilmiş bir ML modelinin çıktısı
> değildir; mevcut modüllerin ham metriklerinden türetilen geçici/sezgisel bir özettir.
> Glare tespiti bu proje kapsamında KİMLİK KARTI/PASAPORT benzeri RENKLİ zeminli belgeler
> için tasarlandı; düz beyaz kağıt kapsam dışıdır (altı ayrı deneme sonrası bilinçli bir
> kapsam kararı) — bu durumda glare "uygulanamaz" olarak işaretlenir. Occlusion'ın OCR tabanlı, alan-bazlı
> bileşeni önceden bilinen şablon gerektirdiğinden bu genel akışa dahil değildir — ama
> konum/renk/dokudan bağımsız ML tespiti dahildir. Detay için
> [Bilinen Sınırlamalar](#bilinen-sınırlamalar).

### Deneyleri çalıştırma

Her modül aynı üç adımlı desene sahiptir: (1) sentetik belge üret, (2) bilinen şiddette
kontrollü bozulma uygula, (3) deneyi çalıştırıp sonuçları `results/<modül>/` altına yaz.

```bash
# Blur
python3 experiments/blur/generate_synthetic_documents.py
python3 experiments/blur/apply_degradation.py
python3 experiments/blur/run_experiment.py

# Glare
python3 experiments/glare/generate_glare_documents.py
python3 experiments/glare/run_experiment.py

# Darkness
python3 experiments/darkness/generate_darkness_documents.py
python3 experiments/darkness/run_experiment.py

# Skew
python3 experiments/skew/generate_skew_documents.py
python3 experiments/skew/run_experiment.py

# Occlusion
python3 experiments/occlusion/generate_occlusion_documents.py
python3 experiments/occlusion/run_experiment.py
```

### Kütüphane olarak kullanma

```python
import cv2
from src.scoring.fusion import compute_document_quality_score

image = cv2.imread("belge.jpg")
result = compute_document_quality_score(image)

print(result["overall_score"])       # 0-100
print(result["components"]["blur"])  # {'raw_value': ..., 'score': ...}
```

## Proje Yapısı

```
document-quality-scoring/
│
├── research/               Literatür taraması ve kaynaklar
│   ├── initial_research.md
│   ├── literature_review.md
│   └── references.md
│
├── data/
│   ├── raw/                 Gerçek, işlenmemiş belge görüntüleri
│   ├── processed/           İşlenmiş/temizlenmiş veri
│   └── synthetic/           Sentetik belge + kontrollü bozulma verisi
│
├── src/                     Yeniden kullanılabilir üretim kodu
│   ├── blur/
│   ├── glare/
│   ├── darkness/
│   ├── skew/
│   ├── occlusion/
│   └── scoring/              Feature fusion (src/scoring/fusion.py)
│
├── experiments/             Çalıştırılabilir doğrulama deneyleri
│   ├── _common/               Paylaşılan sentetik belge üretici
│   ├── blur/ glare/ darkness/ skew/ occlusion/
│
├── results/                 Deney çıktıları (CSV + grafikler)
│
├── app.py                   Flask web arayüzü
├── templates/index.html
│
├── reports/                  Nihai rapor (tüm modüller bittiğinde yazılacak)
├── project_notes.md          Modül bazlı çalışma günlüğü (karar gerekçeleri)
├── requirements.txt
└── README.md
```

## Metodoloji ve Sonuçlar

Her modül, **bilinen şiddette** kontrollü bozulma uygulanmış sentetik belgeler üzerinde
doğrulanır; ölçülen metriğin bu bilinen şiddetle ne kadar tutarlı (monoton) değiştiği
Spearman korelasyonu (rho) ile raporlanır. Öne çıkan bulgular:

| Modül | Metrik | Sonuç | Kaynak |
|---|---|---|---|
| Blur | Tenengrad | rho = −1.00 (mükemmel monoton) | `results/blur/scores.csv` |
| Blur | Laplacian Variance | rho ≈ −0.997 | `results/blur/scores.csv` |
| Darkness | En karanlık blok ortalaması | rho ≈ −0.83 (lokal senaryoda) | `results/darkness/scores_local.csv` |
| Skew | Hough Transform | MAE ≈ 0.91° | `results/skew/scores.csv` |
| Skew | Projection Profile | MAE ≈ 1.82° | `results/skew/scores.csv` |
| Occlusion | length_ratio (alan-bazlı, OCR) | rho ≈ −0.97 | `results/occlusion/scores.csv` |
| Occlusion | ml_occlusion_ratio (konum/renk/dokudan bağımsız) | rho = 1.00 (5 görülmemiş renk/dokuda da), hatalı-pozitif = 0 | `results/occlusion/ml_scores.csv` |
| Glare (beyaz kağıt) | naive_glare_ratio | rho = 1.00, **ancak** severity=0'da ~%85 hatalı-pozitif | `results/glare/false_positive_baseline.csv` |
| Glare (renkli zemin) | glare_ratio (AYNI, değiştirilmemiş fonksiyon) | rho = 1.00, hatalı-pozitif = 0 (3 renk şeması, 72 görüntü) | `results/glare/id_card_scores.csv` |

Glare satırlarındaki çelişki kasıtlı olarak vurgulanmıştır: aynı basit yöntem, düz beyaz
kağıtta *hiç bozulma olmayan* görüntülerde bile yüksek oranda yanlış alarm üretirken,
kimlik kartı gibi RENKLİ zeminde hatasız çalışır — nedeni "dichromatic reflection model"
(parlamanın rengi, yüzeyin kendi renginden ancak yüzey renkliyse ayrışabilir) — bkz.
[Bilinen Sınırlamalar](#bilinen-sınırlamalar)).

Deneylerin tam metodolojisi, alınan kararlar ve karşılaşılan problemler için
[`project_notes.md`](./project_notes.md) dosyasına bakınız.

## Durum ve Yol Haritası

| Aşama | Durum | Not |
|---|---|---|
| 1. Literatür araştırması | ✅ Tamamlandı | `research/` |
| 2. Blur | ✅ Tamamlandı | Font boyutuna duyarlılık bilinen bir sınırlama |
| 3. Darkness | ✅ Tamamlandı | Blok-bazlı analiz, global ortalamanın kaçırdığı lokal karanlığı yakalıyor |
| 3. Skew | ✅ Tamamlandı | Hough, az-metinli belgelerde Projection Profile'dan daha istikrarlı |
| 4. Occlusion (alan-bazlı) | ✅ Tamamlandı | Yalnızca yapılandırılmış (şablonu bilinen) alanlarla sınırlı |
| 4. Occlusion (ML) | ✅ Tamamlandı | Konum/renk/dokudan bağımsız (Random Forest); yalnızca sentetik yamalarla doğrulandı |
| 4. Glare (beyaz kağıt) | ⚠️ Yetersiz bulundu | Altı ayrı deneme (HSV+CC, şekil filtresi, 4 ML varyantı) hiçbiri güvenilir olmadı |
| 4. Glare (renkli zemin) | ✅ Doğrulandı | Kimlik kartı/pasaport benzeri zeminde AYNI basit yöntem rho=1.00, hatalı-pozitif=0 |
| 5. Feature Fusion + Web arayüzü | 🚧 v1.1 (kalibrasyon düzeltmeli) | Kalibrasyon eşikleri henüz gerçek etiketli veriyle doğrulanmadı |
| 6. Nihai rapor | ⏳ Başlanmadı | Tüm modüller ve kalibrasyon netleştiğinde ayrı bir adımda yazılacak |

**Açık noktalar (Aşama 5+ için):**
- Tüm alt-skorların ortak bir yöne (yüksek = iyi) normalize edildiğinin doğrulanması.
- Glare'in beyaz-kağıt sınırlaması artık NEDEN kaynaklı olarak anlaşıldı (dichromatic
  reflection model, renksiz yüzeyde ayrışacak sinyal yok) — kalıcı çözüm ya renkli-zemin
  tespitine (mevcut) güvenmek ya da gerçek CNN'e (Rodin & Orlov, 2019) geçmek.
- Blur/darkness/skew'in normalizasyon eşiklerinin, kimlik kartı gibi düz metin dışı belge
  türlerinde de ayrıca doğrulanması (şu an yalnızca glare bu türde test edildi).
- Ten rengi tabanlı occlusion tespitinin gerçek (sentetik olmayan) fotoğraflarla,
  farklı ışık koşulları ve ten-rengi-benzeri arka planlarla yeniden doğrulanması.
- Normalizasyon eşiklerinin gerçek, etiketli (sentetik olmayan) veriyle kalibrasyonu.
- Tüm modüllerin gerçek belge fotoğrafları üzerinde yeniden doğrulanması.

## Bilinen Sınırlamalar

- **Skorlar mutlak bir "doğruluk oranı" değildir.** Feature Fusion aşamasındaki
  normalizasyon eşikleri, literatür ve sentetik deney gözlemlerinden esinlenen
  geçici/sezgisel değerlerdir; gerçek etiketli veriyle öğrenilmemiştir.
- **Glare tespiti yalnızca RENKLİ zeminli belgeler (kimlik kartı/pasaport benzeri) için
  kapsandı** — bu, projenin hedef kullanım alanına göre bilinçli bir kapsam kararıdır.
  Düz beyaz kağıtta altı ayrı deneme (şekil filtresi + dört ML varyantı) denenmiş, hiçbiri
  güvenilir bulunamamıştır (bkz. `project_notes.md`) — bu yüzden düz beyaz kağıt yüklenirse
  glare "uygulanamaz" olarak işaretlenir, tahmini bir sayı üretilmez. Renkli zeminde ise
  basit HSV+CC yöntemi hatasız çalışır (rho=1.00, hatalı-pozitif=0) — sistem zemin rengini
  otomatik tespit eder (`has_colored_background`).
- **Occlusion modülünün OCR/alan-bazlı bileşeni yalnızca yapılandırılmış alanlarda
  çalışır** — konumu ve beklenen formatı önceden bilinen alanlar (örn. "Belge No")
  gerektirir. ML tabanlı bileşen (Random Forest) konum VE renkten bağımsızdır — 5
  görülmemiş renk/dokuda doğrulanmıştır — ama yalnızca sentetik, PIL ile çizilmiş
  yamalarla test edilmiştir; gerçek el/parmak dokusu, gölgeler ve gerçek fotoğraf
  koşulları henüz test edilmemiştir.
- **Tüm doğrulama sentetik veri üzerinde yapılmıştır.** Gerçek (taranmış/fotoğraflanmış)
  belgelerle yeniden doğrulama henüz yapılmamıştır.

## Dokümantasyon

| Belge | İçerik |
|---|---|
| [`project_notes.md`](./project_notes.md) | Modül bazlı çalışma günlüğü — problem, yöntem, sonuç, alınan kararlar |
| [`research/literature_review.md`](./research/literature_review.md) | Literatür taraması |
| [`research/references.md`](./research/references.md) | Kaynak listesi |
| `src/<modül>/README.md` | Her modülün yöntem açıklaması ve matematiksel gerekçesi |
| `results/<modül>/` | Deney çıktıları — CSV skorları ve grafikler |

## Lisans

Bu proje için henüz bir açık kaynak lisansı belirlenmemiştir; tüm hakları saklıdır.
