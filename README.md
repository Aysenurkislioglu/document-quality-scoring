# Document Quality Scoring

Belge görüntülerindeki kalite problemlerini (**blur, glare, darkness, skew, occlusion**)
ayrı ayrı ölçüp, açıklanabilir bir **Document Quality Score (0-100)** üreten bir sistem.

**Motivasyon:** Kimlik belgelerinde sahtecilikte kullanılan/gizlenen bozulmaları, doğrudan
"kara kutu" bir CNN yerine — daha yüksek doğruluk sağlayan, uygulanabilir ve
gerekçelendirilebilir (açıklanabilir) yöntemlerle tespit edebilmek.

> Proje aktif geliştirme aşamasındadır. Güncel durum, gerekçeler ve deney sonuçları için
> [`project_notes.md`](./project_notes.md) dosyasına bakınız — her modülün "neden bu
> yöntem seçildi", "ne bulundu", "ne işe yaramadı" kaydı orada tutulur.

## Klasör yapısı

```
document-quality-scoring/
│
├── research/              "Literatürde ne yapılmış?"
│   ├── initial_research.md    → İlk (dış) araştırma notları, başlangıç noktası
│   ├── literature_review.md   → Güncellenmiş/genişletilmiş literatür taraması
│   └── references.md          → Doğrulanmış + yeni kaynakların listesi
│
├── data/
│   ├── raw/                → Gerçek, işlenmemiş belge görüntüleri (henüz boş)
│   ├── processed/          → İşlenmiş/temizlenmiş veri (henüz boş)
│   └── synthetic/          → Sentetik belge + kontrollü bozulma verisi (blur/glare/darkness/skew/occlusion)
│
├── src/                    "Biz ne yaptık?" — yeniden kullanılabilir kod
│   ├── blur/                   ✅ Laplacian Variance, Tenengrad
│   ├── glare/                   ⚠️ HSV+CC baseline denendi, mevcut haliyle yetersiz bulundu (bkz. project_notes.md)
│   ├── darkness/                ✅ global/percentile/blok-bazlı yerel analiz
│   ├── skew/                    ✅ Hough Transform, Projection Profile
│   ├── occlusion/                ✅ OCR + beklenen alan deseni (yalnızca yapılandırılmış alanlar)
│   └── scoring/                  ⏳ feature fusion + ML skorlama (ileri aşama, henüz başlanmadı)
│
├── experiments/            "Deneylerde ne oldu?" — çalıştırılabilir deney scriptleri
│   ├── _common/                 Ortak sentetik belge üretici (tüm modüllerce paylaşılır)
│   ├── blur/                    ✅
│   ├── glare/                   ✅
│   ├── darkness/                ✅
│   ├── skew/                    ✅
│   └── occlusion/                ✅
│
├── results/                "Sonuçlarımız ne?" — CSV + grafik çıktıları
│   └── blur/, glare/, darkness/, skew/, occlusion/
│
├── reports/                "Bütün bunlardan ne öğrendik?" — nihai rapor (henüz oluşturulmadı)
│
├── project_notes.md        Modül bazlı çalışma günlüğü (problem/yöntem/sonuç/karar)
├── README.md                Bu dosya
└── requirements.txt          Python bağımlılıkları
```

## Kurulum

```bash
pip install -r requirements.txt
# Occlusion modülü için ayrıca sistem paketleri gerekir:
#   apt-get install tesseract-ocr tesseract-ocr-tur
```

Python 3.11 ile test edilmiştir.

## Modülleri çalıştırma

Her modül aynı üç adımlı desene sahiptir: (1) sentetik veri üret, (2) kontrollü bozulma
uygula (bazı modüllerde 1-2 birleşik), (3) deneyi çalıştır.

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

Çıktılar her modül için `results/<modül>/` altına kaydedilir (skorlar, monotonluk /
dynamic-range analizleri ve grafikler).

Her modülün yöntemi nasıl çalıştığına dair anlaşılır açıklama, ilgili `src/<modül>/README.md`
dosyasındadır.

## Web arayüzü (basit demo)

Bir belge fotoğrafı yükleyip birleşik Document Quality Score (0-100) görmek için:

```bash
python3 app.py
# tarayıcıda http://127.0.0.1:5000 aç
```

Bu arayüz `src/scoring/fusion.py` üzerinden blur/darkness/skew (+ isteğe bağlı glare)
alt-skorlarını birleştirir. **Önemli:** bu skor gerçek etiketli veriyle kalibre edilmiş
bir ML modelinin çıktısı değildir — mevcut modüllerin ham metriklerinden türetilen
geçici/sezgisel bir özet skordur (detay için `src/scoring/fusion.py` docstring'i ve
"Aşama 5" notları). Occlusion modülü, yalnızca önceden bilinen alan şablonları
gerektirdiğinden bu genel yüklemeye dahil değildir.

## Durum

| Modül | Durum | Not |
|---|---|---|
| Literatür araştırması | ✅ Tamamlandı | `research/` |
| Blur | ✅ Baseline çalışıyor | Laplacian+Tenengrad, monoton (rho≈-1), ama font boyutuna duyarlı |
| Glare | ⚠️ Baseline yetersiz | HSV+CC, gri tonlamalı veride S kanalı işe yaramıyor — bkz. project_notes.md |
| Darkness | ✅ Baseline çalışıyor | Blok-bazlı yerel analiz, global ortalamanın kaçırdığı lokal karanlığı yakalıyor |
| Skew | ✅ Baseline çalışıyor | Hough istikrarlı; Projection Profile az-metinli belgelerde başarısız olabiliyor |
| Occlusion | ✅ Baseline çalışıyor | OCR + beklenen uzunluk, güçlü monoton sinyal (yalnızca yapılandırılmış alanlar) |
| Feature Fusion / ML Scoring | ⏳ Başlanmadı | Aşama 5 |
| Nihai Rapor | ⏳ Başlanmadı | Kullanıcı isteğiyle ayrı bir adımda yazılacak |
