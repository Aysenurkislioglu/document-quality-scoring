# Occlusion Detection — Yöntem Açıklaması

## Temel fikir

Occlusion, belgenin bir kısmının parmak, el, sticker veya başka bir nesne tarafından
kapatılmasıdır. initial_research.md'nin verdiği örnek tam olarak bu modülün
uyguladığı yaklaşımı özetliyor:

```
Beklenen text:   AYŞENUR KIŞLIOĞLU
OCR sonucu:      AYŞENUR K______
```

## Yöntem: OCR + "beklenen alan deseni" karşılaştırması

Bu modülde literatürün önerdiği **OCR + layout analysis** baseline'ı, kimlik belgesi
alanlarının **bilinen şeması (schema)** ile birleştirilerek uygulanıyor. Üretimde, bir
kimlik belgesinin hangi alanları içermesi gerektiği (Ad Soyad, Belge No, Tarih) ve bu
alanların **formatı** (örn. Belge No = 10 haneli sayı) zaten bilinir — bu, "referans
görüntüye" ihtiyaç duymadan (yani gerçek/doğru değeri bilmeden) kullanılabilecek meşru bir
no-reference sinyaldir.

```
Belge görüntüsü
      │
      ▼
Bilinen alan konumu (layout: "Belge No" nerede?)
      │
      ▼
O bölgeyi kırp (crop) + OCR uygula
      │
      ▼
   ┌──────────────┬──────────────────┐
   │ OCR sonucunun │ OCR güven skoru  │
   │ beklenen      │ (tesseract'ın    │
   │ UZUNLUĞA      │ kendi confidence │
   │ ne kadar      │ değeri)          │
   │ yaklaştığı    │                  │
   └──────────────┴──────────────────┘
      │
      ▼
Occlusion şüphe skoru (0-100, 100 = kapanma yok)
```

**İki tamamlayıcı sinyal:**

1. **Length ratio (uzunluk oranı):** OCR'ın okuduğu karakter sayısının, o alan için
   beklenen karakter sayısına (örn. Belge No için 10) oranı. Kapanma arttıkça karakterler
   "kaybolur", OCR daha kısa (veya boş) bir sonuç üretir.
2. **OCR confidence:** Tesseract'ın kendi ürettiği, her tanıma için 0-100 arası güven
   skoru. Kapanma, karakterlerin bozulmasına/belirsizleşmesine yol açtığı için confidence
   düşer.

**Neden yalnızca OCR confidence yeterli değil?** README'nin başındaki initial_research.md
alıntısında da belirtildiği gibi: düşük OCR confidence tek başına kesin occlusion kanıtı
değildir (blur, glare, darkness, skew de confidence'ı düşürebilir). Bu yüzden bu modül,
occlusion'a daha özgü bir sinyal olan **"beklenen uzunluktan sapma"**yı da ayrıca
hesaplayıp iki sinyali birlikte raporluyor.

## Sınırlama

Bu yöntem yalnızca **konumu ve formatı önceden bilinen, yapılandırılmış alanlar** (kimlik
numarası, tarih gibi) için çalışır. Serbest metin (paragraflar) için "beklenen uzunluk"
tanımlı olmadığından, bu modül şu an yalnızca yapılandırılmış kimlik alanlarına
uygulanmıştır. Genel/serbest metin occlusion tespiti (örn. nesne tespiti / segmentasyon)
literatürün "ileri yöntem" olarak işaret ettiği ayrı bir konudur (bkz.
`research/literature_review.md`, Bölüm 2.5).

## Ek yöntem: Ten Rengi (Skin-Color) Tespiti — konumdan bağımsız

`skin_detection.py`, yukarıdaki sınırlamayı KISMEN gideren, ayrı bir modüldür: belge
üzerinde parmak/el ile kapatılmış bir bölgeyi, **konumunu önceden bilmeden** tespit etmeyi
hedefler.

**Yöntem:** Görüntü YCrCb renk uzayına çevrilir; ten rengine tipik olan Cr/Cb aralığında
kalan piksel oranı hesaplanır (glare modülündeki HSV eşiklemesiyle aynı aile — bkz.
`src/glare/metrics.py`). Y (parlaklık) kanalı kasıtlı olarak sınırlanmaz, çünkü ten farklı
aydınlatmalarda geniş bir parlaklık aralığına yayılabilir; asıl ayırt edici sinyal Cr/Cb
(renk) kanallarındadır.

```
Görüntü (BGR)
     │
     ▼
YCrCb'ye çevir
     │
     ├── Cr kanalı [133, 173] aralığında  ┐
     │                                     ├─►  Ten rengi adayı piksel
     └── Cb kanalı [77, 127] aralığında    ┘
     │
     ▼
Bağlı bileşen analizi (küçük gürültüyü ele)
     │
     ▼
skin_occlusion_ratio = ten rengi alanı / toplam alan
```

**Doğrulama (bkz. `experiments/occlusion/generate_skin_occlusion_documents.py` ve
`run_skin_experiment.py`):** 12 belge × 3 ten tonu (açık/orta/koyu) × 6 kapanma seviyesi
(216 görüntü) üzerinde test edildi. Yama, konumu KASITLI OLARAK RASTGELE seçildi (yöntemin
gerçekten konumdan bağımsız çalıştığını doğrulamak için).

| Ten tonu | Spearman rho | Hatalı-pozitif (kapanma yokken) |
|---|---|---|
| Açık | 1.0000 | 0.0000 |
| Orta | 1.0000 | 0.0000 |
| Koyu | 1.0000 | 0.0000 |

Üçü de mükemmel monoton ve sıfır hatalı-pozitif — bkz. `results/occlusion/skin_scores.csv`.

**Bilinen sınırlama (henüz test edilmedi):** Bu doğrulama yalnızca SENTETİK, düz renkli
(tek tonlu) dikdörtgen yamalarla yapıldı. Gerçek bir el/parmak fotoğrafında doku, gölgeler,
eklem kıvrımları ve değişken aydınlatma vardır — bunlar test edilmedi. Ayrıca ten rengine
yakın renkte gerçek arka plan nesneleri (örn. ahşap masa, bej duvar) gerçek fotoğraflarda
hatalı-pozitife yol açabilir; bu senaryo da sentetik veri setinde yoktur. Bu yüzden
`src/scoring/fusion.py` bu sinyali kullanırken bu sınırlamayı docstring'inde açıkça belirtir.
