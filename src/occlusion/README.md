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
