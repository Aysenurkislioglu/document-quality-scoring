# Darkness / Illumination — Yöntem Açıklaması

## Temel fikir: neden tek bir "ortalama parlaklık" yeterli değil

İki görüntü aynı ortalama parlaklığa sahip olabilir ama kullanılabilirlik açısından çok
farklı olabilir:

```
Görüntü A: Her yer biraz karanlık (homojen)   → ortalama = 140
Görüntü B: Genel olarak aydınlık, ama         → ortalama = 140
           ID number bölgesi çok karanlık
```

Görüntü B'de kritik bir alan (örn. kimlik numarası) okunamayacak kadar karanlık olabilir,
ama görüntünün geneli aydınlık olduğu için **global ortalama bu sorunu gizler.**

## Bu modülde üç farklı bakış açısı birlikte kullanılıyor

```
Görüntü
   │
   ├── 1) Global istatistikler (mean, median)
   │       → "Genel olarak görüntü ne kadar karanlık?"
   │
   ├── 2) Percentile analizi (P5, P25, P50, P75, P95)
   │       → "En karanlık %5'lik dilim ne durumda?"
   │       → NOT: percentile'lar da TÜM görüntü üzerinden hesaplanırsa,
   │         görüntünün çok küçük bir bölümünü (örn. tek bir kimlik alanı)
   │         kaplayan lokal karanlık bölgeleri KAÇIRABİLİR (bkz. aşağıdaki
   │         deneyde bu durum özellikle test edilmiştir).
   │
   └── 3) Blok-bazlı yerel (local) analiz
           → Görüntü küçük bloklara (örn. 32x32 piksel) bölünür, her
             bloğun kendi ortalaması hesaplanır.
           → "En karanlık BLOK hangisi, ne kadar karanlık?"
           → Bu, küçük ve kritik bir bölgenin (örn. ID number) karanlık
             kalmasını, görüntünün geri kalanı aydınlık olsa bile yakalar.
```

**Neden blok-bazlı analiz percentile'dan farklı/tamamlayıcı?** Percentile, TÜM görüntü
pikselleri havuzuna bakar — eğer karanlık bölge görüntünün örneğin %1'ini kaplıyorsa, P5
(en karanlık %5) bu bölgeyi büyük ölçüde "görebilir" ama bölge daha da küçükse (örn.
%0.4) P5 bile bunu tamamen kaçırabilir. Blok-bazlı analiz ise, bölgenin toplam görüntüye
oranından bağımsız olarak, **her bloğu kendi içinde** değerlendirdiği için küçük ama
kritik bölgeleri yakalamada daha güvenilirdir.

## Yerel kontrast (local contrast)

Her blok için ayrıca standart sapma (contrast) da hesaplanır. Düşük yerel kontrast +
düşük yerel ortalama birlikte, "bu blokta hem karanlık hem de detay kaybı var" anlamına
gelir — yalnızca karanlık olup hâlâ okunabilir olan bir bölgeden ayırt edilebilir.
