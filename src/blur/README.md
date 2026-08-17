# Blur Detection — Yöntem Açıklaması

Bu modül iki klasik, eğitim gerektirmeyen (training-free) sharpness ölçüm yöntemi
uygular: **Laplacian Variance** ve **Tenengrad (Gradient Magnitude)**. İkisi de aynı temel
fikre dayanır ama farklı matematiksel araçlar kullanır.

## Ortak fikir: keskinlik = kenar gücü

Bir görüntüde "bulanıklık" aslında şunu ifade eder: komşu piksellerin değerleri arasındaki
geçişler yumuşamıştır. Keskin bir görüntüde siyah bir harften beyaz arka plana geçiş
1-2 piksel içinde aniden olur. Bulanık bir görüntüde bu geçiş 5-10 piksele yayılır, yani
kenarlar "yumuşar".

```
Keskin kenar (sharp):        Bulanık kenar (blur):

255 255   0   0             255 200 120  60   0
█████░░░░░░                 █████▒▒▒░░░░░

hızlı geçiş                  yavaş, yumuşak geçiş
```

Her iki yöntem de görüntüdeki bu "geçiş hızını" farklı matematiksel araçlarla ölçüp tek bir
sayıya indirger. Sayı ne kadar yüksekse görüntü o kadar keskin (az bulanık) kabul edilir.

---

## 1. Laplacian Variance

**Sezgi:** Laplacian, bir görüntünün *ikinci türevidir*. Düz/homojen bölgelerde
(arka plan gibi) neredeyse sıfır değer üretir; ani değişim olan yerlerde (kenarlarda) büyük
mutlak değerler üretir (pozitif veya negatif).

```
Görüntü
   │
   ▼
Laplacian filtresi uygula  (∂²I/∂x² + ∂²I/∂y²)
   │
   ▼
Her piksel için "kenar gücü" değeri elde edilir
   │
   ▼
Bütün pikseller üzerinden VARYANS hesapla
   │
   ▼
Sharpness skoru
```

**Neden varyans?** Keskin bir görüntüde bazı pikseller (kenarlar) çok yüksek, bazıları
(düz bölgeler) sıfıra yakın değer alır → değerler geniş bir aralığa yayılır → **yüksek
varyans**. Bulanık bir görüntüde kenarlar zayıfladığı için neredeyse tüm pikseller birbirine
yakın (düşük) değerler alır → değerler dar bir aralıkta toplanır → **düşük varyans**.

**Formül (kavramsal):**

```
score = Var( Laplacian(gray_image) )
```

**Bilinen sınırlama (literatürden):** Laplacian ikinci türev olduğu için gürültüye
(noise) karşı oldukça hassastır — rastgele piksel gürültüsü de "ani değişim" gibi
algılanıp skoru yapay olarak yükseltebilir. Ayrıca metin yoğunluğu ve font büyüklüğü
farklı belgelerde doğal olarak farklı kenar miktarı ürettiği için, **tek bir sabit
threshold farklı belge türleri arasında taşınabilir değildir.** Bu proje bu sınırlamayı
sonraki bir deneyde ayrıca test edecek (bkz. `experiments/blur/`).

---

## 2. Tenengrad (Gradient Magnitude)

**Sezgi:** Laplacian yerine, görüntünün *birinci türevini* (gradyanını) kullanır. Sobel
operatörü ile hem yatay (Gx) hem dikey (Gy) yöndeki değişim hızı ayrı ayrı hesaplanır,
sonra bu iki bileşen birleştirilerek her pikselin "ne kadar dik bir kenarda olduğu"
bulunur.

```
Görüntü
   │
   ├── Sobel (x yönü) → Gx
   └── Sobel (y yönü) → Gy
              │
              ▼
   Gradyan büyüklüğü = √(Gx² + Gy²)   (her piksel için)
              │
              ▼
   Tüm piksellerin ortalaması (veya kare ortalaması)
              │
              ▼
        Tenengrad skoru
```

**Formül (kavramsal):**

```
Gx = Sobel_x(gray_image)
Gy = Sobel_y(gray_image)
score = mean( Gx² + Gy² )
```

**Laplacian'dan farkı:** Birinci türev, ikinci türeve göre gürültüye karşı biraz daha
dayanıklıdır (gürültü genelde ikinci türevde daha fazla büyütülür). Bu yüzden Tenengrad,
Laplacian'ı doğrulamak/çapraz kontrol etmek için iyi bir ikinci ölçüttür: iki yöntem aynı
yönde hareket ediyorsa (ikisi de düşükse) blur konusunda daha yüksek güven duyulabilir;
aralarında büyük fark varsa bu, gürültü veya başka bir bozulmanın (örn. JPEG artefaktı)
karıştığına işaret edebilir.

---

## Bu projede nasıl kullanılıyor?

Her iki skor da ham haliyle "birim bağımsız" ve görüntü boyutuna/içeriğine göre değişken
ölçeklere sahiptir. Bu yüzden ilk deneyde amacımız mutlak bir "iyi/kötü" eşiği bulmak değil,
**skorların bilinen (kontrollü) blur şiddeti arttıkça tutarlı biçimde azalıp azalmadığını**
(monotonluk) ve **hangi yöntemin bu azalmayı daha güvenilir/doğrusal şekilde yakaladığını**
ölçmektir. Bkz. `experiments/blur/README.md`.
