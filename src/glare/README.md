# Glare Detection — Yöntem Açıklaması

## Temel fikir

Glare (parlama), belge yüzeyinden gelen güçlü bir ışığın kamera sensörünü doyurması
(saturate etmesi) sonucu oluşur. O bölgedeki piksellerin gerçek renk/doku bilgisi kaybolur
ve neredeyse tamamen **beyaza yakın, renksiz (düşük saturasyon)** bir alana dönüşür.

```
Normal metin bölgesi:      Glare bölgesi:

Koyu harf + beyaz zemin    Her şey beyaza yakın,
(yüksek kontrast)          detay/kontrast kayıp
```

## Yöntem: HSV eşikleme + Connected Components

Görüntü **HSV** (Hue, Saturation, Value) renk uzayına çevrilir:

```
Görüntü (BGR/Gray)
        │
        ▼
   HSV'ye çevir
        │
        ├── V (Value/parlaklık) YÜKSEK   ┐
        │                                 ├─►  Glare adayı piksel
        └── S (Saturation) DÜŞÜK          ┘
        │
        ▼
  Connected Components (bağlı bileşen analizi)
        │
        ▼
  Çok küçük (gürültü) bileşenleri ele
        │
        ▼
  Glare mask + Glare Ratio = Glare alanı / Belge alanı
```

**Neden hem V hem S?** Yalnızca V (parlaklık) kullanmak yeterli değildir — çünkü glare
sadece "parlak" değil, aynı zamanda **renksiz/doygunluğu düşük**tür. S eşiği eklemek,
örneğin sarımsı/renkli parlak bir yüzeyi (ki bu glare olmayabilir) ayırt etmeye yardımcı
olur.

**Neden Connected Components?** Tek tük, dağınık parlak pikseller (örn. sensör
gürültüsü) gerçek bir glare bölgesi değildir. Bağlı bileşen analizi, yalnızca belirli bir
alan büyüklüğünü aşan **bitişik** parlak bölgeleri "glare" olarak sayar.

## ⚠️ Bilinen ve bu projede DOĞRULANAN sınırlama

Literatür şu uyarıyı yapıyor: **"Beyaz belge alanı da yüksek parlaklığa sahip olabilir."**
Yani yukarıdaki yöntem, gerçek bir glare ile sıradan beyaz kağıt/margin alanını
**ayırt edemez** — ikisi de HSV uzayında yüksek V + düşük S özelliğine sahiptir.

Bu projede bu sınırlama, deneysel olarak nicel biçimde doğrulanmıştır: hiç glare
eklenmemiş belgelerde bile, yöntem metin bölgesindeki satır aralarını/boşlukları "glare"
olarak işaretlemektedir (bkz. `project_notes.md`, Glare bölümü, "false positive baseline").
Bu, yöntemin salt bir baseline olduğunu ve tek başına production'a hazır olmadığını
gösteriyor — literatürün "ileri yöntem: CNN segmentation" önerisinin neden gerekli
olabileceğine dair somut bir kanıt.

## Bu projede nasıl kullanılıyor?

Glare oranı yalnızca belgenin **içerik kutusu** (content bounding box — başlık ve
metin satırlarının kapladığı alan) içinde hesaplanır; büyük boş kenar boşlukları (margin)
hesaba katılmaz. Bu, saf beyaz kenarlıkların skoru şişirmesini kısmen engeller, ama satır
aralarındaki boşluklar hâlâ yanlış pozitif üretebilir (bkz. yukarıdaki sınırlama).

## Kırılma noktası: RENKLİ zeminde (kimlik kartı/pasaport) aynı yöntem mükemmel çalışıyor

Yukarıdaki sınırlama, düz **beyaz kağıt** için geçerlidir. Altı ayrı düzeltme denemesi
(şekil filtresi, dört farklı ML varyantı — bkz. `project_notes.md`, "Glare ML v1-v5")
beyaz kağıtta bu sınırı aşamadı. Dış araştırma nedenini açıkladı: klasik specular-highlight
yöntemleri **dichromatic reflection model**e dayanır — parlamanın rengi (ışık kaynağının
rengi, beyaza yakın) yüzeyin KENDİ renginden ayrışır. Beyaz kağıtta yüzeyin "gerçek rengi"
zaten beyaz olduğu için ayrışacak bir fark yok; ama **kimlik kartı/pasaport gibi renkli
zeminli belgelerde** bu fark gerçek ve ölçülebilir.

Bu, **hiçbir kod değişikliği yapılmadan**, yukarıdaki aynı `glare_ratio` (HSV+CC)
fonksiyonuyla test edildi (`experiments/glare/generate_id_card_documents.py` +
`run_id_card_experiment.py`): 3 renk şeması × 4 varyant × 6 şiddet = 72 görüntüde
**rho=1.00, hatalı-pozitif=0** — hem glare-yok hem bulanıklık+glare-yok durumunda (beyaz
kağıttaki en büyük ikinci sorun olan bulanıklık karışması burada da yok). Bkz.
`results/glare/id_card_scores.csv`.

**Sonuç:** `src/scoring/fusion.py`, her görüntü için `has_colored_background()` ile
zeminin renkli mi düz beyaz mı olduğunu tahmin eder; yalnızca renkli zeminde glare
skorunu nihai ortalamaya güvenilir şekilde dahil eder.
