# Gerçek Veri Doğrulama İş Akışı

Bu klasördeki script'ler, projeyi kendi gerçek (kimlik/pasaport gibi hassas) belge
fotoğraflarınla doğrulamak için tasarlandı. **Hiçbir görüntü ya da gerçek dosya adı
bu projenin dışına (git, Claude, başka bir yer) çıkmaz** — her şey senin
bilgisayarında kalır.

## Neden bu sırayla?

Önce ETİKETLE, sonra SKORLA. Eğer önce sistemin skorlarına bakıp sonra etiketlersen,
bilinçsizce kendi yargını sistemin dediğine göre kaydırabilirsin. Bağımsız etiketleme,
gerçek bir doğruluk ölçümü sağlar.

## Adımlar

```bash
source .venv/bin/activate

# 1) Fotoğraflarını anonimleştir (anon_id ata)
python3 experiments/real_data/1_prepare_dataset.py /path/to/kimlik/klasoru

# 2) Kendi gözünle etiketle (yarıda bırakıp devam edebilirsin)
python3 experiments/real_data/2_label_tool.py

# 3) Sistemle toplu skorla
python3 experiments/real_data/3_batch_score.py

# 4) Karşılaştır — gerçek doğruluk raporu
python3 experiments/real_data/4_analyze_accuracy.py
```

## Gizlilik — nerede ne duruyor

| Dosya | İçerik | Paylaşılabilir mi? |
|---|---|---|
| `data/raw/anon_mapping.csv` | anon_id ↔ **gerçek dosya yolu** | ❌ ASLA — yalnızca senin referansın için |
| `results/real_data/labels.csv` | anon_id + senin etiketlerin | Görüntü/isim yok, ama yine de `.gitignore`'da |
| `results/real_data/scores.csv` | anon_id + sistem skorları | Görüntü/isim yok, ama yine de `.gitignore`'da |
| `results/real_data/accuracy_report.txt` | Toplu istatistikler | Görüntü/isim yok — istersen bunu (yalnızca bu dosyayı) paylaşıp yorumlatabilirsin |

`results/real_data/` klasörünün tamamı `.gitignore`'da — hiçbiri commit edilmez, GitHub'a
gitmez. `anon_mapping.csv` da `data/raw/` altında olduğu için aynı şekilde korunuyor.

## Etiketleme tuşları (2_label_tool.py)

| Tuş | Anlamı |
|---|---|
| `1` / `2` / `3` | Genel kalite: kötü / orta / iyi (onaylar, sıradaki görüntüye geçer) |
| `b` | Bulanık (işaretle/kaldır) |
| `p` | Parlama/glare (işaretle/kaldır) |
| `k` | Karanlık (işaretle/kaldır) |
| `o` | Kapanma/örtülü (işaretle/kaldır) |
| `e` | Eğik (işaretle/kaldır) |
| `r` | Bu görüntü için işaretleri sıfırla |
| `u` | Bir önceki görüntüye geri dön |
| `s` | Bu görüntüyü atla |
| `q` | Kaydet ve çık (istediğin an; ilerleme kaybolmaz) |

## Analiz raporu ne gösterir

- Genel kalite: senin yargın ile sistemin skoru arasındaki korelasyon (Spearman rho)
- Her defekt için: "sen bulanık dediğinde, sistemin blur skoru gerçekten düşük mü çıkıyor?"
- "En kötü modül" tespiti: sistemin işaret ettiği modül, senin işaretlediğin defektle uyuşuyor mu?
- Yanlış alarm: hiç sorun görmediğin ama sistemin "kötü" dediği görüntüler
