"""
Document Quality Scoring — basit Flask web arayüzü.

Bir belge fotoğrafı yükle; sistem blur/darkness/skew/occlusion/glare
alt-skorlarını hesaplayıp birleşik bir Document Quality Score (0-100)
üretir.

ÖNEMLİ: Bu skor, gerçek etiketli veriyle kalibre edilmiş bir ML modelinin
"doğruluk oranı" DEĞİLDİR — mevcut modüllerin ham metriklerini okunabilir bir
ölçeğe taşıyan, geçici/sezgisel bir özet skordur. Detaylar için
src/scoring/fusion.py ve project_notes.md.

GİZLİLİK (gerçek kimlik/pasaport gibi hassas belgelerle test edilebileceği
için önemli): Yüklenen görüntü DİSKE, VERİTABANINA YA DA GİT'E HİÇBİR ŞEKİLDE
YAZILMAZ — yalnızca tek bir isteğin (request) ömrü boyunca bellekte tutulur,
skorlar hesaplanır, yanıt (önizleme dahil) tarayıcıya gönderilir ve işlem
bitince Python çöp toplayıcısı belleği serbest bırakır. Sunucu yalnızca
127.0.0.1'i (kendi bilgisayarın) dinler — ağ üzerinden hiçbir yere gitmez.
Tek istisna: çok büyük dosyalarda Werkzeug (Flask'ın alt katmanı), isteği
ayrıştırırken belleği aşan kısmı OS'un geçici klasörüne (macOS'ta kullanıcıya
özel, diğer kullanıcılarca okunamayan bir dizin) kısa süreliğine yazıp istek
bitince otomatik siler — bu, Flask'ın standart davranışıdır, bizim kodumuzun
parçası değildir.

Çalıştırma:
    source .venv/bin/activate
    python3 app.py
    # tarayıcıda http://127.0.0.1:5000 aç
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
from flask import Flask, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from src.scoring.fusion import compare_module_methods, compute_document_quality_score

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB üst sınır


@app.after_request
def _no_store(response):
    """Yanıtta (önizleme görüntüsü dahil) tarayıcının/ara sunucuların hiçbir
    şey diske/önbelleğe yazmamasını garanti eder — hassas belge içeriği
    barındırdığı için."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response

MODULE_LABELS = {
    "blur": ("Bulanıklık", "Yüksek skor = keskin/net görüntü"),
    "darkness": ("Aydınlatma", "Yüksek skor = yeterince aydınlık"),
    "skew": ("Eğiklik", "Yüksek skor = belge düz, az eğik"),
    "occlusion": ("Kapanma", "Yüksek skor = yabancı nesne (el/sticker/vb.) tespit edilmedi"),
    "glare": ("Parlama", "Yüksek skor = parlama tespit edilmedi"),
}
MODULE_ORDER = ["blur", "darkness", "skew", "occlusion", "glare"]


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_error):
    """16 MB üst sınırı aşan yüklemelerde Werkzeug'un varsayılan/İngilizce hata
    sayfası yerine, mevcut şablonu kullanan anlaşılır bir Türkçe mesaj göster."""
    return (
        render_template(
            "index.html",
            result={"error": "Dosya çok büyük (üst sınır: 16 MB). Daha küçük bir görüntü dene."},
            image_data_uri=None,
            module_labels=MODULE_LABELS,
            module_order=MODULE_ORDER,
        ),
        413,
    )


def _verdict(overall: float) -> tuple[str, str]:
    if overall >= 70:
        return "good", "İyi kalite"
    if overall >= 40:
        return "warn", "Orta kalite — bazı problemler olabilir"
    return "bad", "Düşük kalite — belge yeniden çekilmeli"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_data_uri = None

    if request.method == "POST":
        file = request.files.get("document")
        if file and file.filename:
            raw_bytes = file.read()
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if image_bgr is None:
                result = {"error": "Görüntü okunamadı. Desteklenen formatlar: jpg, png, bmp."}
            else:
                scored = compute_document_quality_score(image_bgr)
                method_comparison = compare_module_methods(image_bgr)
                for module_data in method_comparison.values():
                    module_data["methods"] = {
                        name: ("tespit edilemedi" if value is None else f"{value:.3f}")
                        for name, value in module_data["methods"].items()
                    }
                verdict_class, verdict_text = _verdict(scored["overall_score"])
                result = {
                    "overall_score": scored["overall_score"],
                    "verdict_class": verdict_class,
                    "verdict_text": verdict_text,
                    "components": scored["components"],
                    "calibration_note": scored["calibration_note"],
                    "occlusion_note": scored["occlusion_note"],
                    "glare_note": scored["glare_note"],
                    "method_comparison": method_comparison,
                }
                b64 = base64.b64encode(raw_bytes).decode("ascii")
                mime = file.mimetype or "image/jpeg"
                image_data_uri = f"data:{mime};base64,{b64}"
        else:
            result = {"error": "Lütfen bir dosya seç."}

    return render_template(
        "index.html",
        result=result,
        image_data_uri=image_data_uri,
        module_labels=MODULE_LABELS,
        module_order=MODULE_ORDER,
    )


if __name__ == "__main__":
    # debug=False: gerçek kimlik/pasaport gibi hassas belgeler test edilirken
    # Werkzeug'un interaktif hata ayıklayıcısını (hata anında bellek
    # durumunu tarayıcıda gösterir, ayrıca uzaktan kod çalıştırma riski
    # taşır) kasıtlı olarak KAPALI tutuyoruz. Geliştirme sırasında otomatik
    # yeniden yükleme istersen: `FLASK_DEBUG=1 python3 app.py`.
    import os
    debug_mode = os.environ.get("FLASK_DEBUG") == "1"
    app.run(debug=debug_mode, port=5000)
