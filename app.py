"""
Document Quality Scoring — basit Flask web arayüzü.

Bir belge fotoğrafı yükle; sistem blur/darkness/skew (ve isteğe bağlı olarak
glare) alt-skorlarını hesaplayıp birleşik bir Document Quality Score (0-100)
üretir.

ÖNEMLİ: Bu skor, gerçek etiketli veriyle kalibre edilmiş bir ML modelinin
"doğruluk oranı" DEĞİLDİR — mevcut modüllerin ham metriklerini okunabilir bir
ölçeğe taşıyan, geçici/sezgisel bir özet skordur. Detaylar için
src/scoring/fusion.py ve project_notes.md.

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

MODULE_LABELS = {
    "blur": ("Bulanıklık", "Yüksek skor = keskin/net görüntü"),
    "darkness": ("Aydınlatma", "Yüksek skor = yeterince aydınlık"),
    "skew": ("Eğiklik", "Yüksek skor = belge düz, az eğik"),
    "glare": ("Parlama", "Güvenilmez — yalnızca bilgi amaçlı"),
}


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_error):
    """16 MB üst sınırı aşan yüklemelerde Werkzeug'un varsayılan/İngilizce hata
    sayfası yerine, mevcut şablonu kullanan anlaşılır bir Türkçe mesaj göster."""
    return (
        render_template(
            "index.html",
            result={"error": "Dosya çok büyük (üst sınır: 16 MB). Daha küçük bir görüntü dene."},
            image_data_uri=None,
            include_glare=False,
            module_labels=MODULE_LABELS,
            module_order=["blur", "darkness", "skew", "glare"],
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
    include_glare = request.form.get("include_glare") == "on"

    if request.method == "POST":
        file = request.files.get("document")
        if file and file.filename:
            raw_bytes = file.read()
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if image_bgr is None:
                result = {"error": "Görüntü okunamadı. Desteklenen formatlar: jpg, png, bmp."}
            else:
                scored = compute_document_quality_score(image_bgr, include_glare=include_glare)
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
                    "glare_included": scored["glare_included_in_overall"],
                    "calibration_note": scored["calibration_note"],
                    "occlusion_note": scored["occlusion_note"],
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
        include_glare=include_glare,
        module_labels=MODULE_LABELS,
        module_order=["blur", "darkness", "skew", "glare"],
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
