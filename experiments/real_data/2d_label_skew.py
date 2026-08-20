"""
ADIM 2d (ek) — Eğiklik ŞİDDETİNİ etiketler.

GEREKÇE: Blur ve darkness kalibrasyonları gerçek veriyle düzeltildikten
sonra skew, "en kötü modül" olarak en sık çıkan modül hâline geldi
(bkz. project_notes.md, "Blur kalibrasyonu"). Ama skew HİÇ gerçek şiddet
etiketiyle doğrulanmadı — occlusion/darkness'ta yapılan aynı yöntemi
(dereceli ground-truth + aday metrik/kalibrasyon testi) skew için de
uygulamak gerekiyor.

Tıpkı 2b/2c gibi tarayıcı tabanlı (localhost, tamamen yerel). Çıktı:
results/real_data/skew_severity.csv (yalnızca anon_id + eğiklik derecesi
— görüntü/dosya adı yok).

Kullanım:
    python3 experiments/real_data/2d_label_skew.py
Sonra tarayıcıda: http://127.0.0.1:5053
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
from flask import Flask, Response, redirect, render_template_string, request, url_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = PROJECT_ROOT / "data" / "raw" / "anon_mapping.csv"
SEVERITY_PATH = PROJECT_ROOT / "results" / "real_data" / "skew_severity.csv"

SEVERITY_LABELS = {"duz": "Düz (eğik değil)", "az": "Az eğik", "cok": "Çok eğik"}

app = Flask(__name__)


def load_mapping():
    if not MAPPING_PATH.exists():
        return []
    with open(MAPPING_PATH, newline="", encoding="utf-8") as f:
        rows = [(int(r["anon_id"]), r["absolute_path"]) for r in csv.DictReader(f)]
    return sorted(rows)


def load_severity():
    if not SEVERITY_PATH.exists():
        return {}
    with open(SEVERITY_PATH, newline="", encoding="utf-8") as f:
        return {int(r["anon_id"]): r["egiklik_siddet"] for r in csv.DictReader(f)}


def save_severity(data: dict):
    SEVERITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEVERITY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["anon_id", "egiklik_siddet"])
        for anon_id in sorted(data):
            writer.writerow([anon_id, data[anon_id]])


MAPPING = load_mapping()
PATH_BY_ID = dict(MAPPING)
ORDER = [anon_id for anon_id, _ in MAPPING]
_state = {"pos": 0, "history": []}

TEMPLATE = """
<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<title>Eğiklik Şiddeti</title>
<style>
  :root { color-scheme: dark; }
  body { background:#111318; color:#e8e8ec; font-family:-apple-system,'Segoe UI',sans-serif;
         margin:0; padding:24px; display:flex; flex-direction:column; align-items:center; }
  .top { width:100%; max-width:900px; display:flex; justify-content:space-between;
         color:#9a9ba5; font-size:14px; margin-bottom:14px; }
  img.doc { max-width:900px; max-height:70vh; border-radius:10px; box-shadow:0 4px 24px rgba(0,0,0,.5); }
  .row { display:flex; gap:14px; margin-top:24px; }
  .btn { padding:18px 26px; font-size:16px; border-radius:12px; border:none; cursor:pointer;
         font-weight:700; color:#fff; }
  .b-duz { background:#4caf7d; } .b-az { background:#d9a441; color:#1a1400; } .b-cok { background:#e05252; }
  .btn:hover { filter:brightness(1.1); }
  .controls { margin-top:20px; }
  .ctrl-btn { padding:9px 16px; border-radius:8px; border:1px solid #33353f; background:#1a1c23;
              color:#c9cad2; cursor:pointer; font-size:13px; }
  .hint { color:#6b6c78; font-size:13px; margin-top:18px; text-align:center; max-width:600px; }
  .done { text-align:center; margin-top:80px; }
</style>
</head>
<body>
{% if finished %}
  <div class="done">
    <h1>Tamamlandı!</h1>
    <p>{{ done }}/{{ total }}</p>
    <p class="hint">Şimdi: <code>python3 experiments/real_data/4_analyze_accuracy.py</code></p>
  </div>
{% else %}
  <div class="top"><span>#{{ anon_id }} ({{ idx }}/{{ total }})</span><span>{{ done }} etiketlendi</span></div>
  <img class="doc" src="{{ url_for('image', anon_id=anon_id) }}" alt="belge">
  <form method="post" action="{{ url_for('label') }}">
    <input type="hidden" name="anon_id" value="{{ anon_id }}">
    <div class="row">
      <button type="submit" name="sev" value="duz" class="btn b-duz">Düz</button>
      <button type="submit" name="sev" value="az" class="btn b-az">Az Eğik</button>
      <button type="submit" name="sev" value="cok" class="btn b-cok">Çok Eğik</button>
    </div>
  </form>
  <div class="controls">
    <form method="post" action="{{ url_for('back') }}" id="backForm">
      <button class="ctrl-btn" {{ 'disabled' if not can_back else '' }}>◀ Önceki</button>
    </form>
  </div>
  <p class="hint">Kart/belge ne kadar döndürülmüş/eğik duruyor (dikey/yatay hizadan sapma)?
    Düz = neredeyse tam hizalı, Çok Eğik = belirgin döndürülmüş. Klavye: 1=Düz 2=Az 3=Çok, ← önceki</p>
  <script>
    document.addEventListener('keydown', (e) => {
      const map = {'1':'duz','2':'az','3':'cok'};
      if (map[e.key]) document.querySelector(`button[value="${map[e.key]}"]`).click();
      if (e.key === 'ArrowLeft') {
        const btn = document.querySelector('#backForm button');
        if (!btn.disabled) document.getElementById('backForm').submit();
      }
    });
  </script>
{% endif %}
</body>
</html>
"""


@app.route("/")
def index():
    data = load_severity()
    if not ORDER:
        return "anon_mapping.csv bulunamadı — önce 1_prepare_dataset.py çalıştırılmalı.", 400
    if _state["pos"] >= len(ORDER):
        return render_template_string(TEMPLATE, finished=True, done=len(data), total=len(ORDER))
    anon_id = ORDER[_state["pos"]]
    return render_template_string(
        TEMPLATE, finished=False, anon_id=anon_id, idx=_state["pos"] + 1,
        total=len(ORDER), done=len(data), can_back=bool(_state["history"]),
    )


@app.route("/image/<int:anon_id>")
def image(anon_id):
    path = PATH_BY_ID.get(anon_id)
    if not path:
        return "not found", 404
    img = cv2.imread(path)
    if img is None:
        return "okunamadı", 404
    h, w = img.shape[:2]
    max_h = 1000
    if h > max_h:
        scale = max_h / h
        img = cv2.resize(img, (int(w * scale), max_h))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/label", methods=["POST"])
def label():
    data = load_severity()
    anon_id = int(request.form["anon_id"])
    sev = request.form.get("sev")
    if sev in SEVERITY_LABELS:
        data[anon_id] = sev
        save_severity(data)
        _state["history"].append(_state["pos"])
        _state["pos"] = min(_state["pos"] + 1, len(ORDER))
    return redirect(url_for("index"))


@app.route("/back", methods=["POST"])
def back():
    if _state["history"]:
        _state["pos"] = _state["history"].pop()
    return redirect(url_for("index"))


def main():
    if not ORDER:
        print(f"Önce 1_prepare_dataset.py çalıştırılmalı ({MAPPING_PATH} yok).")
        return
    data = load_severity()
    _state["pos"] = len(ORDER)
    for i, anon_id in enumerate(ORDER):
        if anon_id not in data:
            _state["pos"] = i
            break
    _state["history"] = list(range(_state["pos"]))
    print(f"Toplam {len(ORDER)} görüntü, {len(data)} tanesi zaten etiketli.")
    print("Tarayıcıda aç -> http://127.0.0.1:5053")
    app.run(host="127.0.0.1", port=5053, debug=False)


if __name__ == "__main__":
    main()
