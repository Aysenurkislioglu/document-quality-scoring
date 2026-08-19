"""
ADIM 2 (web sürümü) — Tarayıcı tabanlı, TIKLANABİLİR etiketleme aracı.

Klavye kısayolları ezberlemek yerine butonlara tıklayarak etiketlersin.
Yalnızca kendi bilgisayarında (127.0.0.1 / localhost) çalışır — hiçbir
görüntü ya da veri internete/başka bir yere çıkmaz. Görüntüler yalnızca
tarayıcın ile bu script arasında, kendi makinende dolaşır (app.py'deki
ana Flask uygulamasıyla birebir aynı gizlilik modeli).

Çıktı dosyası (labels.csv) 2_label_tool.py (OpenCV/klavye sürümü) ile
TAM OLARAK AYNI formatta — ikisini karışık kullanabilir, istediğin
zaman birinden diğerine geçebilirsin, ilerleme ortak.

Kullanım:
    python3 experiments/real_data/2_label_web.py
Sonra tarayıcıda aç:
    http://127.0.0.1:5050
Durdurmak için terminalde Ctrl+C.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
from flask import Flask, Response, redirect, render_template_string, request, url_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = PROJECT_ROOT / "data" / "raw" / "anon_mapping.csv"
LABELS_PATH = PROJECT_ROOT / "results" / "real_data" / "labels.csv"

FLAG_NAMES = ["bulanik", "parlama", "karanlik", "kapanma", "egik"]
FLAG_LABELS = {
    "bulanik": "Bulanık",
    "parlama": "Parlama",
    "karanlik": "Karanlık",
    "kapanma": "Kapanma / örtülü",
    "egik": "Eğik",
}
QUALITY_LABELS = {"kotu": "Kötü", "orta": "Orta", "iyi": "İyi"}

app = Flask(__name__)


def load_mapping():
    if not MAPPING_PATH.exists():
        return []
    with open(MAPPING_PATH, newline="", encoding="utf-8") as f:
        rows = [(int(row["anon_id"]), row["absolute_path"]) for row in csv.DictReader(f)]
    return sorted(rows)


def load_labels():
    if not LABELS_PATH.exists():
        return {}
    labels = {}
    with open(LABELS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[int(row["anon_id"])] = row
    return labels


def save_labels(labels: dict):
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["anon_id", "genel_kalite"] + FLAG_NAMES
    with open(LABELS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for anon_id in sorted(labels):
            writer.writerow(labels[anon_id])


MAPPING = load_mapping()
PATH_BY_ID = dict(MAPPING)
ORDER = [anon_id for anon_id, _ in MAPPING]

# Basit, dosyaya yazılmayan oturum durumu (tek kullanıcı için yeterli).
_state = {"pos": 0, "history": []}


def _first_unlabeled_pos(labels: dict) -> int:
    for i, anon_id in enumerate(ORDER):
        if anon_id not in labels:
            return i
    return max(len(ORDER) - 1, 0)


TEMPLATE = """
<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<title>Etiketleme</title>
<style>
  :root { color-scheme: dark; }
  body { background:#111318; color:#e8e8ec; font-family:-apple-system,'Segoe UI',sans-serif;
         margin:0; padding:24px; display:flex; flex-direction:column; align-items:center; }
  .top { width:100%; max-width:900px; display:flex; justify-content:space-between;
         align-items:center; margin-bottom:14px; font-size:14px; color:#9a9ba5; }
  .progress-bar { width:100%; max-width:900px; height:6px; background:#23252e; border-radius:3px;
                   overflow:hidden; margin-bottom:18px; }
  .progress-fill { height:100%; background:#5b8def; }
  img.doc { max-width:900px; max-height:70vh; border-radius:10px; box-shadow:0 4px 24px rgba(0,0,0,.5);
            display:block; }
  .flags { display:flex; flex-wrap:wrap; gap:10px; justify-content:center; margin:20px 0; }
  .flag-btn { padding:12px 18px; border-radius:10px; border:2px solid #33353f; background:#1a1c23;
              color:#e8e8ec; font-size:15px; cursor:pointer; user-select:none; transition:.12s; }
  .flag-btn.active { background:#5b8def; border-color:#5b8def; color:#0d0f14; font-weight:600; }
  .quality-row { display:flex; gap:14px; margin-top:10px; }
  .q-btn { padding:16px 30px; font-size:17px; border-radius:12px; border:none; cursor:pointer;
           font-weight:700; color:#fff; }
  .q-kotu { background:#e05252; } .q-orta { background:#d9a441; color:#1a1400; }
  .q-iyi { background:#4caf7d; }
  .q-btn:hover { filter:brightness(1.1); }
  .controls { display:flex; gap:12px; margin-top:22px; }
  .ctrl-btn { padding:10px 18px; border-radius:8px; border:1px solid #33353f; background:#1a1c23;
              color:#c9cad2; cursor:pointer; font-size:14px; }
  .ctrl-btn:hover { border-color:#5b8def; color:#fff; }
  .hint { color:#6b6c78; font-size:13px; margin-top:20px; text-align:center; max-width:600px; }
  .done { text-align:center; margin-top:80px; }
</style>
</head>
<body>
{% if finished %}
  <div class="done">
    <h1>🎉 Tüm görüntüler etiketlendi!</h1>
    <p>{{ done }}/{{ total }} tamamlandı.</p>
    <p class="hint">Şimdi 3. adıma geçebilirsin: <code>python3 experiments/real_data/3_batch_score.py</code></p>
  </div>
{% else %}
  <div class="top">
    <span>#{{ anon_id }} — ({{ idx }}/{{ total }})</span>
    <span>{{ done }} etiketlendi</span>
  </div>
  <div class="progress-bar"><div class="progress-fill" style="width:{{ pct }}%"></div></div>

  <img class="doc" src="{{ url_for('image', anon_id=anon_id) }}" alt="belge">

  <form method="post" action="{{ url_for('label') }}" id="labelForm">
    <input type="hidden" name="anon_id" value="{{ anon_id }}">
    <div class="flags">
      {% for f in flags %}
        <div class="flag-btn {{ 'active' if f in current_flags else '' }}" data-flag="{{ f }}"
             onclick="toggleFlag(this)">{{ flag_labels[f] }}</div>
        <input type="checkbox" name="{{ f }}" id="chk_{{ f }}"
               {{ 'checked' if f in current_flags else '' }} style="display:none">
      {% endfor %}
    </div>
    <div class="quality-row">
      <button type="submit" name="quality" value="kotu" class="q-btn q-kotu">Kötü</button>
      <button type="submit" name="quality" value="orta" class="q-btn q-orta">Orta</button>
      <button type="submit" name="quality" value="iyi" class="q-btn q-iyi">İyi</button>
    </div>
  </form>

  <div class="controls">
    <form method="post" action="{{ url_for('back') }}" id="backForm"><button class="ctrl-btn" {{ 'disabled' if not can_back else '' }}>◀ Önceki</button></form>
    <form method="post" action="{{ url_for('skip') }}" id="skipForm"><button class="ctrl-btn">Atla ▶</button></form>
    <button type="button" class="ctrl-btn" onclick="clearFlags()">Bayrakları temizle</button>
  </div>

  <p class="hint">
    Önce üstteki bayraklardan gördüğün sorunları işaretle (istersen hiç işaretlemeyebilirsin),
    sonra Kötü / Orta / İyi'ye tıkla — bu otomatik kaydeder ve sıradaki görüntüye geçer.
    İstediğin an sekmeyi kapatabilirsin, ilerleme kaybolmaz.<br>
    Klavye: <b>1</b>/<b>2</b>/<b>3</b> = Kötü/Orta/İyi &nbsp;·&nbsp;
    <b>←</b> = önceki &nbsp;·&nbsp; <b>→</b> = atla
  </p>

  <script>
    function toggleFlag(el) {
      el.classList.toggle('active');
      const chk = document.getElementById('chk_' + el.dataset.flag);
      chk.checked = el.classList.contains('active');
    }
    function clearFlags() {
      document.querySelectorAll('.flag-btn').forEach((el) => {
        el.classList.remove('active');
        document.getElementById('chk_' + el.dataset.flag).checked = false;
      });
    }
    document.addEventListener('keydown', (e) => {
      const qualityMap = {'1':'kotu','2':'orta','3':'iyi'};
      if (qualityMap[e.key]) {
        document.querySelector(`button[value="${qualityMap[e.key]}"]`).click();
        return;
      }
      if (e.key === 'ArrowLeft') {
        const btn = document.querySelector('#backForm button');
        if (!btn.disabled) document.getElementById('backForm').submit();
      } else if (e.key === 'ArrowRight') {
        document.getElementById('skipForm').submit();
      }
    });
  </script>
{% endif %}
</body>
</html>
"""


@app.route("/")
def index():
    labels = load_labels()
    if not ORDER:
        return "anon_mapping.csv bulunamadı ya da boş — önce 1_prepare_dataset.py çalıştırılmalı.", 400

    if _state["pos"] >= len(ORDER):
        return render_template_string(TEMPLATE, finished=True, done=len(labels), total=len(ORDER))

    anon_id = ORDER[_state["pos"]]
    existing = labels.get(anon_id)
    current_flags = {f for f in FLAG_NAMES if existing and existing.get(f) == "1"}

    return render_template_string(
        TEMPLATE,
        finished=False,
        anon_id=anon_id,
        idx=_state["pos"] + 1,
        total=len(ORDER),
        done=len(labels),
        pct=round(100 * len(labels) / len(ORDER), 1),
        flags=FLAG_NAMES,
        flag_labels=FLAG_LABELS,
        current_flags=current_flags,
        can_back=bool(_state["history"]),
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
    labels = load_labels()
    anon_id = int(request.form["anon_id"])
    quality = request.form.get("quality")
    if quality not in QUALITY_LABELS:
        return redirect(url_for("index"))

    row = {"anon_id": anon_id, "genel_kalite": quality}
    for f in FLAG_NAMES:
        row[f] = "1" if request.form.get(f) else "0"
    labels[anon_id] = row
    save_labels(labels)  # her onayda diske yaz — ilerleme hiç kaybolmaz

    _state["history"].append(_state["pos"])
    _state["pos"] = min(_state["pos"] + 1, len(ORDER))
    return redirect(url_for("index"))


@app.route("/skip", methods=["POST"])
def skip():
    _state["history"].append(_state["pos"])
    _state["pos"] = min(_state["pos"] + 1, len(ORDER))
    return redirect(url_for("index"))


@app.route("/back", methods=["POST"])
def back():
    if _state["history"]:
        _state["pos"] = _state["history"].pop()
    return redirect(url_for("index"))


def main():
    labels = load_labels()
    if not ORDER:
        print(f"Önce 1_prepare_dataset.py çalıştırılmalı ({MAPPING_PATH} yok).")
        return
    _state["pos"] = _first_unlabeled_pos(labels)
    # Zaten etiketlenmiş görüntülere de "◀ Önceki" ile dönülebilsin diye
    # geçmişi önceden dolduruyoruz (sunucu yeniden başlasa bile).
    _state["history"] = list(range(_state["pos"]))
    print(f"Toplam {len(ORDER)} görüntü, {len(labels)} tanesi zaten etiketli.")
    print("Tarayıcıda aç -> http://127.0.0.1:5050")
    print("Durdurmak için Ctrl+C.")
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
