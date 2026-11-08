"""Flask API untuk prediksi promosi karyawan.

Lokal    : python api/index.py     (http://127.0.0.1:5000)
Vercel   : otomatis memakai variabel `app` di file ini.
"""
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from flask import Flask, jsonify, request

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

app = Flask(__name__)

# Model dimuat SEKALI saat proses start (bukan di setiap request).
preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
model = joblib.load(MODELS_DIR / "model.pkl")
meta = json.loads((MODELS_DIR / "metadata.json").read_text())

FEATURES = meta["features"]
CATEGORIES = meta["categories"]
RANGES = meta["numeric_ranges"]
THRESHOLD = meta["threshold"]

trained_with = meta["versions"]["scikit-learn"]
if trained_with != sklearn.__version__:
    warnings.warn(f"scikit-learn training={trained_with} vs serving="
                  f"{sklearn.__version__}: samakan versinya!")


def validate(payload) -> list[str]:
    """Kembalikan daftar pesan error (kosong = valid)."""
    if not isinstance(payload, dict):
        return ["Body harus berupa JSON object."]
    errors = []
    for name in FEATURES:
        if name not in payload:
            errors.append(f"'{name}' wajib diisi.")
            continue
        value = payload[name]
        if name in CATEGORIES:
            if value not in CATEGORIES[name]:
                errors.append(f"'{name}' harus salah satu dari {CATEGORIES[name]}.")
        else:
            lo, hi = RANGES[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not lo <= value <= hi:
                errors.append(f"'{name}' harus angka {lo}-{hi}.")
    return errors


@app.get("/")
def index():
    return jsonify(service="hr-promotion-api",
                   endpoints=["GET /health", "GET /schema", "POST /predict"])


@app.get("/health")
def health():
    return jsonify(status="ok", trained_at=meta["trained_at"],
                   metrics=meta["metrics"], threshold=THRESHOLD)


@app.get("/schema")
def schema():
    """Dipakai Streamlit untuk membangun form tanpa menduplikasi daftar opsi."""
    return jsonify(features=FEATURES, categories=CATEGORIES, numeric_ranges=RANGES)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    errors = validate(payload)
    if errors:
        return jsonify(error="validation_error", details=errors), 400

    row = pd.DataFrame([{name: payload[name] for name in FEATURES}])
    proba = float(model.predict_proba(preprocessor.transform(row))[0, 1])
    promoted = proba >= THRESHOLD
    return jsonify(
        prediction=int(promoted),
        label="Promoted" if promoted else "Not Promoted",
        probability=round(proba, 4),
        threshold=THRESHOLD,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
