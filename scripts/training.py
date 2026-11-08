"""Training + evaluasi model promosi karyawan.

Jalankan dari root project:
    python scripts/training.py

Output (folder models/):
    preprocessor.pkl  -> ColumnTransformer (imputasi + encoding)
    model.pkl         -> GradientBoostingClassifier
    metadata.json     -> metrik, threshold, kategori, versi library
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "human_capital.csv"
MODELS_DIR = ROOT / "models"
DATA_URL = ("https://raw.githubusercontent.com/densaiko/data_science_learning/"
            "main/dataset/Human%20Capital.csv")

SEED = 42

# ---- Skema fitur (satu sumber kebenaran, ikut disimpan ke metadata.json) ----
CATEGORICAL = {
    "department": ["Sales & Marketing", "Operations", "Technology", "Analytics",
                   "R&D", "Procurement", "Finance", "HR", "Legal"],
    "region": [f"region_{i}" for i in range(1, 35)],
    "education": ["Below Secondary", "Bachelor's", "Master's & above"],
    "gender": ["Male", "Female"],
    "recruitment_channel": ["Referred", "Sourcing", "Others"],
}
NUMERIC = {  # nama: (min, max)
    "no_of_trainings": (1, 10),
    "age": (20, 60),
    "previous_year_rating": (1, 5),
    "length_of_service": (1, 37),
    "awards_won": (0, 1),
    "avg_training_score": (0, 100),
}
FEATURES = list(CATEGORICAL) + list(NUMERIC)
TARGET = "is_promoted"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print(f"Mengunduh dataset ke {DATA_PATH} ...")
        DATA_PATH.parent.mkdir(exist_ok=True)
        pd.read_csv(DATA_URL).to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Samakan label dataset mentah dengan label yang dipakai API/UI."""
    df = df.copy()
    df["gender"] = df["gender"].map({"m": "Male", "f": "Female"})
    df["recruitment_channel"] = df["recruitment_channel"].map(
        {"referred": "Referred", "sourcing": "Sourcing", "other": "Others"})
    return df[FEATURES + [TARGET]]


def build_preprocessor() -> ColumnTransformer:
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=list(CATEGORICAL.values()))),
    ])
    num_pipe = SimpleImputer(strategy="median")
    return ColumnTransformer([
        ("cat", cat_pipe, list(CATEGORICAL)),
        ("num", num_pipe, list(NUMERIC)),
    ])


def best_threshold(y_true, proba) -> float:
    """Threshold yang memaksimalkan F1 (kelas promosi hanya ~8.5%)."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    return float(t[np.argmax(f1)])


def main() -> None:
    df = clean(load_data())
    X, y = df[FEATURES], df[TARGET]
    print(f"Data: {len(df):,} baris | rasio promosi: {y.mean():.2%}")

    # 60% train / 20% validation (cari threshold) / 20% test (evaluasi akhir)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)

    pre = build_preprocessor()
    Xt_train = pre.fit_transform(X_train)          # fit HANYA di data train
    Xt_val, Xt_test = pre.transform(X_val), pre.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4,
        subsample=0.8, random_state=SEED)
    model.fit(Xt_train, y_train)

    thr = best_threshold(y_val, model.predict_proba(Xt_val)[:, 1])
    proba_test = model.predict_proba(Xt_test)[:, 1]
    pred_default = (proba_test >= 0.5).astype(int)
    pred_tuned = (proba_test >= thr).astype(int)

    print("\n=== Evaluasi di data TEST ===")
    print(f"ROC-AUC           : {roc_auc_score(y_test, proba_test):.4f}")
    print(f"F1 @ threshold 0.5: {f1_score(y_test, pred_default):.4f}")
    print(f"F1 @ threshold {thr:.2f}: {f1_score(y_test, pred_tuned):.4f}")
    print(f"\nClassification report (threshold {thr:.2f}):")
    print(classification_report(y_test, pred_tuned,
                                target_names=["Not Promoted", "Promoted"]))
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, pred_tuned))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pre, MODELS_DIR / "preprocessor.pkl")
    joblib.dump(model, MODELS_DIR / "model.pkl")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": round(thr, 4),
        "metrics": {
            "roc_auc": round(float(roc_auc_score(y_test, proba_test)), 4),
            "f1_threshold_0.5": round(float(f1_score(y_test, pred_default)), 4),
            "f1_tuned": round(float(f1_score(y_test, pred_tuned)), 4),
        },
        "features": FEATURES,
        "categories": CATEGORICAL,
        "numeric_ranges": {k: list(v) for k, v in NUMERIC.items()},
        "versions": {"python": sys.version.split()[0],
                     "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nTersimpan di {MODELS_DIR}/ (model.pkl, preprocessor.pkl, metadata.json)")


if __name__ == "__main__":
    main()
