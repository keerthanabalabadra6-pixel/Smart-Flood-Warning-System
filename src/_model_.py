import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from config import FEATURES, TARGET, MODEL_PATH, METRICS_PATH
from src.data_utils import load_dataset


def train_and_save(dataset_path=None):
    frame = load_dataset(Path(dataset_path) if dataset_path else __import__('config').DATASET_PATH)
    X, y = frame[FEATURES], frame[TARGET]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "test_rows": int(len(y_test)),
        "training_rows": int(len(y_train)),
        "classes": sorted(str(item) for item in y.unique()),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "dataset_note": "Synthetic demonstration data" if "synthetic" in str(dataset_path or "").lower() else "Check the dataset source before interpreting metrics.",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model is missing. Run `python train_model.py` first.")
    return joblib.load(MODEL_PATH)
