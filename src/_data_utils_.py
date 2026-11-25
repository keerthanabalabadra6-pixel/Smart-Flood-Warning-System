from pathlib import Path
import numpy as np
import pandas as pd
from config import DATASET_PATH, FEATURES, TARGET


def create_demo_dataset(path: Path = DATASET_PATH, rows: int = 1600, seed: int = 42) -> Path:
    """Create a synthetic dataset for software demonstration only.

    Replace this file with a verified, locally relevant dataset before making
    scientific or operational claims about prediction performance.
    """
    rng = np.random.default_rng(seed)
    rainfall = rng.gamma(shape=2.2, scale=28.0, size=rows).clip(0, 260)
    water = (0.35 + rainfall / 105 + rng.normal(0, 0.45, rows)).clip(0.1, 8.0)
    temperature = rng.normal(28, 5, rows).clip(8, 45)
    humidity = rng.normal(72, 16, rows).clip(20, 100)
    soil = (rainfall / 2.8 + rng.normal(35, 18, rows)).clip(0, 100)
    history = rng.binomial(1, 0.22, rows)

    # Synthetic labels intentionally represent a simple demonstration rule.
    score = (rainfall / 95) + (water / 2.8) + (humidity / 180) + (soil / 160) + history * 0.8
    labels = np.select(
        [score < 2.0, score < 3.1, score < 4.2],
        ["Low", "Moderate", "High"],
        default="Critical",
    )
    frame = pd.DataFrame({
        "rainfall_mm": rainfall.round(2),
        "water_level_m": water.round(2),
        "temperature_c": temperature.round(2),
        "humidity_pct": humidity.round(2),
        "soil_moisture_pct": soil.round(2),
        "recent_flood_history": history,
        "flood_risk": labels,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run `python train_model.py --make-demo-data` "
            "to create a synthetic demonstration dataset."
        )
    frame = pd.read_csv(path)
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    frame = frame[FEATURES + [TARGET]].copy()
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[TARGET] = frame[TARGET].astype("string").str.strip().str.title()
    frame = frame.dropna(subset=FEATURES + [TARGET])
    frame = frame[frame[TARGET].isin(["Low", "Moderate", "High", "Critical"])]
    if len(frame) < 20:
        raise ValueError("Dataset must contain at least 20 valid rows after cleaning.")
    return frame
