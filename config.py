from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATASET_PATH = DATA_DIR / "flood_dataset.csv"
MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
FEATURES = [
    "rainfall_mm",
    "water_level_m",
    "temperature_c",
    "humidity_pct",
    "soil_moisture_pct",
    "recent_flood_history",
]
TARGET = "flood_risk"
RISK_LABELS = ["Low", "Moderate", "High", "Critical"]
