from src.model import load_model
from config import FEATURES

RISK_MESSAGES = {
    "Low": ("Conditions indicate a lower predicted risk.", "Continue monitoring local conditions and official weather updates."),
    "Moderate": ("Conditions indicate a moderate predicted risk.", "Stay alert, review local advisories and prepare essential items."),
    "High": ("Conditions indicate a high predicted risk.", "Avoid low-lying areas, check official alerts and prepare to move to safety if instructed."),
    "Critical": ("Conditions indicate a critical predicted risk.", "Move to a safer or higher location if advised by authorities. Follow official emergency instructions."),
}


def predict_risk(values):
    model = load_model()
    row = [[float(values[name]) for name in FEATURES]]
    prediction = str(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    classes = model.named_steps["classifier"].classes_
    confidence = {str(label): round(float(prob) * 100, 1) for label, prob in zip(classes, probabilities)}
    message, advice = RISK_MESSAGES.get(prediction, RISK_MESSAGES["Moderate"])
    return {"risk": prediction, "confidence": confidence, "message": message, "advice": advice}
