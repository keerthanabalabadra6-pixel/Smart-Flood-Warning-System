from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
from flask import Flask, flash, redirect, render_template, request, session, url_for
from config import METRICS_PATH, MODEL_PATH, DATASET_PATH
from src.prediction import predict_risk

app = Flask(__name__)
app.secret_key = __import__("os").environ.get("SECRET_KEY", "change-this-development-secret")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


def ensure_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("The trained model is not available. Run `python train_model.py --make-demo-data` before starting the website.")


def get_history():
    return session.get("prediction_history", [])


@app.route("/")
def home():
    history = get_history()
    latest = history[0] if history else None
    return render_template("index.html", history=history[:5], latest=latest, page="home")


@app.route("/predict", methods=["GET", "POST"])
def prediction():
    result = None
    values = {"rainfall_mm": "", "water_level_m": "", "temperature_c": "", "humidity_pct": "", "soil_moisture_pct": "", "recent_flood_history": "0"}
    if request.method == "POST":
        try:
            values = {key: request.form.get(key, "") for key in values}
            numeric = {
                "rainfall_mm": (float(values["rainfall_mm"]), 0, 1000),
                "water_level_m": (float(values["water_level_m"]), 0, 100),
                "temperature_c": (float(values["temperature_c"]), -20, 60),
                "humidity_pct": (float(values["humidity_pct"]), 0, 100),
                "soil_moisture_pct": (float(values["soil_moisture_pct"]), 0, 100),
            }
            for name, (number, minimum, maximum) in numeric.items():
                if not minimum <= number <= maximum:
                    raise ValueError(f"{name.replace('_', ' ').title()} must be between {minimum} and {maximum}.")
                values[name] = number
            if values["recent_flood_history"] not in ("0", "1"):
                raise ValueError("Choose Yes or No for recent flood history.")
            values["recent_flood_history"] = int(values["recent_flood_history"])
            result = predict_risk(values)
            record = {**values, **result, "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
            history = [record] + get_history()
            session["prediction_history"] = history[:30]
        except (ValueError, TypeError) as exc:
            flash(str(exc), "error")
        except (FileNotFoundError, RuntimeError) as exc:
            flash(str(exc), "error")
    return render_template("prediction.html", values=values, result=result, page="prediction")


@app.route("/monitoring")
def monitoring():
    return render_template("monitoring.html", page="monitoring")


@app.route("/alerts")
def alerts():
    return render_template("alerts.html", history=get_history(), page="alerts")


@app.route("/analytics")
def analytics():
    history = get_history()
    risk_counts = {}
    for record in history:
        risk_counts[record["risk"]] = risk_counts.get(record["risk"], 0) + 1
    chart_html = None
    if risk_counts:
        fig = px.bar(x=list(risk_counts.keys()), y=list(risk_counts.values()), labels={"x": "Predicted risk", "y": "Number of predictions"}, title="Your prediction history by risk level")
        chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    metrics = {}
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metrics = {}
    return render_template("analytics.html", history=history, risk_counts=risk_counts, chart_html=chart_html, metrics=metrics, page="analytics")


@app.post("/clear-history")
def clear_history():
    session.pop("prediction_history", None)
    flash("Prediction history cleared for this browser session.", "success")
    return redirect(url_for("analytics"))


@app.route("/safety")
def safety():
    return render_template("safety.html", page="safety")


@app.route("/about")
def about():
    return render_template("about.html", page="about")


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html", page=""), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
