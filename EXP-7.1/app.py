from pathlib import Path
import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("knn_model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

FEATURES = model["features"]
CLASS_LABELS = {"B": "Benign", "M": "Malignant"}


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", features=FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = np.asarray([float(request.form[name]) for name in FEATURES])
        scaled = (values - np.asarray(model["means"])) / np.asarray(model["scales"])
        training_x = np.asarray(model["training_features"])
        distances = np.linalg.norm(training_x - scaled, axis=1)
        nearest = np.argpartition(distances, model["k"] - 1)[: model["k"]]
        nearest_labels = np.asarray(model["training_labels"])[nearest]
        labels, counts = np.unique(nearest_labels, return_counts=True)
        winner = int(np.argmax(counts))
        predicted = str(labels[winner])
        confidence = 100.0 * counts[winner] / model["k"]

        return render_template(
            "index.html",
            features=FEATURES,
            prediction_text=f"Prediction: {CLASS_LABELS.get(predicted, predicted)}",
            confidence_text=f"Nearest-neighbor vote: {confidence:.0f}%",
        )
    except (KeyError, TypeError, ValueError) as error:
        return render_template(
            "index.html", features=FEATURES, error_text=f"Please check your inputs: {error}"
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
