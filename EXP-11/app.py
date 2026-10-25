import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

try:
    with open(BASE_DIR / "models.pkl", "rb") as file:
        MODELS = pickle.load(file)
    LOAD_ERROR = None
except (FileNotFoundError, pickle.UnpicklingError, EOFError) as error:
    MODELS = None
    LOAD_ERROR = str(error)


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/predict")
def predict():
    if MODELS is None:
        return jsonify(
            error="models.pkl is unavailable. Run train_model.py first.",
            detail=LOAD_ERROR
        ), 503

    values = {
        column: request.form.get(column, "").strip() or None
        for column in MODELS["feature_columns"]
    }

    for column in MODELS["numeric_columns"]:
        if values[column] is not None:
            try:
                values[column] = float(values[column])
            except ValueError:
                return jsonify(error=f"{column} must be a number."), 400

    vehicle = pd.DataFrame(
        [values],
        columns=MODELS["feature_columns"]
    )

    return jsonify(
        price_predictions={
            name: round(float(model.predict(vehicle)[0]), 2)
            for name, model in MODELS["regressors"].items()
        },
        price_categories={
            name: str(model.predict(vehicle)[0])
            for name, model in MODELS["classifiers"].items()
        },
        classifier_metrics=MODELS.get("classifier_metrics", {}),
        band_definition=MODELS["price_band_definition"],
    )


if __name__ == "__main__":
    app.run(debug=True)