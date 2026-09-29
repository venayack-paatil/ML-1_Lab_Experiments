from flask import Flask, render_template, request
import json
import pickle
from pathlib import Path

import numpy as np

app = Flask(__name__)

MODEL_PATH = Path(__file__).with_name("MLRModel.pkl")

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Keep these names and order consistent with the trained model.
        values = [
            float(request.form["Holiday_Flag"]),
            float(request.form["Temperature"]),
            float(request.form["Fuel_Price"]),
            float(request.form["CPI"]),
            float(request.form["Unemployment"]),
        ]

        scaled_values = (
            np.asarray(values) - np.asarray(model["means"])
        ) / np.asarray(model["scales"])

        prediction = model["intercept"] + np.dot(
            scaled_values,
            np.asarray(model["coefficients"])
        )

        prediction_text = f"Predicted Weekly Sales: ${prediction:,.2f}"

        return render_template(
            "index.html",
            prediction_text=prediction_text
        )

    except (KeyError, TypeError, ValueError) as error:
        return render_template(
            "index.html",
            error_text=f"Please check the entered values: {error}"
        ), 400


if __name__ == "__main__":
    app.run(debug=True)