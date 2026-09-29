from pathlib import Path
import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).with_name("BanknoteModel.pkl")

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Keep this order the same as the order used to train the model.
        features = [
            float(request.form["variance"]),
            float(request.form["skewness"]),
            float(request.form["curtosis"]),
            float(request.form["entropy"]),
        ]

        prediction = int(model.predict(np.array([features]))[0])

        if prediction == 0:
            result = "Prediction: Class 0"
        else:
            result = "Prediction: Class 1"

        return render_template("index.html", prediction_text=result)

    except (KeyError, TypeError, ValueError) as error:
        return render_template(
            "index.html",
            error_text=f"Please enter valid values for all four features. {error}"
        ), 400


if __name__ == "__main__":
    app.run(debug=True)