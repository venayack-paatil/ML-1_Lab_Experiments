
from flask import Flask, render_template, request
import pickle
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load the trained Linear SVM model
MODEL_PATH = Path(__file__).parent / "banknote_svm_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

FEATURES = ["variance", "skewness", "curtosis", "entropy"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    values = {feature: "" for feature in FEATURES}

    if request.method == "POST":
        try:
            # Read input values from the HTML form
            for feature in FEATURES:
                values[feature] = request.form.get(feature, "").strip()

            if any(not values[feature] for feature in FEATURES):
                raise ValueError("Please enter all four feature values.")

            input_values = [float(values[feature]) for feature in FEATURES]

            if not np.isfinite(input_values).all():
                raise ValueError("Please enter valid finite numbers.")

            # Predict using the trained Linear SVM pipeline
            result = int(model.predict([input_values])[0])

            if result == 0:
                prediction = {
                    "label": "Genuine Banknote",
                    "message": "The model classified this sample as genuine.",
                    "class": "genuine"
                }
            else:
                prediction = {
                    "label": "Counterfeit Banknote",
                    "message": "The model classified this sample as counterfeit.",
                    "class": "counterfeit"
                }

        except ValueError as exc:
            error = str(exc) or "Please enter valid numerical values."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)
