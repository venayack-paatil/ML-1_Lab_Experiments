
from flask import Flask, render_template, request
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# Project directory
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
MODEL_PATH = BASE_DIR / "dry_bean_nonlinear_svm_model.pkl"

# Configure Flask template location
app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR)
)

# Load trained model artifact
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Run train_model.py first."
    )

with open(MODEL_PATH, "rb") as file:
    artifact = pickle.load(file)

model = artifact["model"]
label_encoder = artifact["label_encoder"]
feature_names = artifact["feature_names"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    # Preserve submitted values in the form
    values = {feature: "" for feature in feature_names}

    if request.method == "POST":
        try:
            # Read all 16 features in the training order
            for feature in feature_names:
                values[feature] = request.form.get(
                    feature, ""
                ).strip()

            # Validate empty fields
            if any(not values[f] for f in feature_names):
                raise ValueError(
                    "Please fill in all 16 feature fields."
                )

            # Convert inputs to numbers
            numbers = [float(values[f]) for f in feature_names]

            if not np.isfinite(numbers).all():
                raise ValueError(
                    "Please enter valid finite numerical values."
                )

            # Create input with the correct feature names and order
            input_data = pd.DataFrame(
                [numbers],
                columns=feature_names
            )

            # Predict bean variety
            predicted_class = int(model.predict(input_data)[0])

            predicted_name = label_encoder.inverse_transform(
                [predicted_class]
            )[0]

            prediction = {
                "label": str(predicted_name),
                "message": (
                    f"The Non-Linear SVM predicts this bean "
                    f"variety is {predicted_name}."
                )
            }

        except (ValueError, TypeError) as exc:
            error = str(exc) or "Please enter valid numerical values."

    return render_template(
        "index.html",
        feature_names=feature_names,
        prediction=prediction,
        error=error,
        values=values
    )


if __name__ == "__main__":
    # Debug mode is suitable for local development only
    app.run(debug=True)
