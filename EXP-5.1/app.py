from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

DATA_PATH = Path(__file__).with_name("wine.data.csv")
TARGET = "Class"
FEATURES = [
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]


def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Put wine.data.csv next to app.py."
        )

    data = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + [TARGET]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    model.fit(data[FEATURES], data[TARGET])
    return model


model = train_model()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {feature: float(request.form[feature]) for feature in FEATURES}
        input_data = pd.DataFrame([values], columns=FEATURES)
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(max(probabilities)) * 100

        return render_template(
            "index.html",
            prediction_text=f"Predicted wine class: {prediction}",
            confidence_text=f"Model confidence: {confidence:.1f}%",
        )
    except (KeyError, TypeError, ValueError) as error:
        return render_template("index.html", error_text=f"Please check your inputs: {error}"), 400


if __name__ == "__main__":
    app.run(debug=True)
