from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

MODEL_PATH = "naive_bayes_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a college message."}), 400

    features = vectorizer.transform([message])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    confidence = float(max(probabilities) * 100)
    return jsonify({
        "message": message,
        "prediction": prediction,
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
