from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    message = ""

    if request.method == "POST":

        message = request.form["message"]

        if message.strip():

            # Convert text into numerical features
            message_vector = vectorizer.transform([message])

            # Predict
            prediction = model.predict(message_vector)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)