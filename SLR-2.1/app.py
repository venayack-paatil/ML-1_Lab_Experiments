from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("SLRModel.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    fuel_price = float(request.form['fuel_price'])

    prediction = model.predict(np.array([[fuel_price]]))

    return render_template(
        "index.html",
        prediction_text=f"Predicted Weekly Sales : {prediction[0]:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)