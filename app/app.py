from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle entraîné
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking", "mainroad"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in FEATURES]
            prediction = model.predict(np.array([values]))[0]
        except:
            prediction = "Erreur dans les valeurs"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
