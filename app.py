from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Charge ton modèle
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Récupère les données du formulaire HTML
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Fait la prédiction
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Prix estimé: {} DH'.format(output))

if __name__ == "__main__":
    # host='0.0.0.0' est OBLIGATOIRE pour Docker
    app.run(host='0.0.0.0', port=5000)