import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Chargement du modèle
# On utilise la méthode simple qui fonctionne bien
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Fonction de prédiction
    '''
    try:
        # Récupère les valeurs du formulaire
        # Note : Assure-toi que l'ordre des inputs dans ton HTML correspond à l'entraînement
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        
        return render_template('index.html', prediction_text='Prix estimé: {} DH'.format(output))
    except Exception as e:
        return render_template('index.html', prediction_text='Erreur: Vérifiez les valeurs saisies')

if __name__ == "__main__":
    # TRES IMPORTANT POUR DOCKER : host='0.0.0.0'
    # Sans ça, ton site sera inaccessible depuis l'extérieur du conteneur
    app.run(host='0.0.0.0', port=5000)