import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# --- CONFIGURATION ---
DATA_FILE = 'Housing.csv'  # Ton fichier de données doit être ici
MODEL_FILE = 'model.pkl'

# --- FONCTION POUR ENTRAÎNER LE MODÈLE ---
def train_and_save_model():
    """
    Cette fonction lit le CSV, entraîne le modèle et sauvegarde le fichier pkl.
    Elle est appelée au démarrage ou quand on clique sur 'Actualiser'.
    """
    try:
        if not os.path.exists(DATA_FILE):
            return False, "Fichier CSV introuvable."

        # 1. Chargement des données
        df = pd.read_csv(DATA_FILE)

        # 2. Prétraitement (Adapté à ton dataset Housing)
        # On garde les colonnes numériques principales pour l'exemple
        # Assure-toi que ces colonnes existent dans ton CSV
        features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
        
        # Gestion simple des erreurs si colonnes manquantes
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            return False, "Colonnes incorrectes dans le CSV."

        X = df[available_features]
        y = df['price'] # Assure-toi que la colonne cible s'appelle 'price'

        # 3. Entraînement
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 4. Sauvegarde
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
            
        score = model.score(X_test, y_test)
        return True, f"Modèle actualisé avec succès ! Précision (R2): {round(score, 2)}"
    
    except Exception as e:
        return False, str(e)

# --- ROUTES FLASK ---

@app.route('/')
def home():
    # On vérifie si le modèle existe, sinon on l'entraîne
    if not os.path.exists(MODEL_FILE):
        success, msg = train_and_save_model()
        if not success:
            return f"Erreur critique : {msg}"
    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Charger le modèle
        model = pickle.load(open(MODEL_FILE, 'rb'))
        
        # Récupérer les valeurs
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        
        # Prédire
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        
        # Formatage du prix (ex: 1 000 000 DH)
        output_formatted = "{:,.2f}".format(output).replace(",", " ")

        return render_template('index.html', 
                               prediction_text=f'Prix estimé : {output_formatted} DH',
                               scroll='prediction') # Pour scroller vers le résultat
    except Exception as e:
        return render_template('index.html', prediction_text=f'Erreur : {str(e)}')

@app.route('/update_model', methods=['POST'])
def update_model():
    # C'est ici que le bouton "Actualiser" nous amène
    success, message = train_and_save_model()
    
    if success:
        return render_template('index.html', update_text=message, scroll='update')
    else:
        return render_template('index.html', update_text=f"Erreur: {message}", scroll='update')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)