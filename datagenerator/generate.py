import pandas as pd
import numpy as np
import time
import os
import random
import json

# Chemins des fichiers
JSON_FILE = '/app/data/latest_house.json'
CSV_FILE = '/app/data/housing.csv'

def generate_house():
    """
    Génère une maison avec les colonnes exactes du fichier cleaned_housing_data.csv
    Colonnes : area, bedrooms, bathrooms, stories, parking, mainroad, price
    """
    # 1. Génération des caractéristiques (Features)
    area = random.randint(1650, 16200)       # Surface selon tes données min/max
    bedrooms = random.randint(1, 6)
    bathrooms = random.randint(1, 4)
    stories = random.randint(1, 4)
    parking = random.randint(0, 3)
    mainroad = random.choice([0, 1])         # 0 = Non, 1 = Oui

    # 2. Calcul du prix réaliste (basé sur une régression de tes données réelles)
    # Ces coefficients viennent de l'analyse de ton fichier cleaned_housing_data.csv
    coef_const = -60000
    coef_area = 305
    coef_bedrooms = 205000
    coef_bathrooms = 1150000
    coef_stories = 507000
    coef_parking = 343000
    coef_mainroad = 654000

    # Prix théorique
    price_estim = (coef_const + 
                   (area * coef_area) + 
                   (bedrooms * coef_bedrooms) + 
                   (bathrooms * coef_bathrooms) + 
                   (stories * coef_stories) + 
                   (parking * coef_parking) + 
                   (mainroad * coef_mainroad))
    
    # Ajout d'un peu de bruit (aléatoire) pour que ce ne soit pas trop parfait
    # Variation de +/- 10%
    noise = random.uniform(-0.10, 0.10) * price_estim
    final_price = int(price_estim + noise)

    # S'assurer que le prix est positif
    if final_price < 1750000: final_price = 1750000

    # Retourne un dictionnaire (format JSON)
    return {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad": mainroad,
        "price": final_price
    }

def save_to_json(data, filepath):
    """Sauvegarde les données dans un fichier JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"-> Données sauvegardées en JSON : {filepath}")

def convert_json_to_csv(json_filepath, csv_filepath):
    """Lit le JSON et l'ajoute au fichier CSV."""
    # Lire le JSON
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    
    # Convertir en DataFrame pandas (une seule ligne)
    df = pd.DataFrame([data])
    
    # Vérifier si le CSV existe pour écrire l'en-tête ou non
    file_exists = os.path.isfile(csv_filepath)
    
    # Ajouter au CSV (mode 'a' pour append)
    df.to_csv(csv_filepath, mode='a', header=not file_exists, index=False)
    print(f"-> Converti et ajouté au CSV : {csv_filepath}")

def main():
    print("Démarrage du générateur de données v2...")
    print(f"Simulation des données basée sur 'cleaned_housing_data.csv'")
    
    # Créer le dossier data s'il n'existe pas
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    while True:
        # Étape 1 : Générer la donnée en mémoire
        house_data = generate_house()
        
        # Étape 2 : Créer le fichier JSON
        save_to_json(house_data, JSON_FILE)
        
        # Étape 3 : Convertir ce JSON en CSV (Append)
        convert_json_to_csv(JSON_FILE, CSV_FILE)
        
        print("-" * 30)
        
        # Attendre 5 secondes
        time.sleep(5)

if __name__ == "__main__":
    main()