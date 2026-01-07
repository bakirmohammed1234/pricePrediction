# Utiliser une image Python légère
FROM python:3.9-slim

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les librairies (pandas, numpy)
RUN pip install --no-cache-dir -r requirements.txt

# Copier le script python
COPY generate.py .

# Créer le dossier qui recevra les données
RUN mkdir -p /app/data

# Commande par défaut au lancement du conteneur
CMD ["python", "generate.py"]