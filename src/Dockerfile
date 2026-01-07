# Image Python
FROM python:3.10-slim


WORKDIR /app

# Copier les dépendances
COPY requirements.txt .

# Installer les librairies
RUN pip install --no-cache-dir -r requirements.txt

# Copier TOUS les fichiers du projet
COPY . .

# Commande d'exécution
CMD ["python", "train.py"]
