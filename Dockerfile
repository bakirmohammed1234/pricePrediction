# 1. On part d'une version légère de Python
FROM python:3.9-slim

# 2. On crée un dossier de travail dans le conteneur
WORKDIR /app

# 3. On copie tout ton dossier actuel vers le conteneur
COPY . /app

# 4. On installe les librairies (Flask, Numpy...)
RUN pip install -r requirements.txt

# 5. On ouvre le port 5000
EXPOSE 5000

# 6. La commande pour lancer l'app quand le conteneur démarre
CMD ["python", "app.py"]