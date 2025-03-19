import requests
import pandas as pd
import os
from git import Repo  

# 🔑 Clé API
API_KEY = "0a02712560b8b20b7f2bd451ce49a1891e86dc219b16e4d76a2a581fe90e6a05"

# 📌 Dictionnaire des ligues par pays
pays_ligue1 = {
    "France": "168",
    "Espagne": "302",
    "Allemagne": "175",
    "Angleterre": "152",
    "Italie": "207"
}

# 📁 Dossier de stockage des fichiers CSV
dossier_csv = "data_api"
os.makedirs(dossier_csv, exist_ok=True)  # Crée le dossier s'il n'existe pas

# 📊 Stocker les résultats sous forme de dictionnaire de DataFrames
dfs = {}

# 📌 Boucle pour récupérer et stocker les classements des ligues
for pays, league_id in pays_ligue1.items():
    url = f"https://apiv3.apifootball.com/?action=get_standings&league_id={league_id}&APIkey={API_KEY}"
    print(f"🔍 Requête API pour {pays}: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Vérifie les erreurs HTTP (4xx, 5xx)
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)  # Convertir en DataFrame
            dfs[pays] = df
            print(f"✅ Données reçues pour {pays}, {len(df)} équipes enregistrées.\n")
        else:
            print(f"⚠️ Données vides ou incorrectes pour {pays}: {data}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur de connexion pour {pays}: {e}")

# 📌 Sauvegarde des fichiers CSV
csv_files = []  # Stocker les fichiers à commit sur GitHub
for pays, df in dfs.items():
    csv_filename = f"{pays}_api.csv"
    csv_path = os.path.join(dossier_csv, csv_filename)

    try:
        df.to_csv(csv_path, index=False)
        csv_files.append(csv_path)  # Ajouter le fichier pour Git
        print(f"📁 Fichier sauvegardé : {csv_path}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du fichier {csv_filename} : {e}")

# 🔹 Ajouter, commettre et pousser sur GitHub une seule fois (optimisé)
try:
    save_path = os.getcwd()
    repo = Repo(save_path)  # Charge le repo GitHub dans le dossier courant
    origin = repo.remote(name='origin')

    # Ajouter uniquement les nouveaux fichiers CSV
    for file_path in csv_files:
        repo.git.add(file_path)

    # Commit unique
    repo.index.commit("📊 Mise à jour des fichiers CSV des standings")

    # Push vers GitHub
    origin.push()
    print("🚀 Fichiers CSV mis à jour sur GitHub avec succès !")

except Exception as e:
    print(f"❌ Erreur lors de la mise à jour sur GitHub : {e}")
