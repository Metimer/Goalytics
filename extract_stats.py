import zipfile
import requests
import io
import os
from git import Repo  

# URL du fichier ZIP des données de football
zipfile_url = {'2024-2025':'https://www.football-data.co.uk/mmz4281/2425/data.zip',
               '2023-2024': 'https://www.football-data.co.uk/mmz4281/2324/data.zip'
}
# Dossier de sortie pour extraire les fichiers
dossier_de_sortie = "data_football"
os.makedirs(dossier_de_sortie, exist_ok=True)  # Crée le dossier s'il n'existe pas
for key,value in zipfile_url.items():
    # Télécharger le fichier ZIP
    response = requests.get(value, timeout=30)

    # Vérifier que le téléchargement a réussi (code 200)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Lister les fichiers contenus dans l'archive
            zip_list = zip_ref.namelist()

            # Dictionnaire de correspondance pour renommer les fichiers
            correspondance_noms = {
                'D1.csv': f'allemagne_stats_score_{key}.csv',
                'E0.csv': f'angleterre_stats_score_{key}.csv',
                'F1.csv': f'france_stats_score_{key}.csv',
                'SP1.csv': f'espagne_stats_score_{key}.csv',
                'I1.csv': f'italie_stats_score_{key}.csv'
            }

            # Extraire et renommer les fichiers
            for fichier in zip_list:
                if fichier in correspondance_noms:
                    nouveau_nom = correspondance_noms[fichier]
                    chemin_sortie = os.path.join(dossier_de_sortie, nouveau_nom)

                    with zip_ref.open(fichier) as fichier_source, open(chemin_sortie, 'wb') as fichier_destination:
                        fichier_destination.write(fichier_source.read())

                    print(f"✅ Fichier extrait et renommé : {fichier} -> {chemin_sortie}")

        # 🔹 Ajouter, commettre et pousser sur GitHub
        try:
            # Charger le repo GitHub
            save_path = os.getcwd()  # Assure-toi que le script est dans le repo Git
            repo = Repo(save_path)  # Repos GitHub cloné dans le répertoire courant
            origin = repo.remote(name='origin')  # Définir le remote 'origin'

            # Ajouter chaque fichier extrait au suivi de Git
            for fichier in correspondance_noms.values():
                file_path = os.path.join(dossier_de_sortie, fichier)
                repo.git.add(file_path)  # Ajouter chaque fichier extrait

            # Commit des fichiers avec un message
            repo.index.commit("📊 Mise à jour des fichiers CSV extraits du ZIP")

            # Push les changements sur GitHub
            origin.push()  # Pousser les changements vers GitHub
            print("🚀 Fichiers CSV mis à jour sur GitHub avec succès !")

        except Exception as e:
            print(f"❌ Erreur lors de la mise à jour sur GitHub : {e}")

    else:
        print(f"❌ Échec du téléchargement du fichier. Code HTTP: {response.status_code}")

