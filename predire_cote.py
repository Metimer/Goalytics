
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import os
from git import Repo

# Chargement des données
cotes_jour = pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/cotes_du_jour.csv')

all=pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_football/allemagne_stats_score_2024-2025.csv')
all1=pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_football/allemagne_stats_score_2023-2024.csv')
esp=pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_football/espagne_stats_score_2024-2025.csv')
esp1=pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_football/espagne_stats_score_2023-2024.csv')
fra= pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_football/france_stats_score_2024-2025.csv')
fra1=pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_football/france_stats_score_2023-2024.csv')
ita=pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_football/italie_stats_score_2024-2025.csv')
ita1=pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_football/italie_stats_score_2023-2024.csv')
ang=pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_football/angleterre_stats_score_2024-2025.csv')
ang1=pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_football/angleterre_stats_score_2023-2024.csv')

pays_cotes_scores = {
    'allemagne': pd.concat([all, all1], axis=0, ignore_index=True),
        'espagne': pd.concat([esp, esp1], axis=0, ignore_index=True),
        'france': pd.concat([fra, fra1], axis=0, ignore_index=True),
        'italie':pd.concat([ita, ita1], axis=0, ignore_index=True),
        'angleterre': pd.concat([ang, ang1], axis=0, ignore_index=True)
    }

pays_stats_scores = {
    'allemagne': pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_api/Allemagne_api.csv'),
    'espagne': pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_api/Espagne_api.csv'),
    'france': pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_api/France_api.csv'),
    'italie': pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_api/Italie_api.csv'),
    'angleterre': pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_api/Angleterre_api.csv')
}

# Correction de la normalisation des noms d'équipes
pays_stats_scores['france']['team_name'] = pays_stats_scores['france']['team_name'].str.replace('PSG', 'Paris SG')
pays_stats_scores['allemagne']['team_name'] = pays_stats_scores['allemagne']['team_name'].str.replace('B. Monchengladbach',"M'gladbach")
pays_stats_scores['allemagne']['team_name'] = pays_stats_scores['allemagne']['team_name'].str.replace('St. Pauli',"St Pauli")
pays_stats_scores['allemagne']['team_name'] = pays_stats_scores['allemagne']['team_name'].str.replace('Bayer Leverkusen',"Leverkusen")
pays_stats_scores['allemagne']['team_name'] = pays_stats_scores['allemagne']['team_name'].str.replace('Eintracht Frankfurt',"Ein Frankfurt")
pays_stats_scores['espagne']['team_name'] = pays_stats_scores['espagne']['team_name'].str.replace('Celta Vigo',"Celta")
pays_stats_scores['espagne']['team_name'] = pays_stats_scores['espagne']['team_name'].str.replace('Real Sociedad',"Sociedad")
pays_stats_scores['espagne']['team_name'] = pays_stats_scores['espagne']['team_name'].str.replace('Espanyol',"Espanol")
pays_stats_scores['espagne']['team_name'] = pays_stats_scores['espagne']['team_name'].str.replace('Atl.Madrid',"Ath Madrid")
pays_stats_scores['espagne']['team_name'] = pays_stats_scores['espagne']['team_name'].str.replace('Rayo Vallecano',"Vallecano")
pays_stats_scores['italie']['team_name'] = pays_stats_scores['italie']['team_name'].str.replace('AC Milan',"Milan")
pays_stats_scores['italie']['team_name'] = pays_stats_scores['italie']['team_name'].str.replace('AS Roma',"Roma")
pays_stats_scores['angleterre']['team_name'] = pays_stats_scores['angleterre']['team_name'].str.replace('Manchester Utd',"Man United")
pays_stats_scores['angleterre']['team_name'] = pays_stats_scores['angleterre']['team_name'].str.replace('Nottingham',"Nott'm Forest")
pays_stats_scores['angleterre']['team_name'] = pays_stats_scores['angleterre']['team_name'].str.replace('Manchester City',"Man City")


cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Hellas Vérone', 'Verona')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Parme', 'Parma')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Como 1907', 'Como')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Milan AC', 'Milan')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Lazio Rome', 'Lazio')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Venezia FC', 'Venezia')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('AC Monza', 'Monza')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Naples', 'Napoli')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Juventus Turin', 'Juventus')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Inter Milan', 'Inter')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Rayo Vallecano', 'Rayo')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Celta Vigo', 'Celta')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('FC Barcelone', 'Barcelona')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('CF Valence', 'Valencia')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Atl.Madrid', 'Ath Madrid')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Espanyol', 'Espanol')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Gérone', 'Girona')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Majorque', 'Mallorca')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Bétis Séville', 'Betis')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Real Sociedad', 'Sociedad')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Rayo Vallecano', 'Rayo')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('FC Séville', 'Sevilla')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('CD Leganes', 'Leganes')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('FC Heidenheim', 'Heidenheim')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Fribourg', 'Freiburg')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Mayence', 'Mainz')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Werder Brême', 'Werder Bremen')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Ein.Francfort', 'Ein Frankfurt')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace("M'Gladbach", "M'gladbach")
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Wolverhampton', 'Wolves')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace("Nottingham F.", "Nott'm Forest")
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Man. United', 'Man United')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Man. City', 'Man City')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Brighton Hove', 'Brighton')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('Bologne', 'Bologna')
cotes_jour[['Equipe Domicile','Equipe Extérieure']] = cotes_jour[['Equipe Domicile','Equipe Extérieure']].replace('AS Rome', 'Roma')
cotes_jour['Pays'] = cotes_jour['Pays'].str.lower()




# Nettoyage des données et création des datasets ML
pays_stats_ml = {}
pays_cotes_ml2 = {}

for pays, df in pays_stats_scores.items():
    pays_stats_ml[pays] = df.drop(
        ['fk_stage_key', 'stage_name', 'league_round', 'home_promotion', 'away_promotion', 
         'league_id','team_badge', 'team_id'],
        axis=1, errors='ignore'
    )
    
    
    # Vérification et remplissage des valeurs NaN pour 'overall_promotion'
    if 'overall_promotion' in pays_stats_ml[pays].columns:
        pays_stats_ml[pays]['overall_promotion'] = pays_stats_ml[pays]['overall_promotion'].fillna('No Qualification')
        
        # Vérification et encodage numérique
    if 'overall_promotion' in pays_stats_ml[pays].columns:
        pays_stats_ml[pays]['overall_promotion'],_ = pd.factorize(pays_stats_ml[pays]['overall_promotion'])

# Stocker les features des datasets
pays_features = pays_stats_ml['angleterre'].drop('team_name', axis=1, errors='ignore').columns.tolist() 

for payscotes, dfcotes in pays_cotes_scores.items():
    pays_cotes_ml2[payscotes] = dfcotes.drop(['Div', 'Date', 'Time','Attendance','Referee'], axis=1, errors='ignore').copy()
    
    
pays_final_ml = {}

for pays in pays_cotes_ml2.keys():
    df_cotes = pays_cotes_ml2[pays]
    df_stats = pays_stats_ml[pays]

    # Fusion des données en associant les stats des équipes "domicile" et "extérieur"
    df_merged = df_cotes.merge(df_stats, left_on="HomeTeam", right_on="team_name", how="left")
    df_merged = df_merged.merge(df_stats, left_on="AwayTeam", right_on="team_name", how="left", suffixes=("_home", "_away"))

    # Stocker dans un dictionnaire
    pays_final_ml[pays] = df_merged
    
    
pays_final_ml_encoded = {}

for pays, df in pays_final_ml.items():
    df_encoded = df.copy()  # Copie du dataframe original
    df_encoded.columns =df_encoded.columns.str.replace(r'[<>[\]]', '', regex=True)
    cols = pd.Series(df_encoded.columns)

    # Renommer les colonnes en double en ajoutant "_dup"
    for idx, col in enumerate(cols):
        if cols[:idx].eq(col).any():
            cols[idx] = f"{col}_dup"

    df_encoded.columns = cols
    
    # Sélection des colonnes catégorielles à encoder
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Exclure HomeTeam et AwayTeam
    categorical_cols = [col for col in categorical_cols if col not in ['HomeTeam', 'AwayTeam','country_name_home','country_name_away','league_name_home','league_name_away']]
    
    # Encodage avec LabelEncoder pour chaque colonne catégorielle
    label_encoders = {}  
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le  # Stocker l'encodeur si besoin de retrouver les valeurs originales

    # Stocker le dataframe encodé
    pays_final_ml_encoded[pays] = df_encoded
    
    
df_final = pd.concat(pays_final_ml_encoded.values(), ignore_index=True)

categorical_cols2 = ['league_name_home', 'league_name_away', 'country_name_home', 'country_name_away']
label_encoders2 = {}  # Pour stocker les encodeurs si besoin de conversion inverse
for col in categorical_cols2:
    le = LabelEncoder()
    df_final[col] = le.fit_transform(df_final[col].astype(str))
    label_encoders2[col] = le  # Stocker l'encodeur si besoin de retrouver les valeurs originales    
    
for col in pays_features:
    df_final[f"diff_{col}"] = df_final[f"{col}_home"] - df_final[f"{col}_away"]
    

features_name=df_final.drop(['B365H','B365D','B365A','HomeTeam','AwayTeam','league_name_home', 'league_name_away', 'country_name_home', 'country_name_away'],axis=1).columns

# Définition des features et des cibles
X = df_final[features_name]

y_home = df_final["B365H"]  
y_draw = df_final["B365D"]  
y_away = df_final["B365A"]  

# Division des données en train et test
X_train, X_test, y_train_home, y_test_home, y_train_draw, y_test_draw, y_train_away, y_test_away = train_test_split(
    X, y_home, y_draw, y_away, test_size=0.2, random_state=42
)

# Grille de paramètres pour RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'learning_rate': np.linspace(0.01, 0.3, 5),
    'max_depth': np.arange(3, 11),
    'subsample': np.linspace(0.6, 1.0, 5),
    'colsample_bytree': np.linspace(0.6, 1.0, 5)
}

# Initialisation du modèle
xgb = XGBRegressor()

# RandomizedSearchCV avec 20 itérations aléatoires
random_search_home = RandomizedSearchCV(xgb, param_distributions=param_dist, 
                                        n_iter=20, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)

random_search_draw = RandomizedSearchCV(xgb, param_distributions=param_dist, 
                                        n_iter=20, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)

random_search_away = RandomizedSearchCV(xgb, param_distributions=param_dist, 
                                        n_iter=20, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)

# Entraînement avec RandomizedSearchCV
random_search_home.fit(X_train, y_train_home)
random_search_draw.fit(X_train, y_train_draw)
random_search_away.fit(X_train, y_train_away)

# Meilleurs hyperparamètres trouvés
best_params_home = random_search_home.best_params_
best_params_draw = random_search_draw.best_params_
best_params_away = random_search_away.best_params_

# Réentraîner avec les meilleurs paramètres
model_home = XGBRegressor(**best_params_home)
model_draw = XGBRegressor(**best_params_draw)
model_away = XGBRegressor(**best_params_away)

model_home.fit(X_train, y_train_home)
model_draw.fit(X_train, y_train_draw)
model_away.fit(X_train, y_train_away)





def predire_cotes(equipe_domicile, equipe_exterieur, pays, df_final, xgb_home, xgb_draw, xgb_away, pays_stats_scores):
    
    # Récupération de la position de chaque équipe
    position_home = pays_stats_scores[pays].loc[pays_stats_scores[pays]['team_name'] == equipe_domicile, 'overall_league_position'].values
    position_away = pays_stats_scores[pays].loc[pays_stats_scores[pays]['team_name'] == equipe_exterieur, 'overall_league_position'].values
    
    # Vérifier si les positions existent
    if position_home.size == 0 or position_away.size == 0:
        return {
            'Cote domicile': None,
            'Cote nulle': None,
            'Cote extérieur': None
        }
    
    position_home = position_home.item()
    position_away = position_away.item()

    # Extraction des données des deux équipes
    stats_domicile = df_final[df_final['HomeTeam'] == equipe_domicile]
    stats_exterieur = df_final[df_final['AwayTeam'] == equipe_exterieur]

    # Création du DataFrame des caractéristiques pour la prédiction
    df_match = pd.concat([stats_exterieur, stats_domicile])

    # Suppression des colonnes inutiles pour la prédiction
    df_match = df_match.drop(['HomeTeam', 'AwayTeam', 'league_name_home', 'league_name_away', 
                              'country_name_home', 'country_name_away', 'B365H', 'B365A', 'B365D'], 
                              axis=1, errors='ignore')

    # Vérification que df_match n'est pas vide avant la prédiction
    if df_match.empty:
        return {
            'Cote domicile': None,
            'Cote nulle': None,
            'Cote extérieur': None
        }

    # Faire la prédiction des cotes
    y_pred_home = model_home.predict(df_match)
    y_pred_draw = model_draw.predict(df_match)
    y_pred_away = model_away.predict(df_match)

    # Appliquer la pondération en fonction du classement
    cote_dom = y_pred_home[0]
    cote_ext = y_pred_away[0]
    
    diff = abs(position_home - position_away)

    if position_home < position_away:  
        if 10> diff > 5:
            cote_dom *= 0.85
            cote_ext *= 1.75
        elif 10<diff:
            cote_dom *= 0.65     
            cote_ext *= 1.90
        else:
            cote_dom *= 0.80
            cote_ext *= 1.30
    else:  
        if 10>diff > 5:
            cote_ext *= 0.85
            cote_dom *= 1.75
        elif 10<diff:
            cote_ext *= 0.65           
            cote_dom *= 1.90
        else:
            cote_ext *= 0.90
    if cote_dom < 1.10 :
        cote_dom = 1.10
    if cote_ext < 1.10:
        cote_ext=1.10  

    # Retourner les cotes 
    return {
        'Cote domicile': cote_dom,
        'Cote nulle': y_pred_draw[0],
        'Cote extérieur': cote_ext
    }

pred_jour=cotes_jour.dropna().copy()
equipes_domicile=pred_jour['Equipe Domicile'].tolist()
equipes_exterieur=pred_jour['Equipe Extérieure'].tolist()
liste_pays=pred_jour['Pays'].tolist()


# Création des listes pour stocker les résultats
cotes_domicile = []
cotes_nul = []
cotes_exterieur = []

# Boucle sur chaque match pour obtenir les prédictions
for domicile, exterieur, pays in zip(equipes_domicile, equipes_exterieur, liste_pays):
    resultats = predire_cotes(domicile, exterieur, pays, df_final, model_home, model_draw, model_away, pays_stats_scores)
    cotes_domicile.append(resultats['Cote domicile'])
    cotes_nul.append(resultats['Cote nulle'])
    cotes_exterieur.append(resultats['Cote extérieur'])

# Ajout des colonnes au DataFrame
pred_jour['Victoire Domicile pred'] = cotes_domicile
pred_jour['Match Nul pred'] = cotes_nul
pred_jour['Victoire Extérieur pred'] = cotes_exterieur


# Enregistrer le CSV dans le répertoire du projet (à la racine)
csv_filename = "predictions_du_jour.csv"
save_path = os.getcwd()  # S'assure que les fichiers sont enregistrés à la racine du repo GitHub

try:
    # Sauvegarde du fichier CSV
    pred_jour.to_csv(csv_filename, index=False)  # index=False pour ne pas inclure l'index du DataFrame dans le fichier CSV
    print(f"📁 Fichier sauvegardé : {csv_filename}")
except Exception as e:
    print(f"❌ Erreur lors de la sauvegarde du fichier {csv_filename} : {e}")

# 🔹 Ajouter, commettre et pousser sur GitHub
try:
    # Charger le repo GitHub
    repo = Repo(save_path)  # Repos GitHub cloné dans le répertoire courant
    origin = repo.remote(name='origin')  # Définir le remote 'origin'

    # Ajouter le fichier CSV au suivi de Git
    file_path = os.path.join(save_path, csv_filename)
    repo.git.add(file_path)  # Ajouter chaque fichier CSV

    # Commit des fichiers avec un message
    repo.index.commit("Mise à jour des fichiers CSV des prédictions")

    # Push les changements sur GitHub
    origin.push()  # Pousse les changements vers GitHub
    print("🚀 Fichiers CSV mis à jour sur GitHub avec succès !")

except Exception as e:
    print(f"❌ Erreur lors de la mise à jour sur GitHub : {e}")


    
    



