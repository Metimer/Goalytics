import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv



# Configuration de la page
st.set_page_config(
    page_title="Goalytics - Decode The Game !",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS pour améliorer le design
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Teko:wght@400;700&display=swap');
    /* Fond général de l'application */
    .stApp {
        background-color: #0A0A2A !important;
        color: #FFD700 !important;
    }
    label {
    color: #FFD700 !important;
}

    /* Style de la sidebar */
    [data-testid="stSidebar"] {
        background-color: #191970 !important;
        color: black !important;
        border: 2px solid #FFD700;    
    }
    
    /* Modifier la couleur des labels */
    [data-testid="stSidebar"] label {
        color: white !important; /* Texte des labels en blanc */
    }
    
    /* Modifier le style des boutons */
    div.stButton > button {
    color: black !important;  /* Texte en noir */
    background-color: #FFD700 !important;  /* Fond doré */
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    transition: 0.3s;
    }

    div.stButton > button:hover {
    background-color: #E5C100 !important; /* Couleur légèrement plus foncée au survol */
    }
    
    /* Style pour le DataFrame */
    [data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 2px solid #FFD700;
    background-color: black;
    color: white;
    font-size: 18px !important;
    font-weight: bold;
    color: white !important;
    }

    [data-testid="stTable"] tbody tr td {
    font-size: 18px;
    text-align: center;
     }

    /* Augmenter la taille du texte dans les colonnes */
    div[data-testid="stDataEditor"] td {
            font-size: 14px !important;
            font-weight: bold;
    }
    div.stAlert {
        background-color: #191970;  
        color: #FF0000 !important;  
        border-left: 5px solid #FFD700;  
        font-size: 50px;
        font-family: "Montserrat", sans-serif;
        letter-spacing: 2px;
        font-weight: bold;
        -webkit-text-stroke: 1px white;
    }
    /* Style des titres */
    .title {
        text-align: center;
        font-size: 70px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #FFD700;
        font-family: 'Teko', sans-serif; /* Police élégante et disponible par défaut */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* Ombre pour du relief */
        letter-spacing: 2px; /* Espacement des lettres pour un effet premium */
        text-transform: uppercase; /* Mise en majuscules pour plus d'impact */
        -webkit-text-stroke: 1px black; /* Contour fin pour améliorer la lisibilité */
    }

    /* Style des cartes */
    .film-card {
        background-color: #191970 ;
        border-radius: 10px;
        border: 2px solid #FFD700;
        padding: 15px;
        margin: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        text-align: center;
    }

    /* Style des affiches */
    .film-card img {
        border-radius: 10px;
        margin-bottom: 10px;
    }

    /* Style du texte des cartes */
    .film-card h3 {
        text-align: center;
        color: #FFD700;
        font-family: 'Teko', sans-serif; /* Police élégante et disponible par défaut */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* Ombre pour du relief */
        letter-spacing: 2px; /* Espacement des lettres pour un effet premium */       
        -webkit-text-stroke: 1px black; /* Contour fin pour améliorer la lisibilité */
    }

    .film-card p {
        text-align: center;
        font-size: 20px;
        color: #FFD700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* Ombre pour du relief */
        font-family: "Montserrat", sans-serif; /* Police élégante et disponible par défaut */
    }

    .film-card h4 {
    font-size: 45px;
    color: #FFD700;
    font-family: "Teko", sans-serif;
    height: 50px;  
    text-align: center;
    display: flex;
    align-items: center; 
    justify-content: center;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* Ombre pour du relief */
    letter-spacing: 2px; /* Espacement des lettres pour un effet premium */       
    -webkit-text-stroke: 1px black; /* Contour fin pour améliorer la lisibilité */
}

    """,
    unsafe_allow_html=True
)
load_dotenv()
API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def query_huggingface(question):
    # Forcer la réponse en français
    prompt = f"Answer to this question in french : {question}"
    
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 200}
    }
    
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    
    print("Statut HTTP :", response.status_code)
    print("Réponse brute :", response.text)
    
    try:
        response_json = response.json()
        if isinstance(response_json, list) and "generated_text" in response_json[0]:
            generated_text = response_json[0]["generated_text"]
            # Extraction de la réponse après "Answer :"
            response_text = generated_text.split("Answer :")[-1].strip()
            return response_text
        elif isinstance(response_json, dict) and "error" in response_json:
            return f"❌ Erreur API : {response_json['error']}"
        else:
            return "⚠️ Réponse inattendue de l'API."
    except Exception as e:
        return f"🚨 Erreur lors de l'appel à l'API : {str(e)}"

# Logo dans la sidebar
logo_url = "https://i.postimg.cc/g03Cmvnp/image-2.png"
st.sidebar.image(logo_url, use_container_width=True)

# Chargement des données
loading_gif = "https://www.photofunky.net/output/image/5/1/3/c/513c2f/photofunky.gif"  
gif_container = st.empty()

# Afficher le GIF dans le conteneur
gif_container.image(loading_gif, use_container_width=True)

# Chargement des données
pays_stats_scores = {
    'Allemagne': pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_api/Allemagne_api.csv'),
    'Espagne': pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_api/Espagne_api.csv'),
    'France': pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_api/France_api.csv'),
    'Italie': pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_api/Italie_api.csv'),
    'Angleterre': pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/data_api/Angleterre_api.csv')
}

# Nettoyage et renommage
for key, df in pays_stats_scores.items():  # ✅ Récupérer clé et DataFrame
    # Suppression des colonnes inutiles
    df = df.drop(columns=['league_id', 'team_id', 'league_round', 'fk_stage_key', 
                          'stage_name', 'home_promotion', 'away_promotion','country_name'], errors='ignore')
    
    # Renommage des colonnes
    df = df.rename(columns={
        'league_name': 'Ligue', 'team_name': 'Équipe', 
        'overall_promotion': 'Promotion', 'overall_league_position': 'Classement', 
        'overall_league_payed': 'Matchs joués', 'overall_league_W': 'Victoires', 
        'overall_league_D': 'Nuls', 'overall_league_L': 'Défaites', 
        'overall_league_GF': 'Buts marqués', 'overall_league_GA': 'Buts encaissés', 
        'overall_league_PTS': 'Points', 'home_league_position': 'Classement à domicile', 
        'home_league_payed': 'Matchs joués à domicile', 'home_league_W': 'Victoires à domicile', 
        'home_league_D': 'Match nul à domicile', 'home_league_L': 'Défaite à domicile', 
        'home_league_GF': 'Buts marqués à domicile', 'home_league_GA': 'Buts encaissés à domicile', 
        'home_league_PTS': 'Points à domicile', 'away_league_position': 'Classement extérieur', 
        'away_league_payed': 'Matchs joués extérieur', 'away_league_W': 'Victoires extérieurs', 
        'away_league_D': 'Match nul extérieur', 'away_league_L': 'Défaite extérieur', 
        'away_league_GF': 'Buts marqués extérieur', 'away_league_GA': 'Buts encaissés extérieur', 
        'away_league_PTS': 'Points extérieur', 'team_badge': 'Écusson club'
    })
    df['Pays'] = key
    df['Promotion']=df['Promotion'].fillna('Non Qualifié')
    # ✅ Mise à jour du dictionnaire avec le DataFrame modifié
    pays_stats_scores[key] = df
    
cotes_du_jour=pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/cotes_du_jour.csv')
cotes_du_jour.dropna(inplace=True)
cotes_du_jour['Ligue']=cotes_du_jour['Ligue'].replace(['l1-mcdonald-s','serie-a','laliga','bundesliga-1','premier-league'],['Ligue 1','Serie A','La Liga','Bundesliga','Premier League'])

pred_du_jour=pd.read_csv('https://raw.githubusercontent.com/Metimer/Goalytics/refs/heads/main/predictions_du_jour.csv')
pred_du_jour['Pays']=pred_du_jour['Pays'].str.capitalize()
pred_du_jour['Ligue']=pred_du_jour['Ligue'].replace(['l1-mcdonald-s','serie-a','laliga','bundesliga-1','premier-league'],['Ligue 1','Serie A','La Liga','Bundesliga','Premier League'])
pred_du_jour[['Victoire Domicile pred','Match Nul pred','Victoire Extérieur pred']]=pred_du_jour[['Victoire Domicile pred','Match Nul pred','Victoire Extérieur pred']].round(2)
pred_du_jour[['Cote Domicile', 'Cote Nul', 'Cote Extérieur']] = pred_du_jour[['Cote Domicile', 'Cote Nul', 'Cote Extérieur']].replace({",": "."}, regex=True)
pred_du_jour[['Cote Domicile', 'Cote Nul', 'Cote Extérieur']] = pred_du_jour[['Cote Domicile', 'Cote Nul', 'Cote Extérieur']].astype(float).round(2)
gif_container.empty()  # Efface le GIF

#Chargement logo des ligues 

logos={'Ligue 1':'https://images.bfmtv.com/a19IeNS3RDSs1zVTUOBUFfoyK8k=/0x0:0x0/96x0/site_manager_images/noeud/rmcsport/1724226757701_logo_ligue_1_png_ligue_1_1724226764356.png',
       'Serie A':'https://upload.wikimedia.org/wikipedia/en/thumb/a/ab/Serie_A_ENILIVE_logo.svg/210px-Serie_A_ENILIVE_logo.svg.png',
       'La Liga':'https://brandlogos.net/wp-content/uploads/2021/01/la-liga-logo-768x768.png',
       'Premier League':'https://www.topbets.com.gh/wp-content/uploads/2024/07/clipboard_783ebef9-c928-488f-a0cb-aff2d2d4ab2a.png',
       'Bundesliga':'https://upload.wikimedia.org/wikipedia/en/thumb/d/df/Bundesliga_logo_%282017%29.svg/270px-Bundesliga_logo_%282017%29.svg.png'}
# Menu

# Menu de navigation
menu_options = ["Accueil", "Statistiques et cotes", "Nos prédictions", "About us"]
selection = st.sidebar.selectbox("Choissisez une section", menu_options)

# Bouton de login
if st.sidebar.button("Login"):
    st.login("google")  # Lance l'authentification Google

if st.sidebar.button("Logout"):
    st.logout()  # Déconnecte l'utilisateur

if st.experimental_user is not None:
    is_logged_in = getattr(st.experimental_user, "is_logged_in", False)
    user_name = getattr(st.experimental_user, "name", "Utilisateur anonyme")
else:
    is_logged_in = False  # Par défaut, l'utilisateur n'est pas connecté
    user_name = "Utilisateur anonyme"

# Stocker l'état de connexion dans la session
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = is_logged_in

# Vérifier l'accès
if not st.session_state.is_logged_in:
    st.warning("Vous devez être connecté pour accéder à certaines pages.")
    

# Afficher le contenu protégé
st.success(f"Bienvenue {user_name}!")

# Accueil

if selection == "Accueil":
    st.markdown('<div class="title"> Goalytics - Decode The Game ©</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="film-card">
            <img src="{logo_url}" width="700">
            <p>Bienvenue dans l'application communautaire de conseil au pari !!</p>
            <p>Notre équipe vous fournit des prédictions d'après les analyses détaillées des 5 grands championnats !! </p>
            <p>Les données sont mises à jour et rafraichies automatiquement toutes les nuits ( 02:00 UTC )</p> 
            <p>Goalytics , Decode The Game !! © </p>            
        </div>
        """,
        unsafe_allow_html=True
    )
    
# Statistiques et cotes

elif selection == "Statistiques et cotes":
    st.markdown('<div class="title"> Goalytics - Decode The Game ©</div>', unsafe_allow_html=True)

    # Sélection du pays
    pays_input = st.selectbox("Choissisez un pays", list(pays_stats_scores.keys()))
    pays_df = pays_stats_scores[pays_input]
    #Sélection du club
    equipe_input=st.selectbox("Choissisez un club", list(pays_df['Équipe']))
    #Selection de l'url du logo
    url_logo = pays_df[pays_df['Équipe'] == equipe_input]['Écusson club'].iloc[0]
    ligue=pays_df[pays_df['Équipe'] == equipe_input]['Ligue'].iloc[0]
    if ligue in logos:
        logo_ligue = logos[ligue]
    else:
        logo_ligue = None
    pred_du_jour['Pays']=pred_du_jour['Pays'].str.capitalize()
    club_pred=pred_du_jour.drop(['Victoire Domicile pred','Match Nul pred','Victoire Extérieur pred'],axis=1).copy()
    pays_df2=cotes_du_jour[cotes_du_jour['Pays']==pays_input]
    club_df=pays_df[pays_df['Équipe']==equipe_input]
    club_cote = pays_df2[(pays_df2['Equipe Domicile'] == equipe_input) | (pays_df2['Equipe Extérieure'] == equipe_input)]
    #Affichage des statistiques et cotes du pays sélectionné
    st.markdown(
        f"""
        <div class="film-card">
        <h4>Classements et stats : {ligue}</h4>
            <img src="{logo_ligue}" width="100">          
        </div>
        """,
        unsafe_allow_html=True
    )
    #Statistiques et classements
    st.data_editor(pays_df, 
                    hide_index=True,  
    ) 
    st.markdown(
        f"""
        <div class="film-card">
        <p>Classements et stats : {equipe_input}</p>
            <img src="{url_logo}" width="100">          
        </div>
        """,
        unsafe_allow_html=True
    )
    st.data_editor(club_df, 
                    hide_index=True,  
    ) 
    st.markdown(
        f"""
        <div class="film-card">
        <h4>Côtes à venir : {ligue}</h4>
            <img src="{logo_ligue}" width="100">          
        </div>
        """,
        unsafe_allow_html=True
    )
    #Cotes à venir du pays sélectionné 
    st.data_editor(pays_df2, 
                    hide_index=True,  
    ) 
    st.markdown(
        f"""
        <div class="film-card">
        <p>Prochain match ( championnat ) : {equipe_input}</p>
            <img src="{url_logo}" width="100">          
        </div>
        """,
        unsafe_allow_html=True
    )
    st.data_editor(club_cote, 
                    hide_index=True,  
    )                 

# Nos préditctions
elif selection == "Nos prédictions":
    if not st.session_state.is_logged_in:
        st.markdown('<div class="title"> Nos Prédictions </div>', unsafe_allow_html=True)
        st.markdown(
        f"""
        <div class="film-card">
            <img src="{logo_url}" width="700">
            <p>Veuillez vous connecter</p>
            <p>Pour profiter de nos prédictions! </p>
            <p>Goalytics , Decode The Game !! © </p>            
        </div>
        """,
        unsafe_allow_html=True
    )
        
    else:
        st.success(f"Bienvenue {user_name}! , vous avez accès à cette page réservée aux membres!")
        st.markdown('<div class="title"> Goalytics - Decode The Game © </div>', unsafe_allow_html=True)
        
        def highlight_cells(val):       
            if val>=0.275:
                return "background-color: yellow"
            return ""
        
        # Sélection du pays
        pays_input = st.selectbox("Choissisez un pays", list(pays_stats_scores.keys()))
        pays_df = pred_du_jour[pred_du_jour['Pays']==pays_input].copy()
        ligue=pays_df[pays_df['Pays'] == pays_input]['Ligue'].iloc[0]
        if ligue in logos:
            logo_ligue = logos[ligue]
        else:
            logo_ligue = None
        pays_df['diff_dom'] = pays_df['Cote Domicile'] - pays_df['Victoire Domicile pred']
        pays_df['diff_nul'] = pays_df['Cote Nul'] - pays_df['Match Nul pred']
        pays_df['diff_ext'] = pays_df['Cote Extérieur'] - pays_df['Victoire Extérieur pred']
        styled_pays_df = pays_df.style.applymap(highlight_cells, subset=['diff_dom', 'diff_nul', 'diff_ext'])
        st.markdown(
        f"""
        <div class="film-card">
        <h4>Nos prédictions : {ligue}</h4>
            <img src="{logo_ligue}" width="100">
            <p>Les valeurs surlignées sont des bons coups à jouer ( d'après notre algorithme )</p>          
        </div>
        """,
        unsafe_allow_html=True
    )
        
        st.dataframe(styled_pays_df) 
        st.markdown(
        f"""
        <div class="film-card">
        <h4>Le conseiller ( Béta ) </h4>
            <img src="https://cdn-icons-png.flaticon.com/512/4301/4301882.png" width="100">         
        </div>
        """,
        unsafe_allow_html=True
    )
        st.write("💬 **Posez une question et obtenez une réponse !**")

        user_input = st.text_area("Pose une question sur le sport :")
        if st.button("Envoyer") and user_input:
            with st.spinner("Le conseiller réfléchit... 🤔"):
                st.write("🧠 Réponse :", query_huggingface(user_input))

# About us
elif selection == "About us":
    st.markdown('<div class="title"> Goalytics - Decode The Game © </div>', unsafe_allow_html=True)
    st.markdown('<div class="film-card"> <h4>Nos partenaires</h4> </div>', unsafe_allow_html=True)
    part_url2='https://www.hautsdefrance-id.fr/wp-content/uploads/2022/04/wild-code-school-logo-1024x614-1.jpg'
    
    
    st.markdown(
                f"""
                <div class="film-card">
                    <h3> Goalytics - Decode The Game ! </h3>
                    <a href="{'https://github.com/Metimer'}" target="_blank">
                    <img src="{logo_url}" width="550"></a>
                    <p>Nous sommes une équipe </p>  
                    <p>à taille humaine qui saura</p> 
                    <p>relever tout les challenges digitaux </p> 
                    <p>Pour votre entreprise !!</p> 
                    <p>N'hésitez plus , contactez nous !</p>                  
                </div>
                """,
                unsafe_allow_html=True
            )
        
    st.markdown(
                f"""                
                <div class="film-card">   
                    <h3> Wild Code School </h3> 
                    <a href="{'https://www.wildcodeschool.com/'}" target="_blank">                
                    <img src="{part_url2}" width="850"></a>
                    <p>Depuis plus de 10 ans, la Wild Code School forme des talents aux métiers de la tech et de l'IA.</p>
                    <p>Avec plus de 8 000 alumni, des formations adaptées au marché, et une pédagogie innovante,</p>
                    <p>nous préparons les professionnels de demain.</p>
                    <p>Découvrez nos spécialités pour réussir : développement web, data et IA,</p>
                    <p>IT et cybersécurité, design et gestion de projet.</p>
                    <p>Vous aurez les codes ! </p>                   
                    </div>
                """,
        unsafe_allow_html=True
            )