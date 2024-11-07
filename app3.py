import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd
import hashlib
from datetime import datetime
from tensorflow.keras.models import load_model
import numpy as np

# Charger le modèle combiné de prédiction d'âge et de sexe
model = load_model('model_prediction.keras')

# Gestion des utilisateurs
users_db = {
    "Blanche": hashlib.sha256("pass1".encode()).hexdigest(),
    "Kader": hashlib.sha256("pass2".encode()).hexdigest(),
    "Joyeux": hashlib.sha256("pass3".encode()).hexdigest()
}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    hashed_password = hash_password(password)
    return users_db.get(username) == hashed_password

def login():
    st.sidebar.header("Connexion")
    username = st.sidebar.text_input("Nom d'utilisateur")
    password = st.sidebar.text_input("Mot de passe", type="password")
    
    if st.sidebar.button("Se connecter"):
        if authenticate_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.sidebar.success("Connexion réussie !")
        else:
            st.sidebar.error("Nom d'utilisateur ou mot de passe incorrect")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
else:
    st.title('Application pour la détection de l\'âge et du sexe d\'une personne')

    st.markdown("""
    <style>
    .description {
        background-color: #f0f0f0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        font-size: 16px;
        color: #333;
    }
    .description h2 {
        font-size: 20px;
        margin-top: 0;
        color: #00f;
    }
    .description p {
        margin: 0;
    }
    </style>
    <div class="description">
        <h2>Objectif de l'application</h2>
        <p>
            Cette application permet de détecter l'âge et le sexe d'une personne à partir d'une image.
            L'application affichera les résultats directement sur l'image et vous permettra de les sauvegarder.
        </p>
    </div>
    """, unsafe_allow_html=True)

    theme = st.sidebar.selectbox("Choisir le thème", ["Clair", "Sombre"])
    if theme == "Sombre":
        st.write('<style>body {background-color: #2e2e2e; color: white;}</style>', unsafe_allow_html=True)

    st.sidebar.header("Options")
    uploaded_file = st.sidebar.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=["Image", "Âge", "Sexe", "Utilisateur", "Date"])

    tab1, tab2, tab3, tab4 = st.tabs(["Prédiction", "Historique", "Statistiques", "Documentation"])

    def preprocess_image(image):
        image = image.resize((84, 84))  # Redimensionner selon ce que le modèle attend
        image = np.array(image) / 255.0  # Normaliser les pixels
        image = np.expand_dims(image, axis=0)  # Ajouter une dimension batch
        return image

    def predict_age_and_gender(image):
        try:
            predictions = model.predict(image)
            age = int(np.round(predictions[1][0]))  # Supposant que l'âge est la deuxième sortie
            gender = "Masculin" if np.round(predictions[0][0]) == 0 else "Féminin"  # Supposant que le sexe est la première sortie
            return age, gender
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            return None, None

    with tab1:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Image chargée', use_column_width=True)

            if st.button("Prédire"):
                processed_image = preprocess_image(image)
                age, gender = predict_age_and_gender(processed_image)
                
                if age is not None and gender is not None:
                    draw = ImageDraw.Draw(image)
                    font = ImageFont.load_default(60)
                    draw.text((50, 50), f"Âge: {age}, Sexe: {gender}", fill="white", font=font)
                    
                    st.image(image, caption='Résultats de la prédiction', use_column_width=True)

                    new_entry = pd.DataFrame({
                        "Image": [uploaded_file.name],
                        "Âge": [age],
                        "Sexe": [gender],
                        "Utilisateur": [st.session_state["username"]],
                        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                    })
                    st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)

                    buf = io.BytesIO()
                    image.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="Télécharger l'image annotée",
                        data=byte_im,
                        file_name="result.jpg",
                        mime="image/jpeg",
                    )
                else:
                    st.error("Erreur dans la prédiction. Veuillez vérifier l'image ou le modèle.")
        else:
            st.info("Veuillez charger une image pour continuer.")

    with tab2:
        st.header("Historique des Prédictions")
        st.dataframe(st.session_state.history, use_container_width=True, height=200)

    with tab3:
        st.header("Statistiques des Prédictions")
        if len(st.session_state.history) > 0:
            st.subheader("Distribution des Âges")
            age_counts = st.session_state.history['Âge'].value_counts().sort_index()
            st.bar_chart(age_counts)

            st.subheader("Répartition des Sexes")
            gender_counts = st.session_state.history['Sexe'].value_counts()
            st.bar_chart(gender_counts)

            st.subheader("Âge Moyen")
            mean_age = st.session_state.history['Âge'].mean()
            st.write(f"L'âge moyen des utilisateurs est de {mean_age:.2f} ans.")

            st.subheader("Activité des Utilisateurs")
            activity_counts = st.session_state.history['Utilisateur'].value_counts()
            st.bar_chart(activity_counts)
        else:
            st.info("Aucune donnée dans l'historique pour afficher les statistiques.")

    with tab4:
        st.header("Documentation")
        
        st.subheader("Introduction")
        st.write("""
        Cette application permet de prédire l'âge et le sexe d'une personne à partir d'une image. 
        Elle utilise des modèles de machine learning pour effectuer les prédictions et propose une interface utilisateur simple pour charger des images, visualiser les résultats, et consulter l'historique des prédictions.
        """)
        
        st.subheader("Fonctionnalités")
        st.write("""
        - Prédiction d'âge et de sexe à partir d'une image.
        - Historique des prédictions avec la possibilité de consulter les résultats passés.
        - Visualisation des statistiques des prédictions effectuées.
        - Téléchargement des images annotées.
        """)
        
        st.subheader("Technologies Utilisées")
        st.write("""
        - **Streamlit**: Pour l'interface utilisateur.
        - **PIL**: Pour la manipulation des images.
        - **TensorFlow/Keras**: Pour l'entraînement et l'exécution des modèles de machine learning.
        - **Pandas**: Pour la gestion des données et des historiques.
        """)
        
        st.subheader("Modèles de Prédiction")
        st.write("""
        Le modèle de prédiction d'âge et de sexe est basé sur un réseau de neurones convolutif (CNN). Il a été entraîné sur la base de données UTKFace, qui contient des images annotées avec l'âge, le sexe, et l'origine ethnique.
        """)
        
        st.subheader("Auteurs")
        st.write("""
        Cette application a été développée dans le cadre d'un projet de fin d'études. 
        """)

    if st.sidebar.button("Se déconnecter"):
        st.session_state["authenticated"] = False
        st.experimental_rerun()
