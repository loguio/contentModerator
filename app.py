import tempfile
import streamlit as st
import boto3
from dotenv import load_dotenv
import os
from moderation import process_media,get_aws_session,check_filetype  # Importe la fonction process_media depuis moderation.py
import botocore

st.set_page_config(page_title="Content Moderator Pro", page_icon=":guardsman:", layout="wide")
# Style du bandeau bleu
st.markdown(
    """
    <style>
        .labels-container {
            display: flex;
            flex-direction: row;  /* Assure l'affichage en ligne */
            flex-wrap: wrap; /* Permet le retour à la ligne si nécessaire */
            gap: 10px;
        }
        .label {
            display: inline-block;
            background-color: #87CEEB;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.header("Configuration")
    load_dotenv_button = st.button("Charger credentials depuis .env")
    if load_dotenv_button:
        load_dotenv()
        st.session_state.access_key = os.getenv("ACCESS_KEY")
        st.session_state.secret_key = os.getenv("SECRET_KEY")
    access_key = st.text_input("Access Key", value=st.session_state.get("access_key", ""))
    secret_key = st.text_input("Secret Key", value=st.session_state.get("secret_key", ""), type="password")
    bucket_name = st.text_input("Nom du bucket S3", value=st.session_state.get("bucket_name", "test-sdv-ia"))

st.title("Content Moderator Pro")
st.subheader("Modération de contenu avec AWS Rekognition")
st.subheader("Par Jules Croizier et Marius Bourse")
st.warning("Veuillez configurer vos credentials AWS dans la barre latérale.")
uploaded_file = st.file_uploader("Choisissez un fichier (image ou vidéo)", type=["jpg", "jpeg", "png", "mp4"])
 # Vérifier si la variable 'show_transcription' existe dans session_state

if uploaded_file is not None:
    results = None
    with st.spinner("Analyse en cours..."):
        # Définir le chemin du fichier local
        file_path = os.path.join("uploads", uploaded_file.name)
        
        # Sauvegarder le fichier en local
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Crée une session AWS avec les credentials
        session = get_aws_session(access_key, secret_key)
        rekognition = session.client('rekognition')
        transcribe = session.client('transcribe')
        comprehend = session.client('comprehend')

        # Appelle process_media avec le chemin d'accès du fichier enregistré
        
        try:
            # Appel de ta fonction process_media
            results = process_media(file_path, rekognition, transcribe, comprehend, bucket_name)
        except botocore.exceptions.NoCredentialsError as e:
        # Si l'erreur est due au manque de credentials, on affiche un message spécifique
            st.error("Erreur AWS : Impossible de localiser les credentials. Veuillez vérifier vos credentials AWS.")
    
        except botocore.exceptions.ClientError as e:
            # Vérification si l'erreur correspond à un problème de token de sécurité
            if 'UnrecognizedClientException' in str(e):
                st.error("Erreur de connexion AWS : Le jeton de sécurité inclus dans la demande est invalide. Veuillez vérifier vos credentials AWS.")
            else:
                # Si c'est une autre erreur ClientError, on l'affiche dans un message générique
                st.error(f"Erreur AWS : {e}")
                
    if results:
        if "hashtags" in results:  # Vérifie si results est défini
            media_type = check_filetype(uploaded_file.name)
            if media_type == "image":
                st.image(file_path, caption="Image téléchargée", use_container_width=True)
            if media_type == "vidéo":
                st.video(file_path,start_time=0)
                if 'show_transcription' not in st.session_state:
                    st.session_state.show_transcription = False  # Initialisation
                if results["subtitles"]:
                    st.markdown(results["subtitles"])
            # Affichage des labels en ligne
            st.markdown('<div class="labels-container">' + 
                "".join(f'<span class="label">{word}</span>' for word in results["hashtags"]) + 
                '</div>', unsafe_allow_html=True)
            st.success("Contenu approprié")
        else:
            print(results)
            st.error('### Contenu inapproprié Détecté\n### Cette publication a été bloquée')

            st.error("Thème sensible détecté :")
            st.markdown(','.join(results['moderation']))