# PATCH POUR STREAMLIT CLOUD - À placer en tout début de app.py
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
import zipfile
import shutil
from pathlib import Path
from typing import List
import requests

# CORRECTION: Import corrigé pour Chroma (version compatible)
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_mistralai import ChatMistralAI
from prompting import (
    retrieve_documents,
    generate_context_response,
    generate_example_training_plan,
    generate_pedagogical_engineering_advice,
    reformulate_competencies_apc,
    generate_structured_training_scenario
)

# Import de la page RAG Personnel
from user_rag_page import user_rag_page

# CORRECTION: Import de l'API Mistral pour les embeddings
from mistralai.client import MistralClient

# ==========================================
# UTILISATION DU COMPOSANT STREAMLIT-OAUTH
# ==========================================

# Vérifier si streamlit-oauth est installé
try:
    from streamlit_oauth import OAuth2Component
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
    st.error("❌ Le package streamlit-oauth n'est pas installé. Ajoutez 'streamlit-oauth' à votre requirements.txt")

# Configuration des APIs - Utilise les secrets Streamlit
try:
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
    
    # Configuration OAuth Google
    GOOGLE_CLIENT_ID = st.secrets["auth"]["client_id"]
    GOOGLE_CLIENT_SECRET = st.secrets["auth"]["client_secret"]
    GOOGLE_REDIRECT_URI = st.secrets["auth"]["redirect_uri"]
    
except Exception as e:
    st.error(f"❌ Erreur de configuration des secrets: {e}")
    st.stop()

# Configurer le token HuggingFace
if HUGGINGFACE_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HUGGINGFACE_TOKEN

# ==========================================
# GESTION DE L'AUTHENTIFICATION OAUTH
# ==========================================

def init_oauth():
    """Initialise le composant OAuth2 Google"""
    if not OAUTH_AVAILABLE:
        return None
    
    return OAuth2Component(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        refresh_token_endpoint="https://oauth2.googleapis.com/token",
        revoke_token_endpoint="https://oauth2.googleapis.com/revoke",
    )

def check_authentication():
    """Vérifie l'authentification avec streamlit-oauth"""
    if not OAUTH_AVAILABLE:
        return False
    
    # Vérifier si l'utilisateur est déjà authentifié
    if 'auth_token' in st.session_state and 'user_info' in st.session_state:
        return True
    
    return False

def show_login_page():
    """Affiche la page de connexion avec Google OAuth"""
    st.markdown("""
    <div style="text-align: center; margin: 50px 0;">
        <h1>🎓 Assistant FPA</h1>
        <h2>Ingénierie de Formation</h2>
        <p style="font-size: 1.2rem; margin: 30px 0;">
            Connectez-vous avec votre compte Google pour accéder à votre espace personnel de formation
        </p>
        
        <div style="margin: 40px 0;">
            <h3>✨ Fonctionnalités personnalisées :</h3>
            <div style="text-align: left; display: inline-block; margin: 20px 0;">
                <p>📚 • Base de connaissances commune en formation</p>
                <p>🎯 • Scénarisation pédagogique intelligente</p>
                <p>📄 • Votre propre RAG personnel</p>
                <p>💾 • Sauvegarde automatique de vos documents</p>
                <p>🔒 • Données privées et sécurisées</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not OAUTH_AVAILABLE:
        st.error("❌ Le composant OAuth n'est pas disponible. Vérifiez votre requirements.txt")
        return
    
    # Centrer le bouton de connexion
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        oauth2 = init_oauth()
        if oauth2:
            result = oauth2.authorize_button(
                name="🔐 Se connecter avec Google",
                redirect_uri=GOOGLE_REDIRECT_URI,
                scope="openid email profile",
                key="google_auth",
                use_container_width=True
            )
            
            if result and 'token' in result:
                # Récupérer les informations utilisateur
                user_info = get_user_info(result['token']['access_token'])
                if user_info:
                    # Stocker dans la session
                    st.session_state.auth_token = result['token']
                    st.session_state.user_info = user_info
                    st.rerun()
                else:
                    st.error("❌ Erreur lors de la récupération des informations utilisateur")

def get_user_info(access_token):
    """Récupère les informations utilisateur avec le token d'accès"""
    try:
        headers = {'Authorization': f'Bearer {access_token}'}
        response = requests.get("https://www.googleapis.com/oauth2/v2/userinfo", headers=headers)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Erreur API Google: {e}")
    return None

def get_user_identifier():
    """Récupère l'identifiant utilisateur sécurisé"""
    if 'user_info' in st.session_state:
        return st.session_state.user_info['email']
    return None

def logout_user():
    """Déconnecte l'utilisateur"""
    keys_to_remove = ['auth_token', 'user_info']
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

# ==========================================
# CLASSE EMBEDDINGS MISTRAL
# ==========================================

class MistralEmbeddings:
    """
    Wrapper LangChain pour Mistral Embed (1024 dims).
    Compatible avec la base vectorielle créée par rag_formation.py
    """
    def __init__(self, api_key: str, model: str = "mistral-embed"):
        self.client = MistralClient(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                resp = self.client.embeddings(model=self.model, input=batch)
                embeddings.extend([d.embedding for d in resp.data])
            except Exception as e:
                st.error(f"Erreur embedding lot {i//batch_size+1}: {e}")
                embeddings.extend([[0.0]*1024 for _ in batch])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            resp = self.client.embeddings(model=self.model, input=[text])
            return resp.data[0].embedding
        except Exception as e:
            st.error(f"Erreur embedding requête: {e}")
            return [0.0]*1024

# Définition des couleurs
COLORS = {
    "primary": "#1D5B68",
    "secondary": "#E6525E", 
    "light_blue": "#94B7BD",
    "very_light_blue": "#DDE7E9",
    "dark_gray": "#3F3F3F",
    "light_gray": "#F5F5F6",
    "background": "#1A1A1A",
    "text": "#FFFFFF"
}

# Configuration CSS personnalisée
def local_css():
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {COLORS["background"]};
            color: {COLORS["text"]};
        }}
        
        h1, h2, h3 {{
            color: {COLORS["light_blue"]};
            font-family: 'Helvetica Neue', sans-serif;
        }}
        
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
            background-color: {COLORS["dark_gray"]};
            color: {COLORS["text"]};
            border: 1px solid {COLORS["light_blue"]};
        }}
        
        .stSelectbox>div>div>div {{
            background-color: {COLORS["dark_gray"]};
            color: {COLORS["text"]};
            border-radius: 6px;
            border: 1px solid {COLORS["light_blue"]};
        }}
        
        .stButton>button {{
            background-color: {COLORS["primary"]};
            color: {COLORS["text"]};
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            transition: all 0.3s;
        }}
        
        .stButton>button:hover {{
            background-color: {COLORS["light_blue"]};
        }}
        
        [data-testid="stSidebar"] {{
            background-color: {COLORS["dark_gray"]};
            color: {COLORS["text"]};
        }}
        
        .scenario-card, .user-message, .assistant-message {{
            background-color: {COLORS["dark_gray"]};
            color: {COLORS["text"]};
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid {COLORS["secondary"]};
            margin-bottom: 10px;
        }}
        
        .upload-box {{
            background-color: {COLORS["dark_gray"]};
            border: 2px dashed {COLORS["light_blue"]};
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
        }}
        
        .user-info {{
            background-color: {COLORS["primary"]};
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            margin: 10px 0;
        }}
        
        .guide-section {{
            background-color: {COLORS["dark_gray"]};
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid {COLORS["light_blue"]};
        }}
        
        .column-selector {{
            background-color: {COLORS["dark_gray"]};
            padding: 10px;
            border-radius: 8px;
            margin: 5px 0;
        }}
    </style>
    """, unsafe_allow_html=True)

# Configuration de l'application Streamlit
st.set_page_config(
    page_title="Assistant FPA - Ingénierie de Formation",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

local_css()

# ==========================================
# VÉRIFICATION AUTH ET POINT D'ENTRÉE PRINCIPAL
# ==========================================

# Vérification de l'authentification
if not check_authentication():
    show_login_page()
    st.stop()

# ==========================================
# FONCTIONS SYSTÈME (identiques à avant)
# ==========================================

def save_user_rag_state(user_id: str):
    """Sauvegarde l'état du RAG utilisateur (persistance automatique avec Chroma)"""
    pass

def load_user_rag_state(user_id: str):
    """Charge l'état du RAG utilisateur spécifique"""
    user_rag_dir = f"chroma_db_user_{user_id.replace('@', '_').replace('.', '_')}"
    
    if os.path.exists(user_rag_dir) and os.listdir(user_rag_dir):
        try:
            embeddings = load_embedding_model()
            if embeddings:
                vectorstore = Chroma(
                    persist_directory=user_rag_dir,
                    embedding_function=embeddings
                )
                st.session_state[f'RAG_user_{user_id}'] = vectorstore
                return vectorstore
        except Exception as e:
            st.error(f"Erreur lors du chargement du RAG utilisateur: {e}")
    
    st.session_state[f'RAG_user_{user_id}'] = None
    return None

def clean_corrupted_chromadb(db_path):
    """Nettoie automatiquement une base ChromaDB corrompue"""
    try:
        st.warning("🔧 Détection d'une base ChromaDB incompatible...")
        st.info("🗑️ Suppression automatique de l'ancienne base...")
        
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        st.success("✅ Base corrompue supprimée avec succès !")
        st.info("📋 Veuillez re-uploader votre fichier chromadb_formation.zip")
        
        return True
    except Exception as e:
        st.error(f"❌ Erreur lors du nettoyage: {e}")
        return False

@st.cache_resource
def extract_database_if_needed():
    """Décompresse automatiquement la base vectorielle si nécessaire"""
    
    db_path = "chromadb_formation"
    zip_path = "chromadb_formation.zip"
    
    if os.path.exists(db_path) and os.listdir(db_path):
        st.success("✅ Base vectorielle déjà disponible")
        return True
    
    if os.path.exists(zip_path):
        st.info("📦 Décompression de la base vectorielle...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            st.success("✅ Base vectorielle décompressée avec succès")
            return True
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la décompression: {e}")
            return False
    
    return False

def database_upload_interface():
    """Interface d'upload de la base vectorielle"""
    
    st.markdown("""
    <div class="upload-box">
        <h3>📤 Upload de votre base vectorielle</h3>
        <p>La base vectorielle ChromaDB est nécessaire pour le fonctionnement de l'assistant.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choisir le fichier chromadb_formation.zip",
        type="zip",
        help="Uploadez votre base vectorielle compressée au format ZIP"
    )
    
    if uploaded_file is not None:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("💾 Sauvegarde du fichier...")
            progress_bar.progress(25)
            
            with open("chromadb_formation.zip", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            status_text.text("📦 Décompression en cours...")
            progress_bar.progress(50)
            
            with zipfile.ZipFile("chromadb_formation.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            progress_bar.progress(100)
            status_text.text("✅ Base vectorielle installée avec succès!")
            
            st.success("🎉 Votre base vectorielle a été installée ! L'application va redémarrer...")
            st.balloons()
            
            import time
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'installation: {e}")
            st.info("💡 Assurez-vous que le fichier ZIP contient bien le dossier 'chromadb_formation'")
    
    return False

@st.cache_resource
def load_embedding_model():
    """Charge le modèle d'embedding Mistral compatible avec la base vectorielle"""
    try:
        if not MISTRAL_API_KEY:
            st.error("❌ Clé API Mistral manquante")
            return None
            
        return MistralEmbeddings(api_key=MISTRAL_API_KEY, model="mistral-embed")
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle d'embedding: {e}")
        return None

@st.cache_resource
def load_vector_store():
    """Charge la base vectorielle avec mise en cache et gestion d'erreurs"""
    try:
        embeddings = load_embedding_model()
        if embeddings is None:
            return None
            
        db_path = "chromadb_formation"
        if not os.path.exists(db_path):
            st.error("❌ Base vectorielle 'chromadb_formation' non trouvée")
            return None
            
        try:
            vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
            # Test de la base
            vectorstore.similarity_search("test", k=1)
            return vectorstore
            
        except Exception as chroma_error:
            error_msg = str(chroma_error).lower()
            
            if "no such column: collections.topic" in error_msg:
                st.error("❌ Base ChromaDB incompatible détectée")
                
                if clean_corrupted_chromadb(db_path):
                    return "needs_reupload"
                else:
                    return None
                    
            elif "embedding dimension" in error_msg and "does not match" in error_msg:
                st.error("❌ Incompatibilité de dimensions d'embeddings détectée")
                st.info("💡 La base vectorielle a été créée avec des embeddings différents")
                
                if clean_corrupted_chromadb(db_path):
                    return "needs_reupload"
                else:
                    return None
            else:
                st.error(f"❌ Erreur ChromaDB: {chroma_error}")
                return None
                
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de la base vectorielle: {e}")
        return None

@st.cache_resource
def create_mistral_llm():
    """Crée l'instance Mistral avec mise en cache"""
    try:
        if not MISTRAL_API_KEY:
            st.error("❌ Clé API Mistral manquante")
            return None
            
        return ChatMistralAI(
            mistral_api_key=MISTRAL_API_KEY,
            model="open-mistral-7b",
            temperature=0.1,
            max_tokens=4000
        )
    except Exception as e:
        st.error(f"❌ Erreur lors de la création du modèle Mistral: {e}")
        return None

def initialize_system():
    """Initialise le système avec gestion automatique de la base et des erreurs"""
    
    if not extract_database_if_needed():
        return None, None, "database_missing"
    
    with st.spinner("🚀 Initialisation de l'Assistant FPA..."):
        vectorstore = load_vector_store()
        
        if vectorstore == "needs_reupload":
            return None, None, "database_missing"
        
        llm = create_mistral_llm()
        
        if vectorstore is None:
            return None, None, "vectorstore_error"
        if llm is None:
            return None, None, "llm_error"
            
        return vectorstore, llm, "success"

# ==========================================
# UTILISATEUR CONNECTÉ - APPLICATION PRINCIPALE
# ==========================================

user_id = get_user_identifier()
user_info = st.session_state.user_info

# Initialisation du système (une seule fois)
if 'initialized' not in st.session_state:
    vectorstore, llm, status = initialize_system()
    st.session_state.vectorstore = vectorstore
    st.session_state.llm = llm
    st.session_state.initialization_status = status
    st.session_state.conversation_history = []
    st.session_state.scenarisation_history = []
    st.session_state.initialized = True

# Chargement du RAG utilisateur spécifique
if user_id and f'RAG_user_{user_id}' not in st.session_state:
    load_user_rag_state(user_id)

# Gestion des erreurs d'initialisation
if st.session_state.initialization_status == "database_missing":
    st.markdown("""
    <div class="banner">
        <h1>🎓 Assistant FPA - Ingénierie de Formation</h1>
        <p>Configuration initiale requise</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ Base vectorielle non trouvée ou incompatible")
    st.info("📋 Veuillez uploader votre base vectorielle pour commencer à utiliser l'assistant.")
    
    if not database_upload_interface():
        st.stop()

elif st.session_state.initialization_status in ["vectorstore_error", "llm_error"]:
    st.error("❌ Erreur lors de l'initialisation du système")
    st.stop()

# ==========================================
# FONCTIONS DE PAGES (identiques à avant mais raccourcies)
# ==========================================

def get_default_scenario_columns():
    """Retourne les colonnes par défaut pour la scénarisation"""
    return [
        "DURÉE", "HORAIRES", "CONTENU", "OBJECTIFS PÉDAGOGIQUES", "MÉTHODE",
        "RÉPARTITION DES APPRENANTS", "ACTIVITÉS - Formateur", "ACTIVITÉS - Apprenants", 
        "RESSOURCES et MATÉRIEL", "ÉVALUATION - Type", "ÉVALUATION - Sujet"
    ]

def column_selector_interface():
    """Interface pour sélectionner les colonnes du tableau de scénarisation"""
    st.markdown("""
    <div class="scenario-card">
        <h3>📋 Personnalisation du tableau de scénarisation</h3>
        <p>Sélectionnez les colonnes que vous souhaitez inclure dans votre tableau de scénarisation :</p>
    </div>
    """, unsafe_allow_html=True)
    
    default_columns = get_default_scenario_columns()
    
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = default_columns.copy()
    
    if 'custom_columns' not in st.session_state:
        st.session_state.custom_columns = []
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**📊 Colonnes disponibles :**")
        
        selected_defaults = []
        for col in default_columns:
            if st.checkbox(col, value=col in st.session_state.selected_columns, key=f"default_{col}"):
                selected_defaults.append(col)
        
        if st.session_state.custom_columns:
            st.markdown("**✨ Colonnes personnalisées :**")
            selected_customs = []
            for col in st.session_state.custom_columns:
                if st.checkbox(col, value=col in st.session_state.selected_columns, key=f"custom_{col}"):
                    selected_customs.append(col)
        else:
            selected_customs = []
        
        st.session_state.selected_columns = selected_defaults + selected_customs
    
    with col2:
        st.markdown("**➕ Ajouter une colonne personnalisée :**")
        
        new_column = st.text_input(
            "Nom de la nouvelle colonne",
            placeholder="Ex: MATÉRIEL SPÉCIFIQUE",
            key="new_column_input"
        )
        
        if st.button("➕ Ajouter", type="secondary", use_container_width=True):
            if new_column and new_column not in default_columns and new_column not in st.session_state.custom_columns:
                st.session_state.custom_columns.append(new_column)
                st.session_state.selected_columns.append(new_column)
                st.rerun()
            elif new_column in default_columns or new_column in st.session_state.custom_columns:
                st.warning("⚠️ Cette colonne existe déjà")
        
        if st.session_state.custom_columns:
            st.markdown("**🗑️ Gérer les colonnes personnalisées :**")
            col_to_remove = st.selectbox(
                "Supprimer une colonne",
                [""] + st.session_state.custom_columns,
                key="remove_column_select"
            )
            
            if st.button("🗑️ Supprimer", type="secondary", use_container_width=True):
                if col_to_remove:
                    st.session_state.custom_columns.remove(col_to_remove)
                    if col_to_remove in st.session_state.selected_columns:
                        st.session_state.selected_columns.remove(col_to_remove)
                    st.rerun()
        
        if st.button("🔄 Réinitialiser", type="secondary", use_container_width=True):
            st.session_state.selected_columns = default_columns.copy()
            st.session_state.custom_columns = []
            st.rerun()
    
    if st.session_state.selected_columns:
        st.markdown("**✅ Colonnes sélectionnées pour le tableau :**")
        cols_text = " | ".join(st.session_state.selected_columns)
        st.info(f"📋 {cols_text}")
        return st.session_state.selected_columns
    else:
        st.warning("⚠️ Veuillez sélectionner au moins une colonne")
        return []

def convert_columns_to_csv_structure(selected_columns):
    """Convertit la liste des colonnes sélectionnées en structure CSV pour le prompt"""
    header = "\t".join(selected_columns)
    
    example_row = []
    for col in selected_columns:
        if "DURÉE" in col.upper():
            example_row.append("20 min")
        elif "HORAIRES" in col.upper():
            example_row.append("9h00-9h20")
        elif "CONTENU" in col.upper():
            example_row.append("Introduction à la formation")
        elif "OBJECTIFS" in col.upper():
            example_row.append("Identifier le niveau initial des participants")
        elif "MÉTHODE" in col.upper():
            example_row.append("transmissive")
        elif "RÉPARTITION" in col.upper():
            example_row.append("groupe entier")
        elif "FORMATEUR" in col.upper() or ("ACTIVITÉS" in col.upper() and "FORMATEUR" in col.upper()):
            example_row.append("présentation du formateur, du programme")
        elif "APPRENANT" in col.upper() or ("ACTIVITÉS" in col.upper() and "APPRENANT" in col.upper()):
            example_row.append("écoute active, questions")
        elif "RESSOURCES" in col.upper() or "MATÉRIEL" in col.upper():
            example_row.append("présentation PowerPoint, liste des participants")
        elif "ÉVALUATION" in col.upper() and "TYPE" in col.upper():
            example_row.append("diagnostique")
        elif "ÉVALUATION" in col.upper() and "SUJET" in col.upper():
            example_row.append("connaissances préalables")
        elif "ÉVALUATION" in col.upper():
            example_row.append("formative")
        else:
            example_row.append("À compléter")
    
    example_line = "\t".join(example_row)
    return f"{header}\n{example_line}"

def main_chat_page():
    """Page principale de chat avec l'assistant FPA"""
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.conversation_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    <strong>Vous:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>Assistant FPA:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)

        if prompt := st.chat_input("Posez votre question sur la formation professionnelle"):
            st.session_state.conversation_history.append({
                'role': 'user', 
                'content': prompt
            })

            st.markdown(f"""
            <div class="user-message">
                <strong>Vous:</strong><br>
                {prompt}
            </div>
            """, unsafe_allow_html=True)

            with st.status("🔍 Recherche en cours...", expanded=True) as status:
                st.write("📚 Recherche des documents pertinents...")
                retrieved_docs = retrieve_documents(
                    st.session_state.vectorstore, 
                    prompt
                )
                
                st.write("🧠 Génération de la réponse...")
                response = generate_context_response(
                    st.session_state.llm, 
                    prompt, 
                    retrieved_docs,
                    st.session_state.conversation_history
                )
                
                status.update(label="✅ Recherche terminée", state="complete", expanded=False)
                
            st.markdown(f"""
            <div class="assistant-message">
                <strong>Assistant FPA:</strong><br>
                {response}
            </div>
            """, unsafe_allow_html=True)

            st.session_state.conversation_history.append({
                'role': 'assistant', 
                'content': response
            })

            with st.expander("📚 Documents sources"):
                for i, doc in enumerate(retrieved_docs, 1):
                    st.markdown(f"""
                    <div class="scenario-card">
                        <h4>Document {i}</h4>
                        <p><span class="badge badge-blue">Score: {doc['score']:.2f}</span></p>
                        <p><strong>Titre:</strong> {doc['title']}</p>
                        <hr>
                        {doc['content']}
                    </div>
                    """, unsafe_allow_html=True)

def scenarisation_page():
    """Page de scénarisation de formation avec colonnes personnalisables"""
    
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.markdown("""
        <div class="scenario-card">
            <h3>📋 Paramètres du scénario</h3>
        """, unsafe_allow_html=True)
        
        input_type = st.selectbox(
            "Type d'entrée",
            ["Programme", "Compétences"]
        )
        
        input_data = st.text_area(f"Contenu de {input_type.lower()}", 
            height=150,
            placeholder=f"Saisissez ici votre {input_type.lower()} de formation..."
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="scenario-card">
            <h3>⏱️ Durée de formation</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            duration_hours = st.number_input("Heures", min_value=0, max_value=40, value=3, step=1)
        with col2:
            duration_minutes = st.number_input("Minutes supplémentaires", min_value=0, max_value=59, value=30, step=5)
        
        total_duration_minutes = (duration_hours * 60) + duration_minutes
        
        st.markdown(f"""
        <div style="margin-top: 10px; margin-bottom: 10px;">
            <span style="background: {COLORS['primary']}; color: white; padding: 5px 10px; border-radius: 5px; font-size: 1rem;">
                ⏱️ Durée totale: {duration_hours}h{duration_minutes if duration_minutes > 0 else ''} ({total_duration_minutes} minutes)
            </span>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        selected_columns = column_selector_interface()
        
        if st.button("✨ Générer le scénario de formation", use_container_width=True):
            if input_data and selected_columns:
                user_content = f"""
                <div>
                    <p><strong>Type d'entrée:</strong> {input_type}</p>
                    <p><strong>Contenu:</strong> {input_data}</p>
                    <p><strong>Durée:</strong> {duration_hours}h{duration_minutes if duration_minutes > 0 else ''} ({total_duration_minutes} minutes)</p>
                    <p><strong>Colonnes du tableau:</strong> {', '.join(selected_columns)}</p>
                </div>
                """
                
                st.session_state.scenarisation_history.append({
                    'role': 'user', 
                    'content': user_content
                })
                
                st.markdown(f"""
                <div class="user-message">
                    <strong>Votre demande:</strong><br>
                    {user_content}
                </div>
                """, unsafe_allow_html=True)
                
                with st.status("🎯 Création de votre scénario de formation...", expanded=True) as status:
                    input_type_lower = input_type.lower()
                    
                    if input_type_lower == 'competences':
                        st.write("🔄 Reformulation des compétences selon l'approche par compétences...")
                        reformulated_competencies = reformulate_competencies_apc(
                            st.session_state.llm,
                            st.session_state.vectorstore,
                            input_data
                        )
                        input_data = reformulated_competencies
                    
                    st.write("📝 Génération du scénario pédagogique...")
                    
                    csv_structure = convert_columns_to_csv_structure(selected_columns)
                    
                    scenario = generate_structured_training_scenario(
                        st.session_state.llm,
                        st.session_state.vectorstore,
                        input_data,
                        input_type_lower,
                        total_duration_minutes,
                        custom_csv_structure=csv_structure
                    )
                    
                    status.update(label="✅ Scénario terminé!", state="complete", expanded=False)
                
                st.markdown(f"""
                <div class="assistant-message">
                    <h3>📋 Votre Scénario de Formation</h3>
                    <div class="info-box">
                        Ce scénario a été généré en fonction de vos paramètres et colonnes sélectionnées.
                    </div>
                    {scenario}
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.scenarisation_history.append({
                    'role': 'assistant', 
                    'content': scenario
                })
            elif not input_data:
                st.warning("⚠️ Veuillez saisir un contenu pour générer le scénario.")
            elif not selected_columns:
                st.warning("⚠️ Veuillez sélectionner au moins une colonne pour le tableau.")
                
    with right_col:
        st.markdown("""
        <div class="scenario-card">
            <h3>💡 Guide de scénarisation</h3>
            <p>Pour créer un scénario de formation efficace:</p>
            <ol>
                <li><strong>Choisissez un type d'entrée</strong></li>
                <li><strong>Définissez le contenu</strong> avec détails</li>
                <li><strong>Ajustez la durée</strong> selon vos contraintes</li>
                <li><strong>Personnalisez les colonnes</strong> du tableau</li>
                <li><strong>Générez votre scénario</strong></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# INTERFACE PRINCIPALE
# ==========================================

# Banner principal
st.markdown(f"""
<div class="banner">
    <h1>🎓 Assistant FPA - Ingénierie de Formation</h1>
    <p>Votre partenaire intelligent pour la conception et l'amélioration de vos formations professionnelles</p>
    <div class="user-info">
        👤 Connecté en tant que : {user_info['name']} ({user_info['email']})
        <img src="{user_info.get('picture', '')}" width="30" height="30" style="border-radius: 50%; margin-left: 10px;" />
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR AVEC OUTILS ET DÉCONNEXION
# ==========================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <div class="logo" style="margin: 0 auto;">FPA</div>
        <h3>Assistant Formation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Informations utilisateur et déconnexion
    st.markdown("---")
    
    # Photo de profil
    if user_info.get('picture'):
        st.image(user_info['picture'], width=80)
    
    st.markdown(f"**👤 Connecté :** {user_info['name']}")
    st.markdown(f"**📧 Email :** {user_info['email']}")
    
    if st.button("🚪 Se déconnecter", use_container_width=True):
        if user_id:
            save_user_rag_state(user_id)
        logout_user()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Outils supplémentaires")

    if st.button("📝 Exemple de plan de formation"):
        with st.spinner("📝 Génération d'un exemple de plan..."):
            exemple_plan = generate_example_training_plan(st.session_state.llm)
            st.markdown(f"""
            <div class="scenario-card">
                <h2>📝 Exemple de Plan de Formation</h2>
                <div class="info-box">
                    Ce plan peut servir de modèle pour vos propres formations.
                </div>
                {exemple_plan}
            </div>
            """, unsafe_allow_html=True)

    if st.button("🔍 Aide à l'ingénierie pédagogique"):
        with st.spinner("🔍 Génération de conseils..."):
            aide_ingenierie = generate_pedagogical_engineering_advice(st.session_state.llm)
            st.markdown(f"""
            <div class="scenario-card">
                <h2>🔍 Conseils d'Ingénierie Pédagogique</h2>
                <div class="info-box">
                    Conseils pour améliorer vos méthodes d'ingénierie pédagogique.
                </div>
                {aide_ingenierie}
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# ONGLETS DE NAVIGATION PRINCIPAL
# ==========================================

tab1, tab2, tab3 = st.tabs(["💬 Assistant FPA", "🎯 Scénarisation", f"📚 Mon RAG Personnel"])

with tab1:
    main_chat_page()

with tab2:
    scenarisation_page()

with tab3:
    if user_id:
        st.session_state.RAG_user = st.session_state.get(f'RAG_user_{user_id}')
        user_rag_page()
        st.session_state[f'RAG_user_{user_id}'] = st.session_state.RAG_user
        save_user_rag_state(user_id)
    else:
        st.error("❌ Erreur lors de la récupération de l'identifiant utilisateur")