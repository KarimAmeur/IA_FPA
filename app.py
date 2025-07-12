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
import secrets
import hashlib
import hmac
import base64
import json
from datetime import datetime, timedelta
import urllib.parse

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
# AUTHENTIFICATION AVEC AUTHLIB
# ==========================================

try:
    from authlib.integrations.requests_client import OAuth2Session
    from authlib.oauth2.rfc6749 import OAuth2Token
    AUTHLIB_AVAILABLE = True
except ImportError:
    AUTHLIB_AVAILABLE = False
    st.error("❌ Le package Authlib n'est pas installé. Ajoutez 'Authlib>=1.3.2' à votre requirements.txt")

# Configuration des APIs - Utilise les secrets Streamlit
try:
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
    
    # Configuration OAuth Google
    GOOGLE_CLIENT_ID = st.secrets["auth"]["client_id"]
    GOOGLE_CLIENT_SECRET = st.secrets["auth"]["client_secret"]
    GOOGLE_REDIRECT_URI = st.secrets["auth"]["redirect_uri"]
    COOKIE_SECRET = st.secrets["auth"]["cookie_secret"]
    
except Exception as e:
    st.error(f"❌ Erreur de configuration des secrets: {e}")
    st.stop()

# Configurer le token HuggingFace
if HUGGINGFACE_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HUGGINGFACE_TOKEN

# ==========================================
# CLASSE GOOGLE OAUTH AVEC AUTHLIB
# ==========================================

class GoogleOAuthManager:
    """Gestionnaire OAuth Google avec Authlib"""
    
    def __init__(self):
        self.client_id = GOOGLE_CLIENT_ID
        self.client_secret = GOOGLE_CLIENT_SECRET
        self.redirect_uri = GOOGLE_REDIRECT_URI
        self.scope = 'openid email profile'
        self.authorization_endpoint = 'https://accounts.google.com/o/oauth2/v2/auth'
        self.token_endpoint = 'https://oauth2.googleapis.com/token'
        self.userinfo_endpoint = 'https://www.googleapis.com/oauth2/v2/userinfo'
    
    def create_oauth_session(self, state=None):
        """Crée une session OAuth2 avec Authlib"""
        return OAuth2Session(
            client_id=self.client_id,
            client_secret=self.client_secret,
            scope=self.scope,
            redirect_uri=self.redirect_uri,
            state=state
        )
    
    def generate_authorization_url(self):
        """Génère l'URL d'autorisation avec état sécurisé"""
        # Générer un état sécurisé
        state = secrets.token_urlsafe(32)
        
        # Créer la session OAuth
        oauth = self.create_oauth_session(state=state)
        
        # Générer l'URL d'autorisation
        authorization_url, state = oauth.create_authorization_url(
            self.authorization_endpoint,
            access_type='offline',
            prompt='consent'
        )
        
        # Stocker l'état dans la session
        st.session_state.oauth_state = state
        
        return authorization_url
    
    def handle_callback(self, authorization_response_url):
        """Traite le callback d'autorisation et récupère le token"""
        try:
            # Vérifier l'état
            parsed_url = urllib.parse.urlparse(authorization_response_url)
            params = urllib.parse.parse_qs(parsed_url.query)
            
            received_state = params.get('state', [None])[0]
            stored_state = st.session_state.get('oauth_state')
            
            if not received_state or not stored_state or received_state != stored_state:
                return None, "État OAuth invalide"
            
            # Créer la session OAuth avec l'état stocké
            oauth = self.create_oauth_session(state=stored_state)
            
            # Échanger le code contre un token
            token = oauth.fetch_token(
                self.token_endpoint,
                authorization_response=authorization_response_url
            )
            
            return token, None
            
        except Exception as e:
            return None, f"Erreur lors du callback OAuth: {str(e)}"
    
    def get_user_info(self, token):
        """Récupère les informations utilisateur avec le token"""
        try:
            # Créer une session avec le token
            oauth = OAuth2Session(
                client_id=self.client_id,
                token=token
            )
            
            # Récupérer les informations utilisateur
            resp = oauth.get(self.userinfo_endpoint)
            if resp.status_code == 200:
                return resp.json()
            else:
                return None
                
        except Exception as e:
            st.error(f"Erreur lors de la récupération des infos utilisateur: {e}")
            return None

# ==========================================
# GESTION DES SESSIONS SÉCURISÉES
# ==========================================

def create_secure_session_token(user_info):
    """Crée un token de session sécurisé"""
    data = {
        'user_id': user_info['id'],
        'email': user_info['email'],
        'name': user_info['name'],
        'picture': user_info.get('picture', ''),
        'timestamp': datetime.now().isoformat()
    }
    
    # Créer un token signé avec HMAC
    message = base64.b64encode(json.dumps(data).encode()).decode()
    signature = hmac.new(
        COOKIE_SECRET.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return f"{message}.{signature}"

def verify_session_token(token):
    """Vérifie et décode un token de session"""
    try:
        if not token or '.' not in token:
            return None
            
        message, signature = token.rsplit('.', 1)
        
        # Vérifier la signature
        expected_signature = hmac.new(
            COOKIE_SECRET.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(signature, expected_signature):
            return None
        
        # Décoder les données
        data = json.loads(base64.b64decode(message).decode())
        
        # Vérifier l'expiration (24h)
        token_time = datetime.fromisoformat(data['timestamp'])
        if datetime.now() - token_time > timedelta(hours=24):
            return None
        
        return data
        
    except Exception:
        return None

# ==========================================
# GESTION DE L'AUTHENTIFICATION
# ==========================================

def check_authentication():
    """Vérifie l'authentification de l'utilisateur"""
    if not AUTHLIB_AVAILABLE:
        return False
    
    # Vérifier d'abord le callback OAuth dans l'URL
    query_params = st.query_params
    
    if 'code' in query_params and 'state' in query_params:
        handle_oauth_callback()
        return st.session_state.get('authenticated', False)
    
    # Vérifier le token de session existant
    if 'session_token' in st.session_state:
        user_data = verify_session_token(st.session_state.session_token)
        if user_data:
            st.session_state.user_info = user_data
            st.session_state.authenticated = True
            return True
        else:
            # Token expiré ou invalide
            clear_session()
    
    return False

def handle_oauth_callback():
    """Traite le callback OAuth de Google"""
    oauth_manager = GoogleOAuthManager()
    
    # Construire l'URL de réponse d'autorisation
    query_params = st.query_params
    current_url = st.context.headers.get('host', 'localhost:8501')
    protocol = 'https://' if 'streamlit.app' in current_url else 'http://'
    
    callback_url = f"{protocol}{current_url}{st.context.pathname}"
    if query_params:
        callback_url += '?' + '&'.join([f"{k}={v}" for k, v in query_params.items()])
    
    # Traiter le callback
    token, error = oauth_manager.handle_callback(callback_url)
    
    if error:
        st.error(f"❌ Erreur d'authentification: {error}")
        return
    
    if not token:
        st.error("❌ Aucun token reçu")
        return
    
    # Récupérer les informations utilisateur
    user_info = oauth_manager.get_user_info(token)
    
    if not user_info:
        st.error("❌ Impossible de récupérer les informations utilisateur")
        return
    
    # Créer le token de session sécurisé
    session_token = create_secure_session_token(user_info)
    
    # Stocker dans la session
    st.session_state.session_token = session_token
    st.session_state.user_info = user_info
    st.session_state.authenticated = True
    
    # Nettoyer l'URL
    st.query_params.clear()
    st.rerun()

def clear_session():
    """Nettoie la session utilisateur"""
    keys_to_remove = ['session_token', 'user_info', 'authenticated', 'oauth_state']
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

def show_login_page():
    """Affiche la page de connexion avec Google OAuth"""
    if not AUTHLIB_AVAILABLE:
        st.error("❌ Authlib n'est pas installé. Ajoutez 'Authlib>=1.3.2' à votre requirements.txt")
        return
    
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
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔐 Se connecter avec Google", 
                    type="primary", 
                    use_container_width=True):
            
            # Créer le gestionnaire OAuth
            oauth_manager = GoogleOAuthManager()
            
            # Générer l'URL d'autorisation
            auth_url = oauth_manager.generate_authorization_url()
            
            # Rediriger vers Google
            st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', unsafe_allow_html=True)
            st.info("🔄 Redirection vers Google...")

def get_user_identifier():
    """Récupère l'identifiant utilisateur sécurisé"""
    if st.session_state.get('authenticated') and 'user_info' in st.session_state:
        return st.session_state.user_info['email']
    return None

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
# FONCTIONS DE PAGES (raccourcies pour l'exemple)
# ==========================================

def get_default_scenario_columns():
    """Retourne les colonnes par défaut pour la scénarisation"""
    return [
        "DURÉE", "HORAIRES", "CONTENU", "OBJECTIFS PÉDAGOGIQUES", "MÉTHODE",
        "RÉPARTITION DES APPRENANTS", "ACTIVITÉS - Formateur", "ACTIVITÉS - Apprenants", 
        "RESSOURCES et MATÉRIEL", "ÉVALUATION - Type", "ÉVALUATION - Sujet"
    ]

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

def scenarisation_page():
    """Page de scénarisation simplifiée"""
    st.markdown("### 🎯 Scénarisation de Formation")
    st.info("Fonctionnalité en cours d'implémentation avec Authlib...")

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
        clear_session()
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