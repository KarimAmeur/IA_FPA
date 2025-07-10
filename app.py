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
from authlib.integrations.requests_client import OAuth2Session
import requests
from langchain_chroma import Chroma  # Updated import
from langchain_mistralai import ChatMistralAI
from prompting import (
    retrieve_documents,
    generate_context_response,
    generate_example_training_plan,
    generate_pedagogical_engineering_advice,
    reformulate_competencies_apc,
    generate_structured_training_scenario
)
from user_rag_page import user_rag_page
from mistralai.client import MistralClient

try:
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
except:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

try:
    CLIENT_ID = st.secrets["auth"]["client_id"]
    CLIENT_SECRET = st.secrets["auth"]["client_secret"]
    REDIRECT_URI = st.secrets["auth"]["redirect_uri"]
except Exception as e:
    st.error("Erreur récupération secrets OAuth")
    st.stop()
    
if HUGGINGFACE_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HUGGINGFACE_TOKEN

class MistralEmbeddings:
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
        
        .auth-container {{
            max-width: 500px;
            margin: 50px auto;
            padding: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            text-align: center;
            color: white;
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

st.set_page_config(
    page_title="Assistant FPA - Ingénierie de Formation",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

local_css()

def show_usage_guide():
    st.markdown("""
    <div class="guide-section">
        <h2>📖 Guide d'utilisation de l'Assistant FPA</h2>
        <p>Votre assistant intelligent pour l'ingénierie de formation professionnelle</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💬 **Onglet 1 : Assistant FPA**")
    st.markdown("""
    <div class="guide-section">
        <p><strong>🎯 Objectif :</strong> Poser des questions sur la formation professionnelle et obtenir des réponses basées sur une base de connaissances spécialisée.</p>
        
        <p><strong>🔧 Comment utiliser :</strong></p>
        <ul>
            <li>Tapez votre question dans le champ de saisie en bas</li>
            <li>L'assistant recherche dans la base de connaissances commune</li>
            <li>Vous obtenez une réponse détaillée avec les sources</li>
            <li>L'historique de conversation est conservé pour le contexte</li>
        </ul>
        
        <p><strong>💡 Exemples de questions :</strong></p>
        <ul>
            <li>"Comment construire un plan de formation efficace ?"</li>
            <li>"Quelles sont les méthodes pédagogiques actives ?"</li>
            <li>"Comment évaluer les compétences des apprenants ?"</li>
            <li>"Qu'est-ce que l'approche par compétences ?"</li>
        </ul>
        
        <p><strong>🛠️ Outils supplémentaires :</strong></p>
        <ul>
            <li><strong>Exemple de plan :</strong> Génère un modèle de plan de formation</li>
            <li><strong>Aide ingénierie :</strong> Conseils pour votre démarche pédagogique</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 **Onglet 2 : Scénarisation**")
    st.markdown("""
    <div class="guide-section">
        <p><strong>🎯 Objectif :</strong> Créer des scénarios pédagogiques détaillés et structurés selon l'Approche Par Compétences (APC).</p>
        
        <p><strong>🔧 Comment utiliser :</strong></p>
        <ol>
            <li><strong>Choisir le type d'entrée :</strong>
                <ul>
                    <li><strong>Programme :</strong> Décrivez le contenu à enseigner</li>
                    <li><strong>Compétences :</strong> Listez les compétences à développer</li>
                </ul>
            </li>
            <li><strong>Saisir le contenu :</strong> Décrivez en détail votre sujet de formation</li>
            <li><strong>Définir la durée :</strong> Précisez les heures et minutes de formation</li>
            <li><strong>Personnaliser les colonnes :</strong> Sélectionnez les colonnes du tableau de scénarisation</li>
            <li><strong>Générer :</strong> L'IA crée un scénario pédagogique complet</li>
        </ol>
        
        <p><strong>📋 Résultat obtenu :</strong></p>
        <ul>
            <li>Tableau de scénarisation détaillé avec timing précis</li>
            <li>Objectifs formulés selon l'APC de TARDIF</li>
            <li>Méthodes pédagogiques variées et adaptées</li>
            <li>Activités formateur/apprenant détaillées</li>
            <li>Ressources et modalités d'évaluation</li>
        </ul>
        
        <p><strong>💡 Conseils :</strong></p>
        <ul>
            <li>Plus votre description est détaillée, meilleur sera le scénario</li>
            <li>La durée sera respectée au minute près</li>
            <li>Les compétences seront automatiquement reformulées selon l'APC</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📚 **Onglet 3 : Mon RAG Personnel**")
    st.markdown("""
    <div class="guide-section">
        <p><strong>🎯 Objectif :</strong> Créer votre propre base de connaissances personnelle en ajoutant vos documents.</p>
        
        <p><strong>🔧 Comment utiliser :</strong></p>
        <ol>
            <li><strong>Upload de documents :</strong>
                <ul>
                    <li>Formats supportés : PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx)</li>
                    <li>Plusieurs fichiers simultanément possibles</li>
                    <li>Extraction automatique du texte</li>
                </ul>
            </li>
            <li><strong>Vectorisation :</strong>
                <ul>
                    <li>Découpage intelligent en chunks de 1024 caractères</li>
                    <li>Même modèle d'embedding que la base principale (Mistral)</li>
                    <li>Compatibilité garantie</li>
                </ul>
            </li>
            <li><strong>Recherche personnelle :</strong>
                <ul>
                    <li>Testez des requêtes dans vos documents</li>
                    <li>Scores de pertinence affichés</li>
                    <li>Extraits des documents sources</li>
                </ul>
            </li>
        </ol>
        
        <p><strong>🔒 Confidentialité :</strong></p>
        <ul>
            <li><strong>Isolation totale :</strong> Vos documents restent privés</li>
            <li><strong>Pas de partage :</strong> Aucun autre utilisateur n'y a accès</li>
            <li><strong>Stockage sécurisé :</strong> Base vectorielle personnelle</li>
        </ul>
        
        <p><strong>💡 Cas d'usage :</strong></p>
        <ul>
            <li>Ajouter vos supports de cours personnels</li>
            <li>Intégrer des documents d'entreprise</li>
            <li>Créer une base de ressources spécialisées</li>
            <li>Rechercher rapidement dans vos archives</li>
        </揃
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔐 **Sécurité et Confidentialité**")
    st.markdown("""
    <div class="guide-section">
        <p><strong>🛡️ Protection des données :</strong></p>
        <ul>
            <li><strong>Authentification Google :</strong> Connexion sécurisée via OAuth 2.0</li>
            <li><strong>Isolation utilisateur :</strong> Chaque utilisateur a son espace privé</li>
            <li><strong>Pas de partage :</strong> Vos données ne sont jamais visibles par d'autres</li>
            <li><strong>Base commune :</strong> Seule la base de formation générale est partagée</li>
        </ul>
        
        <p><strong>💾 Persistance :</strong></p>
        <ul>
            <li>Vos documents RAG personnels sont sauvegardés automatiquement</li>
            <li>Retrouvez vos données à chaque connexion</li>
            <li>Gestion de la suppression si nécessaire</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def get_default_scenario_columns():
    return [
        "DURÉE",
        "HORAIRES", 
        "CONTENU",
        "OBJECTIFS PÉDAGOGIQUES",
        "MÉTHODE",
        "RÉPARTITION DES APPRENANTS",
        "ACTIVITÉS - Formateur",
        "ACTIVITÉS - Apprenants", 
        "RESSOURCES et MATÉRIEL",
        "ÉVALUATION - Type",
        "ÉVALUATION - Sujet"
    ]

def column_selector_interface():
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

def require_google_login():
    from authlib.integrations.requests_client import OAuth2Session
    import requests

    CLIENT_ID = st.secrets["auth"]["client_id"]
    CLIENT_SECRET = st.secrets["auth"]["client_secret"]
    REDIRECT_URI = st.secrets["auth"]["redirect_uri"]

    oauth = OAuth2Session(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scope="openid email profile",
        redirect_uri=REDIRECT_URI,
    )

    if "token" not in st.session_state:
        query_params = st.query_params
        if "code" not in query_params:
            auth_url, state = oauth.create_authorization_url("https://accounts.google.com/o/oauth2/auth")
            st.session_state["oauth_state"] = state
            st.markdown(f"[Se connecter avec Google]({auth_url})", unsafe_allow_html=True)
            st.stop()
        else:
            code = query_params["code"][0]
            token = oauth.fetch_token("https://oauth2.googleapis.com/token", code=code)
            st.session_state["token"] = token
            st.experimental_rerun()

def get_user_identifier():
    if "token" not in st.session_state or "access_token" not in st.session_state["token"]:
        st.error("Token d'authentification manquant. Veuillez vous connecter.")
        return None

    access_token = st.session_state["token"]["access_token"]
    response = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    if response.status_code != 200:
        st.error("Impossible de récupérer les informations utilisateur.")
        return None

    user_info = response.json()
    st.success(f"Connecté : {user_info.get('name', 'Inconnu')} ({user_info.get('email', 'Email inconnu')})")
    return user_info.get("email")

def save_user_rag_state(user_id: str):
    pass

def load_user_rag_state(user_id: str):
    user_rag_dir = f"chroma_db_user_{user_id}"
    
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

if not st.user.is_logged_in:
    st.markdown("""
    <div class="auth-container">
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
            st.login()
    
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #888;">
        <p>🔒 <strong>Sécurité et confidentialité :</strong></p>
        <p>• Vos données sont privées et sécurisées</p>
        <p>• Chaque utilisateur a son propre espace isolé</p>
        <p>• Aucune donnée partagée entre utilisateurs</p>
        <p>• Authentification déléguée à Google (OAuth 2.0)</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

user_id = get_user_identifier()
if 'initialized' not in st.session_state:
    vectorstore, llm, status = initialize_system()
    st.session_state.vectorstore = vectorstore
    st.session_state.llm = llm
    st.session_state.initialization_status = status
    st.session_state.conversation_history = []
    st.session_state.scenarisation_history = []
    st.session_state.initialized = True

if user_id and f'RAG_user_{user_id}' not in st.session_state:
    load_user_rag_state(user_id)

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

def main_chat_page():
    require_google_login()
    st.markdown(f"""
    <div class="banner">
        <h1>🎓 Assistant FPA - Ingénierie de Formation</h1>
        <p>Votre partenaire intelligent pour la conception et l'amélioration de vos formations professionnelles</p>
        <div class="user-info">
            👤 Connecté en tant que : {st.user.name} ({st.user.email})
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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
    st.markdown("""
    <div class="banner">
        <h1>🎯 Scénarisation de Formation</h1>
        <p>Créez des scénarios pédagogiques adaptés à vos objectifs</p>
    </div>
    """, unsafe_allow_html=True)
    
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

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <div class="logo" style="margin: 0 auto;">FPA</div>
        <h3>Assistant Formation</h3>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**👤 Connecté :** {st.user.name}")
    st.markdown(f"**📧 Email :** {st.user.email}")
    if st.button("🚪 Se déconnecter", use_container_width=True):
        if user_id:
            save_user_rag_state(user_id)
        st.logout()
    st.markdown("---")
    if st.button("📖 Guide d'utilisation", use_container_width=True, type="secondary"):
        st.session_state.show_guide = not st.session_state.get('show_guide', False)
    if st.session_state.get('show_guide', False):
        with st.expander("📖 Guide complet", expanded=True):
            show_usage_guide()
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