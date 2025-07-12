# PATCH POUR STREAMLIT CLOUD - À placer en tout début de app.py
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # pysqlite3 n'est pas disponible, continuer avec sqlite3 normal
    pass

import streamlit as st
import os
import zipfile
import shutil
from pathlib import Path
from typing import List

# CORRECTION: Import corrigé pour Chroma
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

# Configuration des APIs - Utilise les secrets Streamlit
try:
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
except:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Configurer le token HuggingFace
if HUGGINGFACE_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HUGGINGFACE_TOKEN

# NOUVELLE CLASSE: Embeddings Mistral compatible avec LangChain
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

# Configuration CSS personnalisée - OPTIMISÉE MOBILE
def local_css():
    st.markdown(f"""
    <style>
        /* Configuration de base */
        .stApp {{
            background-color: {COLORS["background"]};
            color: {COLORS["text"]};
        }}
        
        /* Headers responsive */
        h1, h2, h3 {{
            color: {COLORS["light_blue"]};
            font-family: 'Helvetica Neue', sans-serif;
        }}
        
        /* Responsive pour mobile */
        @media (max-width: 768px) {{
            .stApp {{
                padding: 0.5rem !important;
            }}
            
            /* Réduire l'espace entre les éléments sur mobile */
            .element-container {{
                margin-bottom: 0.5rem !important;
            }}
            
            /* Adapter la taille du texte */
            h1 {{
                font-size: 1.5rem !important;
            }}
            
            h2 {{
                font-size: 1.3rem !important;
            }}
            
            h3 {{
                font-size: 1.1rem !important;
            }}
            
            /* Optimiser les colonnes sur mobile */
            .stColumn {{
                padding: 0.25rem !important;
            }}
            
            /* Messages plus compacts */
            .user-message, .assistant-message, .scenario-card {{
                padding: 10px !important;
                margin-bottom: 8px !important;
                font-size: 0.9rem !important;
            }}
            
            /* Sidebar plus accessible */
            [data-testid="stSidebar"] {{
                width: 100% !important;
            }}
            
            /* Boutons plus grands sur mobile */
            .stButton > button {{
                width: 100% !important;
                padding: 12px !important;
                font-size: 1rem !important;
            }}
            
            /* Input plus accessibles */
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea {{
                font-size: 16px !important; /* Évite le zoom sur iOS */
            }}
            
            /* Upload box plus compacte */
            .upload-box {{
                padding: 15px !important;
                margin: 10px 0 !important;
            }}
            
            /* Banner plus compact */
            .banner {{
                padding: 1rem !important;
                margin-bottom: 1rem !important;
            }}
            
            .banner h1 {{
                font-size: 1.4rem !important;
                margin-bottom: 0.5rem !important;
            }}
            
            .banner p {{
                font-size: 0.9rem !important;
                margin: 0 !important;
            }}
        }}
        
        /* Inputs et sélecteurs */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
            background-color: {COLORS["dark_gray"]};
            color: {COLORS["text"]};
            border: 1px solid {COLORS["light_blue"]};
            border-radius: 6px;
        }}
        
        .stSelectbox>div>div>div {{
            background-color: {COLORS["dark_gray"]};
            color: {COLORS["text"]};
            border-radius: 6px;
            border: 1px solid {COLORS["light_blue"]};
        }}
        
        /* Boutons */
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
            transform: translateY(-1px);
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {COLORS["dark_gray"]};
            color: {COLORS["text"]};
        }}
        
        /* Cards et messages */
        .scenario-card, .user-message, .assistant-message {{
            background-color: {COLORS["dark_gray"]};
            color: {COLORS["text"]};
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid {COLORS["secondary"]};
            margin-bottom: 10px;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }}
        
        .upload-box {{
            background-color: {COLORS["dark_gray"]};
            border: 2px dashed {COLORS["light_blue"]};
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
        }}
        
        /* Banner principal */
        .banner {{
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
        }}
        
        /* Info box */
        .info-box {{
            background: {COLORS["very_light_blue"]};
            color: {COLORS["dark_gray"]};
            border-left: 4px solid {COLORS["primary"]};
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }}
        
        /* Badge */
        .badge {{
            display: inline-block;
            padding: 0.25em 0.6em;
            font-size: 0.75em;
            font-weight: 700;
            line-height: 1;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 0.25rem;
        }}
        
        .badge-blue {{
            color: #fff;
            background-color: {COLORS["primary"]};
        }}
        
        /* Logo */
        .logo {{
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 1.2rem;
            margin-bottom: 10px;
        }}
        
        /* Optimisations de performance pour mobile */
        * {{
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        
        /* Éviter les débordements */
        .element-container {{
            max-width: 100%;
            overflow-x: hidden;
        }}
        
        /* Chat input optimisé pour mobile */
        .stChatInput {{
            position: sticky;
            bottom: 0;
            background: {COLORS["background"]};
            padding: 10px 0;
            border-top: 1px solid {COLORS["dark_gray"]};
        }}
    </style>
    """, unsafe_allow_html=True)

# Configuration de l'application Streamlit - OPTIMISÉE MOBILE
st.set_page_config(
    page_title="📘 Assistant FPA",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="auto",  # Auto-collapse sur mobile
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Assistant FPA - Ingénierie de Formation"
    }
)

local_css()

# Détection mobile via user agent
def is_mobile():
    """Détecte si l'utilisateur est sur mobile"""
    try:
        # Utiliser les headers de requête si disponibles
        user_agent = st.context.headers.get("user-agent", "").lower()
        mobile_keywords = ["mobile", "android", "iphone", "ipad", "tablet"]
        return any(keyword in user_agent for keyword in mobile_keywords)
    except:
        # Fallback: utiliser JavaScript pour détecter la taille d'écran
        return False

# NOUVELLE FONCTION : Nettoyage automatique ChromaDB
def clean_corrupted_chromadb(db_path):
    """Nettoie automatiquement une base ChromaDB corrompue"""
    try:
        st.warning("🔧 Détection d'une base ChromaDB incompatible...")
        st.info("🗑️ Suppression automatique de l'ancienne base...")
        
        # Supprimer le dossier corrompu
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        st.success("✅ Base corrompue supprimée avec succès !")
        st.info("📋 Veuillez re-uploader votre fichier chromadb_formation.zip")
        
        return True
    except Exception as e:
        st.error(f"❌ Erreur lors du nettoyage: {e}")
        return False

# Fonctions de gestion de la base vectorielle
@st.cache_resource
def extract_database_if_needed():
    """Décompresse automatiquement la base vectorielle si nécessaire"""
    
    db_path = "chromadb_formation"
    zip_path = "chromadb_formation.zip"
    
    # Si le dossier existe déjà et n'est pas vide, pas besoin de décompresser
    if os.path.exists(db_path) and os.listdir(db_path):
        st.success("✅ Base vectorielle déjà disponible")
        return True
    
    # Si le zip existe, le décompresser
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
    """Interface d'upload de la base vectorielle - OPTIMISÉE MOBILE"""
    
    # Interface adaptée mobile
    mobile = is_mobile()
    
    st.markdown(f"""
    <div class="upload-box">
        <h3>📤 Upload de votre base vectorielle</h3>
        <p>{"Base ChromaDB requise" if mobile else "La base vectorielle ChromaDB est nécessaire pour le fonctionnement de l'assistant."}</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Fichier chromadb_formation.zip",
        type="zip",
        help="Base vectorielle au format ZIP",
        label_visibility="visible" if not mobile else "collapsed"
    )
    
    if uploaded_file is not None:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("💾 Sauvegarde...")
            progress_bar.progress(25)
            
            with open("chromadb_formation.zip", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            status_text.text("📦 Décompression...")
            progress_bar.progress(50)
            
            with zipfile.ZipFile("chromadb_formation.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            progress_bar.progress(100)
            status_text.text("✅ Installation réussie!")
            
            st.success("🎉 Base vectorielle installée ! Redémarrage...")
            st.balloons()
            
            import time
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erreur: {e}")
            st.info("💡 Vérifiez que le ZIP contient 'chromadb_formation'")
    
    return False

# CORRECTION: Fonction de chargement des embeddings Mistral
@st.cache_resource
def load_embedding_model():
    """Charge le modèle d'embedding Mistral compatible avec la base vectorielle"""
    try:
        if not MISTRAL_API_KEY:
            st.error("❌ Clé API Mistral manquante")
            return None
            
        return MistralEmbeddings(api_key=MISTRAL_API_KEY, model="mistral-embed")
        
    except Exception as e:
        st.error(f"❌ Erreur modèle embedding: {e}")
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
            st.error("❌ Base vectorielle non trouvée")
            return None
            
        try:
            vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
            # Test de fonctionnement
            vectorstore.similarity_search("test", k=1)
            return vectorstore
            
        except Exception as chroma_error:
            error_msg = str(chroma_error).lower()
            
            # Détecter l'erreur de colonne manquante
            if "no such column: collections.topic" in error_msg:
                st.error("❌ Base ChromaDB incompatible")
                
                if clean_corrupted_chromadb(db_path):
                    return "needs_reupload"
                else:
                    return None
                    
            # Détecter l'erreur de dimension d'embedding
            elif "embedding dimension" in error_msg and "does not match" in error_msg:
                st.error("❌ Incompatibilité embeddings")
                st.info("💡 Base créée avec des embeddings différents")
                
                if clean_corrupted_chromadb(db_path):
                    return "needs_reupload"
                else:
                    return None
            else:
                st.error(f"❌ Erreur ChromaDB: {chroma_error}")
                return None
                
    except Exception as e:
        st.error(f"❌ Erreur chargement base: {e}")
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
        st.error(f"❌ Erreur création Mistral: {e}")
        return None

# Initialisation automatique du système
def initialize_system():
    """Initialise le système avec gestion automatique de la base et des erreurs"""
    
    if not extract_database_if_needed():
        return None, None, "database_missing"
    
    with st.spinner("🚀 Initialisation..."):
        vectorstore = load_vector_store()
        
        if vectorstore == "needs_reupload":
            return None, None, "database_missing"
        
        llm = create_mistral_llm()
        
        if vectorstore is None:
            return None, None, "vectorstore_error"
        if llm is None:
            return None, None, "llm_error"
            
        return vectorstore, llm, "success"

# Vérification et initialisation
if 'initialized' not in st.session_state:
    vectorstore, llm, status = initialize_system()
    st.session_state.vectorstore = vectorstore
    st.session_state.llm = llm
    st.session_state.initialization_status = status
    st.session_state.conversation_history = []
    st.session_state.scenarisation_history = []
    st.session_state.initialized = True

# Gestion des erreurs d'initialisation
if st.session_state.initialization_status == "database_missing":
    st.markdown("""
    <div class="banner">
        <h1>🎓 Assistant FPA</h1>
        <p>Configuration initiale requise</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ Base vectorielle requise")
    st.info("📋 Uploadez votre base vectorielle pour commencer.")
    
    if not database_upload_interface():
        st.stop()

elif st.session_state.initialization_status in ["vectorstore_error", "llm_error"]:
    st.error("❌ Erreur d'initialisation")
    
    # Interface de diagnostic - COMPACTE POUR MOBILE
    st.subheader("🔧 Diagnostic")
    
    # Version mobile : boutons en vertical
    mobile = is_mobile()
    cols = st.columns(1) if mobile else st.columns(3)
    
    if mobile:
        if st.button("🧪 Test API Mistral", use_container_width=True):
            try:
                client = MistralClient(api_key=MISTRAL_API_KEY)
                models = client.list_models()
                st.success("✅ API Mistral OK")
            except Exception as e:
                st.error(f"❌ API Mistral: {e}")
        
        if st.button("🤗 Test HuggingFace", use_container_width=True):
            try:
                if HUGGINGFACE_TOKEN:
                    from huggingface_hub import whoami
                    info = whoami(token=HUGGINGFACE_TOKEN)
                    st.success(f"✅ Token HF: {info['name']}")
                else:
                    st.warning("⚠️ Token HF manquant")
            except Exception as e:
                st.error(f"❌ Token HF: {e}")
        
        if st.button("📁 Vérifier base", use_container_width=True):
            if os.path.exists("chromadb_formation"):
                files = os.listdir("chromadb_formation")
                st.info(f"📂 {len(files)} fichiers trouvés")
            else:
                st.error("❌ Dossier 'chromadb_formation' absent")
    else:
        # Version desktop : boutons en horizontal
        with cols[0]:
            if st.button("🧪 Test API Mistral"):
                try:
                    client = MistralClient(api_key=MISTRAL_API_KEY)
                    models = client.list_models()
                    st.success("✅ API Mistral accessible")
                except Exception as e:
                    st.error(f"❌ Erreur API Mistral: {e}")
        
        with cols[1]:
            if st.button("🤗 Test HuggingFace"):
                try:
                    if HUGGINGFACE_TOKEN:
                        from huggingface_hub import whoami
                        info = whoami(token=HUGGINGFACE_TOKEN)
                        st.success(f"✅ Token HF valide: {info['name']}")
                    else:
                        st.warning("⚠️ Token HuggingFace non configuré")
                except Exception as e:
                    st.error(f"❌ Token HF invalide: {e}")
        
        with cols[2]:
            if st.button("📁 Vérifier base"):
                if os.path.exists("chromadb_formation"):
                    files = os.listdir("chromadb_formation")
                    st.info(f"📂 Dossier trouvé avec {len(files)} fichiers")
                else:
                    st.error("❌ Dossier 'chromadb_formation' non trouvé")
    
    if st.button("🔄 Réessayer", use_container_width=True):
        st.session_state.initialized = False
        st.rerun()
    
    st.stop()

# Page principale de chat - OPTIMISÉE MOBILE
def main_chat_page():
    """Page principale de chat avec l'assistant FPA"""
    
    mobile = is_mobile()
    
    st.markdown(f"""
    <div class="banner">
        <h1>{"🎓 FPA" if mobile else "🎓 Assistant FPA - Ingénierie de Formation"}</h1>
        <p>{"Assistant formation" if mobile else "Votre partenaire intelligent pour la conception et l'amélioration de vos formations professionnelles"}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Conteneur principal
    chat_container = st.container()
    
    with chat_container:
        # Afficher l'historique de conversation
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
                    <strong>Assistant:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)

        # Input pour le message - optimisé mobile
        placeholder = "Question formation..." if mobile else "Posez votre question sur la formation professionnelle"
        
        if prompt := st.chat_input(placeholder):
            # Ajouter le message de l'utilisateur
            st.session_state.conversation_history.append({
                'role': 'user', 
                'content': prompt
            })

            # Afficher le message
            st.markdown(f"""
            <div class="user-message">
                <strong>Vous:</strong><br>
                {prompt}
            </div>
            """, unsafe_allow_html=True)

            # Générer la réponse
            status_text = "🔍 Recherche..." if mobile else "🔍 Recherche en cours..."
            
            with st.status(status_text, expanded=not mobile) as status:
                if not mobile:
                    st.write("📚 Recherche des documents pertinents...")
                    
                retrieved_docs = retrieve_documents(
                    st.session_state.vectorstore, 
                    prompt
                )
                
                if not mobile:
                    st.write("🧠 Génération de la réponse...")
                    
                response = generate_context_response(
                    st.session_state.llm, 
                    prompt, 
                    retrieved_docs,
                    st.session_state.conversation_history
                )
                
                status.update(label="✅ Terminé", state="complete", expanded=False)
                
            # Afficher la réponse
            st.markdown(f"""
            <div class="assistant-message">
                <strong>Assistant:</strong><br>
                {response}
            </div>
            """, unsafe_allow_html=True)

            # Ajouter à l'historique
            st.session_state.conversation_history.append({
                'role': 'assistant', 
                'content': response
            })

            # Sources - collapsed par défaut sur mobile
            with st.expander("📚 Sources", expanded=not mobile):
                for i, doc in enumerate(retrieved_docs, 1):
                    score_display = f"{doc['score']:.2f}" if not mobile else f"{doc['score']:.1f}"
                    title_display = doc['title'][:50] + "..." if mobile and len(doc['title']) > 50 else doc['title']
                    
                    st.markdown(f"""
                    <div class="scenario-card">
                        <h4>Doc {i}</h4>
                        <p><span class="badge badge-blue">Score: {score_display}</span></p>
                        <p><strong>Titre:</strong> {title_display}</p>
                        <hr>
                        {doc['content'][:300] + "..." if mobile and len(doc['content']) > 300 else doc['content']}
                    </div>
                    """, unsafe_allow_html=True)

    # Sidebar avec outils - adapté mobile
    if not mobile:
        st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <div class="logo" style="margin: 0 auto;">FPA</div>
            <h3>Assistant Formation</h3>
        </div>
        """, unsafe_allow_html=True)
    
    sidebar_container = st.sidebar if not mobile else st
    
    with sidebar_container:
        if mobile:
            st.markdown("### 🛠️ Outils")
        else:
            st.markdown("### 🛠️ Outils supplémentaires")

        col1, col2 = st.columns(2) if mobile else (None, None)
        
        button_width = not mobile  # use_container_width pour desktop
        
        if mobile:
            with col1:
                plan_button = st.button("📝 Exemple plan", use_container_width=True)
            with col2:
                aide_button = st.button("🔍 Aide ingé.", use_container_width=True)
        else:
            plan_button = st.button("📝 Exemple de plan de formation")
            aide_button = st.button("🔍 Aide à l'ingénierie pédagogique")

        if plan_button:
            with st.spinner("📝 Génération..."):
                exemple_plan = generate_example_training_plan(st.session_state.llm)
                st.markdown(f"""
                <div class="scenario-card">
                    <h2>📝 Exemple de Plan</h2>
                    <div class="info-box">
                        Ce plan peut servir de modèle.
                    </div>
                    {exemple_plan}
                </div>
                """, unsafe_allow_html=True)

        if aide_button:
            with st.spinner("🔍 Génération..."):
                aide_ingenierie = generate_pedagogical_engineering_advice(st.session_state.llm)
                st.markdown(f"""
                <div class="scenario-card">
                    <h2>🔍 Conseils Ingénierie</h2>
                    <div class="info-box">
                        Conseils pour améliorer vos méthodes.
                    </div>
                    {aide_ingenierie}
                </div>
                """, unsafe_allow_html=True)

# Page de scénarisation - OPTIMISÉE MOBILE
def scenarisation_page():
    """Page de scénarisation de formation"""
    
    mobile = is_mobile()
    
    st.markdown(f"""
    <div class="banner">
        <h1>{"🎯 Scénarisation" if mobile else "🎯 Scénarisation de Formation"}</h1>
        <p>{"Créez des scénarios adaptés" if mobile else "Créez des scénarios pédagogiques adaptés à vos objectifs"}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout adaptatif
    if mobile:
        # Sur mobile : tout en vertical
        main_col = st.container()
        side_col = st.container()
    else:
        # Sur desktop : 2 colonnes
        main_col, side_col = st.columns([2, 1])
    
    with main_col:
        st.markdown("""
        <div class="scenario-card">
            <h3>📋 Paramètres du scénario</h3>
        """, unsafe_allow_html=True)
        
        input_type = st.selectbox(
            "Type d'entrée",
            ["Titre", "Programme", "Compétences"],
            help="Choisissez le type de contenu à utiliser"
        )
        
        # TextArea adapté mobile
        placeholder_text = {
            "Titre": "Ex: Formation à la communication interpersonnelle",
            "Programme": "Ex: Module 1: Les bases\nModule 2: Pratique\n...",
            "Compétences": "Ex: Savoir communiquer efficacement\nÊtre capable de..."
        }
        
        input_data = st.text_area(
            f"Contenu de {input_type.lower()}", 
            height=120 if mobile else 150,
            placeholder=placeholder_text[input_type],
            help=f"Décrivez votre {input_type.lower()} de formation"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="scenario-card">
            <h3>⏱️ Durée de formation</h3>
        """, unsafe_allow_html=True)
        
        # Layout durée adapté mobile
        if mobile:
            duration_hours = st.number_input("Heures", min_value=0, max_value=40, value=3, step=1)
            duration_minutes = st.number_input("Minutes", min_value=0, max_value=59, value=30, step=5)
        else:
            col1, col2 = st.columns(2)
            with col1:
                duration_hours = st.number_input("Heures", min_value=0, max_value=40, value=3, step=1)
            with col2:
                duration_minutes = st.number_input("Minutes supplémentaires", min_value=0, max_value=59, value=30, step=5)
        
        total_duration_minutes = (duration_hours * 60) + duration_minutes
        
        # Affichage durée total adapté
        duration_text = f"{duration_hours}h{duration_minutes if duration_minutes > 0 else ''}"
        
        st.markdown(f"""
        <div style="margin: 10px 0;">
            <span style="background: {COLORS['primary']}; color: white; padding: 5px 10px; border-radius: 5px; font-size: {'0.9rem' if mobile else '1rem'};">
                ⏱️ Durée: {duration_text} ({total_duration_minutes} min)
            </span>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Bouton de génération
        button_text = "✨ Générer le scénario" if mobile else "✨ Générer le scénario de formation"
        
        if st.button(button_text, use_container_width=True):
            if input_data:
                user_content = f"""
                <div>
                    <p><strong>Type:</strong> {input_type}</p>
                    <p><strong>Contenu:</strong> {input_data[:100] + "..." if mobile and len(input_data) > 100 else input_data}</p>
                    <p><strong>Durée:</strong> {duration_text} ({total_duration_minutes} min)</p>
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
                
                # Génération adaptée mobile
                status_text = "🎯 Création..." if mobile else "🎯 Création de votre scénario de formation..."
                
                with st.status(status_text, expanded=not mobile) as status:
                    input_type_lower = input_type.lower()
                    
                    if input_type_lower == 'competences':
                        if not mobile:
                            st.write("🔄 Reformulation des compétences...")
                        reformulated_competencies = reformulate_competencies_apc(
                            st.session_state.llm,
                            st.session_state.vectorstore,
                            input_data
                        )
                        input_data = reformulated_competencies
                    
                    if not mobile:
                        st.write("📝 Génération du scénario pédagogique...")
                    scenario = generate_structured_training_scenario(
                        st.session_state.llm,
                        st.session_state.vectorstore,
                        input_data,
                        input_type_lower,
                        total_duration_minutes
                    )
                    
                    status.update(label="✅ Terminé!", state="complete", expanded=False)
                
                st.markdown(f"""
                <div class="assistant-message">
                    <h3>📋 {"Scénario" if mobile else "Votre Scénario de Formation"}</h3>
                    <div class="info-box">
                        {"Scénario généré selon vos paramètres." if mobile else "Ce scénario a été généré en fonction de vos paramètres."}
                    </div>
                    {scenario}
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.scenarisation_history.append({
                    'role': 'assistant', 
                    'content': scenario
                })
            else:
                st.warning("⚠️ Veuillez saisir un contenu pour générer le scénario.")
                
    with side_col:
        if mobile:
            st.markdown("---")  # Séparateur sur mobile
            
        st.markdown(f"""
        <div class="scenario-card">
            <h3>💡 {"Guide" if mobile else "Guide de scénarisation"}</h3>
            <p>{"Pour créer un scénario efficace:" if mobile else "Pour créer un scénario de formation efficace:"}</p>
            <ol>
                <li><strong>{"Type d'entrée" if mobile else "Choisissez un type d'entrée"}</strong></li>
                <li><strong>{"Contenu détaillé" if mobile else "Définissez le contenu"}</strong> {"avec détails" if not mobile else ""}</li>
                <li><strong>{"Durée adaptée" if mobile else "Ajustez la durée"}</strong> {"selon contraintes" if mobile else "selon vos contraintes"}</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Historique des scénarisations sur mobile
        if mobile and st.session_state.scenarisation_history:
            with st.expander("📜 Historique scénarios"):
                for i, message in enumerate(st.session_state.scenarisation_history[-3:]):  # 3 derniers sur mobile
                    if message['role'] == 'assistant':
                        preview = message['content'][:150] + "..." if len(message['content']) > 150 else message['content']
                        st.markdown(f"""
                        <div style="background: {COLORS['dark_gray']}; padding: 8px; border-radius: 5px; margin: 5px 0; font-size: 0.8rem;">
                            {preview}
                        </div>
                        """, unsafe_allow_html=True)

# CORRECTION: Onglets de navigation avec gestion mobile
def main_navigation():
    """Navigation principale avec adaptation mobile"""
    
    mobile = is_mobile()
    
    # Labels adaptés mobile
    if mobile:
        tab_labels = ["💬 Chat", "🎯 Scénario", "📚 RAG"]
    else:
        tab_labels = ["💬 Assistant FPA", "🎯 Scénarisation", "📚 RAG Personnel"]
    
    tab1, tab2, tab3 = st.tabs(tab_labels)

    with tab1:
        main_chat_page()

    with tab2:
        scenarisation_page()

    with tab3:
        try:
            user_rag_page()  # Appel de la fonction pour afficher la page RAG Personnel
        except Exception as e:
            st.error(f"❌ Erreur page RAG: {e}")
            st.info("💡 Assurez-vous que le module user_rag_page est disponible")

# Point d'entrée principal
if __name__ == "__main__":
    main_navigation()