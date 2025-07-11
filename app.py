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

# Définition des couleurs - CHARTE GRAPHIQUE EDSET
COLORS = {
    "primary": "#1D5B68",        # Bleu principal Edset
    "secondary": "#E6525E",      # Rouge accent Edset  
    "light_blue": "#94B7BD",     # Bleu ciel Edset
    "very_light_blue": "#DDE7E9", # Bleu très clair Edset
    "dark_gray": "#3F3F3F",      # Gris foncé Edset
    "light_gray": "#F5F5F6",     # Gris clair Edset
    "background": "#FFFFFF",     # Fond blanc (charte Edset)
    "text": "#3F3F3F"            # Texte gris foncé (charte Edset)
}

# Configuration CSS personnalisée - CHARTE GRAPHIQUE EDSET
def local_css():
    st.markdown(f"""
    <style>
        /* TYPOGRAPHIE EDSET : Omnes + Roboto */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,300;0,400;0,500;0,700;1,400&display=swap');
        
        .stApp {{
            background-color: {COLORS["background"]};
            color: {COLORS["text"]};
            font-family: 'Roboto', sans-serif;
        }}
        
        /* TITRES : Roboto Medium (selon charte Edset) */
        h1, h2, h3 {{
            color: {COLORS["primary"]};
            font-family: 'Roboto', sans-serif;
            font-weight: 500;
            line-height: 1.2;
        }}
        
        h1 {{
            font-size: 2.5rem;
            font-weight: 600;
        }}
        
        h2 {{
            font-size: 2rem;
            font-weight: 500;
        }}
        
        h3 {{
            font-size: 1.5rem;
            font-weight: 500;
        }}
        
        /* INPUTS : Style Edset */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
            background-color: {COLORS["background"]};
            color: {COLORS["text"]};
            border: 2px solid {COLORS["very_light_blue"]};
            border-radius: 8px;
            font-family: 'Roboto', sans-serif;
            font-weight: 300;
            transition: border-color 0.3s ease;
        }}
        
        .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {{
            border-color: {COLORS["primary"]};
            box-shadow: 0 0 0 2px {COLORS["light_blue"]}40;
        }}
        
        .stSelectbox>div>div>div {{
            background-color: {COLORS["background"]};
            color: {COLORS["text"]};
            border: 2px solid {COLORS["very_light_blue"]};
            border-radius: 8px;
            font-family: 'Roboto', sans-serif;
        }}
        
        /* BOUTONS : Style Edset moderne */
        .stButton>button {{
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["light_blue"]} 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-family: 'Roboto', sans-serif;
            font-weight: 500;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(29, 91, 104, 0.2);
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(29, 91, 104, 0.3);
            background: linear-gradient(135deg, {COLORS["light_blue"]} 0%, {COLORS["primary"]} 100%);
        }}
        
        /* SIDEBAR : Style Edset */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS["very_light_blue"]} 0%, {COLORS["light_gray"]} 100%);
            border-right: 1px solid {COLORS["very_light_blue"]};
        }}
        
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: {COLORS["primary"]};
        }}
        
        /* CARDS : Style Edset moderne */
        .scenario-card, .user-message, .assistant-message {{
            background: {COLORS["background"]};
            color: {COLORS["text"]};
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid {COLORS["secondary"]};
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            font-family: 'Roboto', sans-serif;
            font-weight: 300;
            line-height: 1.6;
        }}
        
        .user-message {{
            border-left-color: {COLORS["primary"]};
            background: linear-gradient(135deg, {COLORS["very_light_blue"]}20 0%, {COLORS["background"]} 100%);
        }}
        
        .assistant-message {{
            border-left-color: {COLORS["secondary"]};
            background: linear-gradient(135deg, {COLORS["light_gray"]}40 0%, {COLORS["background"]} 100%);
        }}
        
        .upload-box {{
            background: linear-gradient(135deg, {COLORS["very_light_blue"]}30 0%, {COLORS["background"]} 100%);
            border: 2px dashed {COLORS["light_blue"]};
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
            transition: all 0.3s ease;
        }}
        
        .upload-box:hover {{
            border-color: {COLORS["primary"]};
            background: linear-gradient(135deg, {COLORS["very_light_blue"]}50 0%, {COLORS["background"]} 100%);
        }}
        
        /* AUTH CONTAINER : Style Edset */
        .auth-container {{
            max-width: 500px;
            margin: 50px auto;
            padding: 40px;
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["light_blue"]} 100%);
            border-radius: 20px;
            text-align: center;
            color: white;
            box-shadow: 0 10px 30px rgba(29, 91, 104, 0.3);
        }}
        
        .user-info {{
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["light_blue"]} 100%);
            color: white;
            padding: 12px 24px;
            border-radius: 25px;
            margin: 15px 0;
            font-family: 'Roboto', sans-serif;
            font-weight: 400;
        }}
        
        .guide-section {{
            background: {COLORS["background"]};
            color: {COLORS["text"]};
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            border-left: 4px solid {COLORS["primary"]};
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            font-family: 'Roboto', sans-serif;
            font-weight: 300;
        }}
        
        .column-selector {{
            background: {COLORS["very_light_blue"]}30;
            padding: 15px;
            border-radius: 10px;
            margin: 8px 0;
            border: 1px solid {COLORS["very_light_blue"]};
        }}
        
        /* BANNER : Style sans dégradé */
        .banner {{
            background: {COLORS["primary"]};
            color: white;
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(29, 91, 104, 0.2);
            position: relative;
            overflow: hidden;
        }}
        
        .banner::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="white" stroke-opacity="0.1" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
            pointer-events: none;
        }}
        
        .banner h1 {{
            color: white;
            font-size: 2.8rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: relative;
            z-index: 1;
        }}
        
        .banner p {{
            font-size: 1.2rem;
            font-weight: 300;
            margin-bottom: 0;
            opacity: 0.95;
            position: relative;
            z-index: 1;
        }}
        
        /* LOGO : Style Edset */
        .logo {{
            width: 70px;
            height: 70px;
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 1.4rem;
            margin: 0 auto 15px auto;
            box-shadow: 0 4px 15px rgba(29, 91, 104, 0.3);
            font-family: 'Roboto', sans-serif;
        }}
        
        /* INFO BOX : Style Edset */
        .info-box {{
            background: linear-gradient(135deg, {COLORS["very_light_blue"]}60 0%, {COLORS["light_gray"]}40 100%);
            color: {COLORS["text"]};
            border-left: 4px solid {COLORS["primary"]};
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1.5rem 0;
            font-family: 'Roboto', sans-serif;
            font-weight: 300;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        /* BADGE : Style Edset */
        .badge {{
            display: inline-block;
            padding: 0.4em 0.8em;
            font-size: 0.8rem;
            font-weight: 500;
            line-height: 1;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 15px;
            font-family: 'Roboto', sans-serif;
        }}
        
        .badge-blue {{
            color: white;
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["light_blue"]} 100%);
            box-shadow: 0 2px 8px rgba(29, 91, 104, 0.2);
        }}
        
        /* TABS : Style Edset */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background: {COLORS["very_light_blue"]}30;
            border-radius: 15px;
            padding: 5px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px;
            padding: 12px 24px;
            font-family: 'Roboto', sans-serif;
            font-weight: 500;
            color: {COLORS["text"]};
            transition: all 0.3s ease;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["light_blue"]} 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(29, 91, 104, 0.2);
        }}
        
        /* CORRECTION MOBILE UNIQUEMENT */
        @media (max-width: 768px) {{
            .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
                font-size: 16px !important; /* Évite le zoom sur iOS */
            }}
            
            .scenario-card, .user-message, .assistant-message {{
                word-wrap: break-word;
                overflow-wrap: break-word;
                padding: 15px;
            }}
            
            .banner {{
                padding: 2rem 1rem;
                border-radius: 15px;
            }}
            
            .banner h1 {{
                font-size: 2rem;
            }}
            
            .banner p {{
                font-size: 1rem;
            }}
            
            .logo {{
                width: 60px;
                height: 60px;
                font-size: 1.2rem;
            }}
        }}
        
        /* ANIMATIONS ET TRANSITIONS */
        * {{
            transition: color 0.3s ease, background-color 0.3s ease, border-color 0.3s ease;
        }}
        
        /* SCROLLBAR : Style Edset */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {COLORS["light_gray"]};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["light_blue"]} 100%);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: linear-gradient(135deg, {COLORS["light_blue"]} 0%, {COLORS["primary"]} 100%);
        }}
    </style>
    """, unsafe_allow_html=True)

# Configuration de l'application Streamlit
st.set_page_config(
    page_title="Assistant Formation - Ingénierie pédagogique",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

local_css()

# ==========================================
# GUIDE D'UTILISATION
# ==========================================

def show_usage_guide():
    """Affiche le guide d'utilisation de l'assistant"""
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
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# GESTION DES COLONNES DE SCÉNARISATION
# ==========================================

def get_default_scenario_columns():
    """Retourne les colonnes par défaut pour la scénarisation"""
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
    """Interface pour sélectionner les colonnes du tableau de scénarisation"""
    st.markdown("""
    <div class="scenario-card">
        <h3>📋 Personnalisation du tableau de scénarisation</h3>
        <p>Sélectionnez les colonnes que vous souhaitez inclure dans votre tableau de scénarisation :</p>
    </div>
    """, unsafe_allow_html=True)
    
    default_columns = get_default_scenario_columns()
    
    # Initialiser les colonnes sélectionnées dans session_state si pas déjà fait
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = default_columns.copy()
    
    if 'custom_columns' not in st.session_state:
        st.session_state.custom_columns = []
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**📊 Colonnes disponibles :**")
        
        # Checkboxes pour les colonnes par défaut
        selected_defaults = []
        for col in default_columns:
            if st.checkbox(col, value=col in st.session_state.selected_columns, key=f"default_{col}"):
                selected_defaults.append(col)
        
        # Afficher les colonnes personnalisées ajoutées
        if st.session_state.custom_columns:
            st.markdown("**✨ Colonnes personnalisées :**")
            selected_customs = []
            for col in st.session_state.custom_columns:
                if st.checkbox(col, value=col in st.session_state.selected_columns, key=f"custom_{col}"):
                    selected_customs.append(col)
        else:
            selected_customs = []
        
        # Mettre à jour la sélection
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
        
        # Bouton pour supprimer les colonnes personnalisées
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
        
        # Bouton de reset
        if st.button("🔄 Réinitialiser", type="secondary", use_container_width=True):
            st.session_state.selected_columns = default_columns.copy()
            st.session_state.custom_columns = []
            st.rerun()
    
    # Afficher les colonnes sélectionnées
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
    # Créer l'en-tête CSV
    header = "\t".join(selected_columns)
    
    # Créer une ligne d'exemple pour chaque colonne
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

# ==========================================
# GESTION DE L'UTILISATEUR (STREAMLIT CLOUD)
# ==========================================

def get_user_identifier():
    """Récupère l'identifiant utilisateur de Streamlit Cloud"""
    try:
        # Utilise l'API native de Streamlit Cloud pour l'utilisateur connecté
        if hasattr(st, 'user') and st.user is not None:
            return st.user.email
        else:
            return None
    except Exception as e:
        st.error(f"Erreur lors de la récupération de l'utilisateur: {e}")
        return None

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

# ==========================================
# FONCTIONS ORIGINALES (CORRECTIONS APPLIQUÉES)
# ==========================================

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
# VÉRIFICATION AUTH ET POINT D'ENTRÉE PRINCIPAL
# ==========================================

# Vérification de l'authentification AVANT tout le reste
if not hasattr(st, 'user') or st.user is None or not st.user.is_logged_in:
    st.markdown("""
    <div class="auth-container">
        <h1>🎓 Assistant Formation</h1>
        <h2 style="font-style: italic; font-weight: 300; opacity: 0.9;">Ingénierie pédagogique</h2>
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
            st.switch_page("login")
    
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

# ==========================================
# UTILISATEUR CONNECTÉ - APPLICATION PRINCIPALE
# ==========================================

user_id = get_user_identifier()

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
        <h1>🎓 Assistant Formation</h1>
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
# PAGES PRINCIPALES
# ==========================================

def main_chat_page():
    """Page principale de chat avec l'assistant FPA"""
    
    st.markdown(f"""
    <div class="banner">
        <h1>🎓 Assistant Formation</h1>
        <p>Votre partenaire intelligent pour la formation professionnelle</p>
        <div class="user-info">
            👤 Connecté en tant que : {st.user.name} ({st.user.email})
        </div>
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
                    <strong>Assistant FPA:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)

        # Input pour le message de l'utilisateur
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
    
    st.markdown("""
    <div class="banner">
        <h1>🎯 Scénarisation</h1>
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
        
        # Interface de sélection des colonnes
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
                    
                    # Conversion des colonnes sélectionnées en structure CSV
                    csv_structure = convert_columns_to_csv_structure(selected_columns)
                    
                    # Appel modifié avec la structure CSV personnalisée
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
# SIDEBAR AVEC OUTILS ET DÉCONNEXION
# ==========================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <div class="logo">AF</div>
        <h3 style="color: #1D5B68; margin: 0; font-weight: 500;">Assistant Formation</h3>
        <p style="color: #94B7BD; font-size: 0.9rem; margin: 5px 0 0 0; font-style: italic;">Ingénierie pédagogique</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Informations utilisateur et déconnexion
    st.markdown("---")
    st.markdown(f"**👤 Connecté :** {st.user.name}")
    st.markdown(f"**📧 Email :** {st.user.email}")
    
    if st.button("🚪 Se déconnecter", use_container_width=True):
        if user_id:
            save_user_rag_state(user_id)
        # Correction pour Streamlit Cloud
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Guide d'utilisation
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

# ==========================================
# ONGLETS DE NAVIGATION PRINCIPAL
# ==========================================

tab1, tab2, tab3 = st.tabs(["💬 Assistant", "🎯 Scénarisation", f"📚 Mon RAG Personnel"])

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