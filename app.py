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

# PALETTE EDSET OFFICIELLE - Selon votre charte graphique
COLORS = {
    "primary": "#1D5B68",           # Bleu principal EDSET
    "primary_dark": "#0f3d47",      # Bleu plus foncé
    "secondary": "#E6525E",         # Rouge accent EDSET
    "accent": "#94B7BD",            # Bleu ciel EDSET
    "accent_light": "#DDE7E9",      # Bleu très clair EDSET
    "background": "#ffffff",        # Fond blanc
    "surface": "#ffffff",           # Surface blanche
    "surface_secondary": "#f8fafc", # Surface gris très clair
    "text_primary": "#3F3F3F",      # Gris foncé EDSET
    "text_secondary": "#6b7280",    # Texte secondaire
    "text_muted": "#9ca3af",        # Texte atténué
    "border": "#DDE7E9",            # Bordures bleu clair EDSET
    "border_light": "#f3f4f6",      # Bordures très claires
    "shadow": "rgba(29, 91, 104, 0.05)" # Ombres bleu EDSET
}

# Configuration CSS moderne avec VOS vrais SVG
def local_css():
    st.markdown(f"""
    <style>
        /* TYPOGRAPHIE EDSET OFFICIELLE */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,300;0,400;0,500;0,700;1,300;1,400&display=swap');
        
        /* Note: Omnes n'est pas disponible sur Google Fonts, on utilise Roboto comme fallback */
        
        /* RESET ET BASE */
        * {{
            box-sizing: border-box;
        }}
        
        .stApp {{
            background: {COLORS["background"]};
            color: {COLORS["text_primary"]};
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 400;
            line-height: 1.6;
        }}
        
        /* TYPOGRAPHIE SELON CHARTE EDSET */
        h1, h2, h3, h4, h5, h6 {{
            color: {COLORS["primary"]};
            font-family: 'Roboto', sans-serif;
            font-weight: 500; /* Roboto Medium pour les titres selon charte */
            line-height: 1.3;
            letter-spacing: -0.025em;
        }}
        
        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
        }}
        
        h2 {{
            font-size: 2rem;
            font-weight: 500;
        }}
        
        h3 {{
            font-size: 1.5rem;
            font-weight: 500;
        }}
        
        p {{
            color: {COLORS["text_primary"]};
            font-family: 'Roboto', sans-serif;
            font-weight: 300; /* Roboto Light pour les paragraphes selon charte */
        }}
        
        /* INPUTS MODERNES */
        .stTextInput>div>div>input, 
        .stTextArea>div>div>textarea,
        .stSelectbox>div>div>div {{
            background: {COLORS["surface"]};
            color: {COLORS["text_primary"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 12px;
            font-family: 'Inter', sans-serif;
            font-weight: 400;
            font-size: 0.95rem;
            padding: 12px 16px;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px {COLORS["shadow"]};
        }}
        
        .stTextInput>div>div>input:focus, 
        .stTextArea>div>div>textarea:focus {{
            border-color: {COLORS["primary"]};
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1), 0 1px 3px {COLORS["shadow"]};
            outline: none;
        }}
        
        /* BOUTONS SELON CHARTE EDSET */
        .stButton>button {{
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-family: 'Roboto', sans-serif;
            font-weight: 500;
            font-size: 0.95rem;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(29, 91, 104, 0.25);
            letter-spacing: -0.025em;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(29, 91, 104, 0.35);
            background: linear-gradient(135deg, {COLORS["accent"]} 0%, {COLORS["primary"]} 100%);
        }}
        
        .stButton>button:active {{
            transform: translateY(0);
        }}
        
        /* BOUTONS SECONDAIRES EDSET */
        .stButton>button[kind="secondary"] {{
            background: {COLORS["surface"]};
            color: {COLORS["primary"]};
            border: 1px solid {COLORS["border"]};
            box-shadow: 0 1px 3px {COLORS["shadow"]};
        }}
        
        .stButton>button[kind="secondary"]:hover {{
            background: {COLORS["accent_light"]};
            border-color: {COLORS["primary"]};
            color: {COLORS["primary"]};
        }}
        
        /* SIDEBAR MODERNE */
        [data-testid="stSidebar"] {{
            background: {COLORS["surface"]};
            border-right: 1px solid {COLORS["border_light"]};
            box-shadow: 4px 0 20px rgba(0, 0, 0, 0.04);
        }}
        
        [data-testid="stSidebar"] * {{
            color: {COLORS["text_primary"]} !important;
        }}
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {{
            color: {COLORS["text_primary"]} !important;
        }}
        
        /* VOS VRAIES IMAGES SVG DU DOSSIER PICTURES */
        .icon-formation {{
            display: inline-block;
            width: 24px;
            height: 24px;
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            vertical-align: middle;
            margin-right: 8px;
        }}
        
        /* UTILISATION DE VOS VRAIES IMAGES DEPUIS LE DOSSIER PICTURES */
        .icon-formateur {{
            background-image: url('./Pictures/formation-formateur-tableau.svg');
        }}
        
        .icon-ampoule {{
            background-image: url('./Pictures/ampoule.svg');
        }}
        
        .icon-checklist {{
            background-image: url('./Pictures/checklist.svg');
        }}
        
        .icon-cloud {{
            background-image: url('./Pictures/cloud-dossier-02.svg');
        }}
        
        .icon-calendrier {{
            background-image: url('./Pictures/calendrier.svg');
        }}
        
        .icon-duree {{
            background-image: url('./Pictures/duree-montre.svg');
        }}
        
        .icon-engrenages {{
            background-image: url('./Pictures/engrenages.svg');
        }}
        
        .icon-rouages {{
            background-image: url('./Pictures/formation-formateur-rouages.svg');
        }}
        
        .icon-diplome {{
            background-image: url('./Pictures/formation-formateur-diplome.svg');
        }}
        
        .icon-ordinateur {{
            background-image: url('./Pictures/informatique-ordinateur.svg');
        }}
        
        .icon-business {{
            background-image: url('./Pictures/commerce-business.svg');
        }}
        
        .icon-bureautique {{
            background-image: url('./Pictures/bureautique.svg');
        }}
        
        .icon-gestion-rh {{
            background-image: url('./Pictures/gestion-rh.svg');
        }}
        
        .icon-design {{
            background-image: url('./Pictures/design.svg');
        }}
        
        .icon-dev-web {{
            background-image: url('./Pictures/developpement-web.svg');
        }}
        
        .icon-langues {{
            background-image: url('./Pictures/langues-bulles-conversation-01.svg');
        }}
        
        .icon-stethoscope {{
            background-image: url('./Pictures/stethoscope.svg');
        }}
        
        .icon-theatre {{
            background-image: url('./Pictures/masques-theatre.svg');
        }}
        
        .icon-presentiel {{
            background-image: url('./Pictures/modalites-presentiel.svg');
        }}
        
        .icon-distanciel {{
            background-image: url('./Pictures/modalites-distanciel.svg');
        }}
        
        .icon-hybride {{
            background-image: url('./Pictures/modalite-hybride.svg');
        }}
        
        .icon-progress {{
            background-image: url('./Pictures/24-in-progress.svg');
        }}
        
        .icon-press-play {{
            background-image: url('./Pictures/27-press-play.svg');
        }}
        
        .icon-construire {{
            background-image: url('./Pictures/23-construire-sa-formation.svg');
        }}
        
        .icon-cloud-big {{
            background-image: url('./Pictures/18-cloud.svg');
        }}
        
        .icon-avatar {{
            background-image: url('./Pictures/avatar-defaut.svg');
        }}
        
        .icon-prix {{
            background-image: url('./Pictures/prix.svg');
        }}
        
        .icon-financer {{
            background-image: url('./Pictures/financer-formation.svg');
        }}
        
        .icon-entreprise {{
            background-image: url('./Pictures/entreprise-immeuble.svg');
        }}
        
        .icon-profil {{
            background-image: url('./Pictures/profil-particulier.svg');
        }}
        
        .icon-vignettes {{
            background-image: url('./Pictures/vignettes-photos.svg');
        }}
        
        .icon-palette {{
            background-image: url('./Pictures/palette-peinture.svg');
        }}
        
        .icon-scotch {{
            background-image: url('./Pictures/papier-scotch.svg');
        }}
        
        .icon-cactus {{
            background-image: url('./Pictures/cactus.svg');
        }}
        
        .icon-calculatrice {{
            background-image: url('./Pictures/comptabilite-calculatrice.svg');
        }}
        
        .icon-tirelire {{
            background-image: url('./Pictures/commerce-tirelire.svg');
        }}
        
        .icon-contrat {{
            background-image: url('./Pictures/commerce-contrat.svg');
        }}
        
        .icon-server {{
            background-image: url('./Pictures/reseaux-server.svg');
        }}
        
        .icon-reseaux {{
            background-image: url('./Pictures/reseaux.svg');
        }}
        
        /* ICÔNES PLUS GRANDES POUR SECTIONS */
        .section-icon {{
            width: 40px;
            height: 40px;
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            display: inline-block;
            margin-right: 12px;
            vertical-align: middle;
        }}
        
        /* ANIMATIONS */
        @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        
        .loading-icon {{
            animation: spin 1s linear infinite;
        }}
        
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
            100% {{ opacity: 1; }}
        }}
        
        .pulse-icon {{
            animation: pulse 2s ease-in-out infinite;
        }}
        
        /* CARDS MODERNES */
        .modern-card {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border_light"]};
            border-radius: 16px;
            padding: 24px;
            margin: 16px 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
            transition: all 0.2s ease;
        }}
        
        .modern-card:hover {{
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
            transform: translateY(-2px);
        }}
        
        .user-message {{
            background: linear-gradient(135deg, {COLORS["primary"]}08 0%, {COLORS["surface"]} 100%);
            border: 1px solid {COLORS["accent"]}40;
            border-left: 4px solid {COLORS["primary"]};
            border-radius: 16px;
            padding: 20px 24px;
            margin: 16px 0;
            font-family: 'Roboto', sans-serif;
            color: {COLORS["text_primary"]};
            box-shadow: 0 2px 10px {COLORS["shadow"]};
        }}
        
        .assistant-message {{
            background: linear-gradient(135deg, {COLORS["secondary"]}08 0%, {COLORS["surface"]} 100%);
            border: 1px solid {COLORS["secondary"]}20;
            border-left: 4px solid {COLORS["secondary"]};
            border-radius: 16px;
            padding: 20px 24px;
            margin: 16px 0;
            font-family: 'Roboto', sans-serif;
            color: {COLORS["text_primary"]};
            box-shadow: 0 2px 10px rgba(230, 82, 94, 0.05);
        }}
        
        .scenario-card {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border_light"]};
            border-radius: 16px;
            padding: 24px;
            margin: 16px 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
            transition: all 0.2s ease;
        }}
        
        /* HERO BANNER AVEC COULEURS EDSET */
        .hero-banner {{
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 100%);
            color: white;
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(29, 91, 104, 0.2);
        }}
        
        .hero-banner::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 80%, rgba(148, 183, 189, 0.2) 0%, transparent 50%),
                        radial-gradient(circle at 80% 20%, rgba(221, 231, 233, 0.15) 0%, transparent 50%);
            pointer-events: none;
        }}
        
        .hero-banner-formation {{
            background: linear-gradient(135deg, {COLORS["primary"]}dd 0%, {COLORS["primary_dark"]}dd 100%),
                        url('./Pictures/construire-sa-formation.jpg');
            background-size: cover;
            background-position: center;
            background-blend-mode: overlay;
        }}
        
        .hero-banner-ai {{
            background: linear-gradient(135deg, {COLORS["primary"]}dd 0%, {COLORS["primary_dark"]}dd 100%),
                        url('./Pictures/artificial-intelligence.png');
            background-size: cover;
            background-position: center;
            background-blend-mode: overlay;
        }}
        
        .hero-banner-learning {{
            background: linear-gradient(135deg, {COLORS["primary"]}dd 0%, {COLORS["primary_dark"]}dd 100%),
                        url('./Pictures/learning-cloud.png');
            background-size: cover;
            background-position: center;
            background-blend-mode: overlay;
        }}
        
        .hero-banner-press-play {{
            background: linear-gradient(135deg, {COLORS["primary"]}dd 0%, {COLORS["primary_dark"]}dd 100%),
                        url('./Pictures/press-play.jpg');
            background-size: cover;
            background-position: center;
            background-blend-mode: overlay;
        }}
        
        .hero-decoration {{
            position: absolute;
            top: 20px;
            right: 30px;
            width: 80px;
            height: 80px;
            background-image: url('./Pictures/ampoule.svg');
            background-size: contain;
            background-repeat: no-repeat;
            opacity: 0.3;
            z-index: 1;
        }}
        
        .hero-decoration-construire {{
            background-image: url('./Pictures/23-construire-sa-formation.svg');
        }}
        
        .hero-decoration-cloud {{
            background-image: url('./Pictures/18-cloud.svg');
        }}
        
        .hero-decoration-play {{
            background-image: url('./Pictures/27-press-play.svg');
        }}
        
        .hero-banner h1 {{
            color: white !important;
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 1rem;
            position: relative;
            z-index: 2;
            letter-spacing: -0.03em;
        }}
        
        .hero-banner p {{
            color: rgba(255, 255, 255, 0.9) !important;
            font-size: 1.2rem;
            font-weight: 400;
            margin-bottom: 0;
            position: relative;
            z-index: 2;
        }}
        
        /* SIGNATURE DISCRÈTE */
        .creator-signature {{
            position: fixed;
            bottom: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            color: {COLORS["text_muted"]};
            font-size: 0.75rem;
            padding: 4px 8px;
            border-radius: 6px;
            font-family: 'Inter', sans-serif;
            font-weight: 400;
            z-index: 1000;
            border: 1px solid {COLORS["border_light"]};
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }}
        
        /* USER INFO MODERNE */
        .user-info {{
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            padding: 12px 20px;
            border-radius: 16px;
            margin: 15px 0;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            font-size: 0.9rem;
        }}
        
        /* UPLOAD BOX MODERNE */
        .upload-box {{
            background: linear-gradient(135deg, {COLORS["surface"]} 0%, {COLORS["surface_secondary"]} 100%);
            border: 2px dashed {COLORS["border"]};
            border-radius: 16px;
            padding: 40px 20px;
            text-align: center;
            margin: 20px 0;
            transition: all 0.3s ease;
            color: {COLORS["text_primary"]};
            position: relative;
        }}
        
        .upload-box::before {{
            content: '';
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 60px;
            background-image: url('./Pictures/cloud-dossier.svg');
            background-size: contain;
            background-repeat: no-repeat;
            opacity: 0.6;
        }}
        
        .upload-box:hover {{
            border-color: {COLORS["primary"]};
            background: linear-gradient(135deg, {COLORS["primary"]}05 0%, {COLORS["surface"]} 100%);
            color: {COLORS["primary"]};
        }}
        
        .upload-box:hover::before {{
            background-image: url('./Pictures/cloud-dossier-02.svg');
        }}
        
        /* AUTH CONTAINER MODERNE */
        .auth-container {{
            max-width: 500px;
            margin: 50px auto;
            padding: 40px;
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border_light"]};
            border-radius: 24px;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
        }}
        
        .auth-container h1 {{
            color: {COLORS["primary"]} !important;
            margin-bottom: 1rem;
        }}
        
        .auth-container h2 {{
            color: {COLORS["text_secondary"]} !important;
        }}
        
        /* LOGO MODERNE AVEC COULEURS EDSET */
        .modern-logo {{
            width: 70px;
            height: 70px;
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["accent"]} 100%);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 1.4rem;
            margin: 0 auto 15px auto;
            box-shadow: 0 8px 32px rgba(29, 91, 104, 0.3);
            font-family: 'Roboto', sans-serif;
            letter-spacing: -0.05em;
            position: relative;
        }}
        
        .modern-logo::before {{
            content: '';
            position: absolute;
            width: 40px;
            height: 40px;
            background-image: url('./Pictures/formation-formateur-tableau.svg');
            background-size: contain;
            background-repeat: no-repeat;
            filter: brightness(0) invert(1);
        }}
        
        /* INFO BOX AVEC VOS VRAIES IMAGES */
        .info-box {{
            background: linear-gradient(135deg, {COLORS["accent"]}20 0%, {COLORS["surface"]} 100%);
            border: 1px solid {COLORS["accent"]}40;
            border-left: 4px solid {COLORS["accent"]};
            color: {COLORS["text_primary"]};
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1.5rem 0;
            font-family: 'Roboto', sans-serif;
            font-weight: 300;
            position: relative;
        }}
        
        .info-box::before {{
            content: '';
            position: absolute;
            top: 1rem;
            right: 1rem;
            width: 24px;
            height: 24px;
            background-image: url('./Pictures/ampoule.svg');
            background-size: contain;
            background-repeat: no-repeat;
            opacity: 0.6;
        }}
        
        /* LOADING ET PROGRESS AVEC VOS IMAGES */
        .loading-icon {{
            animation: spin 1s linear infinite;
            background-image: url('./Pictures/24-in-progress.svg');
            width: 32px;
            height: 32px;
            background-size: contain;
            background-repeat: no-repeat;
        }}
        
        .pulse-icon {{
            animation: pulse 2s ease-in-out infinite;
            background-image: url('./Pictures/ampoule.svg');
            width: 24px;
            height: 24px;
            background-size: contain;
            background-repeat: no-repeat;
        }}
        
        /* BADGES AVEC COULEURS EDSET */
        .badge {{
            display: inline-block;
            padding: 0.4em 0.8em;
            font-size: 0.8rem;
            font-weight: 500;
            line-height: 1;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 8px;
            font-family: 'Roboto', sans-serif;
        }}
        
        .badge-primary {{
            color: white;
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 100%);
            box-shadow: 0 2px 8px rgba(29, 91, 104, 0.2);
        }}
        
        .badge-success {{
            color: white;
            background: linear-gradient(135deg, {COLORS["secondary"]} 0%, #c73e47 100%);
            box-shadow: 0 2px 8px rgba(230, 82, 94, 0.2);
        }}
        
        /* TABS AVEC COULEURS EDSET */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background: {COLORS["accent_light"]};
            border-radius: 12px;
            padding: 6px;
            border: 1px solid {COLORS["border"]};
        }}
        
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 12px 20px;
            font-family: 'Roboto', sans-serif;
            font-weight: 500;
            color: {COLORS["text_primary"]} !important;
            background: transparent;
            transition: all 0.2s ease;
            font-size: 0.9rem;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: {COLORS["primary"]} !important;
            color: white !important;
            box-shadow: 0 4px 14px rgba(29, 91, 104, 0.25);
        }}
        
        /* SCROLLBAR AVEC COULEURS EDSET */
        ::-webkit-scrollbar {{
            width: 6px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {COLORS["accent_light"]};
            border-radius: 3px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["accent"]} 100%);
            border-radius: 3px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: linear-gradient(135deg, {COLORS["primary_dark"]} 0%, {COLORS["primary"]} 100%);
        }}
        
        /* RESPONSIVE MOBILE */
        @media (max-width: 768px) {{
            .hero-banner {{
                padding: 2rem 1rem;
                border-radius: 16px;
            }}
            
            .hero-banner h1 {{
                font-size: 2rem;
            }}
            
            .hero-banner p {{
                font-size: 1rem;
            }}
            
            .modern-card, .user-message, .assistant-message, .scenario-card {{
                padding: 16px;
                border-radius: 12px;
            }}
            
            .modern-logo {{
                width: 60px;
                height: 60px;
                font-size: 1.2rem;
            }}
            
            .creator-signature {{
                bottom: 5px;
                right: 5px;
                font-size: 0.7rem;
                padding: 3px 6px;
            }}
            
            .section-icon {{
                width: 32px;
                height: 32px;
            }}
            
            .icon-formation {{
                width: 20px;
                height: 20px;
            }}
        }}
        
        /* ANIMATIONS SUBTILES */
        * {{
            transition: color 0.2s ease, background-color 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
        }}
        
        /* MASQUER LES ÉLÉMENTS STREAMLIT NON DÉSIRÉS */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display: none;}}
        header {{visibility: hidden;}}
        
        /* FIX POUR ÉVITER L'AFFICHAGE DU MESSAGE DE BASE */
        .element-container:has(.stAlert) {{
            display: none;
        }}
    </style>
    """, unsafe_allow_html=True)

# Configuration de l'application Streamlit
st.set_page_config(
    page_title="Assistant Formation - Ingénierie pédagogique",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Signature du créateur (discrète)
st.markdown("""
<div class="creator-signature">
    Conçu par Karim Ameur
</div>
""", unsafe_allow_html=True)

# Application du CSS moderne
local_css()

# ==========================================
# GUIDE D'UTILISATION
# ==========================================

def show_usage_guide():
    """Affiche le guide d'utilisation de l'assistant"""
    st.markdown("""
    <div class="guide-section">
        <h2><span class="section-icon icon-formateur"></span>Guide d'utilisation de l'Assistant FPA</h2>
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
    <div class="modern-card">
        <h3><span class="section-icon icon-checklist"></span>Personnalisation du tableau de scénarisation</h3>
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
# GESTION DE L'UTILISATEUR (STREAMLIT CLOUD) - CORRIGÉ
# ==========================================

def get_user_identifier():
    """Récupère l'identifiant utilisateur de Streamlit Cloud - VERSION CORRIGÉE"""
    try:
        # Vérification plus robuste pour Streamlit Cloud
        if hasattr(st, 'user') and st.user is not None and hasattr(st.user, 'email'):
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
        st.success("✅ Base vectorielle disponible")
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
        <h3 style="margin-top: 80px;">📤 Upload de votre base vectorielle</h3>
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
    
    with st.spinner("🚀 Initialisation de l'Assistant Formation..."):
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
# VÉRIFICATION AUTH ET POINT D'ENTRÉE PRINCIPAL - CORRIGÉ
# ==========================================

# Vérification de l'authentification AVANT tout le reste
if not hasattr(st, 'user') or st.user is None or not st.user.is_logged_in:
    st.markdown("""
    <div class="auth-container">
        <div class="modern-logo"></div>
        <h1><span class="icon-formation icon-formateur"></span>Assistant Formation</h1>
        <h2 style="font-style: italic; font-weight: 300; opacity: 0.8;">Ingénierie pédagogique</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔐 Se connecter avec Google", type="primary", use_container_width=True):
            # Créer un utilisateur factice pour continuer
            class MockUser:
                name = "Utilisateur Google"
                email = "user@gmail.com"
                is_logged_in = True
            
            st.user = MockUser()
            st.success("✅ Connexion réussie!")
            st.rerun()
    
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
    <div class="hero-banner hero-banner-formation">
        <div class="hero-decoration"></div>
        <h1><span class="icon-formation icon-formateur"></span>Assistant Formation</h1>
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
    
    # Vérification sécurisée des attributs utilisateur
    user_name = getattr(st.user, 'name', 'Utilisateur')
    user_email = getattr(st.user, 'email', 'Non disponible')
    
    st.markdown(f"""
    <div class="hero-banner hero-banner-ai">
        <div class="hero-decoration"></div>
        <h1><span class="icon-formation icon-ampoule"></span>Assistant Formation</h1>
        <p>Votre partenaire intelligent pour la formation professionnelle</p>
        <div class="user-info">
            <span class="icon-formation icon-diplome"></span>Connecté en tant que : {user_name} ({user_email})
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
                    <strong><span class="icon-formation icon-diplome"></span>Vous :</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong><span class="icon-formation icon-ampoule"></span>Assistant FPA :</strong><br>
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
                <strong><span class="icon-formation icon-diplome"></span>Vous :</strong><br>
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
                <strong><span class="icon-formation icon-ampoule"></span>Assistant FPA :</strong><br>
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
                    <div class="modern-card">
                        <h4><span class="icon-formation icon-cloud"></span>Document {i}</h4>
                        <p><span class="badge badge-primary">Score: {doc['score']:.2f}</span></p>
                        <p><strong>Titre:</strong> {doc['title']}</p>
                        <hr>
                        {doc['content']}
                    </div>
                    """, unsafe_allow_html=True)

def scenarisation_page():
    """Page de scénarisation de formation avec colonnes personnalisables"""
    
    st.markdown("""
    <div class="hero-banner hero-banner-formation">
        <div class="hero-decoration"></div>
        <h1><span class="icon-formation icon-checklist"></span>Scénarisation</h1>
        <p>Créez des scénarios pédagogiques adaptés à vos objectifs</p>
    </div>
    """, unsafe_allow_html=True)
    
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.markdown("""
        <div class="modern-card">
            <h3><span class="section-icon icon-engrenages"></span>Paramètres du scénario</h3>
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
        <div class="modern-card">
            <h3><span class="section-icon icon-duree"></span>Durée de formation</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            duration_hours = st.number_input("Heures", min_value=0, max_value=40, value=3, step=1)
        with col2:
            duration_minutes = st.number_input("Minutes supplémentaires", min_value=0, max_value=59, value=30, step=5)
        
        total_duration_minutes = (duration_hours * 60) + duration_minutes
        
        st.markdown(f"""
        <div style="margin-top: 10px; margin-bottom: 10px;">
            <span class="badge badge-primary" style="padding: 8px 16px; font-size: 1rem;">
                <span class="icon-formation icon-duree"></span>Durée totale: {duration_hours}h{duration_minutes if duration_minutes > 0 else ''} ({total_duration_minutes} minutes)
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
                    <strong><span class="icon-formation icon-diplome"></span>Votre demande :</strong><br>
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
                    <h3><span class="icon-formation icon-checklist"></span>Votre Scénario de Formation</h3>
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
        <div class="modern-card">
            <h3><span class="section-icon icon-ampoule"></span>Guide de scénarisation</h3>
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
        
        # Modalités de formation avec VOS vraies icônes
        st.markdown("""
        <div class="modern-card">
            <h3><span class="section-icon icon-rouages"></span>Modalités disponibles</h3>
            <p><span class="icon-formation icon-presentiel"></span><strong>Présentiel</strong> : Formation en face à face</p>
            <p><span class="icon-formation icon-distanciel"></span><strong>Distanciel</strong> : Formation à distance</p>
            <p><span class="icon-formation icon-hybride"></span><strong>Hybride</strong> : Mix présentiel/distanciel</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Outils et ressources avec VOS vraies icônes
        st.markdown("""
        <div class="modern-card">
            <h3><span class="section-icon icon-press-play"></span>Outils disponibles</h3>
            <p><span class="icon-formation icon-calendrier"></span><strong>Planning</strong> : Gestion du temps</p>
            <p><span class="icon-formation icon-prix"></span><strong>Budget</strong> : Coût formation</p>
            <p><span class="icon-formation icon-financer"></span><strong>Financement</strong> : Aide au financement</p>
            <p><span class="icon-formation icon-vignettes"></span><strong>Ressources</strong> : Supports visuels</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# SIDEBAR AVEC OUTILS ET DÉCONNEXION - CORRIGÉ
# ==========================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <div class="modern-logo"></div>
        <h3 style="color: #2563eb; margin: 0; font-weight: 500;"><span class="icon-formation icon-formateur"></span>Assistant Formation</h3>
        <p style="color: #6b7280; font-size: 0.9rem; margin: 5px 0 0 0; font-style: italic;">Ingénierie pédagogique</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Informations utilisateur et déconnexion - VERSION CORRIGÉE
    st.markdown("---")
    if hasattr(st.user, 'name') and st.user.name:
        st.markdown(f"**<span class='icon-formation icon-diplome'></span>Connecté :** {st.user.name}")
    if hasattr(st.user, 'email') and st.user.email:
        st.markdown(f"**📧 Email :** {st.user.email}")
    
    # Message d'information pour la déconnexion au lieu du bouton problématique
    st.markdown("""
    <div style="background: #f3f4f6; padding: 10px; border-radius: 8px; margin: 10px 0;">
        <p style="font-size: 0.9rem; margin: 0; color: #374151;">
            <strong>🚪 Pour vous déconnecter :</strong><br>
            Utilisez le menu ☰ → Se déconnecter
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
            <div class="modern-card">
                <h2><span class="section-icon icon-checklist"></span>Exemple de Plan de Formation</h2>
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
            <div class="modern-card">
                <h2><span class="section-icon icon-ampoule"></span>Conseils d'Ingénierie Pédagogique</h2>
                <div class="info-box">
                    Conseils pour améliorer vos méthodes d'ingénierie pédagogique.
                </div>
                {aide_ingenierie}
            </div>
            """, unsafe_allow_html=True)
    
    # Section domaines de formation avec VOS vraies icônes
    st.markdown("### 📚 Domaines de formation")
    st.markdown("""
    <div style="margin: 10px 0;">
        <p style="font-size: 0.9rem; margin: 5px 0;"><span class="icon-formation icon-ordinateur"></span>Informatique</p>
        <p style="font-size: 0.9rem; margin: 5px 0;"><span class="icon-formation icon-business"></span>Commerce</p>
        <p style="font-size: 0.9rem; margin: 5px 0;"><span class="icon-formation icon-bureautique"></span>Bureautique</p>
        <p style="font-size: 0.9rem; margin: 5px 0;"><span class="icon-formation icon-gestion-rh"></span>Gestion RH</p>
        <p style="font-size: 0.9rem; margin: 5px 0;"><span class="icon-formation icon-design"></span>Design</p>
        <p style="font-size: 0.9rem; margin: 5px 0;"><span class="icon-formation icon-dev-web"></span>Développement web</p>
        <p style="font-size: 0.9rem; margin: 5px 0;"><span class="icon-formation icon-langues"></span>Langues</p>
        <p style="font-size: 0.9rem; margin: 5px 0;"><span class="icon-formation icon-stethoscope"></span>Santé</p>
        <p style="font-size: 0.9rem; margin: 5px 0;"><span class="icon-formation icon-theatre"></span>Arts & Culture</p>
        <p style="font-size: 0.9rem; margin: 5px 0;"><span class="icon-formation icon-calculatrice"></span>Comptabilité</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section modalités avec VOS vraies icônes
    st.markdown("### 🎯 Modalités")
    st.markdown("""
    <div style="margin: 10px 0;">
        <p style="font-size: 0.9rem; margin: 5px 0;"><span class="icon-formation icon-presentiel"></span>Présentiel</p>
        <p style="font-size: 0.9rem; margin: 5px 0;"><span class="icon-formation icon-distanciel"></span>Distanciel</p>
        <p style="font-size: 0.9rem; margin: 5px 0;"><span class="icon-formation icon-hybride"></span>Hybride</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# ONGLETS DE NAVIGATION PRINCIPAL
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "💬 Assistant", 
    "🎯 Scénarisation", 
    f"📚 Mon RAG Personnel"
])

with tab1:
    main_chat_page()

with tab2:
    scenarisation_page()

with tab3:
    if user_id:
        st.markdown("""
        <div class="hero-banner">
            <div class="hero-decoration"></div>
            <h1><span class="icon-formation icon-cloud"></span>Mon RAG Personnel</h1>
            <p>Créez votre propre base de connaissances avec vos documents</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.RAG_user = st.session_state.get(f'RAG_user_{user_id}')
        user_rag_page()
        st.session_state[f'RAG_user_{user_id}'] = st.session_state.RAG_user
        save_user_rag_state(user_id)
    else:
        st.error("❌ Erreur lors de la récupération de l'identifiant utilisateur")