import os
import sys
import argparse
import subprocess
import traceback
import torch
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI

# Configuration de l'API Mistral
MISTRAL_API_KEY = "d7uUh67a9yj6lz9hABWRCvXRuoiescRF"

# Cache pour le modèle d'embeddings (singleton)
_EMBEDDING_MODEL = None

PERSIST_DIRECTORY = "chromadb_formation"

def check_mistral_api():
    """Vérifie si l'API Mistral est accessible"""
    try:
        from mistralai.client import MistralClient
        client = MistralClient(api_key=MISTRAL_API_KEY)
        # Test simple pour vérifier la connectivité
        models = client.list_models()
        return True
    except Exception as e:
        print(f"Erreur de connexion à l'API Mistral: {e}")
        return False

def get_embedding_model():
    """Retourne l'instance du modèle d'embedding, en le chargeant une seule fois."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print("Chargement du modèle d'embedding Salesforce/SFR-Embedding-Mistral...")
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name="Salesforce/SFR-Embedding-Mistral",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("Modèle d'embedding chargé avec succès")
    return _EMBEDDING_MODEL

def load_vector_store(persist_directory=PERSIST_DIRECTORY):
    """Charge la base de données vectorielle existante."""
    print(f"Chargement de la base depuis '{persist_directory}'")
    
    if not os.path.exists(persist_directory):
        print(f"Erreur: La base de données vectorielle n'existe pas dans '{persist_directory}'")
        return None
    
    try:
        # Utilise le modèle d'embedding mis en cache
        embeddings = get_embedding_model()
        
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        print(f"Base de données vectorielle chargée depuis '{persist_directory}'")
        return vectorstore
    except Exception as e:
        print(f"Erreur lors du chargement de la base de données vectorielle: {e}")
        traceback.print_exc()
        return None

def setup_mistral_rag(vectorstore, model_name="mistral-large-latest"):
    """Configure le système RAG avec l'API Mistral."""
    try:
        from langchain.chains import RetrievalQA
        
        # Configuration du modèle Mistral
        llm = ChatMistralAI(
            mistral_api_key=MISTRAL_API_KEY,
            model=model_name,
            temperature=0.1,
            max_tokens=4000
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        print(f"Erreur lors de la configuration du système RAG: {e}")
        traceback.print_exc()
        return None

def create_mistral_llm(model_name="mistral-large-latest"):
    """Crée une instance du modèle Mistral."""
    try:
        llm = ChatMistralAI(
            mistral_api_key=MISTRAL_API_KEY,
            model=model_name,
            temperature=0.1,
            max_tokens=4000
        )
        return llm
    except Exception as e:
        print(f"Erreur lors de la création du modèle Mistral: {e}")
        traceback.print_exc()
        return None

def query_rag(qa_chain, question):
    """Interroge le système RAG."""
    try:
        result = qa_chain({"query": question})
        return result
    except Exception as e:
        print(f"Erreur lors de l'interrogation du système RAG: {e}")
        traceback.print_exc()
        return {"result": "Erreur lors de l'interrogation du système RAG", "source_documents": []}

def launch_streamlit_app():
    """Lance l'application Streamlit"""
    print("Lancement de l'application Streamlit...")
    try:
        subprocess.Popen(["streamlit", "run", "app.py"])
        print("Application Streamlit lancée!")
    except Exception as e:
        print(f"Erreur lors du lancement de l'application Streamlit: {e}")
        print("Veuillez installer Streamlit avec 'pip install streamlit' et réessayer.")

def main():
    parser = argparse.ArgumentParser(description="Système RAG avec Mistral API")
    parser.add_argument("--app", action="store_true", help="Lancer l'application Streamlit")
    parser.add_argument("--test", action="store_true", help="Tester le système RAG avec une question")
    parser.add_argument("--question", type=str, help="Question à poser lors du test")
    
    args = parser.parse_args()
    
    # Si aucun argument n'est fourni, afficher l'aide
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Lancer l'application Streamlit
    if args.app:
        launch_streamlit_app()
    
    # Tester le système RAG
    if args.test:
        # Vérifier si l'API Mistral est accessible
        if not check_mistral_api():
            print("Erreur: L'API Mistral n'est pas accessible.")
            print("Vérifiez votre clé API et votre connexion internet.")
            return
        
        try:
            vector_db = load_vector_store()
            
            if not vector_db:
                print("La base de données vectorielle n'a pas pu être chargée correctement.")
                return
            
            qa_chain = setup_mistral_rag(vector_db, model_name="mistral-large-latest")
            
            if not qa_chain:
                print("Le système RAG n'a pas pu être configuré correctement.")
                return
            
            question = args.question if args.question else "Quelle est la principale information contenue dans les données?"
            print(f"\nQuestion: {question}")
            
            result = query_rag(qa_chain, question)
            
            if "result" in result:
                print("\nRéponse:")
                print(result["result"])
                
                if "source_documents" in result:
                    print("\nSources:")
                    for doc in result["source_documents"]:
                        print(f"- {doc.metadata.get('file_name', 'Source inconnue')}")
            else:
                print("La requête n'a pas retourné de résultat valide.")
                
        except Exception as e:
            print(f"Erreur générale lors de l'exécution du test: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()