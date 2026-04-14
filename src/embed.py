import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load .env mainly if it hasn't been loaded
load_dotenv()

def get_embedding_model():
    """
    Initializes and returns the embedding model.
    Uses sentence-transformers locally to avoid API costs.
    """
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # HuggingFaceEmbeddings performs local inference using sentence-transformers
    print(f"Initializing embedding model: {model_name}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Defaulting to CPU for wider compatibility
        encode_kwargs={'normalize_embeddings': True} # Normalization helps in cosine similarity
    )
    
    return embeddings
