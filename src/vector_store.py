import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

FAISS_DB_DIR = "vectorstore"

def create_and_save_vectorstore(chunks: List[Document], embeddings) -> FAISS:
    """
    Creates a FAISS vector database from text chunks and saves it locally.
    """
    print(f"Generating embeddings and building FAISS index for {len(chunks)} chunks...")
    # This automatically batches the embedding generation
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    # Ensure directory exists
    if not os.path.exists(FAISS_DB_DIR):
        os.makedirs(FAISS_DB_DIR)
        
    print(f"Saving vector database to ./{FAISS_DB_DIR}...")
    vectorstore.save_local(FAISS_DB_DIR)
    
    return vectorstore

def load_vectorstore(embeddings) -> FAISS:
    """
    Loads the FAISS vector database from disk.
    Requires the same embedding model that was used to create it.
    """
    if not os.path.exists(FAISS_DB_DIR) or not os.path.exists(os.path.join(FAISS_DB_DIR, "index.faiss")):
        raise FileNotFoundError(f"FAISS index not found in ./{FAISS_DB_DIR}. Please run index.py first.")
        
    print(f"Loading vector database from ./{FAISS_DB_DIR}...")
    vectorstore = FAISS.load_local(FAISS_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore
