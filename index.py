import os
from src.ingest import load_documents
from src.chunk import chunk_documents
from src.embed import get_embedding_model
from src.vector_store import create_and_save_vectorstore

DATA_DIR = "data"

def main():
    print("=== Document Ingestion and Vector DB Indexing ===")
    
    # 1. Load documents
    if not os.path.exists(DATA_DIR):
        print(f"Creating '{DATA_DIR}' directory. Please place your PDFs/TXTs/DOCXs here and run again.")
        os.makedirs(DATA_DIR)
        return
        
    docs = load_documents(DATA_DIR)
    if not docs:
        print(f"No documents found in current '{DATA_DIR}' directory. Please add some files.")
        return
        
    # 2. Chunk documents
    chunks = chunk_documents(docs)
    
    # 3. Load embedding model
    embeddings = get_embedding_model()
    
    # 4. Create and save FAISS DB
    create_and_save_vectorstore(chunks, embeddings)
    
    print("=== Indexing Complete! ===")
    print("You can now run 'python main.py' to query your documents.")

if __name__ == "__main__":
    main()
