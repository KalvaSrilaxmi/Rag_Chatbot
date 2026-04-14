import sys
from src.embed import get_embedding_model
from src.vector_store import load_vectorstore
from src.retriever import retrieve_with_threshold
from src.qa_chain import get_llm, generate_answer_from_docs

def main():
    print("=== Welcome to the RAG Document Q&A Bot ===")
    
    print("Initializing components...")
    try:
        # 1. Get embedding model
        embeddings = get_embedding_model()
        
        # 2. Load FAISS db
        vectorstore = load_vectorstore(embeddings)
        
        # 3. Initialize LLM
        llm = get_llm()
    
    except Exception as e:
        print(f"\n[Error] Initialization Failed: {e}")
        print("Did you run 'python index.py' first?")
        sys.exit(1)
        
    print("\nSystem ready! Type 'exit' or 'quit' to stop.\n")
    print("-" * 50)
    
    # Chat Loop
    while True:
        try:
            query = input("\nUser Query: ")
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if not query.strip():
                continue
                
            print("\nThinking...")
            docs, _ = retrieve_with_threshold(vectorstore, query, k=5, max_distance=1.2)
            response = generate_answer_from_docs(query, docs, llm=llm)
            
            print(f"\nBot Output: \n{response}")
            if docs:
                print("\nRetrieved Chunks:")
                for i, doc in enumerate(docs, 1):
                    filename = doc.metadata.get('filename', 'Unknown')
                    page = doc.metadata.get('page', 'N/A')
                    if isinstance(page, int):
                        page = page + 1
                    snippet = doc.page_content[:150].replace("\n", " ")
                    print(f"  [{i}] {filename} | Page {page} | {snippet}...")
            print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()
