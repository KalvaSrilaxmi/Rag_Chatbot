import os
import streamlit as st
from dotenv import load_dotenv

# Import components from src
from src.embed import get_embedding_model
from src.vector_store import load_vectorstore, create_and_save_vectorstore
from src.retriever import retrieve_with_threshold
from src.qa_chain import get_llm, generate_answer_from_docs
from src.ingest import load_documents
from src.chunk import chunk_documents

# Load environment configuration
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="RAG Q&A Bot", page_icon="🤖", layout="wide")

DATA_DIR = "data"

# ==========================================
# Caching & Initialization Functions
# ==========================================

@st.cache_resource
def init_rag_pipeline():
    """Initializes core RAG components once to prevent reloading."""
    try:
        embeddings = get_embedding_model()
        vectorstore = load_vectorstore(embeddings)
        llm = get_llm()
        return vectorstore, llm
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        return None

def reindex_data():
    """Runs the indexing pipeline for newly uploaded files."""
    try:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        docs = load_documents(DATA_DIR)
        if not docs:
            return "No documents found to index."
            
        chunks = chunk_documents(docs)
        embeddings = get_embedding_model()
        create_and_save_vectorstore(chunks, embeddings)
        
        # Clear the cached pipeline so it reloads with the new vectors
        init_rag_pipeline.clear()
        return "Indexing Complete! New documents are now available in the database."
    except Exception as e:
        return f"Error during indexing: {e}"

# ==========================================
# Sidebar UI & Ingestion
# ==========================================

with st.sidebar:
    st.title("📂 Document Management")
    st.write("Upload PDFs, TXTs, or DOCXs and index them for the bot.")
    
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=['pdf', 'txt', 'docx'])
    
    if st.button("Save & Process Files", use_container_width=True):
        if uploaded_files:
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)
                
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DATA_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
            with st.spinner("Processing texts and generating embeddings..."):
                status_msg = reindex_data()
            st.success(status_msg)
        else:
            st.warning("Please upload a file first.")
            
    st.divider()
    st.markdown("### ⚙️ System Info")
    st.markdown(f"**LLM:** `{os.getenv('LLM_PROVIDER', 'ollama').upper()}`")
    model_name = os.getenv('OLLAMA_MODEL', 'llama3') if os.getenv('LLM_PROVIDER') == 'ollama' else os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    st.markdown(f"**Model:** `{model_name}`")
    st.markdown(f"**Embedder:** `{os.getenv('EMBEDDING_MODEL', 'sentence-transformers')}`")

# ==========================================
# Main Chat Interface
# ==========================================

st.title("🤖 Document Q&A Bot")
st.markdown("Ask natural language questions about your indexed documents. Answers are generated **strictly from the retrieved context** using Retrieval-Augmented Generation (RAG).")

# Setup session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chats
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "docs" in msg and msg["docs"]:
            with st.expander("📚 View Retrieved Chunks"):
                for i, doc in enumerate(msg["docs"]):
                    filename = doc.metadata.get('filename', 'Unknown')
                    page = doc.metadata.get('page', 'N/A')
                    if isinstance(page, int):
                        page = page + 1
                    st.markdown(f"**Chunk {i+1}** (Source: `{filename}`, Page: `{page}`)")
                    st.caption(doc.page_content)
                    st.divider()

# Pipeline init
rag_components = init_rag_pipeline()

# Chat input
if prompt := st.chat_input("E.g., What is the main topic of the document?"):
    if not rag_components:
        st.error("RAG pipeline is not initialized. Please ensure the vector database exists (upload documents using the sidebar).")
        st.stop()
    vectorstore, llm = rag_components
        
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents & thinking..."):
            try:
                # Retrieve chunks using a relevance threshold to reduce hallucinations
                context_docs, _ = retrieve_with_threshold(
                    vectorstore=vectorstore,
                    query=prompt,
                    k=4,
                    max_distance=1.2
                )

                # Generate answer strictly from retained context
                response = generate_answer_from_docs(prompt, context_docs, llm=llm)
                
                st.markdown(response)
                
                # Show retrieved context as an expander
                if context_docs:
                    with st.expander("📚 View Retrieved Chunks"):
                        for i, doc in enumerate(context_docs):
                            filename = doc.metadata.get('filename', 'Unknown')
                            page = doc.metadata.get('page', 'N/A')
                            if isinstance(page, int):
                                page = page + 1
                            st.markdown(f"**Chunk {i+1}** (Source: `{filename}`, Page: `{page}`)")
                            st.caption(doc.page_content)
                            st.divider()
                
                # Save assistant response
                st.session_state.messages.append({"role": "assistant", "content": response, "docs": context_docs})
                
            except Exception as e:
                st.error(f"Error generating answer: {e}")

