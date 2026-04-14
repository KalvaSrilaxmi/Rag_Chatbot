import os
import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document

def clean_text(text: str) -> str:
    """
    Cleans the extracted text.
    Removes sequence of multiple newlines and extra spaces.
    """
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def load_documents(data_dir: str) -> List[Document]:
    """
    Loads all supported documents (PDF, TXT, DOCX) from the given directory.
    """
    documents = []
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return documents

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        file_ext = os.path.splitext(filename)[1].lower()

        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded PDF: {filename} ({len(docs)} pages/parts)")
            
            elif file_ext == '.txt':
                loader = TextLoader(filepath, encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded TXT: {filename}")
                
            elif file_ext == '.docx':
                loader = Docx2txtLoader(filepath)
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded DOCX: {filename}")
            
            else:
                print(f"Skipping unsupported file format: {filename}")
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    # Clean the loaded page content
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    return documents
