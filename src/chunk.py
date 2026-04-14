from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Splits the given documents into smaller chunks using RecursiveCharacterTextSplitter.
    Optimizes for semantic boundaries (paragraphs, sentences) before falling back to chars.
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    
    print(f"Chunking {len(documents)} documents (size={chunk_size}, overlap={chunk_overlap})...")
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks.")
    
    # Optional: ensure metadata retains filename clearly, which PyPDFLoader usually puts in 'source'
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get('source', 'Unknown source')
        # We can extract just the filename if it's a full path
        import os
        chunk.metadata['filename'] = os.path.basename(source)
        # Ensure page exists
        if 'page' not in chunk.metadata:
            chunk.metadata['page'] = 'N/A'
            
    return chunks
