from typing import List, Tuple
from langchain_core.documents import Document


def get_retriever(vectorstore, k: int = 4):
    """
    Configures and returns a retriever from the FAISS vector store.
    k: Number of relevant chunks to retrieve.
    """
    print(f"Configuring retriever with top_k={k}...")
    
    # Use similarity search
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    return retriever


def retrieve_with_threshold(
    vectorstore,
    query: str,
    k: int = 4,
    max_distance: float = 1.2
) -> Tuple[List[Document], List[Tuple[Document, float]]]:
    """
    Retrieves top-k chunks with vector distances and filters weak matches.
    Lower distance means higher similarity.
    Returns:
        - filtered_docs: docs below distance threshold (for answer generation)
        - scored_results: raw (doc, distance) pairs (for debugging/inspection)
    """
    scored_results = vectorstore.similarity_search_with_score(query, k=k)
    filtered_docs: List[Document] = []

    for doc, distance in scored_results:
        if distance is not None and distance <= max_distance:
            filtered_docs.append(doc)

    return filtered_docs, scored_results
