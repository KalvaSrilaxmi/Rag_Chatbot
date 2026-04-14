import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Define a strict prompt to answer only from context
QA_TEMPLATE = """You are a helpful A.I. assistant for answering questions based on the provided documents.
Please use ONLY the following retrieved context to answer the question.
If you don't know the answer or if the answer is not present in the context, just say: "Answer not found in documents."
Do NOT try to make up an answer.
Always include specific references to the filename and page number available in context metadata to justify your answer.

Context:
{context}

Question: 
{question}

Answer:"""

QA_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)

def get_llm():
    """
    Initializes the LLM based on environment variables.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        if not api_key or api_key == "your-openai-api-key-here":
            raise ValueError("OPENAI_API_KEY is missing or invalid in .env")
        print(f"Initializing OpenAI model: {model}")
        return ChatOpenAI(model=model, temperature=0)
        
    elif provider == "ollama":
        from langchain_community.llms import Ollama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3")
        print(f"Initializing Ollama model ({model}) at {base_url}")
        return Ollama(
            model=model,
            base_url=base_url,
            temperature=0
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def format_docs(docs):
    """
    Formats the retrieved documents to provide context to the LLM,
    including file and page metadata.
    """
    formatted_context = []
    for doc in docs:
        filename = doc.metadata.get('filename', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        if isinstance(page, int):
            page = page + 1
        
        # Include context + metadata
        chunk_text = f"[Source: {filename}, Page: {page}]\n{doc.page_content}"
        formatted_context.append(chunk_text)
        
    return "\n\n".join(formatted_context)


def format_citations(docs):
    """
    Generates deterministic citation lines from retrieved metadata.
    """
    seen = set()
    citation_lines = []
    for doc in docs:
        filename = doc.metadata.get('filename', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        if isinstance(page, int):
            page = page + 1
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            citation_lines.append(f"- {filename}, page {page}")
    return "\n".join(citation_lines)


def generate_answer_from_docs(question, docs, llm=None):
    """
    Builds a grounded answer directly from retrieved docs.
    Adds deterministic citations when model output omits them.
    """
    if not docs:
        return "Answer not found in documents."

    llm = llm or get_llm()
    context = format_docs(docs)
    prompt = QA_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    answer = str(response).strip()

    # Enforce a visible citation block if model forgot explicit sources.
    if "Source:" not in answer and "page" not in answer.lower():
        answer = f"{answer}\n\nSources:\n{format_citations(docs)}"

    return answer

def build_qa_chain(retriever):
    """
    Builds the complete RAG pipeline.
    Question -> Retriever -> Formatting -> Prompt -> LLM -> Output String
    """
    llm = get_llm()
    
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
    
    return qa_chain
