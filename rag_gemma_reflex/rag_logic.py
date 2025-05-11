import os
import reflex as rx
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import ollama as ollama_client
import traceback
import time
import requests
from typing import Optional, Tuple

# Load environment variables
load_dotenv()

# --- Configuration ---
DEFAULT_OLLAMA_MODEL = "gemma3:4b-it-qat"
DATASET_NAME = "neural-bridge/rag-dataset-12000"
DATASET_SUBSET_SIZE = 100  # Keep subset for faster initial load
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
DEFAULT_TEMPERATURE = 0.7

# Ensure you have pulled this model via `ollama pull <model_name>`
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
FAISS_INDEX_PATH = "faiss_index_neural_bridge"  # Path for this dataset's index


# --- Global Variables ---
_retriever = None
_rag_chain = None
_conversation_history = []


def check_ollama_connection(base_url: Optional[str] = None) -> bool:
    """Check if Ollama server is responsive."""
    try:
        url = f"{base_url or 'http://localhost:11434'}/api/health"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def wait_for_ollama_server(max_retries: int = MAX_RETRIES, delay: int = RETRY_DELAY) -> bool:
    """Wait for Ollama server to become available."""
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    for attempt in range(max_retries):
        if check_ollama_connection(base_url):
            print("Successfully connected to Ollama server.")
            return True
        if attempt < max_retries - 1:
            print(f"Waiting for Ollama server... (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
    return False


# --- Helper Functions ---
def load_and_split_data():
    """
    Loads the neural-bridge/rag-dataset-12000 dataset and converts
    contexts into LangChain Documents.
    """
    print(f"Loading dataset '{DATASET_NAME}'...")
    try:
        if DATASET_SUBSET_SIZE:
            print(f"Loading only the first {DATASET_SUBSET_SIZE} entries.")
            dataset = load_dataset(DATASET_NAME, split=f"train[:{DATASET_SUBSET_SIZE}]")
        else:
            print("Loading the full dataset...")
            dataset = load_dataset(DATASET_NAME, split="train")

        documents = [
            Document(
                page_content=row["context"],
                metadata={"question": row["question"], "answer": row["answer"]},
            )
            for row in dataset
            if row.get("context")
        ]
        print(f"Loaded {len(documents)} documents.")
        return documents

    except Exception as e:
        print(f"Error loading dataset '{DATASET_NAME}': {e}")
        print(traceback.format_exc())
        return []


def get_embeddings_model():
    """Initializes and returns the HuggingFace embedding model."""
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    print("Embedding model loaded.")
    return embeddings


def create_or_load_vector_store(documents, embeddings):
    """Creates a FAISS vector store from documents or loads it if it exists."""
    if os.path.exists(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from '{FAISS_INDEX_PATH}'...")
        try:
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
            )
            print("FAISS index loaded.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            print("Attempting to rebuild the index...")
            vector_store = None
    else:
        vector_store = None

    if vector_store is None:
        if not documents:
            print("Error: No documents loaded to create FAISS index.")
            return None
        print("Creating new FAISS index...")
        vector_store = FAISS.from_documents(documents, embeddings)
        print("FAISS index created.")
        print(f"Saving FAISS index to '{FAISS_INDEX_PATH}'...")
        try:
            vector_store.save_local(FAISS_INDEX_PATH)
            print("FAISS index saved.")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")

    return vector_store


def get_ollama_llm(temperature: float = DEFAULT_TEMPERATURE) -> Optional[Ollama]:
    """Initializes and returns the Ollama LLM with retry logic."""
    global OLLAMA_MODEL

    if not wait_for_ollama_server():
        print("Error: Could not connect to Ollama server.")
        return None

    current_ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    if OLLAMA_MODEL != current_ollama_model:
        print(f"Ollama model changed to '{current_ollama_model}'.")
        OLLAMA_MODEL = current_ollama_model
        global _rag_chain
        _rag_chain = None

    print(f"Initializing Ollama LLM with model '{OLLAMA_MODEL}'...")
    try:
        ollama_client.show(OLLAMA_MODEL)
        print(f"Confirmed Ollama model '{OLLAMA_MODEL}' is available locally.")
    except ollama_client.ResponseError as e:
        if "model not found" in str(e).lower():
            print(f"Error: Ollama model '{OLLAMA_MODEL}' not found locally.")
            print(f"Please pull it first using: ollama pull {OLLAMA_MODEL}")
            return None
        else:
            print(f"An error occurred while checking the Ollama model: {e}")
            return None
    except Exception as e:
        print(f"An unexpected error occurred while checking Ollama model: {e}")
        return None

    ollama_base_url = os.getenv("OLLAMA_HOST")
    try:
        if ollama_base_url:
            print(f"Using Ollama host: {ollama_base_url}")
            llm = Ollama(
                model=OLLAMA_MODEL,
                base_url=ollama_base_url,
                temperature=temperature
            )
        else:
            print("Using default Ollama host (http://localhost:11434).")
            llm = Ollama(
                model=OLLAMA_MODEL,
                temperature=temperature
            )
        print("Ollama LLM initialized.")
        return llm
    except Exception as e:
        print(f"Failed to initialize Ollama LLM: {e}")
        return None


def setup_rag_chain(temperature: float = DEFAULT_TEMPERATURE):
    """Sets up the complete RAG chain with conversation memory."""
    global _retriever, _rag_chain
    if _rag_chain is not None:
        current_ollama_model_env = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

        try:
            chain_model_name = _rag_chain.combine_docs_chain.llm_chain.llm.model
            if chain_model_name == current_ollama_model_env:
                print("RAG chain already initialized and model unchanged.")
                return _retriever, _rag_chain
            else:
                print(f"Ollama model has changed. Re-initializing RAG chain.")
                _rag_chain = None
        except AttributeError:
            print("Could not verify model name in existing chain. Re-initializing RAG chain.")
            _rag_chain = None

    print("Setting up RAG chain...")
    documents = load_and_split_data()
    if not documents:
        print("No documents loaded, cannot proceed with RAG chain setup.")
        return None, None

    embeddings = get_embeddings_model()
    vector_store = create_or_load_vector_store(documents, embeddings)
    if vector_store is None:
        print("Vector store creation/loading failed. Cannot create RAG chain.")
        return None, None

    llm = get_ollama_llm(temperature=temperature)
    if llm is None:
        print("LLM initialization failed. Cannot create RAG chain.")
        _rag_chain = None
        _retriever = None
        return _retriever, _rag_chain

    _retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    print("Retriever created.")

    template = """You are a helpful AI assistant engaging in a conversation. Use the following pieces of context and conversation history to provide accurate, relevant, and natural responses.

    Context from documents:
    {context}

    Conversation history:
    {chat_history}

    Current question: {input}

    Instructions:
    1. Use the context and conversation history to provide a coherent response
    2. If the context doesn't contain relevant information, acknowledge what you know and what you don't
    3. Keep responses concise but informative
    4. Maintain a friendly and helpful tone
    5. If referring to previous conversation, do so naturally
    6. Format technical information clearly

    Answer: """

    prompt = ChatPromptTemplate.from_template(template)
    print("Enhanced prompt template created.")

    # Create the RAG chain with the updated prompt
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    print("Enhanced QA chain created.")

    _rag_chain = create_retrieval_chain(_retriever, question_answer_chain)
    print("Enhanced RAG chain created successfully.")

    return _retriever, _rag_chain


def add_to_conversation_history(question: str, answer: str):
    """Add a Q&A pair to the conversation history."""
    global _conversation_history
    _conversation_history.append((question, answer))
    # Keep only the last 5 exchanges to maintain context without overwhelming
    _conversation_history = _conversation_history[-5:]


def get_conversation_history() -> str:
    """Format conversation history for the prompt."""
    if not _conversation_history:
        return "No previous conversation."
    
    history = []
    for i, (q, a) in enumerate(_conversation_history, 1):
        history.append(f"Q{i}: {q}")
        history.append(f"A{i}: {a}")
    return "\n".join(history)


def get_rag_response(question: str, temperature: float = DEFAULT_TEMPERATURE) -> Tuple[str, bool]:
    """Get a response from the RAG chain with conversation history."""
    global _rag_chain
    
    if _rag_chain is None:
        setup_rag_chain(temperature=temperature)
    
    if _rag_chain is None:
        return "System is not available. Please check the logs.", False

    try:
        # Include conversation history in the context
        chat_history = get_conversation_history()
        
        response = _rag_chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        
        answer = response.get("answer", "Sorry, I couldn't generate a response.")
        add_to_conversation_history(question, answer)
        return answer, True
    
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg, False


# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    print("Testing RAG logic setup...")
    try:
        retriever, rag_chain = setup_rag_chain()

        if rag_chain:
            print("\nRAG Chain setup complete. Testing with a sample question...")
            test_question = "What are the benefits of RAG?"
            try:
                response = rag_chain.invoke({"input": test_question})
                print(f"\nQuestion: {test_question}")
                print(f"Answer: {response['answer']}")
            except Exception as e:
                print(f"An error occurred during invocation: {e}")
                print(traceback.format_exc())
                print("Ensure Ollama is running and the model is available.")
        else:
            print("RAG chain initialization failed.")
    except Exception as e:
        print(f"An error occurred during setup: {e}")
        print(traceback.format_exc())
