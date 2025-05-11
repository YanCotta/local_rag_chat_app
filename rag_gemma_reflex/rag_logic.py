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

# Load environment variables (optional, for OLLAMA_HOST)
load_dotenv()

# --- Configuration ---
DEFAULT_OLLAMA_MODEL = "gemma3:4b-it-qat"
DATASET_NAME = "neural-bridge/rag-dataset-12000"
DATASET_SUBSET_SIZE = 100  # Keep subset for faster initial load
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Ensure you have pulled this model via `ollama pull <model_name>`
# You can override this by setting the OLLAMA_MODEL environment variable
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
FAISS_INDEX_PATH = "faiss_index_neural_bridge"  # Path for this dataset's index


# --- Global Variables ---
_retriever = None
_rag_chain = None


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


def get_ollama_llm():
    """Initializes and returns the Ollama LLM using the new package."""
    global OLLAMA_MODEL
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
    if ollama_base_url:
        print(f"Using Ollama host: {ollama_base_url}")
        llm = Ollama(model=OLLAMA_MODEL, base_url=ollama_base_url)
    else:
        print("Using default Ollama host (http://localhost:11434).")
        llm = Ollama(model=OLLAMA_MODEL)
    print("Ollama LLM initialized.")
    return llm


def setup_rag_chain():
    """Sets up the complete RAG chain."""
    global _retriever, _rag_chain
    if _rag_chain is not None:
        current_ollama_model_env = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

        try:
            chain_model_name = _rag_chain.combine_docs_chain.llm_chain.llm.model
            if chain_model_name == current_ollama_model_env:
                print("RAG chain already initialized and model unchanged.")
                return _retriever, _rag_chain
            else:
                print(
                    f"Ollama model has changed (expected '{current_ollama_model_env}', found '{chain_model_name}'). Re-initializing RAG chain."
                )
                _rag_chain = None
        except AttributeError:
            print(
                "Could not verify model name in existing chain. Re-initializing RAG chain."
            )
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

    llm = get_ollama_llm()
    if llm is None:
        print("LLM initialization failed. Cannot create RAG chain.")
        _rag_chain = None
        _retriever = None
        return _retriever, _rag_chain

    _retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("Retriever created.")

    template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    Question: {input}

    Context: {context}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    print("Prompt template created.")

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    print("Stuff documents chain created.")

    _rag_chain = create_retrieval_chain(_retriever, question_answer_chain)
    print("RAG retrieval chain created successfully.")

    return _retriever, _rag_chain


def get_rag_chain():
    """Returns the initialized RAG chain, setting it up if necessary."""
    if _rag_chain is None:
        setup_rag_chain()
    if _rag_chain is None:
        print("Warning: RAG chain is not available.")
    return _rag_chain


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
