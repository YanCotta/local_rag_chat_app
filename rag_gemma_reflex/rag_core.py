"""Core RAG implementation with proper class structure and error handling."""
import logging
import gc
import torch
from typing import List, Dict, Any, Optional
from threading import Lock
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from datasets import load_dataset
import requests
import time
from datetime import datetime

from .config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter implementation."""
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = Lock()
    
    def can_proceed(self) -> bool:
        """Check if request can proceed under rate limits."""
        now = time.time()
        with self._lock:
            # Remove old requests
            self.requests = [req for req in self.requests if now - req < self.time_window]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

class RAGCore:
    """Core RAG implementation with proper error handling and memory management."""
    
    def __init__(self):
        self._retriever = None
        self._rag_chain = None
        self._conversation_history = []
        self._lock = Lock()
        self._rate_limiter = RateLimiter(config.MAX_REQUESTS_PER_MINUTE)
        
        # Initialize text splitter
        self._text_splitter = CharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        
    @staticmethod
    def cleanup_memory() -> None:
        """Clean up memory after heavy operations."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama server is responsive with exponential backoff."""
        for attempt in range(config.MAX_RETRIES):
            try:
                url = f"{config.OLLAMA_BASE_URL}/api/health"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info("Successfully connected to Ollama server")
                    return True
            except requests.exceptions.RequestException as e:
                delay = config.RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                logger.info(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
        return False

    async def initialize(self) -> None:
        """Initialize the RAG system with proper error handling."""
        try:
            if not self.check_ollama_connection():
                raise ConnectionError("Could not connect to Ollama server")
            
            # Initialize embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
            
            # Try to load existing index
            try:
                self._retriever = FAISS.load_local(
                    config.FAISS_INDEX_PATH,
                    embeddings
                ).as_retriever()
                logger.info("Loaded existing FAISS index")
            except Exception as e:
                logger.warning(f"Could not load existing index: {str(e)}. Creating new one...")
                documents = self._load_and_split_data()
                vectorstore = FAISS.from_documents(documents, embeddings)
                vectorstore.save_local(config.FAISS_INDEX_PATH)
                self._retriever = vectorstore.as_retriever()
                logger.info("Created and saved new FAISS index")
            
            # Initialize LLM and chain
            llm = Ollama(model=config.OLLAMA_MODEL)
            prompt = ChatPromptTemplate.from_template("""
                Answer the following question based on the provided context:
                Context: {context}
                Question: {input}
                Answer: Let me help you with that.
            """)
            
            document_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=prompt,
                document_variable_name="context",
                return_source_documents=True
            )
            
            self._rag_chain = create_retrieval_chain(self._retriever, document_chain)
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise
        finally:
            self.cleanup_memory()
    
    def _load_and_split_data(self) -> List[Document]:
        """Load and split dataset with proper error handling."""
        try:
            logger.info(f"Loading dataset '{config.DATASET_NAME}'...")
            dataset = load_dataset(
                config.DATASET_NAME,
                split=f"train[:{config.DATASET_SUBSET_SIZE}]" if config.DATASET_SUBSET_SIZE else "train"
            )
            
            documents = [
                Document(
                    page_content=row["context"],
                    metadata={"question": row["question"], "answer": row["answer"]},
                )
                for row in dataset
                if row.get("context")
            ]
            
            # Apply text splitting for better chunking
            split_documents = []
            for doc in documents:
                splits = self._text_splitter.split_documents([doc])
                split_documents.extend(splits)
            
            logger.info(f"Loaded and split {len(split_documents)} documents")
            return split_documents
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    async def process_query(
        self,
        query: str,
        temperature: float = config.DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a query with rate limiting and error handling."""
        try:
            # Validate input
            if not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Check rate limit
            if not self._rate_limiter.can_proceed():
                raise Exception("Rate limit exceeded. Please wait before trying again.")
            
            # Process query
            with self._lock:
                response = await self._rag_chain.ainvoke({
                    "input": query,
                    "config": {
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                })
                
                # Format and store response
                result = {
                    "answer": response["answer"],
                    "sources": [doc.metadata for doc in response.get("source_documents", [])],
                    "timestamp": datetime.now().isoformat()
                }
                
                self._conversation_history.append({
                    "query": query,
                    **result
                })
                
                return result
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
        finally:
            self.cleanup_memory()
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Safely get conversation history."""
        with self._lock:
            return self._conversation_history.copy()
