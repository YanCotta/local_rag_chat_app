"""Configuration settings for the RAG Chat Assistant."""
import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the application."""
    # Model settings
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:4b-it-qat")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Dataset settings
    DATASET_NAME: str = "neural-bridge/rag-dataset-12000"
    DATASET_SUBSET_SIZE: int = int(os.getenv("DATASET_SUBSET_SIZE", "100"))
    
    # Performance settings
    BATCH_SIZE: int = 32
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # API settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 2
    DEFAULT_TEMPERATURE: float = 0.7
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    # Storage settings
    FAISS_INDEX_PATH: str = os.getenv("VECTOR_STORE_PATH", "faiss_index_neural_bridge")
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE: int = 60
    
    @classmethod
    def validate(cls) -> tuple[bool, Optional[str]]:
        """Validate configuration settings."""
        if not cls.OLLAMA_MODEL:
            return False, "OLLAMA_MODEL not configured"
        if not cls.EMBEDDING_MODEL_NAME:
            return False, "EMBEDDING_MODEL not configured"
        return True, None

config = Config()
