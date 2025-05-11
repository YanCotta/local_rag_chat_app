"""Test suite for the RAG Chat Assistant."""
import pytest
import asyncio
from unittest.mock import Mock, patch
from typing import List, Dict, Any
from langchain.docstore.document import Document

from rag_gemma_reflex.rag_core import RAGCore
from rag_gemma_reflex.error_handling import ModelConnectionError, DatasetError
from rag_gemma_reflex.config import Config

@pytest.fixture
def mock_documents() -> List[Document]:
    """Create mock documents for testing."""
    return [
        Document(
            page_content="Test content 1",
            metadata={"question": "test q1", "answer": "test a1"}
        ),
        Document(
            page_content="Test content 2",
            metadata={"question": "test q2", "answer": "test a2"}
        )
    ]

@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    mock = Mock()
    mock.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock.embed_query.return_value = [0.5, 0.6]
    return mock

@pytest.fixture
def mock_ollama():
    """Create mock Ollama for testing."""
    mock = Mock()
    mock.invoke.return_value = {"response": "Test response"}
    return mock

@pytest.fixture
def rag_core():
    """Create RAGCore instance for testing."""
    return RAGCore()

@pytest.mark.asyncio
async def test_initialization(rag_core, mock_documents, mock_embeddings, mock_ollama):
    """Test RAG system initialization."""
    with patch("rag_gemma_reflex.rag_core.load_dataset", return_value=mock_documents), \
         patch("rag_gemma_reflex.rag_core.HuggingFaceEmbeddings", return_value=mock_embeddings), \
         patch("rag_gemma_reflex.rag_core.Ollama", return_value=mock_ollama):
        
        await rag_core.initialize()
        assert rag_core._retriever is not None

@pytest.mark.asyncio
async def test_ollama_connection_error(rag_core):
    """Test handling of Ollama connection errors."""
    with patch("rag_gemma_reflex.rag_core.requests.get", side_effect=Exception("Connection failed")):
        with pytest.raises(ModelConnectionError):
            await rag_core.initialize()

@pytest.mark.asyncio
async def test_query_processing(rag_core, mock_documents, mock_embeddings, mock_ollama):
    """Test query processing."""
    with patch("rag_gemma_reflex.rag_core.load_dataset", return_value=mock_documents), \
         patch("rag_gemma_reflex.rag_core.HuggingFaceEmbeddings", return_value=mock_embeddings), \
         patch("rag_gemma_reflex.rag_core.Ollama", return_value=mock_ollama):
        
        await rag_core.initialize()
        response = await rag_core.process_query("test query")
        
        assert "answer" in response
        assert "sources" in response
        assert "timestamp" in response

@pytest.mark.asyncio
async def test_rate_limiting(rag_core, mock_documents, mock_embeddings, mock_ollama):
    """Test rate limiting functionality."""
    with patch("rag_gemma_reflex.rag_core.load_dataset", return_value=mock_documents), \
         patch("rag_gemma_reflex.rag_core.HuggingFaceEmbeddings", return_value=mock_embeddings), \
         patch("rag_gemma_reflex.rag_core.Ollama", return_value=mock_ollama):
        
        await rag_core.initialize()
        
        # Make multiple requests quickly
        requests = []
        for _ in range(Config.MAX_REQUESTS_PER_MINUTE + 1):
            requests.append(rag_core.process_query("test query"))
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await asyncio.gather(*requests)

@pytest.mark.asyncio
async def test_memory_cleanup(rag_core, mock_documents, mock_embeddings, mock_ollama):
    """Test memory cleanup functionality."""
    with patch("rag_gemma_reflex.rag_core.load_dataset", return_value=mock_documents), \
         patch("rag_gemma_reflex.rag_core.HuggingFaceEmbeddings", return_value=mock_embeddings), \
         patch("rag_gemma_reflex.rag_core.Ollama", return_value=mock_ollama), \
         patch("rag_gemma_reflex.rag_core.gc.collect") as mock_gc, \
         patch("rag_gemma_reflex.rag_core.torch.cuda.empty_cache") as mock_cuda:
        
        await rag_core.initialize()
        rag_core.cleanup_memory()
        
        mock_gc.assert_called_once()
        mock_cuda.assert_called_once()

@pytest.mark.asyncio
async def test_conversation_history(rag_core, mock_documents, mock_embeddings, mock_ollama):
    """Test conversation history management."""
    with patch("rag_gemma_reflex.rag_core.load_dataset", return_value=mock_documents), \
         patch("rag_gemma_reflex.rag_core.HuggingFaceEmbeddings", return_value=mock_embeddings), \
         patch("rag_gemma_reflex.rag_core.Ollama", return_value=mock_ollama):
        
        await rag_core.initialize()
        await rag_core.process_query("test query 1")
        await rag_core.process_query("test query 2")
        
        history = rag_core.get_conversation_history()
        assert len(history) == 2
        assert "query" in history[0]
        assert "answer" in history[0]
        assert "timestamp" in history[0]

@pytest.mark.asyncio
async def test_input_validation(rag_core):
    """Test input validation."""
    with pytest.raises(ValueError):
        await rag_core.process_query("")  # Empty query
    
    with pytest.raises(ValueError):
        await rag_core.process_query(" " * 10)  # Whitespace only
