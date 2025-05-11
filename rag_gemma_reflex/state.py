import reflex as rx
from . import error_handling
from .rag_core import RAGCore
from .config import config
import traceback
import json
import asyncio
from datetime import datetime
from typing import Optional, List, Any, Dict
import re
import logging

# Configure logging
logger = logging.getLogger(__name__)

class QA(rx.Base):
    """A question and answer pair with metadata."""
    question: str
    answer: str
    sources: List[Dict[str, Any]] = []
    is_loading: bool = False
    is_error: bool = False
    is_system_message: bool = False
    timestamp: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class State(rx.State):
    """Manages the application state for the RAG chat interface with improved error handling."""
    # Core state
    question: str = ""
    chat_history: List[QA] = []
    is_loading: bool = False
    system_status: Optional[str] = None
    show_settings: bool = False
    initialization_status: str = "Not Started"
    initialization_progress: float = 0
    error_message: Optional[str] = None
    
    # Model settings
    temperature: float = config.DEFAULT_TEMPERATURE
    streaming: bool = True
    
    def __init__(self):
        """Initialize the state with the new RAG core."""
        super().__init__()
        self.rag = RAGCore()
        self.initialize_rag_system()
    
    async def initialize_rag_system(self):
        """Initialize the RAG system with progress updates and proper error handling."""
        try:
            self.initialization_status = "Initializing..."
            self.initialization_progress = 0.1
            yield
            
            # Initialize RAG core
            await self.rag.initialize()
            
            self.initialization_status = "Ready"
            self.initialization_progress = 1.0
            self.add_system_message("System initialized successfully!")
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}\n{traceback.format_exc()}")
            self.initialization_status = "Error"
            self.error_message = str(e)
            self.add_system_message(f"Error during initialization: {str(e)}")
    
    async def process_message(self, message: str) -> None:
        """Process a user message with improved error handling and loading states."""
        if not message.strip():
            self.error_message = "Message cannot be empty"
            return
            
        qa_pair = QA(
            question=message,
            answer="",
            is_loading=True
        )
        self.chat_history.append(qa_pair)
        
        try:
            self.is_loading = True
            response = await self.rag.process_query(
                query=message,
                temperature=self.temperature
            )
            
            qa_pair.answer = response["answer"]
            qa_pair.sources = response.get("sources", [])
            qa_pair.is_loading = False
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}\n{traceback.format_exc()}")
            qa_pair.is_error = True
            qa_pair.answer = f"Error: {str(e)}"
            self.error_message = str(e)
        finally:
            self.is_loading = False
            qa_pair.is_loading = False

    def set_temperature(self, value: float):
        """Update the temperature parameter with validation."""
        self.temperature = max(0.0, min(1.0, value))

    def toggle_streaming(self):
        """Toggle streaming responses on/off."""
        self.streaming = not self.streaming

    def toggle_settings(self):
        """Toggle settings panel visibility."""
        self.show_settings = not self.show_settings

    def add_system_message(self, message: str):
        """Add a system message to the chat history."""
        self.chat_history.append(
            QA(
                question="",
                answer=message,
                is_system_message=True
            )
        )

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.chat_history = []
        self.error_message = None
        
    def export_conversation(self) -> Dict[str, Any]:
        """Export the conversation history with metadata."""
        return {
            "messages": [
                {
                    "role": "system" if qa.is_system_message else "user" if qa.question else "assistant",
                    "content": qa.question if qa.question else qa.answer,
                    "timestamp": qa.timestamp,
                    "sources": qa.sources if hasattr(qa, "sources") else []
                }
                for qa in self.chat_history
            ],
            "metadata": {
                "temperature": self.temperature,
                "streaming": self.streaming,
                "exported_at": datetime.now().isoformat()
            }
        }
