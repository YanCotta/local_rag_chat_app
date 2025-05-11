import reflex as rx
from . import rag_logic
from . import error_handling
import traceback
import json
import asyncio
from datetime import datetime
from typing import Optional, List, Any
import re


class QA(rx.Base):
    """A question and answer pair."""
    question: str
    answer: str
    is_loading: bool = False
    is_error: bool = False
    is_system_message: bool = False
    timestamp: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class State(rx.State):
    """Manages the application state for the RAG chat interface."""
    # Core state
    question: str = ""
    chat_history: list[QA] = []
    is_loading: bool = False
    system_status: Optional[str] = None
    show_settings: bool = False
    initialization_status: str = "Not Started"
    initialization_progress: float = 0
    error_message: Optional[str] = None
    
    # Model settings
    temperature: float = 0.7
    streaming: bool = True
    
    def __init__(self):
        """Initialize the state and start async initialization."""
        super().__init__()
        self.initialize_rag_system()
    
    async def initialize_rag_system(self):
        """Initialize the RAG system with progress updates."""
        try:
            self.initialization_status = "Initializing..."
            self.initialization_progress = 0.1
            yield
            
            # Check Ollama connection
            self.initialization_status = "Checking Ollama connection..."
            self.initialization_progress = 0.2
            yield
            
            if not rag_logic.wait_for_ollama_server():
                raise Exception("Could not connect to Ollama server")
            
            # Load documents
            self.initialization_status = "Loading documents..."
            self.initialization_progress = 0.4
            yield
            
            # Initialize RAG chain
            self.initialization_status = "Setting up RAG chain..."
            self.initialization_progress = 0.6
            yield
            
            chain = rag_logic.get_rag_chain()
            if chain is None:
                raise Exception("Failed to initialize RAG chain")
            
            self.initialization_status = "Ready"
            self.initialization_progress = 1.0
            self.add_system_message("System initialized successfully!")
            
        except Exception as e:
            self.initialization_status = "Error"
            self.error_message = str(e)
            self.add_system_message(f"Error during initialization: {str(e)}")
            print(f"Initialization error: {e}")
            print(traceback.format_exc())

    def set_temperature(self, value: float):
        """Update the temperature parameter."""
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
                is_system_message=True,
            )
        )

    def clear_chat(self):
        """Clear the chat history."""
        self.chat_history = []
        self.add_system_message("Chat history cleared.")
        rag_logic._conversation_history = []

    def export_chat(self):
        """Export chat history to JSON."""
        export_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "conversations": [
                {
                    "question": qa.question,
                    "answer": qa.answer,
                    "timestamp": qa.timestamp,
                    "type": "system" if qa.is_system_message else "chat"
                }
                for qa in self.chat_history
            ]
        }
        # Return the JSON string for download
        return json.dumps(export_data, indent=2)

    def format_code_blocks(self, text: str) -> str:
        """Format code blocks with syntax highlighting."""
        code_pattern = r"```(\w+)?\n(.*?)\n```"
        
        def replace_code(match):
            lang = match.group(1) or ""
            code = match.group(2)
            return f'<pre class="code-block {lang}">{code}</pre>'
        
        return re.sub(code_pattern, replace_code, text, flags=re.DOTALL)

    async def stream_answer(self, answer: str):
        """Stream the answer word by word."""
        words = answer.split()
        current_answer = ""
        for word in words:
            current_answer += word + " "
            self.chat_history[-1].answer = self.format_code_blocks(current_answer.strip())
            yield    async def handle_submit(self):
        """Handles the user submitting a question."""
        # Input validation
        if not self.question.strip():
            return

        # Check system status
        if self.initialization_status != "Ready":
            self.add_system_message("System is still initializing. Please wait...")
            return

        # Initialize conversation entry
        user_question = self.question
        self.question = ""
        
        # Create error boundary
        async with error_handling.ErrorBoundary(
            on_error=lambda e: self.add_system_message(f"Error: {str(e)}")
        ) as boundary:
            # Add question to chat history
            self.chat_history.append(QA(question=user_question, answer="", is_loading=True))
            yield

            # Configure retry behavior
            retry_config = error_handling.RetryConfig(
                max_retries=3,
                delay=2.0,
                backoff_factor=1.5
            )

            # Process the question with retries
            result, success = await error_handling.with_retries(
                rag_logic.get_rag_response,
                user_question,
                temperature=self.temperature,
                retry_config=retry_config,
                on_retry=lambda attempt, e: self.update_status(f"Retrying ({attempt})...")
            )

            if success:
                if self.streaming:
                    self.chat_history[-1].is_loading = False
                    async for _ in self.stream_answer(result):
                        yield
                else:
                    self.chat_history[-1].answer = self.format_code_blocks(result)
                    self.chat_history[-1].is_loading = False
            else:
                error_msg = error_handling.format_error_message(
                    boundary.error if boundary.error else Exception("Failed to get response"),
                    user_friendly=True
                )
                self.chat_history[-1].answer = error_msg
                self.chat_history[-1].is_loading = False
                self.chat_history[-1].is_error = True

            self.system_status = None
            
        # Ensure loading state is reset
        if self.chat_history and self.chat_history[-1].is_loading:
            self.chat_history[-1].is_loading = False

        max_retries = rag_logic.MAX_RETRIES
        retry_delay = rag_logic.RETRY_DELAY
        attempt = 0

        while attempt < max_retries:
            try:
                self.system_status = f"Processing your question (attempt {attempt + 1}/{max_retries})..."
                yield

                answer, success = rag_logic.get_rag_response(
                    user_question, 
                    temperature=self.temperature
                )

                if success:
                    if self.streaming:
                        qa_entry.is_loading = False
                        async for _ in self.stream_answer(answer):
                            yield
                    else:
                        qa_entry.answer = self.format_code_blocks(answer)
                        qa_entry.is_loading = False

                    self.system_status = None
                    return
                
                attempt += 1
                if attempt < max_retries:
                    self.system_status = f"Retrying in {retry_delay}s ({attempt}/{max_retries})"
                    yield
                    await asyncio.sleep(retry_delay)
                else:
                    raise Exception("Maximum retry attempts reached")

            except Exception as e:
                print(f"Error processing question (attempt {attempt + 1}): {e}")
                print(traceback.format_exc())
                
                if attempt < max_retries - 1:
                    attempt += 1
                    self.system_status = f"Retrying in {retry_delay}s ({attempt}/{max_retries})"
                    yield
                    await asyncio.sleep(retry_delay)
                else:
                    error_message = (
                        "I apologize, but I encountered an error while processing your "
                        "question. This might be due to:\n\n"
                        "1. Connection issues with the language model\n"
                        "2. Problems processing your query\n"
                        "3. System resource constraints\n\n"
                        "Please try again in a moment."
                    )
                    qa_entry.answer = error_message
                    qa_entry.is_loading = False
                    qa_entry.is_error = True
                    self.system_status = "Error occurred"
                    return
                    
        finally:
            if qa_entry.is_loading:
                qa_entry.is_loading = False
                qa_entry.is_error = True
                qa_entry.answer = "Request timed out. Please try again."
                self.system_status = "Timed out"

    def update_status(self, status: str):
        """Update the system status message."""
        self.system_status = status
