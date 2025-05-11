import reflex as rx
from . import rag_logic
import traceback
import json
from datetime import datetime
from typing import Optional
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
    
    # Model settings
    temperature: float = 0.7
    streaming: bool = True

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
            yield

    async def handle_submit(self):
        """Handles the user submitting a question."""
        if not self.question.strip():
            return

        user_question = self.question
        self.chat_history.append(QA(question=user_question, answer="", is_loading=True))
        self.question = ""
        yield

        try:
            # Update system status
            self.system_status = "Processing your question..."
            yield
            
            answer, success = rag_logic.get_rag_response(
                user_question, 
                temperature=self.temperature
            )

            if not success:
                self.chat_history[-1].answer = "Sorry, I couldn't process your question. Please try again."
                self.chat_history[-1].is_loading = False
                self.chat_history[-1].is_error = True
                self.system_status = "Failed to process question"
                return

            if self.streaming:
                self.chat_history[-1].is_loading = False
                async for _ in self.stream_answer(answer):
                    yield
            else:
                self.chat_history[-1].answer = self.format_code_blocks(answer)
                self.chat_history[-1].is_loading = False

            self.system_status = None

        except Exception as e:
            print(f"Error processing question: {e}")
            print(traceback.format_exc())
            self.chat_history[-1].answer = f"An error occurred. Please try again later."
            self.chat_history[-1].is_loading = False
            self.chat_history[-1].is_error = True
            self.system_status = "Error occurred"
        finally:
            if self.chat_history:
                self.chat_history[-1].is_loading = False
