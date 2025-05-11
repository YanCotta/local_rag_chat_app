import reflex as rx
from . import rag_logic
import traceback


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str
    is_loading: bool = False


class State(rx.State):
    """Manages the application state for the RAG chat interface."""

    question: str = ""
    chat_history: list[QA] = []
    is_loading: bool = False
    temperature: float = 0.7
    streaming: bool = True

    def set_temperature(self, value: float):
        """Update the temperature parameter."""
        self.temperature = max(0.0, min(1.0, value))

    def toggle_streaming(self):
        """Toggle streaming responses on/off."""
        self.streaming = not self.streaming

    async def stream_answer(self, answer: str):
        """Stream the answer word by word."""
        words = answer.split()
        current_answer = ""
        for word in words:
            current_answer += word + " "
            self.chat_history[-1].answer = current_answer.strip()
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
            answer, success = rag_logic.get_rag_response(
                user_question, 
                temperature=self.temperature
            )

            if not success:
                self.chat_history[-1].answer = "Sorry, I couldn't process your question. Please try again."
                self.chat_history[-1].is_loading = False
                return

            if self.streaming:
                self.chat_history[-1].is_loading = False
                async for _ in self.stream_answer(answer):
                    yield
            else:
                self.chat_history[-1].answer = answer
                self.chat_history[-1].is_loading = False

        except Exception as e:
            print(f"Error processing question: {e}")
            print(traceback.format_exc())
            self.chat_history[-1].answer = f"An error occurred. Please try again later."
            self.chat_history[-1].is_loading = False
        finally:
            if self.chat_history:
                self.chat_history[-1].is_loading = False
