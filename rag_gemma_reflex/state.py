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

    async def handle_submit(self):
        """Handles the user submitting a question."""
        if not self.question.strip():
            return

        user_question = self.question
        self.chat_history.append(QA(question=user_question, answer="", is_loading=True))
        self.question = ""
        yield

        try:
            rag_chain = rag_logic.get_rag_chain()

            if rag_chain is None:
                raise Exception("RAG chain could not be initialized. Check logs.")

            response = await rag_chain.ainvoke({"input": user_question})
            answer = response.get("answer", "Sorry, I couldn't find an answer.")

            self.chat_history[-1].answer = answer
            self.chat_history[-1].is_loading = False

        except Exception as e:
            print(f"Error processing question: {e}")
            print(traceback.format_exc())
            self.chat_history[
                -1
            ].answer = f"An error occurred: {e}. Check the console logs."
            self.chat_history[-1].is_loading = False
        finally:
            if self.chat_history:
                self.chat_history[-1].is_loading = False
