"""Main UI file for the RAG chat application."""
import reflex as rx
from . import styles
from . import components
from .state import State

# Import styles and components
colors = styles.colors
animations = styles.animations
base_style = styles.base_style
input_style = styles.input_style
button_style = styles.button_style

# --- UI Components ---


def message_bubble(qa: QA):
    """Displays a single question and its answer."""
    return rx.vstack(
        rx.box(qa.question, style=question_style),
        rx.cond(
            qa.is_loading,
            rx.box("Thinking...", style={**answer_style, **loading_style}),
            rx.markdown(qa.answer, style=answer_style),
        ),
        align_items="stretch",
        width="100%",
        spacing="1",
    )


def settings_panel() -> rx.Component:
    """Creates a settings panel component."""
    return rx.hstack(
        rx.vstack(
            rx.text("Temperature:", color=colors["text_secondary"]),
            rx.slider(
                value=State.temperature,
                min_=0.0,
                max_=1.0,
                step=0.1,
                on_change=State.set_temperature,
                width="150px",
            ),
            rx.text(
                f"{State.temperature:.1f}",
                color=colors["text_secondary"],
                font_size="0.8em",
            ),
            align_items="center",
            spacing="2",
        ),
        rx.vstack(
            rx.text("Streaming:", color=colors["text_secondary"]),
            rx.switch(
                is_checked=State.streaming,
                on_change=State.toggle_streaming,
            ),
            align_items="center",
            spacing="2",
        ),
        justify_content="center",
        spacing="8",
        padding="1em",
        background=colors["input_bg"],
        border_radius="md",
        margin="1em 0",
    )


# --- Main Page ---


def index() -> rx.Component:
    """The main chat interface page."""
    heading_style = {
        "size": "7",
        "margin_bottom": "0.25em",
        "font_weight": "400",
        "background_image": f"linear-gradient(to right, {colors['heading_gradient_start']}, {colors['heading_gradient_end']})",
        "background_clip": "text",
        "-webkit-background-clip": "text",
        "color": "transparent",
        "width": "fit-content",
        "animation": animations["fade_in"],
    }

    return rx.container(
        rx.vstack(
            rx.box(
                rx.heading("RAG Chat with Gemma", **heading_style),
                rx.text(
                    "Ask a question based on the loaded context.",
                    color=colors["text_secondary"],
                    font_weight="300",
                ),
                padding_bottom="0.5em",
                width="100%",
                text_align="center",
            ),
            settings_panel(),
            rx.box(
                rx.foreach(State.chat_history, message_bubble),
                style=chat_box_style,
            ),
            rx.form(
                rx.hstack(
                    rx.input(
                        name="question",
                        placeholder="Ask your question...",
                        value=State.question,
                        on_change=State.set_question,
                        style=input_style,
                        flex_grow=1,
                        height="50px",
                    ),
                    rx.button(
                        "Ask",
                        type="submit",
                        style=button_style,
                        is_loading=State.is_loading,
                        height="50px",
                    ),
                    width="100%",
                    align_items="center",
                ),
                on_submit=State.handle_submit,
                width="100%",
            ),
            align_items="center",
            width="100%",
            height="100%",
            padding_x="1em",
            padding_y="1em",
            spacing="4",
        ),
        max_width="900px",
        height="100vh",
        padding=0,
        margin="auto",
    )


# --- App Setup ---
stylesheets = [
    "https://fonts.googleapis.com/css2?family=Roboto:wght@200;300;400;500&display=swap",
]

app = rx.App(style=base_style, stylesheets=stylesheets)
app.add_page(index, title="Reflex Chat")
