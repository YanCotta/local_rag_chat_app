"""UI components for the RAG chat application."""
import reflex as rx
from datetime import datetime
from . import styles
from .state import State, QA

# Import styles and configurations
colors = styles.colors
animations = styles.animations
code_theme = styles.code_theme
input_style = styles.input_style
button_style = styles.button_style
chat_box_style = styles.chat_box_style
qa_style = styles.qa_style
question_style = styles.question_style
answer_style = styles.answer_style
loading_style = styles.loading_style
system_message_style = {
    **qa_style,
    "background_color": colors["accent"],
    "color": colors["text_primary"],
    "align_self": "center",
    "font_style": "italic",
    "max_width": "none",
    "width": "100%",
    "text_align": "center",
}

def loading_dots() -> rx.Component:
    """Animated loading dots."""
    return rx.hstack(
        *[rx.box("•", animation=animations["bounce"], animation_delay=f"{i * 0.2}s")
          for i in range(3)],
        color=colors["loading_text"],
        spacing="2",
    )

def code_block(code: str, language: str = "") -> rx.Component:
    """Syntax highlighted code block with copy button."""
    return rx.box(
        rx.hstack(
            rx.text(
                language.upper() if language else "CODE",
                color=colors["text_secondary"],
                font_size="xs",
                font_family="monospace",
            ),
            rx.button(
                rx.icon("copy"),
                on_click=rx.set_clipboard(code),
                size="xs",
                variant="ghost",
                color=colors["text_secondary"],
                _hover={"color": colors["primary"]},
            ),
            justify_content="flex-end",
            width="100%",
            padding="0.5em 1em",
            background=f"{code_theme['dark']['background']}dd",
            border_top_radius="md",
        ),
        rx.code(
            code,
            language=language,
            theme=code_theme["dark"],
            show_line_numbers=True,
        ),
        border_radius="md",
        background=code_theme["dark"]["background"],
        margin_y="0.5em",
        overflow_x="auto",
        border=f"1px solid {colors['bubble_border']}",
        _hover={
            "border_color": colors["primary"],
            "box_shadow": f"0 0 0 1px {colors['primary']}",
        },
    )

def progress_bar(value: float, color: str = None) -> rx.Component:
    """Animated progress bar component."""
    return rx.box(
        rx.box(
            width=f"{value * 100}%",
            height="2px",
            background=color or colors["primary"],
            transition="width 0.5s ease-in-out",
        ),
        width="100%",
        background=colors["input_bg"],
    )

def system_status_bar() -> rx.Component:
    """System status indicator with progress."""
    return rx.cond(
        State.system_status is not None,
        rx.vstack(
            rx.hstack(
                rx.spinner(color=colors["primary"], size="xs"),
                rx.text(State.system_status),
                spacing="2",
            ),
            rx.cond(
                State.initialization_progress < 1.0,
                progress_bar(State.initialization_progress),
            ),
            rx.cond(
                State.error_message is not None,
                rx.text(
                    State.error_message,
                    color=colors["error"],
                    font_size="sm",
                ),
            ),
            padding="0.5em",
            background=colors["bot_bubble_bg"],
            border_radius="md",
            animation=animations["fade_in"],
            width="100%",
            spacing="2",
        ),
    )

def chat_controls() -> rx.Component:
    """Chat control buttons with loading state handling."""
    return rx.hstack(
        rx.button(
            "Clear Chat",
            on_click=State.clear_chat,
            style=button_style,
            is_disabled=State.is_loading or State.initialization_status != "Ready",
        ),
        rx.button(
            "Export Chat",
            on_click=rx.download(
                State.export_chat(),
                filename=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            ),
            style=button_style,
            is_disabled=State.is_loading or len(State.chat_history) == 0,
        ),
        rx.tooltip(
            rx.icon("info"),
            text="Export chat history as JSON",
            style={"color": colors["text_secondary"]},
        ),
        spacing="2",
        padding="1em",
    )

def message_bubble(qa: QA) -> rx.Component:
    """Enhanced message bubble with animations and code highlighting."""
    return rx.vstack(
        rx.cond(
            qa.is_system_message,
            rx.box(
                rx.text(qa.answer),
                style=system_message_style,
                animation=animations["fade_in"],
            ),
            rx.vstack(
                # Question
                rx.cond(
                    qa.question != "",
                    rx.box(
                        rx.text(qa.question),
                        style=question_style,
                        animation=animations["slide_up"],
                    ),
                ),
                # Answer or Loading State
                rx.cond(
                    qa.is_loading,
                    rx.box(
                        rx.hstack(
                            loading_dots(),
                            rx.text("Processing...", color=colors["text_secondary"]),
                            style=loading_style,
                        ),
                    ),
                    rx.box(
                        rx.markdown(
                            qa.answer,
                            style={
                                **answer_style,
                                "color": colors["error"] if qa.is_error else colors["text_primary"],
                            },
                            custom_components={
                                "code": lambda props: code_block(
                                    props["children"], 
                                    props.get("className", "").replace("language-", "")
                                )
                            },
                        ),
                        animation=animations["slide_up"],
                    ),
                ),
                # Timestamp
                rx.text(
                    qa.timestamp,
                    font_size="xs",
                    color=colors["text_secondary"],
                    align_self="flex-end" if not qa.is_system_message else "center",
                ),
                align_items="stretch",
                width="100%",
                spacing="1",
            ),
        ),
    )
