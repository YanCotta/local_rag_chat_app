"""UI components for the RAG chat application."""
import reflex as rx
from . import styles
from .state import State, QA

# Import styles
colors = styles.colors
animations = styles.animations
code_theme = styles.code_theme

def loading_dots() -> rx.Component:
    """Animated loading dots."""
    return rx.hstack(
        *[rx.box("•", animation=animations["bounce"], animation_delay=f"{i * 0.2}s")
          for i in range(3)],
        color=colors["loading_text"],
        spacing="2",
    )

def code_block(code: str, language: str = "") -> rx.Component:
    """Syntax highlighted code block."""
    return rx.box(
        rx.code(
            code,
            language=language,
            theme=code_theme["dark"],
        ),
        padding="1em",
        border_radius="md",
        background=code_theme["dark"]["background"],
        margin_y="0.5em",
        overflow_x="auto",
    )

def system_status_bar() -> rx.Component:
    """System status indicator."""
    return rx.cond(
        State.system_status is not None,
        rx.box(
            rx.hstack(
                rx.spinner(color=colors["primary"], size="xs"),
                rx.text(State.system_status),
                spacing="2",
            ),
            padding="0.5em",
            background=colors["bot_bubble_bg"],
            border_radius="md",
            animation=animations["fade_in"],
        ),
    )

def chat_controls() -> rx.Component:
    """Chat control buttons."""
    return rx.hstack(
        rx.button(
            "Clear Chat",
            on_click=State.clear_chat,
            style=button_style,
        ),
        rx.button(
            "Export Chat",
            on_click=rx.download(
                State.export_chat(),
                filename=f"chat_export_{rx.now().strftime('%Y%m%d_%H%M%S')}.json",
            ),
            style=button_style,
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
                rx.cond(
                    qa.question != "",
                    rx.box(
                        rx.text(qa.question),
                        style=question_style,
                        animation=animations["slide_up"],
                    ),
                ),
                rx.cond(
                    qa.is_loading,
                    rx.box(
                        loading_dots(),
                        style=loading_style,
                    ),
                    rx.box(
                        rx.markdown(
                            qa.answer,
                            style=answer_style,
                            custom_components={
                                "code": lambda props: code_block(
                                    props["children"], props.get("className", "")
                                )
                            },
                        ),
                        animation=animations["slide_up"],
                    ),
                ),
                align_items="stretch",
                width="100%",
                spacing="1",
            ),
        ),
    )
