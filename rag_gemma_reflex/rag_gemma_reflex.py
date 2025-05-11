import reflex as rx
from .state import State, QA

# --- UI Styles ---

colors = {
    "background": "#0F0F10",
    "text_primary": "#E3E3E3",
    "text_secondary": "#BDC1C6",
    "input_bg": "#1F1F21",
    "input_border": "#3C4043",
    "button_bg": "#8AB4F8",
    "button_text": "#202124",
    "button_hover_bg": "#AECBFA",
    "user_bubble_bg": "#3C4043",
    "bot_bubble_bg": "#1E1F21",
    "bubble_border": "#5F6368",
    "loading_text": "#9AA0A6",
    "heading_gradient_start": "#8AB4F8",
    "heading_gradient_end": "#C3A0F8",
}

base_style = {
    "background_color": colors["background"],
    "color": colors["text_primary"],
    "font_family": "'Roboto', sans-serif",
    "font_weight": "200",
    "height": "100vh",
    "width": "100%",
}

input_style = {
    "background_color": colors["input_bg"],
    "border": f"1px solid {colors['input_border']}",
    "color": colors["text_primary"],
    "border_radius": "24px",
    "padding": "12px 18px",
    "width": "100%",
    "font_weight": "400",
    "_placeholder": {
        "color": colors["text_secondary"],
        "font_weight": "300",
    },
    ":focus": {
        "border_color": colors["button_bg"],
        "box_shadow": f"0 0 0 1px {colors['button_bg']}",
    },
}

button_style = {
    "background_color": colors["button_bg"],
    "color": colors["button_text"],
    "border": "none",
    "border_radius": "24px",
    "padding": "12px 20px",
    "cursor": "pointer",
    "font_weight": "500",
    "font_family": "'Roboto', sans-serif",
    "transition": "background-color 0.2s ease",
    ":hover": {
        "background_color": colors["button_hover_bg"],
    },
}

chat_box_style = {
    "padding": "1em 0",
    "flex_grow": 1,
    "overflow_y": "auto",
    "display": "flex",
    "flex_direction": "column-reverse",
    "width": "100%",
    "&::-webkit-scrollbar": {
        "width": "8px",
    },
    "&::-webkit-scrollbar-track": {
        "background": colors["input_bg"],
        "border_radius": "4px",
    },
    "&::-webkit-scrollbar-thumb": {
        "background": colors["bubble_border"],
        "border_radius": "4px",
    },
    "&::-webkit-scrollbar-thumb:hover": {
        "background": colors["text_secondary"],
    },
}

qa_style = {
    "margin_bottom": "1em",
    "padding": "12px 18px",
    "border_radius": "18px",
    "word_wrap": "break-word",
    "max_width": "85%",
    "box_shadow": "0 1px 3px 0 rgba(0, 0, 0, 0.15)",
    "line_height": "1.6",
    "font_weight": "400",
    "code": {
        "background_color": "rgba(255, 255, 255, 0.1)",
        "padding": "0.2em 0.4em",
        "font_size": "85%",
        "border_radius": "4px",
        "font_family": "monospace",
    },
    "a": {
        "color": colors["button_bg"],
        "text_decoration": "underline",
        ":hover": {
            "color": colors["button_hover_bg"],
        },
    },
    "p": {
        "margin": "0",
    },
}

question_style = {
    **qa_style,
    "background_color": colors["user_bubble_bg"],
    "color": colors["text_primary"],
    "align_self": "flex-end",
    "border_bottom_right_radius": "4px",
}

answer_style = {
    **qa_style,
    "background_color": colors["bot_bubble_bg"],
    "color": colors["text_primary"],
    "align_self": "flex-start",
    "border_bottom_left_radius": "4px",
}

loading_style = {
    "color": colors["loading_text"],
    "font_style": "italic",
    "font_weight": "300",
}

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
