"""Styles for the RAG chat application."""

colors = {
    "bg_dark": "#1a1a1a",
    "bg_light": "#ffffff",
    "primary": "#3B82F6",
    "secondary": "#6B7280",
    "accent": "#10B981",
    "text_primary": "#F3F4F6",
    "text_secondary": "#9CA3AF",
    "bot_bubble_bg": "#2D3748",
    "user_bubble_bg": "#3B82F6",
    "input_bg": "#374151",
    "button_bg": "#3B82F6",
    "button_hover_bg": "#2563EB",
    "loading_text": "#6B7280",
    "bubble_border": "#4B5563",
    "heading_gradient_start": "#3B82F6",
    "heading_gradient_end": "#10B981",
    "error": "#EF4444",
    "warning": "#F59E0B",
    "success": "#10B981",
}

animations = {
    "fade_in": {
        "keyframes": {
            "0%": {"opacity": 0},
            "100%": {"opacity": 1},
        },
        "duration": "0.3s",
    },
    "slide_up": {
        "keyframes": {
            "0%": {"transform": "translateY(10px)", "opacity": 0},
            "100%": {"transform": "translateY(0)", "opacity": 1},
        },
        "duration": "0.3s",
    },
    "bounce": {
        "keyframes": {
            "0%, 100%": {"transform": "translateY(0)"},
            "50%": {"transform": "translateY(-5px)"},
        },
        "duration": "1s",
        "iteration_count": "infinite",
    },
}

code_theme = {
    "dark": {
        "background": "#282C34",
        "text": "#ABB2BF",
        "comment": "#5C6370",
        "keyword": "#C678DD",
        "string": "#98C379",
        "number": "#D19A66",
        "function": "#61AFEF",
        "operator": "#56B6C2",
    }
}

base_style = {
    "background": colors["bg_dark"],
    "min_height": "100vh",
    "color": colors["text_primary"],
    "font_family": "'Roboto', sans-serif",
}
