# Set by main.py before any widget is created.
# Pi 800×480: min side = 480. PC simulation: same 480.
EFFECTIVE_PORTRAIT_WIDTH = 480


def get_font_size(base: int) -> int:
    """Scale font size for portrait orientation.
    Base sizes are calibrated for 480px portrait width (Pi 4.3" display).
    """
    scale = max(0.85, min(EFFECTIVE_PORTRAIT_WIDTH / 480.0, 2.5))
    return max(base, int(base * scale))


# Portrait-specific layout constants (px, at 480px width)
HEADER_HEIGHT = 72
FOOTER_HEIGHT = 58
CHAT_LINE_SPACING = 150  # % of font height


DARK_THEME = """
QMainWindow, QWidget#central {
    background-color: #0d0d0d;
}

QWidget#header {
    background-color: #111111;
    border-bottom: 1px solid #252525;
}

QWidget#footer {
    background-color: #111111;
    border-top: 1px solid #252525;
}

QTextEdit#chat {
    background-color: #0d0d0d;
    color: #e0e0e0;
    border: none;
    padding: 16px;
    selection-background-color: #2a2a2a;
}

QLabel#title {
    color: #ffffff;
    font-weight: bold;
}

QLabel#status {
    color: #00c8a3;
}

QLabel#dot {
    color: #00c8a3;
}

QLabel#footer_text {
    color: #555555;
}

QScrollBar:vertical {
    width: 0px;
}
"""
