from PyQt5.QtWidgets import QApplication


def screen_width() -> int:
    return QApplication.primaryScreen().size().width()


def screen_height() -> int:
    return QApplication.primaryScreen().size().height()


def get_font_size(base: int) -> int:
    """Scale font size for portrait orientation.

    Base sizes are calibrated for 480px width (Pi 7" held portrait).
    Scale is clamped so fonts stay readable across display sizes.
    """
    w = screen_width()
    scale = max(0.85, min(w / 480.0, 2.5))
    return max(base, int(base * scale))


# Portrait-specific layout constants (px)
HEADER_HEIGHT = 56
FOOTER_HEIGHT = 48
CHAT_SIDE_PADDING = 16   # px left/right padding in chat area
CHAT_LINE_SPACING = 140  # % of font height (QTextDocument blockFormat)


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
