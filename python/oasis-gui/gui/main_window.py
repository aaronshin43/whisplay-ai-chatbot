from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal
from .header_widget import HeaderWidget
from .footer_widget import FooterWidget
from .chat_widget import ChatWidget
from .theme import DARK_THEME


class MainWindow(QMainWindow):
    key_pressed = pyqtSignal(int)   # Qt.Key_* value
    key_released = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setStyleSheet(DARK_THEME)

        # Central widget
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = HeaderWidget()
        self.chat = ChatWidget()
        self.footer = FooterWidget()

        layout.addWidget(self.header)
        layout.addWidget(self.chat, stretch=1)
        layout.addWidget(self.footer)

    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            self.key_pressed.emit(event.key())
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            self.key_released.emit(event.key())
        super().keyReleaseEvent(event)

    # ── Convenience delegation ──────────────────────────────────────────────

    def set_status(self, text: str):
        self.header.set_status(text)

    def set_footer(self, text: str):
        self.footer.set_text(text)
