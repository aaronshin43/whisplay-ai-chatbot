from PyQt5.QtWidgets import (
    QMainWindow, QGraphicsView, QGraphicsScene,
    QWidget, QVBoxLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from .header_widget import HeaderWidget
from .footer_widget import FooterWidget
from .chat_widget import ChatWidget
from .theme import DARK_THEME


class ContentWidget(QWidget):
    """Portrait-layout content (header + chat + footer).
    Sized to portrait dimensions (short × long), then rotated by MainWindow.
    """
    def __init__(self, portrait_w: int, portrait_h: int):
        super().__init__()
        self.setFixedSize(portrait_w, portrait_h)
        self.setObjectName("central")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = HeaderWidget()
        self.chat = ChatWidget()
        self.footer = FooterWidget()

        layout.addWidget(self.header)
        layout.addWidget(self.chat, stretch=1)
        layout.addWidget(self.footer)


class MainWindow(QMainWindow):
    key_pressed = pyqtSignal(int)
    key_released = pyqtSignal(int)

    def __init__(self, screen_size: tuple = None):
        """
        Args:
            screen_size: (width, height) of the physical screen.
                         Pass (800, 480) on PC to simulate Pi.
                         Pass None on Pi to auto-detect.
        """
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setStyleSheet(DARK_THEME)

        if screen_size:
            sw, sh = screen_size
        else:
            from PyQt5.QtWidgets import QApplication
            s = QApplication.primaryScreen().size()
            sw, sh = s.width(), s.height()

        # Portrait dimensions: short side = width, long side = height
        pw = min(sw, sh)   # 480
        ph = max(sw, sh)   # 800

        # QGraphicsView fills the physical screen (landscape: sw × sh)
        view = QGraphicsView()
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setFrameShape(QGraphicsView.NoFrame)
        view.setBackgroundBrush(QBrush(QColor("#0d0d0d")))
        view.setScene(QGraphicsScene(0, 0, sw, sh))
        view.setInteractive(True)

        # Portrait content widget (pw × ph)
        # Stylesheet must be applied directly — QGraphicsProxyWidget does not
        # inherit the parent QMainWindow's stylesheet.
        self.content = ContentWidget(pw, ph)
        self.content.setStyleSheet(DARK_THEME)
        proxy = view.scene().addWidget(self.content)

        # Rotate -90° (CCW) so portrait content fits landscape screen.
        # Math: -90° maps local (x,y) → scene (y, -x).
        # Translate by (0, pw) to bring y back to positive range.
        proxy.setRotation(-90)
        proxy.setPos(0, pw)

        self.setCentralWidget(view)

    # ── Convenience delegation ──────────────────────────────────────────────

    @property
    def chat(self):
        return self.content.chat

    def set_status(self, text: str):
        self.content.header.set_status(text)

    def set_footer(self, text: str):
        self.content.footer.set_text(text)

    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            self.key_pressed.emit(event.key())
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            self.key_released.emit(event.key())
        super().keyReleaseEvent(event)
