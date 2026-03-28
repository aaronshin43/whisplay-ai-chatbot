from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
from .theme import get_font_size, HEADER_HEIGHT


class HeaderWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("header")
        self.setFixedHeight(HEADER_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(8)

        # Dot indicator
        self._dot = QLabel("●")
        self._dot.setObjectName("dot")
        dot_font = self._dot.font()
        dot_font.setPointSize(get_font_size(13))
        self._dot.setFont(dot_font)
        self._dot.setFixedWidth(get_font_size(20))

        # Title
        self._title = QLabel("OASIS")
        self._title.setObjectName("title")
        title_font = self._title.font()
        title_font.setPointSize(get_font_size(22))
        title_font.setBold(True)
        self._title.setFont(title_font)

        # Status (right-aligned)
        self._status = QLabel("Starting...")
        self._status.setObjectName("status")
        status_font = self._status.font()
        status_font.setPointSize(get_font_size(16))
        self._status.setFont(status_font)
        self._status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(self._dot)
        layout.addWidget(self._title)
        layout.addStretch()
        layout.addWidget(self._status)

    def set_status(self, text: str):
        self._status.setText(text)
