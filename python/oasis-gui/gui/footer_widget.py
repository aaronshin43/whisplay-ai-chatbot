from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
from .theme import get_font_size, FOOTER_HEIGHT


class FooterWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("footer")
        self.setFixedHeight(FOOTER_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)

        self._label = QLabel("Press and hold button to speak")
        self._label.setObjectName("footer_text")
        self._label.setAlignment(Qt.AlignCenter)
        font = self._label.font()
        font.setPointSize(get_font_size(10))
        self._label.setFont(font)

        layout.addWidget(self._label)

    def set_text(self, text: str):
        self._label.setText(text)
