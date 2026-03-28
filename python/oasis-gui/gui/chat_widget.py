from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QColor, QFont, QTextBlockFormat
from PyQt5.QtCore import Qt, pyqtSlot
from .theme import get_font_size, CHAT_LINE_SPACING


ROLE_USER = "user"
ROLE_OASIS = "oasis"


class ChatWidget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chat")
        self.setReadOnly(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Portrait-optimized font: larger base size for narrow screen readability
        font = QFont()
        font.setFamilies(["Noto Sans", "DejaVu Sans", "Helvetica Neue", "Arial"])
        font.setPointSize(get_font_size(14))
        self.setFont(font)

        # Line spacing for portrait readability
        self._block_fmt = QTextBlockFormat()
        self._block_fmt.setLineHeight(CHAT_LINE_SPACING, QTextBlockFormat.ProportionalHeight)

        self._streaming = False

    def _fmt(self, color: str, bold: bool = False, size_delta: int = 0) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        if bold:
            fmt.setFontWeight(QFont.Bold)
        if size_delta:
            fmt.setFontPointSize(get_font_size(14) + size_delta)
        return fmt

    def _insert_block(self, cursor: QTextCursor):
        """Insert a new paragraph with portrait line spacing applied."""
        cursor.insertBlock(self._block_fmt)

    def add_message(self, role: str, text: str):
        """Append a complete message block."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)

        if not self.document().isEmpty():
            # Blank line between messages
            cursor.insertBlock(self._block_fmt)
            cursor.insertText("")

        if role == ROLE_USER:
            cursor.insertBlock(self._block_fmt)
            cursor.insertText("You", self._fmt("#777777", bold=True, size_delta=-2))
            cursor.insertBlock(self._block_fmt)
            cursor.insertText(text, self._fmt("#bbbbbb"))
        else:
            cursor.insertBlock(self._block_fmt)
            cursor.insertText("OASIS", self._fmt("#00c8a3", bold=True, size_delta=-2))
            cursor.insertBlock(self._block_fmt)
            cursor.insertText(text, self._fmt("#e0e0e0"))

        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def begin_oasis_response(self):
        """Start a new OASIS streaming block."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)

        if not self.document().isEmpty():
            cursor.insertBlock(self._block_fmt)
            cursor.insertText("")

        cursor.insertBlock(self._block_fmt)
        cursor.insertText("OASIS", self._fmt("#00c8a3", bold=True, size_delta=-2))
        cursor.insertBlock(self._block_fmt)
        self.setTextCursor(cursor)
        self._streaming = True

    @pyqtSlot(str)
    def append_token(self, token: str):
        """Append a single streaming token — called from main thread via signal."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token, self._fmt("#e0e0e0"))
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def end_oasis_response(self):
        self._streaming = False

    def clear_chat(self):
        self.clear()
        self._streaming = False
