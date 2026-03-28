from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QColor, QFont, QTextBlockFormat
from PyQt5.QtCore import Qt, pyqtSlot
from .theme import get_font_size, CHAT_LINE_SPACING


ROLE_USER = "user"
ROLE_OASIS = "oasis"

# Font sizes (base values at 480px portrait width on 4.3" display)
_BODY_PT   = 22   # main response text
_LABEL_PT  = 16   # "You" / "OASIS" role labels


class ChatWidget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chat")
        self.setReadOnly(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._drag_start_y = None
        self._drag_start_scroll = 0

        font = QFont()
        font.setFamilies(["Noto Sans", "DejaVu Sans", "Helvetica Neue", "Arial"])
        font.setPointSize(get_font_size(_BODY_PT))
        self.setFont(font)

        # Block format with portrait-optimised line spacing
        self._block_fmt = QTextBlockFormat()
        self._block_fmt.setLineHeight(
            CHAT_LINE_SPACING, QTextBlockFormat.ProportionalHeight
        )

        self._streaming = False
        self._auto_scroll = True  # follows new tokens unless user scrolled up

    # ── Format helpers ──────────────────────────────────────────────────────

    def _fmt(self, color: str, bold: bool = False, pt: int = 0) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        if bold:
            fmt.setFontWeight(QFont.Bold)
        fmt.setFontPointSize(get_font_size(pt or _BODY_PT))
        return fmt

    def _label_fmt(self, color: str) -> QTextCharFormat:
        return self._fmt(color, bold=True, pt=_LABEL_PT)

    def _body_fmt(self, color: str) -> QTextCharFormat:
        return self._fmt(color, pt=_BODY_PT)

    # ── Message blocks ──────────────────────────────────────────────────────

    def add_message(self, role: str, text: str):
        """Append a complete message block."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)

        if not self.document().isEmpty():
            cursor.insertBlock(self._block_fmt)
            cursor.insertText("")

        if role == ROLE_USER:
            cursor.insertBlock(self._block_fmt)
            cursor.insertText("You", self._label_fmt("#777777"))
            cursor.insertBlock(self._block_fmt)
            cursor.insertText(text, self._body_fmt("#bbbbbb"))
        else:
            cursor.insertBlock(self._block_fmt)
            cursor.insertText("OASIS", self._label_fmt("#00c8a3"))
            cursor.insertBlock(self._block_fmt)
            cursor.insertText(text, self._body_fmt("#e0e0e0"))

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
        cursor.insertText("OASIS", self._label_fmt("#00c8a3"))
        cursor.insertBlock(self._block_fmt)
        self.setTextCursor(cursor)
        self._streaming = True

    @pyqtSlot(str)
    def append_token(self, token: str):
        """Append a single streaming token — called from main thread via signal."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token, self._body_fmt("#e0e0e0"))
        if self._auto_scroll:
            self.setTextCursor(cursor)
            self.ensureCursorVisible()

    def end_oasis_response(self):
        self._streaming = False

    def clear_chat(self):
        self.clear()
        self._streaming = False

    # ── Touch / drag scroll ─────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_y = event.pos().y()
            self._drag_start_scroll = self.verticalScrollBar().value()
            self._auto_scroll = False  # freeze immediately on touch

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self._drag_start_y is not None:
            delta = event.pos().y() - self._drag_start_y
            self.verticalScrollBar().setValue(self._drag_start_scroll - delta)

    def mouseReleaseEvent(self, _event):
        self._drag_start_y = None
        # resume auto-scroll only if user dragged back to the bottom
        sb = self.verticalScrollBar()
        self._auto_scroll = sb.value() >= sb.maximum() - 4
