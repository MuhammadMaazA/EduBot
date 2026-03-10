"""Activity window with chatbot, coding area, scorecard, and settings."""
import html
import re
import sys
import os
import tempfile
from pathlib import Path

from PyQt6 import uic
from PyQt6.QtCore import QProcess, QRegularExpression, QTimer
from PyQt6.QtGui import QColor, QPalette, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QWidget, QDialog, QLabel, QHBoxLayout, QSizePolicy
from PyQt6.QtGui import QSyntaxHighlighter

# path to activity_window.ui
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(CURRENT_DIR, "activity_window.ui")

# ──────────────────────────────────────────────────────────────────────────────
# Python syntax highlighter
# ──────────────────────────────────────────────────────────────────────────────

class _PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._formats: dict[str, QTextCharFormat] = {
            "keyword":  self._fmt(QColor("#c678dd")),
            "string":   self._fmt(QColor("#98c379")),
            "comment":  self._fmt(QColor("#5c6370"), italic=True),
            "function": self._fmt(QColor("#61afef")),
            "number":   self._fmt(QColor("#d19a66")),
        }
        self._rules: list[tuple] = []
        for kw in (
            "and as assert break class continue def del elif else except "
            "finally for from global if import in is lambda nonlocal not "
            "or pass raise return try while with yield True False None"
        ).split():
            self._rules.append(
                (QRegularExpression(rf"\b{re.escape(kw)}\b"), "keyword")
            )
        self._rules += [
            (QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"'), "string"),
            (QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"), "string"),
            (QRegularExpression(r'#.*$'),                   "comment"),
            (QRegularExpression(r'\b[0-9]+\b'),             "number"),
            (QRegularExpression(r'\b[A-Za-z_][A-Za-z0-9_]*(?=\s*\()'), "function"),
        ]

    def _fmt(self, color: QColor, bold=False, italic=False) -> QTextCharFormat:
        f = QTextCharFormat()
        f.setForeground(color)
        if bold:
            f.setFontWeight(700)
        if italic:
            f.setFontItalic(True)
        return f

    def highlightBlock(self, text: str):
        for pattern, name in self._rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(
                    m.capturedStart(), m.capturedLength(), self._formats[name]
                )


# ──────────────────────────────────────────────────────────────────────────────
# Settings dialog
# ──────────────────────────────────────────────────────────────────────────────

class ActivitySettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("activity_settings_window.ui", self)
        self.save_settings_button.clicked.connect(self._save)
        self.close_button.clicked.connect(self.accept)
        self.start_listening_button.clicked.connect(lambda: print("Start listening"))
        self.stop_listening_button.clicked.connect(lambda: print("Stop listening"))

    def _save(self):
        print(
            f"Settings saved: Name={self.robot_name.text()}, "
            f"Volume={self.robot_volume.value()}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Chat bubble helper
# ──────────────────────────────────────────────────────────────────────────────

def _append_bubble(scroll_area, bubble_layout, text: str, is_user: bool) -> None:
    """Append a rounded QLabel bubble.  User = right/blue, Robot = left/navy."""
    bg = "#2563eb" if is_user else "#1e3a5f"

    bubble = QLabel(text)
    bubble.setWordWrap(True)
    bubble.setStyleSheet(
        f"background-color:{bg}; color:white;"
        "border-radius:16px; padding:10px 16px;"
        "font-size:13px; font-weight:600;"
    )
    bubble.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    bubble.setMaximumWidth(320)

    row = QWidget()
    row.setStyleSheet("background:transparent;")
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(0)

    if is_user:
        row_layout.addStretch(1)
        row_layout.addWidget(bubble)
    else:
        row_layout.addWidget(bubble)
        row_layout.addStretch(1)

    bubble_layout.insertWidget(max(0, bubble_layout.count() - 1), row)

    QTimer.singleShot(
        30,
        lambda: scroll_area.verticalScrollBar().setValue(
            scroll_area.verticalScrollBar().maximum()
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main activity window
# ──────────────────────────────────────────────────────────────────────────────

class ActivityWindow(QWidget):
    """Main activity window."""

    def __init__(self, activity_number: int = 1, parent=None):
        super().__init__(parent)
        uic.loadUi(UI_PATH, self)

        self._activity_number    = activity_number
        self._passed_count       = 0
        self._failed_count       = 0
        self._code_process: QProcess | None = None
        self._tmp_script: str | None        = None
        self._stopwatch_seconds  = 0

        # Stopwatch
        self._stopwatch_timer = QTimer(self)
        self._stopwatch_timer.timeout.connect(self._tick_stopwatch)

        # Syntax highlighter on code editor
        self._highlighter = _PythonHighlighter(self.code_editor.document())
        p = self.code_editor.palette()
        p.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
        self.code_editor.setPalette(p)
        self.code_editor.setTabStopDistance(16.0)

        # Window / labels
        self.setWindowTitle(f"Activity {self._activity_number}")
        self.activity_title_label.setText(f"Activity {self._activity_number}")
        self.current_activity_label.setText(
            f"Current activity: {self._activity_number}"
        )
        self._update_scorecard_log()

        # Column stretches for coding grid layouts
        # (PyQt6 uic doesn't support columnStretch in .ui files)
        self.coding_toolbar_grid.setColumnStretch(0, 1)
        self.coding_toolbar_grid.setColumnStretch(1, 0)
        self.coding_toolbar_grid.setColumnStretch(2, 1)
        self.coding_editors_grid.setColumnStretch(0, 1)
        self.coding_editors_grid.setColumnStretch(1, 0)
        self.coding_editors_grid.setColumnStretch(2, 1)
        self.stdin_row_grid.setColumnStretch(0, 1)
        self.stdin_row_grid.setColumnStretch(1, 0)
        self.stdin_row_grid.setColumnStretch(2, 1)

        # stdin bar hidden until a process starts
        self._set_stdin_visible(False)

        # Chat scroll: bottom spacer pushes bubbles to top
        self.chat_bubble_layout.addStretch(1)

        self._connect_signals()
        self._stopwatch_timer.start(1000)

    # ── signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.settings_button.clicked.connect(self._open_settings)
        self.send_message_button.clicked.connect(self._send_chat_message)
        self.chat_input.returnPressed.connect(self._send_chat_message)
        self.play_code_button.clicked.connect(self._run_code)
        self.stop_code_button.clicked.connect(self._stop_code)
        self.stdin_input.returnPressed.connect(self._send_stdin)
        self.stdin_send_button.clicked.connect(self._send_stdin)

    # ── stdin visibility ──────────────────────────────────────────────────────

    def _set_stdin_visible(self, visible: bool):
        self.stdin_input.setVisible(visible)
        self.stdin_send_button.setVisible(visible)
        if visible:
            self.stdin_input.setFocus()

    # ── settings ──────────────────────────────────────────────────────────────

    def _open_settings(self):
        ActivitySettingsDialog(self).exec()

    # ── stopwatch ─────────────────────────────────────────────────────────────

    def _tick_stopwatch(self):
        self._stopwatch_seconds += 1
        m, s = divmod(self._stopwatch_seconds, 60)
        self.stopwatch_label.setText(f"Time: {m}:{s:02d}")

    def _update_scorecard_log(self):
        self.scorecard_log_label.setText(
            f"Passed: {self._passed_count}  |  Failed: {self._failed_count}"
        )

    # ── chat ──────────────────────────────────────────────────────────────────

    def _send_chat_message(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        _append_bubble(self.chat_scroll, self.chat_bubble_layout, text, is_user=True)
        self.chat_input.clear()
        _append_bubble(
            self.chat_scroll,
            self.chat_bubble_layout,
            "(response to be filled in later)",
            is_user=False,
        )

    # ── code runner ───────────────────────────────────────────────────────────

    def _run_code(self):
        """Kill any existing process, then run the code editor contents fresh."""
        # Clean up any leftover process first
        if self._code_process is not None:
            if self._code_process.state() != QProcess.ProcessState.NotRunning:
                try:
                    self._code_process.readyReadStandardOutput.disconnect()
                    self._code_process.finished.disconnect()
                except RuntimeError:
                    pass
                self._code_process.kill()
                self._code_process.waitForFinished(500)
            self._code_process = None

        code = self.code_editor.toPlainText().strip()
        if not code:
            self.terminal_output.setPlainText("No code to run.")
            return

        self.terminal_output.clear()

        # Write to a temp file; -u = unbuffered so output arrives immediately
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        tmp.write(code)
        tmp.close()
        self._tmp_script = tmp.name

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        self._code_process = proc

        # Use the same interpreter that runs this app; -u = unbuffered stdout
        proc.start(sys.executable, ["-u", self._tmp_script])

        if not proc.waitForStarted(3000):
            self._append_terminal("Error: could not start Python.\n", colour="error")
            self._cleanup_tmp()
            self._code_process = None
            return

        self._set_stdin_visible(True)

    def _stop_code(self):
        """
        Kill the running process without blocking the GUI.

        Strategy:
          1. Disconnect signals immediately so _on_finished doesn't fire.
          2. Call kill() — guaranteed on Windows (TerminateProcess) and Unix.
          3. Use QTimer to wait + clean up off the main call stack so the GUI
             thread never blocks inside waitForFinished().
        """
        proc = self._code_process
        if proc is None or proc.state() == QProcess.ProcessState.NotRunning:
            return

        # Prevent _on_finished from running after we take control
        try:
            proc.readyReadStandardOutput.disconnect()
            proc.finished.disconnect()
        except RuntimeError:
            pass

        self._code_process = None   # null reference before async wait

        proc.kill()
        # Deferred wait — 300 ms gives the OS time to release the handle
        # without freezing the GUI
        QTimer.singleShot(300, lambda: self._finish_stop(proc))

    def _finish_stop(self, proc: QProcess):
        """Called ~300 ms after kill(); drains + cleans up."""
        proc.waitForFinished(200)
        self._cleanup_tmp()
        self._set_stdin_visible(False)
        self._append_terminal("\n[Program stopped]", colour="stopped")

    def _send_stdin(self):
        """Send the stdin_input line to the running process."""
        proc = self._code_process
        if proc is None or proc.state() != QProcess.ProcessState.Running:
            return
        text = self.stdin_input.text()          # what the user typed
        self.stdin_input.clear()
        # Echo the typed text into the terminal (program won't echo it itself
        # because QProcess doesn't allocate a real TTY)
        self._append_terminal(text + "\n")
        proc.write((text + "\n").encode("utf-8"))

    def _on_output(self):
        proc = self._code_process
        if proc is None:
            return
        data = proc.readAllStandardOutput().data()
        text = data.decode("utf-8", errors="replace")
        if text:
            self._append_terminal(text)

    def _on_finished(self, exit_code: int, _status):
        """Called when the process exits on its own (not via _stop_code)."""
        proc = self._code_process
        if proc is not None:
            # Drain any last bytes
            data = proc.readAllStandardOutput().data()
            text = data.decode("utf-8", errors="replace")
            if text:
                self._append_terminal(
                    text, colour="error" if exit_code != 0 else None
                )
        self._cleanup_tmp()
        self._code_process = None
        self._set_stdin_visible(False)

    # ── terminal display ──────────────────────────────────────────────────────

    def _append_terminal(self, text: str, colour: str | None = None) -> None:
        """
        Append *text* to the read-only terminal display (QTextEdit).
          colour=None      → white  (normal output)
          colour='error'   → red    (#ef4444)
          colour='stopped' → orange (#f97316) bold
        """
        cursor = self.terminal_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.terminal_output.setTextCursor(cursor)

        if colour in ("error", "stopped"):
            hex_col = "#ef4444" if colour == "error" else "#f97316"
            weight  = "bold" if colour == "stopped" else "normal"
            escaped = html.escape(text).replace("\n", "<br>")
            self.terminal_output.insertHtml(
                f'<span style="color:{hex_col}; font-weight:{weight}; '
                f'font-family:Consolas,monospace; font-size:13px;">'
                f"{escaped}</span>"
            )
        else:
            self.terminal_output.insertPlainText(text)

        self.terminal_output.ensureCursorVisible()

    # ── cleanup ───────────────────────────────────────────────────────────────

    def _cleanup_tmp(self):
        if self._tmp_script:
            Path(self._tmp_script).unlink(missing_ok=True)
            self._tmp_script = None
