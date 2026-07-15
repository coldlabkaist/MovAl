from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class TaskProgressPanel(QFrame):
    """Compact progress card shared by data split, training, and inference."""

    def __init__(self, parent: QWidget | None = None, *, show_log: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("StatusStrip")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("SubtleText")
        header.addWidget(self.status_label, 1)

        self.log_button: QPushButton | None = None
        if show_log:
            self.log_button = QPushButton("Show Details")
            self.log_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.log_button.clicked.connect(self._toggle_log)
            header.addWidget(self.log_button)
        layout.addLayout(header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.detail_label = QLabel("")
        self.detail_label.setObjectName("SubtleText")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

        self.log_view: QPlainTextEdit | None = None
        if show_log:
            self.log_view = QPlainTextEdit()
            self.log_view.setReadOnly(True)
            self.log_view.setMaximumBlockCount(1000)
            self.log_view.setMinimumHeight(120)
            self.log_view.setMaximumHeight(220)
            self.log_view.hide()
            layout.addWidget(self.log_view)

        self.hide()

    def reset(self, message: str = "Preparing...") -> None:
        self.show()
        self.status_label.setText(message)
        self.detail_label.clear()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Running...")
        if self.log_view is not None:
            self.log_view.clear()

    def update_progress(
        self,
        done: int,
        total: int,
        message: str,
        detail: str = "",
    ) -> None:
        self.show()
        self.status_label.setText(message or "Running...")
        self.detail_label.setText(detail)
        if total > 0:
            safe_done = max(0, min(int(done), int(total)))
            self.progress_bar.setRange(0, int(total))
            self.progress_bar.setValue(safe_done)
            self.progress_bar.setFormat("%v/%m (%p%)")
        else:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Running...")

    def set_stopping(self, message: str = "Stopping...") -> None:
        self.show()
        self.status_label.setText(message)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Stopping...")

    def set_result(
        self,
        message: str,
        *,
        success: bool,
        detail: str = "",
        cancelled: bool = False,
    ) -> None:
        self.show()
        self.status_label.setText(message)
        self.detail_label.setText(detail)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1 if success else 0)
        if success:
            result_text = "Completed"
        elif cancelled:
            result_text = "Stopped"
        else:
            result_text = "Failed"
        self.progress_bar.setFormat(result_text)

    def append_log(self, text: str) -> None:
        if self.log_view is None:
            return
        clean = str(text or "").strip()
        if clean:
            self.log_view.appendPlainText(clean)

    def _toggle_log(self) -> None:
        if self.log_view is None or self.log_button is None:
            return
        visible = not self.log_view.isVisible()
        self.log_view.setVisible(visible)
        self.log_button.setText("Hide Details" if visible else "Show Details")
