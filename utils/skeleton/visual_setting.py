from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

class NodeVisualSettingDialog(QDialog):
    def __init__(self, node, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Visual Settings – {node.name}")
        self.node = node
        layout = QVBoxLayout(self)

        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Shape:"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["circle", "square", "text"])
        self.shape_combo.setCurrentText(node.shape)
        shape_row.addWidget(self.shape_combo, 1)
        layout.addLayout(shape_row)

        text_row = QHBoxLayout()
        text_row.addWidget(QLabel("Text:"))
        self.text_edit = QLineEdit(node.text or node.name)
        self.text_edit.setEnabled(node.shape == "text")
        text_row.addWidget(self.text_edit, 1)
        layout.addLayout(text_row)

        colour_row = QHBoxLayout()
        colour_row.addWidget(QLabel("Colour:"))
        self.colour_btn = QPushButton("   ") 
        self._update_colour_btn(node.color)
        self.colour_btn.clicked.connect(self._choose_colour)
        colour_row.addWidget(self.colour_btn, 0)
        layout.addLayout(colour_row)

        self.filled_chk = QCheckBox("Filled")
        self.filled_chk.setChecked(node.filled)
        layout.addWidget(self.filled_chk, 0, Qt.AlignmentFlag.AlignLeft)

        thick_row = QHBoxLayout()
        thick_row.addWidget(QLabel("Line thickness:"))
        self.thickness_spin = QSpinBox()
        self.thickness_spin.setRange(1, 20)
        self.thickness_spin.setValue(node.thickness)
        thick_row.addWidget(self.thickness_spin)
        layout.addLayout(thick_row)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.shape_combo.currentTextChanged.connect(self._on_shape_changed)

    def apply_changes(self) -> None:
        self.node.shape = self.shape_combo.currentText()
        if self.node.shape == "text":
            self.node.text = self.text_edit.text().strip() or self.node.name
        else:
            self.node.text = None

        self.node.color = self._current_colour()
        self.node.filled = self.filled_chk.isChecked()
        self.node.thickness = self.thickness_spin.value()

    def _update_colour_btn(self, colour: QColor):
        if not isinstance(colour, QColor):
            colour = QColor(colour)
        self.colour_btn.setStyleSheet(f"background-color: {colour.name()}; border: 1px solid #666;")

    def _choose_colour(self):
        col = QColorDialog.getColor(self._current_colour(), self)
        if col.isValid():
            self._update_colour_btn(col)

    def _current_colour(self) -> QColor:
        sheet = self.colour_btn.styleSheet()
        start = sheet.find("background-color:")
        if start == -1:
            return QColor("#666666")
        start = sheet.find('#', start)
        return QColor(sheet[start:start + 7])

    def _on_shape_changed(self, shape: str):
        self.text_edit.setEnabled(shape == "text")


def edit_node_visual(node_item, parent=None):
    dlg = NodeVisualSettingDialog(node_item.node, parent)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        dlg.apply_changes()
        node_item.update()
        for edge in list(node_item.edges):
            edge.setPen(edge.normal_pen)
