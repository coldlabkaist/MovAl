from __future__ import annotations
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QStyledItemDelegate
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFontMetrics
from PyQt6.QtCore import Qt, QSize

CUTIE_COLOR_BASE = ["#ab1f24", "#36ae37", "#b9b917", "#063391", "#983a91",
                    "#20b6b5", "#c1c0bf", "#5c0d11", "#e71f19", "#60b630",
                    "#f4ba19", "#503390", "#ca4392", "#5eb7b7", "#f6bcbc"]

class NodePreviewDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option, index):
        super().paint(painter, option, index)

        inst_idx = index.data(Qt.ItemDataRole.UserRole + 1)
        if inst_idx is not None:
            chip_color = QColor(CUTIE_COLOR_BASE[inst_idx % len(CUTIE_COLOR_BASE)])

            chip_h = option.rect.height() - 6
            chip_w = 4
            chip_x = option.rect.left() + 2
            chip_y = option.rect.top() + 3

            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(chip_color))
            painter.drawRect(chip_x, chip_y, chip_w, chip_h)
            painter.restore()

        node = index.data(Qt.ItemDataRole.UserRole)
        if node is None:
            return

        rect = option.rect
        size   = min(rect.height(), 7)
        radius = size / 2
        cx = rect.left() + 8 + radius
        cy = rect.center().y()

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(node.color, max(1, int(node.thickness)))
        painter.setPen(pen)
        painter.setBrush(QBrush(node.color) if node.filled else Qt.BrushStyle.NoBrush)

        shape = (node.shape or "circle").lower()
        if shape == "circle":
            painter.drawEllipse(int(cx - radius), int(cy - radius), int(size), int(size))
        elif shape == "square":
            painter.drawRect(int(cx - radius), int(cy - radius), int(size), int(size))
        elif shape == "text":
            txt = node.text or node.name
            fnt = painter.font()
            fnt.setPointSize(int(radius * 3))
            painter.setFont(fnt)
            fm  = QFontMetrics(fnt)
            tx  = int(cx - fm.horizontalAdvance(txt) / 2)
            ty  = int(cy + fm.ascent() / 2)
            painter.drawText(tx, ty, txt)
        else:
            painter.drawEllipse(int(cx - radius), int(cy - radius), int(size), int(size))

        painter.restore()

    def sizeHint(self, option, index):
        base = super().sizeHint(option, index)
        return QSize(base.width(), max(base.height(), 20))

class KeypointListWidget(QListWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setItemDelegate(NodePreviewDelegate(self))

        self._item_map: dict[tuple[str, str], QListWidgetItem] = {}
        self._header_map: dict[str, QListWidgetItem] = {}

    def build(self, tracks, kp_order, skeleton_model):
        self.clear()
        self._item_map.clear()
        self._header_map.clear()

        for idx, track in enumerate(tracks):
            hdr = QListWidgetItem(f"Animal {idx + 1} ({track})")
            hdr.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.addItem(hdr)
            self._header_map[str(track)] = hdr

            for kp in kp_order:
                node = skeleton_model.nodes[kp]

                it = QListWidgetItem(f"    {kp}")
                it.setData(Qt.ItemDataRole.UserRole, node)
                it.setData(Qt.ItemDataRole.UserRole + 1, idx)
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.addItem(it)

                self._item_map[(str(track), kp)] = it

    def highlight(self, track, kp):
        item = self._item_map.get((str(track), kp))
        if item is None:
            return
        self.setCurrentItem(item)
        self.scrollToItem(item)
