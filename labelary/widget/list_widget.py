from __future__ import annotations
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QStyledItemDelegate
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFontMetrics, QFont
from PyQt6.QtCore import Qt, QSize

CUTIE_COLOR_BASE = ["#ab1f24", "#36ae37", "#b9b917", "#063391", "#983a91",
                    "#20b6b5", "#c1c0bf", "#5c0d11", "#e71f19", "#60b630",
                    "#f4ba19", "#503390", "#ca4392", "#5eb7b7", "#f6bcbc"]

def _background_color_track(idx) -> QColor:
    color = QColor(CUTIE_COLOR_BASE[idx])
    other, t = QColor("white"), 0.3
    return QColor(round(color.red()*(1-t)+other.red()*t),
                round(color.green()*(1-t)+other.green()*t),
                round(color.blue()*(1-t)+other.blue()*t))

def _background_color_kpt(idx) -> QColor:
    color = QColor(CUTIE_COLOR_BASE[idx])
    other, t = QColor("white"), 0.7
    return QColor(round(color.red()*(1-t)+other.red()*t),
                round(color.green()*(1-t)+other.green()*t),
                round(color.blue()*(1-t)+other.blue()*t))


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
        
        self.mouse_controller = None

        self._track_order: list[str] = []
        self._kp_order:    list[str] = []

    def build(self, tracks, kp_order, skeleton_model):
        self.clear()
        self._item_map.clear()
        self._header_map.clear()

        self._track_order = [str(t) for t in tracks]
        self._kp_order    = list(kp_order)

        for idx, track in enumerate(tracks):
            hdr = QListWidgetItem(track)
            base_font = self.font()
            bold_font = QFont(base_font.family(),
                    base_font.pointSize(),
                    QFont.Weight.Bold)
            hdr.setFont(bold_font)
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

    def highlight(self, track: str | None, kp: str | None):
        if not self._track_order or not self._kp_order:
            return
        
        for it in self._item_map.values():
            it.setBackground(QBrush(Qt.BrushStyle.NoBrush))
        for it in self._header_map.values():
            it.setBackground(QBrush(Qt.BrushStyle.NoBrush))

        if track:
            track = str(track)
            track_idx = self._track_order.index(track)
            try:
                base_idx = track_idx * (len(self._kp_order) + 1)
            except ValueError:
                return 
            if kp:
                try:
                    offset = 1 + self._kp_order.index(kp)
                except ValueError:
                    return
            else:
                offset = 0
            row = base_idx + offset

            kpt_item = self.item(row)
            if kpt_item:
                kpt_item.setBackground(_background_color_kpt(track_idx))
            track_item = self.item(base_idx)
            if track_item:
                track_item.setBackground(_background_color_track(track_idx))

    def update_list_visibility(self, coords: dict[str, dict[str, tuple]]):
        for (tr, kp), item in self._item_map.items():
            if kp in coords.get(tr, {}):
                item.setForeground(QBrush(QColor("black")))
            else:
                item.setForeground(QBrush(QColor("lightgray")))