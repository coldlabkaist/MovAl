from __future__ import annotations
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QStyledItemDelegate
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFontMetrics, QFont
from PyQt6.QtCore import Qt, QSize

from ..IO.data_loader import DataLoader

CUTIE_COLOR_BASE = ["#ab1f24", "#36ae37", "#b9b917", "#063391", "#983a91",
                    "#20b6b5", "#c1c0bf", "#5c0d11", "#e71f19", "#60b630",
                    "#f4ba19", "#503390", "#ca4392", "#5eb7b7", "#f6bcbc"]

ROLE_NODE = Qt.ItemDataRole.UserRole
ROLE_COLOR_INDEX = Qt.ItemDataRole.UserRole + 1
ROLE_INSTANCE_KEY = Qt.ItemDataRole.UserRole + 2
ROLE_GROUP_ACTIVE = Qt.ItemDataRole.UserRole + 3
ROLE_SHOW_COLOR_CHIP = Qt.ItemDataRole.UserRole + 4

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

        inst_idx = index.data(ROLE_COLOR_INDEX)
        show_color_chip = bool(index.data(ROLE_SHOW_COLOR_CHIP))
        if inst_idx is not None and show_color_chip:
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

        node = index.data(ROLE_NODE)
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
        self._all_keypoint_items: list[QListWidgetItem] = []
        
        self.mouse_controller = None

        self._project_track_order: list[str] = []
        self._kp_order:    list[str] = []
        self._syncing_selection = False

    def _base_color_index(self, base_track: str) -> int:
        try:
            return self._project_track_order.index(base_track)
        except ValueError:
            return 0

    def _instance_sort_key(self, instance_key: str) -> tuple[int, str]:
        instance_id = DataLoader.get_instance_id_from_key(instance_key)
        resolved_id = int(instance_id) if instance_id is not None else 1
        return resolved_id, instance_key

    def _set_item_enabled_state(self, item: QListWidgetItem, enabled: bool) -> None:
        item.setForeground(QBrush(QColor("black") if enabled else QColor("lightgray")))

    def _set_item_flags(self, item: QListWidgetItem, *, enabled: bool, selectable: bool) -> None:
        flags = Qt.ItemFlag.NoItemFlags
        if enabled:
            flags |= Qt.ItemFlag.ItemIsEnabled
        if selectable:
            flags |= Qt.ItemFlag.ItemIsSelectable
        item.setFlags(flags)

    def _add_group(
        self,
        *,
        display_label: str,
        base_track: str,
        instance_key: str | None,
        kp_order: list[str],
        skeleton_model,
        enabled: bool,
    ) -> None:
        color_index = self._base_color_index(base_track)

        hdr = QListWidgetItem(display_label)
        base_font = self.font()
        bold_font = QFont(
            base_font.family(),
            base_font.pointSize(),
            QFont.Weight.Bold,
        )
        hdr.setFont(bold_font)
        hdr.setFlags(Qt.ItemFlag.ItemIsEnabled)
        hdr.setData(ROLE_COLOR_INDEX, color_index)
        hdr.setData(ROLE_INSTANCE_KEY, instance_key)
        hdr.setData(ROLE_GROUP_ACTIVE, enabled)
        hdr.setData(ROLE_SHOW_COLOR_CHIP, False)
        self._set_item_flags(hdr, enabled=True, selectable=instance_key is not None and enabled)
        self._set_item_enabled_state(hdr, enabled)
        self.addItem(hdr)
        if instance_key is not None:
            self._header_map[str(instance_key)] = hdr

        for kp in kp_order:
            node = skeleton_model.nodes.get(kp)
            if node is None:
                continue

            it = QListWidgetItem(f"    {kp}")
            it.setData(ROLE_NODE, node)
            it.setData(ROLE_COLOR_INDEX, color_index)
            it.setData(ROLE_INSTANCE_KEY, instance_key)
            it.setData(ROLE_GROUP_ACTIVE, enabled)
            it.setData(ROLE_SHOW_COLOR_CHIP, True)
            self._set_item_flags(it, enabled=True, selectable=instance_key is not None and enabled)
            self._set_item_enabled_state(it, enabled)
            self.addItem(it)
            self._all_keypoint_items.append(it)

            if instance_key is not None:
                self._item_map[(str(instance_key), kp)] = it

    def build(self, project_tracks, visible_tracks, kp_order, skeleton_model):
        self.clear()
        self._item_map.clear()
        self._header_map.clear()
        self._all_keypoint_items.clear()

        self._project_track_order = [str(t) for t in project_tracks]
        self._kp_order    = list(kp_order)

        visible_by_base: dict[str, list[str]] = {}
        for raw_key in visible_tracks:
            instance_key = str(raw_key)
            base_track = DataLoader.get_base_track_name(instance_key)
            visible_by_base.setdefault(base_track, []).append(instance_key)

        for keys in visible_by_base.values():
            keys.sort(key=self._instance_sort_key)

        ordered_base_tracks = list(self._project_track_order)
        for base_track in visible_by_base:
            if base_track not in ordered_base_tracks:
                ordered_base_tracks.append(base_track)

        for base_track in ordered_base_tracks:
            visible_keys = visible_by_base.get(base_track, [])
            if not visible_keys:
                self._add_group(
                    display_label=base_track,
                    base_track=base_track,
                    instance_key=None,
                    kp_order=self._kp_order,
                    skeleton_model=skeleton_model,
                    enabled=False,
                )
                continue

            if len(visible_keys) == 1:
                self._add_group(
                    display_label=base_track,
                    base_track=base_track,
                    instance_key=visible_keys[0],
                    kp_order=self._kp_order,
                    skeleton_model=skeleton_model,
                    enabled=True,
                )
                continue

            for instance_key in visible_keys:
                instance_id = DataLoader.get_instance_id_from_key(instance_key)
                suffix = f" [{instance_id}]" if instance_id is not None else ""
                self._add_group(
                    display_label=f"{base_track}{suffix}",
                    base_track=base_track,
                    instance_key=instance_key,
                    kp_order=self._kp_order,
                    skeleton_model=skeleton_model,
                    enabled=True,
                )

    def highlight(self, track: str | None, kp: str | None):
        if not self._kp_order:
            return
        
        for it in self._item_map.values():
            it.setBackground(QBrush(Qt.BrushStyle.NoBrush))
        for it in self._header_map.values():
            it.setBackground(QBrush(Qt.BrushStyle.NoBrush))

        if track:
            track = str(track)
            header_item = self._header_map.get(track)
            if header_item is None:
                return
            color_index = header_item.data(ROLE_COLOR_INDEX) or 0
            target_item = header_item
            if kp:
                kpt_item = self._item_map.get((track, kp))
                if kpt_item is not None:
                    kpt_item.setBackground(_background_color_kpt(color_index))
                    target_item = kpt_item
            header_item.setBackground(_background_color_track(color_index))
            self._set_current_item(target_item)
            return

        self._set_current_item(None)

    def _set_current_item(self, item: QListWidgetItem | None) -> None:
        self._syncing_selection = True
        try:
            self.blockSignals(True)
            self.setCurrentItem(item)
            if item is not None:
                item.setSelected(True)
            else:
                self.clearSelection()
        finally:
            self.blockSignals(False)
            self._syncing_selection = False

    def is_syncing_selection(self) -> bool:
        return self._syncing_selection

    def get_item_selection(self, item: QListWidgetItem | None) -> tuple[str | None, str | None]:
        if item is None:
            return None, None

        instance_key = item.data(ROLE_INSTANCE_KEY)
        if not instance_key:
            return None, None

        node = item.data(ROLE_NODE)
        node_name = getattr(node, "name", None) if node is not None else None
        return str(instance_key), node_name

    def update_list_visibility(self, coords: dict[str, dict[str, tuple]]):
        for item in self._header_map.values():
            self._set_item_enabled_state(item, True)

        for item in self._all_keypoint_items:
            instance_key = item.data(ROLE_INSTANCE_KEY)
            group_active = bool(item.data(ROLE_GROUP_ACTIVE))
            if not group_active or not instance_key:
                self._set_item_enabled_state(item, False)
                continue

            node = item.data(ROLE_NODE)
            node_name = getattr(node, "name", None)
            is_visible = bool(node_name and node_name in coords.get(str(instance_key), {}))
            self._set_item_enabled_state(item, is_visible)
