from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import yaml
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from utils.skeleton import EdgeItem, NodeItem, NodeVisualSettingDialog, SkeletonModel, SkeletonScene, SymItem

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = REPO_ROOT / "preset" / "skeleton"


class SkeletonManagerDialog(QDialog):
    def __init__(
        self,
        main_window,
        *,
        project=None,
        allow_structure_edit: bool = True,
        save_callback: Optional[Callable[[SkeletonModel, bool], None]] = None,
    ) -> None:
        super().__init__(main_window)
        self.setWindowTitle("Skeleton Manager")
        self.resize(920, 560)

        self.main_window = main_window
        self.project = project
        self.save_callback = save_callback
        self._preset_dir = PRESET_DIR
        self._preset_dir.mkdir(parents=True, exist_ok=True)
        self._is_project_mode = project is not None
        self._structure_edit_enabled = allow_structure_edit if not self._is_project_mode else allow_structure_edit
        self._structure_edit_unlocked = bool(self._structure_edit_enabled)

        self.model = SkeletonModel()
        self.scene = SkeletonScene(self.model, self)
        self.scene.setSceneRect(-200, -200, 700, 420)
        self.view = QGraphicsView(self.scene)
        self.node_list = QListWidget()
        self.node_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

        self.node_items: dict[str, NodeItem] = {}
        self._sync_list = True
        self._sync_scene = True

        self._build_ui()
        self._connect_signals()
        self._load_initial_model()
        self._apply_structure_edit_state(self._structure_edit_enabled)
        QTimer.singleShot(0, lambda: self._fill_background(Qt.GlobalColor.white))

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        self.mode_label = QLabel(self)
        top_bar.addWidget(self.mode_label)

        self.combo = QComboBox(self)
        self.combo.setEditable(False)
        self.combo.setPlaceholderText("Select config file")
        self.title_edit = QLineEdit(self)
        self.title_edit.setPlaceholderText("Config title")

        self.project_info_label = QLabel(self)
        self.project_info_label.setWordWrap(True)

        if self._is_project_mode:
            self.mode_label.setText("Project Skeleton")
            top_bar.addWidget(self.project_info_label, 1)
            self.project_info_label.setText(
                f"Project: {self.project.title} | Preset base: {self.project.skeleton_name}"
            )
        else:
            self.mode_label.setText("Preset")
            top_bar.addWidget(self.combo, 1)
            top_bar.addWidget(QLabel("Title:", self))
            top_bar.addWidget(self.title_edit, 2)

        main_layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_layout.addWidget(splitter, 1)

        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        splitter.addWidget(left_widget)

        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        btn_row = QHBoxLayout()
        self.btn_load_img = QPushButton("Load Image", self)
        self.btn_white_bg = QPushButton("White BG", self)
        self.btn_black_bg = QPushButton("Black BG", self)
        for button in (self.btn_load_img, self.btn_white_bg, self.btn_black_bg):
            btn_row.addWidget(button)
        left_layout.addLayout(btn_row)
        left_layout.addWidget(self.view, 1)

        self.node_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        right_layout.addWidget(self.node_list, 1)

        self.add_node_radio = QRadioButton("Add Keypoint")
        self.add_skeleton_radio = QRadioButton("Add Skeleton / Symmetry")
        self.add_node_radio.setChecked(True)
        right_layout.addWidget(self.add_node_radio)
        right_layout.addWidget(self.add_skeleton_radio)

        self.unlock_button = QPushButton("Enable Full Skeleton Edit...", self)
        self.unlock_button.setVisible(self._is_project_mode and not self._structure_edit_enabled)
        right_layout.addWidget(self.unlock_button)

        self.btn_save = QPushButton("Save Project Skeleton" if self._is_project_mode else "Save Preset", self)
        right_layout.addWidget(self.btn_save)

    def _connect_signals(self) -> None:
        self.scene.selectionChanged.connect(self._on_scene_selection_changed)
        self.node_list.itemSelectionChanged.connect(self._on_list_selection_changed)
        self.node_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.node_list.customContextMenuRequested.connect(self._on_list_context_menu)

        self.btn_load_img.clicked.connect(self._choose_image)
        self.btn_white_bg.clicked.connect(lambda: self._fill_background(Qt.GlobalColor.white))
        self.btn_black_bg.clicked.connect(lambda: self._fill_background(Qt.GlobalColor.black))
        self.add_node_radio.toggled.connect(self._on_mode_toggled)
        self.add_skeleton_radio.toggled.connect(self._on_mode_toggled)
        self.btn_save.clicked.connect(self._save_config)
        self.unlock_button.clicked.connect(self._enable_full_edit)

        if not self._is_project_mode:
            self._load_combo_items()
            self.combo.currentIndexChanged.connect(self._on_preset_changed)

    def _load_initial_model(self) -> None:
        if self._is_project_mode:
            self.model.load_from_dict(self.project.skeleton_data)
            self._rebuild_scene_from_model()
            return

        if self.combo.count() > 0:
            self._on_preset_changed(self.combo.currentIndex())
        else:
            self._prepare_new_config()

    def allow_structure_edit(self) -> bool:
        return self._structure_edit_enabled

    def _apply_structure_edit_state(self, enabled: bool) -> None:
        self._structure_edit_enabled = enabled
        self.scene.setStructureEditEnabled(enabled)
        self.add_node_radio.setEnabled(enabled)
        self.add_skeleton_radio.setEnabled(enabled)
        if enabled:
            self._structure_edit_unlocked = True
            self._on_mode_toggled()
        else:
            self.scene.setMode("view")

    def _enable_full_edit(self) -> None:
        reply = QMessageBox.warning(
            self,
            "Enable full edit?",
            "Full skeleton editing can invalidate existing TXT labels or trained models.\n\n"
            "Continue only if you understand the compatibility risk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.unlock_button.hide()
        self._apply_structure_edit_state(True)

    def _load_combo_items(self) -> None:
        self.combo.clear()
        files = sorted(path.name for path in self._preset_dir.glob("*.yaml"))
        self.combo.addItems(files)
        self.combo.addItem("Create new config")

    def _on_preset_changed(self, _index: int) -> None:
        text = self.combo.currentText()
        if text == "Create new config" or not text:
            self._prepare_new_config()
        else:
            self._load_yaml_from_preset(text)

    def _prepare_new_config(self) -> None:
        self.scene.clear()
        self.node_list.clear()
        self.node_items.clear()
        self.model = SkeletonModel()
        self.scene.model = self.model
        self.title_edit.clear()

    def _load_yaml_from_preset(self, file_name: str) -> None:
        try:
            path = self._preset_dir / file_name
            self.model.load_from_yaml(path)
            self._rebuild_scene_from_model()
            self.title_edit.setText(Path(file_name).stem)
        except Exception as err:
            QMessageBox.warning(self, "Load Error", str(err))

    def _rebuild_scene_from_model(self) -> None:
        self.scene.clear()
        self.node_list.clear()
        self.node_items.clear()

        for name, node in self.model.nodes.items():
            node_item = NodeItem(node)
            node_item.setPos(node.x, node.y)
            self.scene.addItem(node_item)
            self.node_items[name] = node_item
            self.add_node_to_list(node)

        for n1, n2 in (tuple(edge) for edge in self.model.edges):
            if n1 in self.node_items and n2 in self.node_items:
                edge_item = EdgeItem(self.node_items[n1], self.node_items[n2])
                edge_item.setZValue(3)
                self.scene.addItem(edge_item)

        for n1, n2 in (tuple(sym) for sym in self.model.syms):
            if n1 in self.node_items and n2 in self.node_items:
                sym_item = SymItem(self.node_items[n1], self.node_items[n2])
                sym_item.setZValue(3.1)
                self.scene.addItem(sym_item)

        self.scene.update()

    def add_node_to_list(self, node) -> None:
        self.node_list.addItem(QListWidgetItem(node.name))

    def _fill_background(self, color: Qt.GlobalColor) -> None:
        self.scene.setBackgroundBrush(QBrush(color))

    def _choose_image(self) -> None:
        fname, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg)")
        if not fname:
            return
        image = QImage(fname)
        if image.isNull():
            return
        self.scene.setBackgroundBrush(QBrush(QPixmap.fromImage(image)))

    def _on_list_context_menu(self, pos) -> None:
        if not self.node_list.selectedItems():
            return

        menu = QMenu(self)
        rename_act = menu.addAction("Rename node")
        visual_act = menu.addAction("Visualization option")
        delete_act = menu.addAction("Delete selected")

        selected_count = len(self.node_list.selectedItems())
        rename_act.setEnabled(selected_count == 1 and self.allow_structure_edit())
        visual_act.setEnabled(selected_count == 1)
        delete_act.setEnabled(selected_count >= 1 and self.allow_structure_edit())

        action = menu.exec(self.node_list.mapToGlobal(pos))
        if action == rename_act:
            self._rename_selected_node()
        elif action == visual_act:
            self._visualization_setting()
        elif action == delete_act:
            self._delete_selected_nodes()

    def _rename_selected_node(self) -> None:
        if not self.allow_structure_edit():
            return
        items = self.scene.selectedItems()
        if len(items) != 1:
            return
        node_item = items[0]
        old_name = node_item.node.name

        new_name, ok = QInputDialog.getText(self, "Rename node", "New name:", text=old_name)
        if not ok or not new_name.strip() or new_name == old_name:
            return
        new_name = new_name.strip()

        try:
            self.model.rename_node(old_name, new_name)
        except ValueError as err:
            QMessageBox.warning(self, "Rename error", str(err))
            return

        self.node_items[new_name] = self.node_items.pop(old_name)
        for item in self.node_list.findItems(old_name, Qt.MatchFlag.MatchExactly):
            item.setText(new_name)
        node_item.update()

    def _visualization_setting(self) -> None:
        items = self.scene.selectedItems()
        if len(items) != 1:
            return
        node_item = items[0]
        dialog = NodeVisualSettingDialog(node_item.node, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            dialog.apply_changes()
            node_item.update()
            self.scene.update()

    def _delete_selected_scene_items(self) -> None:
        if not self.allow_structure_edit():
            return
        items = self.scene.selectedItems()
        if not items:
            return

        for item in [obj for obj in items if isinstance(obj, EdgeItem)]:
            n1, n2 = item.node1.node.name, item.node2.node.name
            self.model.remove_edge(n1, n2)
            item.node1.remove_edge(item)
            item.node2.remove_edge(item)
            self.scene.removeItem(item)

        for item in [obj for obj in items if isinstance(obj, SymItem)]:
            n1, n2 = item.node1.node.name, item.node2.node.name
            self.model.remove_sym(n1, n2)
            item.node1.remove_sym(item)
            item.node2.remove_sym(item)
            self.scene.removeItem(item)

        names = [item.node.name for item in items if isinstance(item, NodeItem)]
        if names:
            self._sync_list = False
            self.node_list.clearSelection()
            for name in names:
                for list_item in self.node_list.findItems(name, Qt.MatchFlag.MatchExactly):
                    list_item.setSelected(True)
            self._sync_list = True
            self._delete_selected_nodes()

    def _delete_selected_nodes(self) -> None:
        if not self.allow_structure_edit():
            return
        selected_items = self.node_list.selectedItems()
        if not selected_items:
            return

        for name in [item.text() for item in selected_items]:
            node_item = self.node_items.get(name)
            if node_item is None:
                continue
            for edge in list(node_item.edges):
                self.scene.removeItem(edge)
                edge.node1.remove_edge(edge)
                edge.node2.remove_edge(edge)
                self.model.remove_edge(edge.node1.node.name, edge.node2.node.name)
            for sym in list(node_item.syms):
                self.scene.removeItem(sym)
                sym.node1.remove_sym(sym)
                sym.node2.remove_sym(sym)
                self.model.remove_sym(sym.node1.node.name, sym.node2.node.name)
            self.scene.removeItem(node_item)
            self.model.remove_node(name)
            del self.node_items[name]
            for match in self.node_list.findItems(name, Qt.MatchFlag.MatchExactly):
                self.node_list.takeItem(self.node_list.row(match))

    def _on_scene_selection_changed(self) -> None:
        if not self._sync_scene:
            return
        self._sync_list = False
        self.node_list.clearSelection()
        for obj in self.scene.selectedItems():
            if isinstance(obj, NodeItem):
                for item in self.node_list.findItems(obj.node.name, Qt.MatchFlag.MatchExactly):
                    item.setSelected(True)
        self._sync_list = True

    def _on_list_selection_changed(self) -> None:
        if not self._sync_list:
            return
        self._sync_scene = False
        selected_names = [item.text() for item in self.node_list.selectedItems()]
        for obj in self.scene.selectedItems():
            obj.setSelected(False)
        for name in selected_names:
            node_item = self.node_items.get(name)
            if node_item is not None:
                node_item.setSelected(True)
        self._sync_scene = True

    def _on_mode_toggled(self) -> None:
        if not self.allow_structure_edit():
            self.scene.setMode("view")
        elif self.add_node_radio.isChecked():
            self.scene.setMode("add_node")
        else:
            self.scene.setMode("add_edge")

    def _save_config(self) -> None:
        if self._is_project_mode:
            try:
                if self.save_callback is not None:
                    self.save_callback(self.model, self._structure_edit_unlocked)
                QMessageBox.information(self, "Saved", "Project skeleton updated.")
                self.accept()
            except Exception as err:
                QMessageBox.critical(self, "Save Failed", str(err))
            return

        title = self.title_edit.text().strip()
        if not title:
            QMessageBox.warning(self, "Save Failed", "Please enter a preset title first.")
            return
        if not title.lower().endswith(".yaml"):
            title += ".yaml"
        path = self._preset_dir / title

        try:
            self.model.save_to_yaml(path)
            QMessageBox.information(self, "Save Complete", f"The preset has been saved to\n{path}\n.")
            if hasattr(self.main_window, "load_combo_items"):
                self.main_window.load_combo_items(title)
            self.accept()
        except ValueError as err:
            QMessageBox.warning(
                self,
                "Save Failed",
                f"An error occurred while saving:\n{err}\n"
                "Please use a different preset name or unique node names.",
            )
        except Exception as err:
            QMessageBox.critical(self, "Save Failed", str(err))
