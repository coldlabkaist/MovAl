from __future__ import annotations
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QDialog, QGraphicsView, QListWidget, QPushButton, QRadioButton, 
    QHBoxLayout, QVBoxLayout, QListWidgetItem, QMenu, QFileDialog, QMessageBox, 
    QAbstractItemView, QLabel, QComboBox, QLineEdit, QSplitter, QFrame, QFileDialog, QInputDialog
)
from PyQt6.QtGui     import QBrush, QPixmap, QImage
from utils.skeleton import SkeletonModel, SkeletonScene, NodeItem, EdgeItem, NodeVisualSettingDialog
import os, yaml

class SkeletonManagerDialog(QDialog):
    def __init__(self, main_window) -> None:
        super().__init__()
        self.setWindowTitle("Skeleton Manager")
        self.resize(800, 500)
        self.setFixedSize(self.size())

        self.model = SkeletonModel()
        self.scene = SkeletonScene(self.model, self)
        self.scene.setSceneRect(-200, -200, 600, 350)
        self.view = QGraphicsView(self.scene)
        self.node_list = QListWidget()
        self.node_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

        main_layout = QVBoxLayout(self)
        top_bar      = QHBoxLayout()

        lbl = QLabel("Preset:", self)
        self.combo = QComboBox(self)
        self.combo.setEditable(True)
        self.combo.setPlaceholderText("Select config file")
        self.combo.setEditable(False)

        self.title_edit = QLineEdit(self)
        self.title_edit.setPlaceholderText("Config title")

        top_bar.addWidget(lbl)
        top_bar.addWidget(self.combo, 1)
        top_bar.addWidget(QLabel("Title:", self))
        top_bar.addWidget(self.title_edit, 2)
        main_layout.addLayout(top_bar)
        
        self.main_window = main_window
        self._preset_dir = main_window._preset_dir
        self._load_combo_items()
        self.combo.currentIndexChanged.connect(self._on_preset_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_layout.addWidget(splitter, 1) 

        left_widget  = QWidget(self)
        left_layout  = QVBoxLayout(left_widget)
        splitter.addWidget(left_widget)
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        self.btn_load_img   = QPushButton("Load image", self)
        self.btn_white_bg   = QPushButton("White BG",   self)
        self.btn_black_bg   = QPushButton("Black BG",   self)
        btn_row = QHBoxLayout()
        for b in (self.btn_load_img, self.btn_white_bg, self.btn_black_bg):
            btn_row.addWidget(b)
        left_layout.addLayout(btn_row)

        left_layout.addWidget(self.view, 1)

        self.btn_load_img.clicked.connect(self._choose_image)
        self.btn_white_bg.clicked.connect(lambda: self._fill_background(Qt.GlobalColor.white))
        self.btn_black_bg.clicked.connect(lambda: self._fill_background(Qt.GlobalColor.black))

        self.node_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.node_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        right_layout.addWidget(self.node_list, 1)

        self.node_list.itemSelectionChanged.connect(self._on_list_selection_changed)
        self.node_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.node_list.customContextMenuRequested.connect(self._on_list_context_menu)

        self.add_node_radio = QRadioButton("Add keypoint")
        self.add_skeleton_radio = QRadioButton("Add skeleton")
        self.add_node_radio.setChecked(True)
        right_layout.addWidget(self.add_node_radio)
        right_layout.addWidget(self.add_skeleton_radio)
        self.add_node_radio.toggled.connect(self._on_mode_toggled)
        self.add_skeleton_radio.toggled.connect(self._on_mode_toggled)

        self.btn_save = QPushButton("Save config", self)
        right_layout.addWidget(self.btn_save)
        self.btn_save.clicked.connect(self._save_config)

        self.scene.selectionChanged.connect(self._on_scene_selection_changed)
        self.node_items = {}

        self._sync_list = True
        self._sync_scene = True

        QTimer.singleShot(0, lambda: self._fill_background(Qt.GlobalColor.white))

    def _load_combo_items(self):
        self.combo.clear()
        os.makedirs(self._preset_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(self._preset_dir) if f.endswith(".yaml"))
        self.combo.addItems(files)
        self.combo.addItem("Create new config")

    def _on_preset_changed(self, _idx):
        text = self.combo.currentText()
        if text == "Create new config":
            self._prepare_new_config()
        else:
            self._load_yaml_from_preset(text)

    def _prepare_new_config(self):
        self.scene.clear()
        self.node_list.clear()
        self.node_items.clear()
        self.model = SkeletonModel()
        self.scene.model = self.model
        self.title_edit.clear()

    def _load_yaml_from_preset(self, text: str):
        try:
            path = os.path.join(self._preset_dir, text)
            self.model.load_from_yaml(path)
            self._rebuild_scene_from_model()
            self.title_edit.setText(os.path.splitext(text)[0])
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _rebuild_scene_from_model(self):
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
                self.scene.addItem(edge_item)

    def add_node_to_list(self, node):
        item = QListWidgetItem(node.name)
        self.node_list.addItem(item)

    def _fill_background(self, color: Qt.GlobalColor):
        self.scene.setBackgroundBrush(QBrush(color))

    def _choose_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg)")
        if fname:
            img = QImage(fname)
            if img.isNull():
                return
            pix = QPixmap.fromImage(img)
            self.scene.setBackgroundBrush(QBrush(pix))

    def _on_list_context_menu(self, pos):
        if not self.node_list.selectedItems():
            event.ignore()
            return

        menu = QMenu(self)
        rename_act = menu.addAction("Rename node")
        visual_act = menu.addAction("visuialization option")
        delete_act = menu.addAction("Delete selected")

        action = menu.exec(self.node_list.mapToGlobal(pos))

        if action == rename_act:
            if len(self.node_list.selectedItems()) == 1:
                self._rename_selected_node()
        elif action == visual_act:
            if len(self.selectedItems()) == 1:
                self.main_window._visualization_setting()
        elif action == delete_act:
            self._delete_selected_nodes()

    def _rename_selected_node(self):
        items = self.scene.selectedItems()
        if len(items) != 1:
            return
        node_item = items[0]
        old_name  = node_item.node.name

        new_name, ok = QInputDialog.getText(self, "Rename node", "New name:", text=old_name)
        if not ok or not new_name.strip() or new_name == old_name:
            return
        new_name = new_name.strip()

        try:
            self.model.rename_node(old_name, new_name)
        except ValueError as e:
            QMessageBox.warning(self, "Rename error", str(e))
            return

        self.node_items[new_name] = self.node_items.pop(old_name)
        matches = self.node_list.findItems(old_name, Qt.MatchFlag.MatchExactly)
        for itm in matches:
            itm.setText(new_name)
        node_item.update()

    def _visualization_setting(self):
        items = self.scene.selectedItems()
        if len(items) != 1:
            return
        node_item = items[0]
        
        dialog = NodeVisualSettingDialog(node_item.node) 
        if dialog.exec() == QDialog.DialogCode.Accepted:
            dialog.apply_changes()
            node_item.update()
            self.scene.update()

    def _delete_selected_scene_items(self):
        items = self.scene.selectedItems()
        if not items:
            return

        for it in [i for i in items if isinstance(i, EdgeItem)]:
            n1, n2 = it.node1.node.name, it.node2.node.name
            self.model.remove_edge(n1, n2)
            it.node1.remove_edge(it); it.node2.remove_edge(it)
            self.scene.removeItem(it)

        names = [i.node.name for i in items if isinstance(i, NodeItem)]
        if names:
            self._sync_list = False
            self.node_list.clearSelection()
            for nm in names:
                for li in self.node_list.findItems(nm, Qt.MatchFlag.MatchExactly):
                    li.setSelected(True)
            self._sync_list = True
            self._delete_selected_nodes()

    def _delete_selected_nodes(self):
        selected_items = self.node_list.selectedItems()
        if not selected_items:
            return
        names_to_remove = [item.text() for item in selected_items]
        for name in names_to_remove:
            if name not in self.node_items:
                continue
            node_item = self.node_items[name]
            for edge in list(node_item.edges):
                self.scene.removeItem(edge)
                edge.node1.remove_edge(edge); edge.node2.remove_edge(edge)
                self.model.remove_edge(edge.node1.node.name, edge.node2.node.name)
            self.scene.removeItem(node_item)
            self.model.remove_node(name)
            del self.node_items[name]
            
            matches = self.node_list.findItems(name, Qt.MatchFlag.MatchExactly)
            for item in matches:
                row = self.node_list.row(item)
                self.node_list.takeItem(row)

    def _on_scene_selection_changed(self):
        if not self._sync_scene:
            return
        self._sync_list = False
        self.node_list.clearSelection()
        for obj in self.scene.selectedItems():
            if isinstance(obj, NodeItem):
                name = obj.node.name
                matches = self.node_list.findItems(name, Qt.MatchFlag.MatchExactly)
                for item in matches:
                    item.setSelected(True)
        self._sync_list = True

    def _on_list_selection_changed(self):
        if not self._sync_list:
            return 
        self._sync_scene = False
        selected_names = [item.text() for item in self.node_list.selectedItems()]
        for obj in self.scene.selectedItems():
            obj.setSelected(False)
        for name in selected_names:
            if name in self.node_items:
                node_item = self.node_items[name]
                node_item.setSelected(True)
        self._sync_scene = True

    def _on_mode_toggled(self):
        if self.add_node_radio.isChecked():
            self.scene.setMode('add_node')
        else:
            self.scene.setMode('add_edge')

    def _save_config(self):
        title = self.title_edit.text().strip()

        if not title:
            QMessageBox.warning(self, "Save Failed", "Please enter a preset title first.")
            return

        if not title.lower().endswith(".yaml"):
            title += ".yaml"
        path = os.path.join(self._preset_dir, title)

        try:
            self.model.save_to_yaml(path)
            QMessageBox.information(self, "Save Complete", f"The preset has been saved to\n{path}\n.")
            
            self.main_window.load_combo_items(title)
            self.accept()
            
        except ValueError as ve:
            QMessageBox.warning(
                self,
                f"An error occurred while saving:\n{ve}\n"
                "Please enter a different preset name or modify the node names "
                "so that they are not duplicated."
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))