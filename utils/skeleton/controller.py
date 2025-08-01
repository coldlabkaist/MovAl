from PyQt6.QtWidgets import (
    QGraphicsScene, QGraphicsLineItem, QGraphicsItem, QMenu
)
from PyQt6.QtCore import Qt, QLineF
from PyQt6.QtGui  import QPen, QColor, QTransform
from .item import NodeItem, EdgeItem, SymItem

class SkeletonScene(QGraphicsScene):
    def __init__(self, model, main_window):
        super().__init__()
        self.model = model
        self.main_window = main_window
        self.mode = 'add_node'
        self.temp_edge_start = None  
        self.temp_line = None 
        self.temp_line_is = None 

    def setMode(self, mode):
        self.mode = mode
        for item in self.items():
            if isinstance(item, NodeItem):
                movable = (mode == 'add_node')
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, movable)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == 'add_node':
                item = self.itemAt(event.scenePos(), QTransform())
                if item is None:
                    node = self.model.add_node()
                    pos    = event.scenePos()
                    
                    node_item = NodeItem(node)
                    node_item.setPos(pos)
                    self.addItem(node_item)
                    
                    self.main_window.node_items[node.name] = node_item
                    self.main_window.add_node_to_list(node)
                    self.clearSelection()
                    node_item.setSelected(True)
                    event.accept()
                    return
                if isinstance(item, NodeItem):
                    if item.isSelected():
                        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                            item.setSelected(False)
                        else:
                            pass
                    else:
                        if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                            self.clearSelection()
                        item.setSelected(True)
            elif self.mode == 'add_edge':
                item = self.itemAt(event.scenePos(), QTransform())
                if isinstance(item, NodeItem):
                    self.temp_edge_start = item
                    self.temp_line = QGraphicsLineItem(QLineF(item.scenePos(), item.scenePos()))
                    pen = QPen(QColor("grey"), 2, Qt.PenStyle.SolidLine)
                    self.temp_line.setPen(pen)
                    self.temp_line.setZValue(-1)
                    self.addItem(self.temp_line)
                    self.temp_line_is = "Edge"
                    event.accept()
                    return
                if isinstance(item, EdgeItem):
                    if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                        self.clearSelection()
                    item.setSelected(True)
                    event.accept()
                    return
        if event.button() == Qt.MouseButton.RightButton:
            item = self.itemAt(event.scenePos(), QTransform())
            if self.mode == 'add_node':
                if isinstance(item, NodeItem):
                    if item.isSelected():
                        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                            item.setSelected(False)
                        else:
                            pass
                    else:
                        if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                            self.clearSelection()
                        item.setSelected(True)
                        event.accept()
                        return
            if self.mode == 'add_edge':
                if isinstance(item, NodeItem):
                    self.temp_edge_start = item
                    self.temp_line = QGraphicsLineItem(QLineF(item.scenePos(), item.scenePos()))
                    self.temp_line.setZValue(2)
                    pen = QPen(QColor("cyan"), 1, Qt.PenStyle.DashLine)
                    self.temp_line.setPen(pen)
                    self.temp_line.setZValue(-1)
                    self.addItem(self.temp_line)
                    self.temp_line_is = "Sym"
                    event.accept()
                    return
                if isinstance(item, EdgeItem):
                    if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                        self.clearSelection()
                    item.setSelected(True)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.mode == 'add_edge' and self.temp_edge_start and self.temp_line:
            p1 = self.temp_edge_start.scenePos()
            p2 = event.scenePos()
            self.temp_line.setLine(QLineF(p1, p2))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.mode == 'add_edge' and self.temp_edge_start:
            target_node_item = None
            item = self.itemAt(event.scenePos(), QTransform())
            if isinstance(item, NodeItem) and item is not self.temp_edge_start:
                target_node_item = item
            else:
                release_pos = event.scenePos()
                min_dist2 = float('inf')
                nearest_item = None
                for obj in self.items():
                    if isinstance(obj, NodeItem) and obj is not self.temp_edge_start:
                        dx = obj.scenePos().x() - release_pos.x()
                        dy = obj.scenePos().y() - release_pos.y()
                        dist2 = dx*dx + dy*dy
                        if dist2 < 400 and dist2 < min_dist2:
                            min_dist2 = dist2
                            nearest_item = obj
                target_node_item = nearest_item
            if target_node_item:
                if self.temp_line_is == "Edge":
                    name1 = self.temp_edge_start.node.name
                    name2 = target_node_item.node.name
                    if self.model.add_edge(name1, name2):
                        edge_item = EdgeItem(self.temp_edge_start, target_node_item)
                        self.addItem(edge_item)
                elif self.temp_line_is == "Sym":
                    name1 = self.temp_edge_start.node.name
                    name2 = target_node_item.node.name
                    if self.model.add_sym(name1, name2):
                        sym_item = SymItem(self.temp_edge_start, target_node_item)
                        sym_item.setZValue(1)
                        self.addItem(sym_item)

            if self.temp_line:
                self.removeItem(self.temp_line)
                self.temp_line = None
                self.temp_line_is = None
            self.temp_edge_start = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.main_window._delete_selected_scene_items()
            event.accept()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        if not self.selectedItems():
            return

        menu = QMenu()
        rename_act = menu.addAction("Rename node")
        visual_act = menu.addAction("visuialization option")
        delete_act = menu.addAction("Delete selected")

        sel_cnt = len(self.selectedItems())
        rename_act.setEnabled(sel_cnt == 1)
        visual_act.setEnabled(sel_cnt == 1)
        delete_act.setEnabled(sel_cnt >= 1)

        act = menu.exec(event.screenPos())

        if act == rename_act:
            self.main_window._rename_selected_node()
        elif act == visual_act:
            self.main_window._visualization_setting()
        elif act == delete_act:
            self.main_window._delete_selected_scene_items()