from PyQt6.QtCore import QRectF, QPointF, Qt, QLineF
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsLineItem
from PyQt6.QtGui     import QPen, QBrush, QFont, QFontMetrics, QColor

class NodeItem(QGraphicsItem):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.edges = [] 
        self.syms = [] 
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | 
                      QGraphicsItem.GraphicsItemFlag.ItemIsMovable | 
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(1)
        self.selectAvailable = True
        # Base radius aligned with Labelary's default keypoint radius feel.
        self.r = 5

    def _view_scale(self) -> float:
        scene = self.scene()
        if scene is None:
            return 1.0
        views = scene.views()
        if not views:
            return 1.0
        transform = views[0].transform()
        sx = abs(transform.m11())
        sy = abs(transform.m22())
        scale = (sx + sy) * 0.5
        return max(scale, 1e-6)

    def _effective_radius(self) -> float:
        # Match Labelary's zoom feel: node size grows sub-linearly with zoom
        # so it becomes relatively smaller against the image when zoomed in.
        return self._base_radius() / (self._view_scale() ** 0.5)

    def _base_radius(self) -> float:
        if self.node.filled and self.node.shape in ("circle", "square"):
            # In filled mode, thickness value is interpreted as node size.
            # Keep size=1 visually close to legacy default radius(5).
            return self.r + max(0, self.node.thickness - 1) * 2.0
        return self.r

    def add_edge(self, edge_item):
        self.edges.append(edge_item)

    def remove_edge(self, edge_item):
        if edge_item in self.edges:
            self.edges.remove(edge_item)
            
    def add_sym(self, edge_item):
        self.syms.append(edge_item)

    def remove_sym(self, edge_item):
        if edge_item in self.syms:
            self.syms.remove(edge_item)

    def boundingRect(self) -> QRectF:
        if self.node.shape == 'text':
            font = QFont()
            fm = QFontMetrics(font)
            text = self.node.text if self.node.text is not None else self.node.name
            w = fm.horizontalAdvance(text)
            h = fm.height()
            extra = max(self.node.thickness / 2 + 5, 6) 
            return QRectF(-w/2 - extra, -h/2 - extra, w + 2*extra, h + 2*extra)
        else:
            # Keep rect generous so zoom-dependent rendering does not clip.
            r = self._base_radius() * 3.0
            th = self.node.thickness
            extra = th / 2.0 + 3
            return QRectF(-r - extra, -r - extra, 2*r + 2*extra, 2*r + 2*extra)

    def paint(self, painter, option, widget=None):
        color = self.node.color
        thickness = self.node.thickness
        r = self._effective_radius()

        if self.selectAvailable and self.isSelected():
            hl_pen = QPen(QColor("red"), max(2, thickness + 1))
            hl_pen.setCosmetic(True)
            painter.setPen(hl_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            if self.node.shape == "circle":
                painter.drawEllipse(-r - 3, -r - 3, 2 * (r + 3), 2 * (r + 3))
            elif self.node.shape == "square":
                painter.drawRect(-r - 3, -r - 3, 2 * (r + 3), 2 * (r + 3))
            else:
                painter.drawRect(-r - 3, -r - 3, 2 * (r + 3), 2 * (r + 3))

        if self.node.shape == 'text':
            text_pen = QPen(color, 1)
            text_pen.setCosmetic(True)
            painter.setPen(text_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            text = self.node.text if self.node.text is not None else self.node.name
            painter.drawText(self.boundingRect(), Qt.AlignmentFlag.AlignCenter, text)
        else:
            if self.node.filled:
                # Filled nodes should remain solid (no inner hole).
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
            else:
                pen = QPen(color, thickness)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            if self.node.shape == 'circle':
                painter.drawEllipse(-r, -r, 2*r, 2*r)
            elif self.node.shape == 'square':
                painter.drawRect(-r, -r, 2*r, 2*r)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            newPos = value
            
            self.node.x = newPos.x()
            self.node.y = newPos.y()
            
            for edge in list(self.edges):
                edge.update_line()
            for sym in list(self.syms):
                sym.update_line()
        return super().itemChange(change, value)

class EdgeItem(QGraphicsLineItem):
    def __init__(self, node1: NodeItem, node2: NodeItem):
        super().__init__()
        self.node1 = node1
        self.node2 = node2
        self.update_line()
        
        node1.add_edge(self)
        node2.add_edge(self)

        self.normal_pen = QPen(QColor("black"), 2)
        self.hl_pen     = QPen(QColor("red"),   3)
        self.normal_pen.setCosmetic(True)
        self.hl_pen.setCosmetic(True)
        self.setPen(self.normal_pen)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(-1)

    def update_line(self):
        p1 = self.node1.scenePos()
        p2 = self.node2.scenePos()
        self.setLine(QLineF(p1, p2))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.setPen(self.hl_pen if self.isSelected() else self.normal_pen)
        return super().itemChange(change, value)

class SymItem(QGraphicsLineItem):
    def __init__(self, node1: NodeItem, node2: NodeItem):
        super().__init__()
        self.node1 = node1
        self.node2 = node2
        self.update_line()
        
        node1.add_sym(self)
        node2.add_sym(self)

        self.normal_pen = QPen(QColor("cyan"), 1, Qt.PenStyle.DashLine)
        self.hl_pen     = QPen(QColor("blue"), 2, Qt.PenStyle.DashLine)
        self.normal_pen.setCosmetic(True)
        self.hl_pen.setCosmetic(True)
        self.setPen(self.normal_pen)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(-1)

    def update_line(self):
        p1 = self.node1.scenePos()
        p2 = self.node2.scenePos()
        self.setLine(QLineF(p1, p2))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.setPen(self.hl_pen if self.isSelected() else self.normal_pen)
        return super().itemChange(change, value)
