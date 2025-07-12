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
        self.r = 10

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
            r = self.r
            th = self.node.thickness
            extra = th / 2.0 + 3
            return QRectF(-r - extra, -r - extra, 2*r + 2*extra, 2*r + 2*extra)

    def paint(self, painter, option, widget=None):
        color = self.node.color
        thickness = self.node.thickness

        if self.selectAvailable and self.isSelected():
            hl_pen = QPen(QColor("red"), max(2, thickness + 1))
            painter.setPen(hl_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            r = self.r
            if self.node.shape == "circle":
                painter.drawEllipse(-r - 3, -r - 3, 2 * (r + 3), 2 * (r + 3))
            elif self.node.shape == "square":
                painter.drawRect(-r - 3, -r - 3, 2 * (r + 3), 2 * (r + 3))
            else:
                painter.drawRect(-r - 3, -r - 3, 2 * (r + 3), 2 * (r + 3))

        if self.node.shape == 'text':
            painter.setPen(QPen(color, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            text = self.node.text if self.node.text is not None else self.node.name
            painter.drawText(self.boundingRect(), Qt.AlignmentFlag.AlignCenter, text)
        else:
            pen = QPen(color, thickness)
            painter.setPen(pen)
            if self.node.filled:
                painter.setBrush(QBrush(color))
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            r = 10  
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

    def mouseDoubleClickEvent(self, event):
        if hasattr(self, 'openPropertiesCallback'):
            self.openPropertiesCallback(self.node)
        else:
            super().mouseDoubleClickEvent(event)

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