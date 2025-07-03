from PyQt6.QtCore import QRectF, QPointF, Qt, QLineF
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsLineItem
from PyQt6.QtGui     import QPen, QBrush, QFont, QFontMetrics, QColor

class NodeItem(QGraphicsItem):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.edges = [] 
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | 
                      QGraphicsItem.GraphicsItemFlag.ItemIsMovable | 
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(1)
        # (openPropertiesCallback은 이후에 설정하여 더블클릭으로 속성창을 열 수 있게 함)

    def add_edge(self, edge_item):
        self.edges.append(edge_item)

    def remove_edge(self, edge_item):
        if edge_item in self.edges:
            self.edges.remove(edge_item)

    def boundingRect(self) -> QRectF:
        if self.node.shape == 'text':
            font = QFont()
            fm = QFontMetrics(font)
            text = self.node.text if self.node.text is not None else self.node.name
            w = fm.horizontalAdvance(text)
            h = fm.height()
            return QRectF(-w/2 - 2, -h/2 - 2, w + 4, h + 4)
        else:
            r = 10
            th = self.node.thickness
            extra = th / 2.0
            return QRectF(-r - extra, -r - extra, 2*r + 2*extra, 2*r + 2*extra)

    def paint(self, painter, option, widget=None):
        color = self.node.color
        thickness = self.node.thickness
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
        """아이템 상태 변경 시 호출 (이동 등)"""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            newPos = value
            
            self.node.x = newPos.x()
            self.node.y = newPos.y()
            
            for edge in list(self.edges):
                edge.update_line()
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

        pen = QPen(QColor("black"), 2)
        self.setPen(pen)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(-1)

    def update_line(self):
        p1 = self.node1.scenePos()
        p2 = self.node2.scenePos()
        self.setLine(QLineF(p1, p2))
