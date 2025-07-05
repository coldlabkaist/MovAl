"""skeleton.py — compact skeleton editor (v14)
========================================================
2025‑05‑01
* fix: two‑click edge creation breaks when 트랙이 끼어드는 경우
* feat: Edge 리스트를 **track / kp 순으로 실시간 정렬**

Public classes
--------------
• SkeletonManager — in‑memory graph + JSON I/O
• SkeletonDialog   — Qt dialog for editing / previewing skeletons
"""

import json
from pathlib import Path
from typing import Dict, Set, Tuple, Union, Optional, List

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QDialog,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QMenu,
)

from .data_loader import DataLoader

# ---------------------------------------------------------------------------
Edge = Tuple[str, str]
Graph = Dict[str, Set[Edge]]

def _norm(e: Edge) -> Edge:
    """Make edge direction‑independent (undirected)."""
    a, b = e
    return (a, b) if a <= b else (b, a)

# =============================================================================
# SkeletonManager
# =============================================================================

class SkeletonManager:
    """Static helper for accessing DataLoader.skeleton_data as an undirected graph."""

    _dl = DataLoader  # alias for brevity

    @classmethod
    def graph(cls) -> Graph:
        return {t: set(edges) for t, edges in cls._dl.skeleton_data.items()}

    @classmethod
    def set_graph(cls, g: Graph):
        cleaned = {t: {_norm(e) for e in edges} for t, edges in g.items()}
        cls._dl.skeleton_data = cleaned

    # --------------- convenience wrappers
    @classmethod
    def reset(cls):
        cls._dl.skeleton_data = {}

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            g: Graph = {t: {_norm(tuple(p)) for p in plist} for t, plist in raw.items()}
            cls.set_graph(g)
            return True
        except Exception:
            return False

    @classmethod
    def save_json(cls, path: Union[str, Path]) -> bool:
        try:
            ser = {t: [list(e) for e in sorted(edges)] for t, edges in cls.graph().items()}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(ser, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

# =============================================================================
# SkeletonDialog
# =============================================================================

class SkeletonDialog(QDialog):
    """Modal dialog for creating / editing skeleton graphs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Skeleton Editor")
        self.resize(760, 520)

        lay = QHBoxLayout(self)

        # -------- 좌측: 트리
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Track", "Key‑point"])
        self.tree.itemDoubleClicked.connect(self._on_tree_double)
        lay.addWidget(self.tree, 1)

        # -------- 우측: Edge 리스트 + 버튼 바
        vr = QVBoxLayout()
        self.list_edges = QListWidget()
        self.list_edges.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_edges.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_edges.customContextMenuRequested.connect(self._edge_menu)

        vr.addWidget(QLabel("Edges:"))
        vr.addWidget(self.list_edges, 1)

        btn_bar = QHBoxLayout()
        self.btn_imp = QPushButton("Import JSON …")
        self.btn_exp = QPushButton("Export JSON …")
        self.btn_reset = QPushButton("Reset")
        btn_close = QPushButton("Close")

        btn_bar.addWidget(self.btn_imp)
        btn_bar.addWidget(self.btn_exp)
        btn_bar.addWidget(self.btn_reset)
        btn_bar.addStretch()
        btn_bar.addWidget(btn_close)
        vr.addLayout(btn_bar)
        lay.addLayout(vr, 1)

        # signals
        self.btn_imp.clicked.connect(self._import)
        self.btn_exp.clicked.connect(self._export)
        self.btn_reset.clicked.connect(self._reset)
        btn_close.clicked.connect(self.accept)

        # state
        self._pending: Optional[Tuple[str, str]] = None  # first click track/kp
        self._rebuild()

    # ------------------------------------------------ rebuild UI
    def _rebuild(self):
        self.tree.clear(); self.list_edges.clear()
        mapping = self._infer_tracks() or {"DefaultTrack": []}
        for track, kps in sorted(mapping.items()):
            item = QTreeWidgetItem([track])
            self.tree.addTopLevelItem(item)
            item.setExpanded(True)
            for kp in kps:
                ch = QTreeWidgetItem(["", kp])
                ch.setData(0, Qt.ItemDataRole.UserRole, (track, kp))
                item.addChild(ch)
        for t, edges in SkeletonManager.graph().items():
            for e in sorted(edges):
                self._add_edge_item(t, e, resort=False)
        self._sort_edge_list()

    def _infer_tracks(self) -> Dict[str, List[str]]:
        mp: Dict[str, List[str]] = {}
        dl = DataLoader

        # 1) CSV가 있으면 track 열 고유값 그대로 사용
        if getattr(dl, "data", None) is not None and not dl.data.empty:
            try:
                kps = [c.rsplit(".", 1)[0] for c in dl.data.columns if c.endswith(".x")]
                for t in dl.data["track"].unique():          # ← track_0 대신 원본
                    mp[str(t)] = kps
            except Exception:
                pass

        # 2) Skeleton에 이미 값이 있으면 그대로
        if not mp and dl.skeleton_data:
            for t, es in dl.skeleton_data.items():
                pts = {p for e in es for p in e}
                mp[t] = sorted(pts)
        return mp

    # ------------------------------------------------ tree interaction
    def _on_tree_double(self, item: QTreeWidgetItem, _col: int):
        """Two‑click logic that survives track switching."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        track, kp = data

        # 첫 클릭: pending 설정
        if self._pending is None:
            self._pending = (track, kp)
            return

        prev_track, prev_kp = self._pending

        if prev_track == track and prev_kp != kp:  # 같은 트랙 & 다른 KP
            self._add_edge(track, (prev_kp, kp))
            self._pending = None  # 성공적으로 간선 추가 → 초기화
        else:
            # 다른 트랙을 클릭했거나 같은 KP 재클릭 → 새 첫 클릭으로 교체
            self._pending = (track, kp)

    # ------------------------------------------------ edge helpers
    def _add_edge(self, track: str, e: Edge):
        g = SkeletonManager.graph()
        e = _norm(e)
        if e not in g.get(track, set()):
            g.setdefault(track, set()).add(e)
            SkeletonManager.set_graph(g)
            self._add_edge_item(track, e)
            self._show_preview()

    def _add_edge_item(self, track: str, e: Edge, *, resort: bool = True):
        it = QListWidgetItem(f"{track}: {e[0]} ↔ {e[1]}")
        it.setData(Qt.ItemDataRole.UserRole, (track, e))
        self.list_edges.addItem(it)
        if resort:
            self._sort_edge_list()

    def _sort_edge_list(self):
        items = [self.list_edges.takeItem(0) for _ in range(self.list_edges.count())]
        items.sort(key=lambda it: it.text())  # track & kp lexicographic
        for it in items:
            self.list_edges.addItem(it)

    def _delete_edges(self, items: List[QListWidgetItem]):
        g = SkeletonManager.graph()
        for it in items:
            track, e = it.data(Qt.ItemDataRole.UserRole)
            e = _norm(e)
            if e in g.get(track, set()):
                g[track].remove(e)
        SkeletonManager.set_graph(g)
        self._show_preview()
        self._rebuild()

    def _copy_edges(self, src_track: str, edges: Set[Edge], targets: List[str]):
        g = SkeletonManager.graph()
        for tgt in targets:
            if tgt == src_track:
                continue
            g.setdefault(tgt, set()).update(edges)
        SkeletonManager.set_graph(g)
        self._show_preview()
        self._rebuild()

    # ------------------------------------------------ context menu
    def _edge_menu(self, pos: QPoint):
        sel = self.list_edges.selectedItems()
        if not sel:
            return
        menu = QMenu(self)
        act_del = menu.addAction("Delete selected")
        # copy submenu logic
        tracks = {it.data(Qt.ItemDataRole.UserRole)[0] for it in sel}
        copy_sub = None
        if len(tracks) == 1:
            src_track = next(iter(tracks))
            copy_sub = menu.addMenu("Copy to …")
            act_all = copy_sub.addAction("All other tracks")
            copy_sub.addSeparator()
            for t in sorted(self._infer_tracks().keys()):
                if t != src_track:
                    copy_sub.addAction(t)
        trg = menu.exec(self.list_edges.mapToGlobal(pos))
        if trg is None:
            return
        if trg == act_del:
            self._delete_edges(sel)
        elif copy_sub and trg:
            src_track = next(iter(tracks))
            edges = {it.data(Qt.ItemDataRole.UserRole)[1] for it in sel}
            if trg.text() == "All other tracks":
                tar = [t for t in self._infer_tracks().keys() if t != src_track]
            else:
                tar = [trg.text()]
            self._copy_edges(src_track, edges, tar)

    # ------------------------------------------------ file buttons
    def _import(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load skeleton", "", "JSON (*.json)")
        if fn and SkeletonManager.load_json(fn):
            self._rebuild()
            self._show_preview()
        elif fn:
            QMessageBox.critical(self, "Error", "Failed to load JSON")

    def _export(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save skeleton", "", "JSON (*.json)")
        if not fn:
            return
        if not fn.lower().endswith(".json"):
            fn += ".json"
        if SkeletonManager.save_json(fn):
            QMessageBox.information(self, "Saved", "Skeleton saved successfully.")
        else:
            QMessageBox.critical(self, "Error", "Failed to save JSON")

    def _reset(self):
        if QMessageBox.question(
            self,
            "Reset",
            "Clear all skeleton data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:
            SkeletonManager.reset()
            self._rebuild()
            self._show_preview()

    # ------------------------------------------------ preview button
    def _show_preview(self):
        """부모 위젯(videolabel) 을 강제 update 하여 미리보기를 즉시 반영"""
        parent = self.parent()
        if parent is None:
            return
        if hasattr(parent, "video_label"):
            parent.video_label.update()
        else:
            parent.update()
