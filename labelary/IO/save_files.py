from __future__ import annotations

import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import pandas as pd
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QMessageBox,
    QInputDialog, QWidget, QFrame
)
from PyQt6.QtCore import Qt
import yaml
from dataclasses import asdict, is_dataclass

from .data_loader import DataLoader

class _SaveActionDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save / Export")
        self.setFixedSize(240, 200)

        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        self._choice: str | None = None

        btn_csv = QPushButton("save CSV", self)
        btn_txt = QPushButton("export TXT", self)
        btn_vid = QPushButton("export Video", self)

        btn_csv.clicked.connect(lambda: self._set_choice("csv"))
        btn_txt.clicked.connect(lambda: self._set_choice("txt"))
        btn_vid.clicked.connect(lambda: self._set_choice("video"))

        lay.addWidget(btn_csv)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        lay.addWidget(line)

        lay.addWidget(btn_txt)
        lay.addWidget(btn_vid)

    def _set_choice(self, c: str) -> None:
        self._choice = c
        self.accept()

    def choice(self) -> str | None:
        return self._choice

def save_modified_data(parent: QWidget):
    if DataLoader.loaded_data is None:
        QMessageBox.warning(parent, "Warning", "Load CSV/TXT first")
        return

    dlg = _SaveActionDialog(parent)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    action = dlg.choice()
    if action is None:
        return

    df_orig = _sanitize_index(DataLoader.loaded_data.copy())

    project = _find_project(parent)
    if project is None or not hasattr(project, "project_dir"):
        QMessageBox.critical(parent, "Error", "Project information not found.")
        return
    video_path, video_name = _current_video(parent)
    config_path = Path(project.project_dir) / "config.yaml"
    project_info = parent.project

    if action == "csv":
        base_dir = Path(project.project_dir) / "labels" / video_name / "csv"
        base_dir.mkdir(parents=True, exist_ok=True)

        fname, ok = QInputDialog.getText(
            parent, "Enter CSV file name", "File name (without extension):"
        )
        if not ok or not fname.strip():
            return

        csv_path = base_dir / f"{fname.strip()}.csv"
        if csv_path.exists():
            res = QMessageBox.question(
                parent, "The file already exists",
                f"Overwrite {csv_path.name}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if res != QMessageBox.StandardButton.Yes:
                return

        df_to_save = df_orig.copy()
        for sc in [c for c in df_to_save.columns if c.endswith(".score")]:
            vis = sc.replace(".score", ".visibility")
            if vis not in df_to_save.columns:
                df_to_save[vis] = 2
            df_to_save.drop(columns=[sc], inplace=True)

        try:
            df_to_save.to_csv(csv_path, index=False)
            QMessageBox.information(parent, "Success", f"✅ CSV Saved!:\n{csv_path}")
            modify_yaml(video_path, "csv", csv_path, config_path, project_info)
        except Exception as e:
            QMessageBox.critical(parent, "Error", f"Failed to save CSV:\n{e}")
        parent.update_label_combo(set_text = csv_path)
        return

    if action == "txt":
        txt_dir = Path(project.project_dir) / "labels" / video_name / "txt"
        txt_dir.mkdir(parents=True, exist_ok=True)

        has_existing = any(txt_dir.glob("*.txt"))
        if has_existing:
            res = QMessageBox.question(
                parent, "Confirm overwrite file",
                f"TXT files already exist.\nOverwrite all TXT files?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if res != QMessageBox.StandardButton.Yes:
                return 

        try:
            _export_txt_files(txt_dir, df_orig)
            QMessageBox.information(parent, "Success", f"✅ TXT Exported:\n{txt_dir}")
            modify_yaml(video_path, "txt", txt_dir, config_path, project_info)
        except Exception as e:
            QMessageBox.critical(parent, "Error", f"Failed to export TXT:\n{e}")
        parent.update_label_combo(set_text = txt_dir)
        return
        
    if action == "video":
        from .video_saver import _export_video_stub
        _export_video_stub(parent)
        return

def _current_video(parent: QWidget) -> str:
    if hasattr(parent, "video_combo"):
        p: Path | None = parent.video_combo.currentData(Qt.ItemDataRole.UserRole)
        if p is None:
            p = Path(parent.video_combo.currentText())
        return p, p.stem
    if DataLoader.csv_path:
        p = Path(DataLoader.csv_path)
        return p, p.stem
    return Path(), "unknown_video"

def _sanitize_index(df: pd.DataFrame) -> pd.DataFrame:
    for lvl in list(df.index.names):
        if lvl:
            drop = lvl in df.columns
            df = df.reset_index(level=lvl, drop=drop)
    df.reset_index(drop=True, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def _find_project(parent: QWidget):
    cur = parent
    while cur:
        if hasattr(cur, "project"):
            return getattr(cur, "project")
        cur = cur.parent()
    return None

def modify_yaml(video_path, file_type, file_path, yaml_path, project_info):
    video_path = str(_norm(video_path))
    file_path  = str(_norm(file_path))
    entry = next((fe for fe in project_info.files if _norm(fe.video) == video_path), None)
    if file_type == "csv":
        if file_path not in entry.csv:
            entry.csv.append(file_path)
    elif file_type == "txt":
        if file_path not in entry.txt:
            entry.txt.append(file_path)

    def _serialize(obj):
        if is_dataclass(obj):
            return {k: _serialize(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict): 
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(i) for i in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    data_dict = _serialize(project_info)
    data_dict["skeleton"] = project_info.skeleton_name
    data_dict.pop("skeleton_yaml", None)

    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data_dict,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

def _norm(p: str | Path) -> Path:
    return str(Path(p).expanduser().resolve())

def _export_txt_files(target_dir: Path, df: pd.DataFrame) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    max_f = int(df["frame.idx"].max())
    pad = max(2, len(str(max_f)))

    def _track_num(t) -> int:
        m = re.search(r"(\d+)$", str(t))
        return int(m.group(1)) if m else 0

    for f_idx in sorted(df["frame.idx"].unique()):
        f_df = df[df["frame.idx"] == f_idx].sort_values("track", key=lambda s: s.map(_track_num))
        lines: list[str] = []

        for _, row in f_df.iterrows():
            tn = _track_num(row["track"])
            xs = [row[f"{kp}.x"] for kp in DataLoader.kp_order if f"{kp}.x" in row]
            ys = [row[f"{kp}.y"] for kp in DataLoader.kp_order if f"{kp}.y" in row]
            mid_x, mid_y = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
            w, h = max(xs) - min(xs), max(ys) - min(ys)

            buf = [tn, mid_x, mid_y, w, h]
            for kp in DataLoader.kp_order:
                x, y = row.get(f"{kp}.x"), row.get(f"{kp}.y")
                vis = row.get(f"{kp}.visibility")
                if x is not None and y is not None and vis is not None:
                    buf.extend([x, y, vis])

            line = " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in buf)
            lines.append(line)

        out = target_dir / f"{f_idx:0{pad}d}.txt"
        out.write_text("\n".join(lines), encoding="utf-8")

    print(f"✅ TXT files saved to {target_dir}")

