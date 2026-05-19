from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from datetime import datetime

import cv2
import pandas as pd
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QMessageBox,
    QInputDialog, QWidget, QFrame
)
from PyQt6.QtCore import Qt
from tqdm import tqdm
from .data_loader import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np   
import shutil 

ONLINE_TXT_EXPORT_ROOT = "online_label_exports"

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
            QMessageBox.information(parent, "Success", f"CSV Saved!:\n{csv_path}")
            parent.update_label_combo(
                video_index = (parent.video_combo.currentIndex() if hasattr(parent, "video_combo") else None),
                set_text = csv_path
            )
        except Exception as e:
            QMessageBox.critical(parent, "Error", f"Failed to save CSV:\n{e}")
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
            export_loaded_data_to_txt_dir(txt_dir, df=df_orig, clear_existing=has_existing)
            QMessageBox.information(parent, "Success", f"TXT Exported:\n{txt_dir}")
            parent.update_label_combo(
                video_index = (parent.video_combo.currentIndex() if hasattr(parent, "video_combo") else None),
                set_text = txt_dir
            )
        except Exception as e:
            QMessageBox.critical(parent, "Error", f"Failed to export TXT:\n{e}")
        return
        
    if action == "video":
        from .video_saver import _export_video_stub
        _export_video_stub(parent)
        return


def export_loaded_data_to_txt_dir(
    target_dir: str | Path,
    *,
    df: pd.DataFrame | None = None,
    clear_existing: bool = False,
) -> Path:
    if df is None:
        if DataLoader.loaded_data is None:
            raise ValueError("Load CSV/TXT first")
        df = _sanitize_index(DataLoader.loaded_data.copy())

    target_dir = Path(target_dir)
    if clear_existing and target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    _export_txt_files(target_dir, df)
    return target_dir


def export_current_labels_to_txt_snapshot(
    parent: QWidget,
    target_dir: str | Path | None = None,
) -> Path:
    project = _find_project(parent)
    if project is None or not hasattr(project, "project_dir"):
        raise ValueError("Project information not found.")

    _, video_name = _current_video(parent)
    if target_dir is None:
        stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        target_dir = (
            Path(project.project_dir)
            / "runs"
            / ONLINE_TXT_EXPORT_ROOT
            / video_name
            / f"txt_snapshot_{stamp}"
        )

    return export_loaded_data_to_txt_dir(target_dir)

def _current_video(parent: QWidget) -> tuple[Path, str]:
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

def _export_txt_files(target_dir: Path, df: pd.DataFrame) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    max_f = int(df["frame_idx"].max())
    pad   = max(2, len(str(max_f)))

    tracks_num = df["track"].map({n: i for i, n in enumerate(DataLoader.animals_name)}).to_numpy(np.int32)
    fidx_arr   = df["frame_idx"].to_numpy(np.int32)
    x_cols = [f"{kp}.x" for kp in DataLoader.kp_order if f"{kp}.x" in df.columns]
    y_cols = [f"{kp}.y" for kp in DataLoader.kp_order if f"{kp}.y" in df.columns]
    v_cols = [f"{kp}.visibility" for kp in DataLoader.kp_order if f"{kp}.visibility" in df.columns]

    xs = df[x_cols].to_numpy(np.float32)
    ys = df[y_cols].to_numpy(np.float32)
    vs = df[v_cols].to_numpy(np.int8)
    kp_n = len(v_cols)

    unique_frames = np.unique(fidx_arr)

    def _one_frame(fid: int):
        m    = fidx_arr == fid
        rows = np.nonzero(m)[0]
        out_lines: list[str] = []
        for idx in rows:
            tn = tracks_num[idx]
            xx = xs[idx]; yy = ys[idx]
            mid_x = (xx.min() + xx.max()) / 2
            mid_y = (yy.min() + yy.max()) / 2
            w     = xx.max() - xx.min()
            h     = yy.max() - yy.min()

            buf = [tn, mid_x, mid_y, w, h]
            for k in range(kp_n):
                buf.extend((xx[k], yy[k], int(vs[idx, k])))

            out_lines.append(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v)
                                       for v in buf))

        (target_dir / f"{fid:0{pad}d}.txt").write_text("\n".join(out_lines),
                                                       encoding="utf-8")

    cpu_n = max((os.cpu_count() or 1) * 2, 4)
    with ThreadPoolExecutor(max_workers=cpu_n) as pool:
        futures = [pool.submit(_one_frame, fid) for fid in unique_frames]
        with tqdm(total=len(futures), desc="Exporting TXT") as pbar:
            for _ in as_completed(futures):
                pbar.update(1)

    print(f"TXT files saved to {target_dir}")
