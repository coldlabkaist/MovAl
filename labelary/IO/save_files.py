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
from tqdm import tqdm
from .data_loader import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np   
import shutil 

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
            QMessageBox.information(parent, "Success", f"CSV Saved!:\n{csv_path}")
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
            shutil.rmtree(txt_dir)
            txt_dir.mkdir(parents=True, exist_ok=True)

        try:
            _export_txt_files(txt_dir, df_orig)
            QMessageBox.information(parent, "Success", f"TXT Exported:\n{txt_dir}")
            if not has_existing:
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
    pad   = max(2, len(str(max_f)))

    tracks_num = df["track"].map({n: i for i, n in enumerate(DataLoader.animals_name)}).to_numpy(np.int32)
    fidx_arr   = df["frame.idx"].to_numpy(np.int32)

    xs = df.filter(regex=r"\.x$").to_numpy(np.float32)
    ys = df.filter(regex=r"\.y$").to_numpy(np.float32)
    vs = df.filter(regex=r"\.visibility$").to_numpy(np.int8)
    kp_n = xs.shape[1]

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

    cpu_n = max(os.cpu_count() * 2, 4)
    with ThreadPoolExecutor(max_workers=cpu_n) as pool:
        futures = [pool.submit(_one_frame, fid) for fid in unique_frames]
        counter = 0
        with tqdm(total=len(futures), desc="Exporting TXT") as pbar:
            for _ in as_completed(futures):
                counter += 1
                if counter % 50 == 0:
                    pbar.update(50)

    print(f"TXT files saved to {target_dir}")