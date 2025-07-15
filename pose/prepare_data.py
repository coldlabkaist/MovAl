from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QScrollArea, QComboBox,
    QDialog, QMessageBox, QSpinBox, QFileDialog, QGroupBox, QFormLayout, QSlider, QCheckBox, QLineEdit, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import subprocess
import os
import json
import random
import shutil
import yaml
import cv2
import re
import numpy as np
import pandas as pd
from pathlib import Path
   
class DataSplitDialog(QDialog):
    def __init__(self, current_project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Split")
        self.setFixedSize(500, 400)

        self.current_project = current_project
        self.files = current_project.files

        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner_widget = QWidget()
        self.files_lay = QVBoxLayout(inner_widget)
        scroll.setWidget(inner_widget)
        layout.addWidget(scroll)

        self._populate_file_items()

        layout.addSpacing(40) 
        self.count_label = QLabel("Label number: 0 / Image number: 0")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        count_font = QFont()
        count_font.setPointSize(11) 
        self.count_label.setFont(count_font)
        layout.addWidget(self.count_label)

        layout.addSpacing(20) 

        ratio_layout = QFormLayout()

        self.train_slider = QSlider(Qt.Orientation.Horizontal)
        self.train_slider.setRange(0, 100)
        self.train_slider.setValue(70)
        self.train_spin = QSpinBox()
        self.train_spin.setRange(0, 100)
        self.train_spin.setValue(70)
        self.train_slider.valueChanged.connect(self.train_spin.setValue)
        self.train_spin.valueChanged.connect(self.train_slider.setValue)
        ratio_layout.addRow("Train %", self.create_slider_spinbox_layout(self.train_slider, self.train_spin))

        self.valid_slider = QSlider(Qt.Orientation.Horizontal)
        self.valid_slider.setRange(0, 100)
        self.valid_slider.setValue(20)
        self.valid_spin = QSpinBox()
        self.valid_spin.setRange(0, 100)
        self.valid_spin.setValue(20)
        self.valid_slider.valueChanged.connect(self.valid_spin.setValue)
        self.valid_spin.valueChanged.connect(self.valid_slider.setValue)
        ratio_layout.addRow("Valid %", self.create_slider_spinbox_layout(self.valid_slider, self.valid_spin))

        self.test_slider = QSlider(Qt.Orientation.Horizontal)
        self.test_slider.setRange(0, 100)
        self.test_slider.setValue(10)
        self.test_spin = QSpinBox()
        self.test_spin.setRange(0, 100)
        self.test_spin.setValue(10)
        self.test_slider.valueChanged.connect(self.test_spin.setValue)
        self.test_spin.valueChanged.connect(self.test_slider.setValue)
        ratio_layout.addRow("Test %", self.create_slider_spinbox_layout(self.test_slider, self.test_spin))

        layout.addLayout(ratio_layout)

        layout.addSpacing(20)

        self.frame_type_combo = QComboBox()
        self.frame_type_combo.addItem("davis")
        self.frame_type_combo.addItem("contour")
        layout.addWidget(self.frame_type_combo)

        self.run_btn = QPushButton("Run")
        layout.addWidget(self.run_btn)
        self.run_btn.clicked.connect(self.run_split)

        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

    def _populate_file_items(self) -> None:
        for fe in self.files:
            video_path = Path(fe.video)
            if not video_path.is_absolute():
                video_path = self.current_project.project_dir / video_path

            cap = cv2.VideoCapture(str(video_path))
            frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()

            label_cnt = 0
            for txt_rel in fe.txt:
                txt_full = Path(txt_rel)
                if not txt_full.is_absolute():
                    txt_full = self.current_project.project_dir / txt_full

                if txt_full.is_dir():
                    label_cnt += sum(1 for _ in txt_full.glob("*.txt"))
                elif txt_full.is_file():
                    label_cnt += 1

            row_lay = QHBoxLayout()
            chk = QCheckBox()
            chk.stateChanged.connect(self._update_selection_count)
            chk._frame_cnt  = frame_cnt
            chk._label_cnt  = label_cnt
            chk._file_entry = fe

            name_lbl  = QLabel(video_path.name)
            count_lbl = QLabel(f"({frame_cnt:,} frames, {label_cnt:,} labels)")
            count_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            row_lay.addWidget(chk)
            row_lay.addWidget(name_lbl, 1)
            row_lay.addWidget(count_lbl)

            self.files_lay.addLayout(row_lay)

        self.files_lay.addStretch(1)

    def _update_selection_count(self):
        total_files  = 0
        total_frames = 0
        total_labels = 0

        for i in range(self.files_lay.count() - 1): 
            lay = self.files_lay.itemAt(i)
            if not isinstance(lay, QHBoxLayout):
                continue
            chk = lay.itemAt(0).widget()
            if isinstance(chk, QCheckBox) and chk.isChecked():
                total_files  += 1
                total_frames += getattr(chk, "_frame_cnt", 0)
                total_labels += getattr(chk, "_label_cnt", 0)

        self.count_label.setText(
            f"{total_files} files selected ㆍ "
            f"{total_frames:,} frames ㆍ "
            f"{total_labels:,} labels"
        )

    def get_selected_entries(self):
        selected_entries: list[FileEntry] = []
        for i in range(self.files_lay.count() - 1):
            lay = self.files_lay.itemAt(i)
            if not isinstance(lay, QHBoxLayout):
                continue
            chk = lay.itemAt(0).widget()
            if isinstance(chk, QCheckBox) and chk.isChecked():
                selected_entries.append(chk._file_entry)
        return selected_entries

    def create_slider_spinbox_layout(self, slider, spinbox):
        hlayout = QHBoxLayout()
        hlayout.addWidget(slider)
        hlayout.addWidget(spinbox)
        return hlayout

    def run_split(self):
        selected_entries: List[FileEntry] = self.get_selected_entries()
        if not selected_entries:
            QMessageBox.warning(self, "Error", "First, select a video file.")
            return

        project_dir = Path(self.current_project.project_dir)
        dataset_dir = project_dir / "runs" / "dataset"

        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        for split in ("train", "val", "test"):
            (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        frame_type = self.frame_type_combo.currentText()
        pair_list: List[Tuple[Path, Path, str]] = []

        digit_re = re.compile(r'(\d+)$')        # video1_0000042 → 0000042

        for fe in selected_entries:
            video_path = Path(fe.video)
            video_name = video_path.stem 

            for txt_rel in fe.txt:
                txt_path = Path(txt_rel)
                if not txt_path.is_absolute():
                    txt_path = project_dir / txt_path
                if txt_path.is_dir():
                    txt_files = sorted(txt_path.glob("*.txt"))  
                else:
                    continue
                for lbl_file in txt_files:
                    m = digit_re.search(lbl_file.stem)
                    if not m:
                        continue
                    
                    orig_num_str   = m.group(1)
                    digits_len     = 7
                    base_digit_len = len(orig_num_str)
                    frame_idx      = int(orig_num_str) - 1
                    frame_num      = f"{frame_idx:0{digits_len}d}"
                    base_name      = f"{video_name}_{frame_idx:0{base_digit_len}d}"

                    img_dir  = project_dir / "frames" / video_name / "visualization" / frame_type
                    img_path = img_dir / f"{frame_num}.jpg"

                    pair_list.append((lbl_file, img_path, base_name))

        if not pair_list:
            QMessageBox.warning(self, "Error", "Could not find label-image pair.")
            return

        random.shuffle(pair_list)

        train_ratio = self.train_spin.value() / 100.0
        val_ratio   = self.valid_spin.value() / 100.0

        total = len(pair_list)
        train_end = int(total * train_ratio)
        val_end   = train_end + int(total * val_ratio)

        split_map = {
            "train": pair_list[:train_end],
            "val":   pair_list[train_end:val_end],
            "test":  pair_list[val_end:],
        }

        for split, pairs in split_map.items():
            img_dst_root = dataset_dir / split / "images"
            lbl_dst_root = dataset_dir / split / "labels"

            for lbl_path, img_path, base in pairs:
                print(lbl_path, img_path, base, base_digit_len)
                shutil.copy(lbl_path, lbl_dst_root / f"{base}.txt")
                img_ext = img_path.suffix.lower()
                shutil.copy(img_path, img_dst_root / f"{base}{img_ext}")

        QMessageBox.information(
            self,
            "Success",
            (f"Data Split completed\n"
            f"Train: {len(split_map['train'])}\n"
            f"Val:   {len(split_map['val'])}\n"
            f"Test:  {len(split_map['test'])}")
        )