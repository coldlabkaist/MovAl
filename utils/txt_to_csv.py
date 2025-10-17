from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDialog, QLineEdit, QMessageBox, QFileDialog, QScrollArea,
    QListView, QTreeView, QAbstractItemView
)
from PyQt6.QtCore import Qt
import os
import pandas as pd
import yaml
import numpy as np
import re

def extract_frame_number(filename):
    match = re.search(r'_(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)\.txt$', filename)
    return int(match.group(1)) if match else -1

class TxtToCsvDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TXT to CSV Convert")
        self.setFixedSize(600, 400)

        self.kpt_names = []

        main_layout = QVBoxLayout(self)

        txt_btn = QPushButton("Load TXT Folders")
        txt_btn.clicked.connect(self.load_txt_folders)
        main_layout.addWidget(txt_btn)

        yaml_btn = QPushButton("Read Data YAML (training_config.yaml from MovAl)")
        yaml_btn.clicked.connect(self.load_yaml)
        main_layout.addWidget(yaml_btn)

        self.kpt_names_label = QLabel("Kpt Names: ")
        self.kpt_names_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.kpt_names_label.setWordWrap(True)
        main_layout.addWidget(self.kpt_names_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.inner = QWidget()
        self.inner_layout = QVBoxLayout(self.inner)
        self.scroll.setWidget(self.inner)
        main_layout.addWidget(self.scroll)
        
        btn_layout = QHBoxLayout()
        norm_btn = QPushButton("Convert CSV (normalized)")
        norm_btn.clicked.connect(self.convert_csv_normalized)
        pixel_btn = QPushButton("Convert CSV (pixel)")
        pixel_btn.clicked.connect(self.convert_csv_pixel)
        btn_layout.addWidget(norm_btn)
        btn_layout.addWidget(pixel_btn)
        main_layout.addLayout(btn_layout)

    def load_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YAML File", "", "YAML Files (*.yaml *.yml)")
        if not file_path:
            return

        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        self.kpt_names = data.get('kpt_names', [])

        print("Loaded kpt_names:", self.kpt_names)

        kpt_text = "Kpt Names:\n"
        for idx, name in enumerate(self.kpt_names):
            kpt_text += f"{idx} : {name}\n"
        self.kpt_names_label.setText(kpt_text)

    def load_txt_folders(self):
        self.txt_folders = []

        dialog = QFileDialog(self, "Select TXT Folders")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        for view in dialog.findChildren(QListView) + dialog.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

        if dialog.exec():
            selected = dialog.selectedFiles()
            for folder in selected:
                if folder and folder not in self.txt_folders:
                    self.txt_folders.append(folder)

        if not self.txt_folders:
            return

        self.video_to_txts = {}
        for folder in self.txt_folders:
            collected_any = False
            if os.path.basename(folder).lower() == 'labels':
                video_name = os.path.basename(os.path.dirname(folder))
                txts = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')]
                if txts:
                    self.video_to_txts.setdefault(video_name, []).extend(txts)
                    collected_any = True
            labels_dir = os.path.join(folder, 'labels')
            if os.path.isdir(labels_dir):
                video_name = os.path.basename(folder)
                txts = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.txt')]
                if txts:
                    self.video_to_txts.setdefault(video_name, []).extend(txts)
                    collected_any = True
            for root, dirs, files in os.walk(folder):
                if os.path.basename(root).lower() == 'labels':
                    video_name = os.path.basename(os.path.dirname(root))
                    txts = [os.path.join(root, f) for f in files if f.endswith('.txt')]
                    if txts:
                        self.video_to_txts.setdefault(video_name, []).extend(txts)
                        collected_any = True
            if not collected_any:
                for root, dirs, files in os.walk(folder):
                    for f in files:
                        if f.endswith('.txt'):
                            name_part = "_".join(f.split("_")[:-1])
                            if name_part:
                                self.video_to_txts.setdefault(name_part, []).append(os.path.join(root, f))

        for k, v in list(self.video_to_txts.items()):
            self.video_to_txts[k] = list(set(v))

        video_names = set(self.video_to_txts.keys())

        for i in reversed(range(self.inner_layout.count())):
            item = self.inner_layout.itemAt(i)
            if item is not None and item.widget():
                item.widget().deleteLater()
            elif item is not None and item.layout():
                while item.layout().count():
                    sub_item = item.layout().takeAt(0)
                    if sub_item.widget():
                        sub_item.widget().deleteLater()

        if self.kpt_names:
            kpt_text = ", ".join([f"{i}: {name}" for i, name in enumerate(self.kpt_names)])
            kpt_label = QLabel(f"kpt_names: {kpt_text}")
            self.inner_layout.addWidget(kpt_label)
            
        self.video_widget_map = {}

        for name in sorted(video_names):
            layout = QHBoxLayout()
            name_label = QLabel(name)
            width_edit = QLineEdit()
            width_edit.setPlaceholderText("width")
            height_edit = QLineEdit()
            height_edit.setPlaceholderText("height")
            layout.addWidget(name_label)
            layout.addWidget(width_edit)
            layout.addWidget(height_edit)
            self.inner_layout.addLayout(layout)
            
            self.video_widget_map[name] = (width_edit, height_edit)

        print("Loaded Video Names:", video_names)

    def convert_csv_normalized(self):
        if not hasattr(self, 'txt_folders') or not self.txt_folders:
            QMessageBox.warning(self, "Error", "Load TXT folders first.")
            return

        if not self.kpt_names:
            QMessageBox.warning(self, "Error", "Load YAML file first.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_dir:
            return

        for video_name in self.video_widget_map:
            all_txts = sorted(self.video_to_txts.get(video_name, []), key=lambda x: extract_frame_number(os.path.basename(x)))

            rows = []
            has_instance_id = False
            for idx, txt_path in enumerate(all_txts):
                with open(txt_path, "r") as f:
                    lines = f.readlines()
                    
                detections = []
                for line in lines:
                    items = line.strip().split()
                    if len(items) < 6:
                        continue
                    track_id = int(items[0])
                    raw = items[5:]

                    if len(raw) % 3 == 1:
                        try:
                            instance_id = int(raw[-1])
                            kpt_data = list(map(float, raw[:-1]))
                            has_instance_id = True
                        except:
                            continue
                    else:
                        instance_id = None
                        kpt_data = list(map(float, raw))

                    remapped_id = instance_id if instance_id is not None else ""
                    detections.append((track_id, remapped_id, kpt_data))

                # Use actual frame number from filename instead of sequential index
                frame_num = extract_frame_number(os.path.basename(txt_path))
                if frame_num < 0:
                    frame_num = idx + 1

                # Merge duplicates by (track_id, instance_id) using per-kpt max confidence
                track_data = {}
                for track_id, remapped_id, kpt_data in detections:
                    key = (track_id, remapped_id if remapped_id != "" else None)
                    if key not in track_data:
                        track_data[key] = (kpt_data, remapped_id)
                    else:
                        prev, rid = track_data[key]
                        prev_np = np.array(prev)
                        curr_np = np.array(kpt_data)
                        for kp in range(len(self.kpt_names)):
                            if curr_np[kp*3 + 2] > prev_np[kp*3 + 2]:
                                prev_np[kp*3:kp*3+3] = curr_np[kp*3:kp*3+3]
                        track_data[key] = (prev_np.tolist(), rid)

                for (track_id, _), (kpt_data, remapped_id) in track_data.items():
                    row = [f"track_{track_id}", frame_num, 0.9]
                    for kp in range(len(self.kpt_names)):
                        x, y, conf = kpt_data[kp*3:kp*3+3]
                        row.extend([x, y, conf])
                    if has_instance_id:
                        row.append(remapped_id)
                    rows.append(row)

            columns = ["track", "frame_idx", "instance.score"]
            for name in self.kpt_names:
                columns += [f"{name}.x", f"{name}.y", f"{name}.score"]
            if has_instance_id:
                columns.append("instance.id")

            df = pd.DataFrame(rows, columns=columns)
            save_path = os.path.join(output_dir, f"{video_name}.csv")
            df.to_csv(save_path, index=False)
            print(f"Saved: {save_path}")


    def convert_csv_pixel(self):
        if not hasattr(self, 'txt_folders') or not self.txt_folders:
            QMessageBox.warning(self, "Error", "Load TXT folders first.")
            return

        if not self.kpt_names:
            QMessageBox.warning(self, "Error", "Load YAML file first.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_dir:
            return

        for video_name, (width_edit, height_edit) in self.video_widget_map.items():
            width = width_edit.text()
            height = height_edit.text()
            if not width or not height:
                QMessageBox.warning(self, "Error", f"{video_name} width/height missing.")
                return
            width, height = int(width), int(height)

            all_txts = sorted(self.video_to_txts.get(video_name, []), key=lambda x: extract_frame_number(os.path.basename(x)))

            rows = []
            has_instance_id = False

            for idx, txt_path in enumerate(all_txts):
                with open(txt_path, "r") as f:
                    lines = f.readlines()

                detections = []  # list of tuples: (track_id, remapped_id, kpt_data)
                for line in lines:
                    items = line.strip().split()
                    if len(items) < 6:
                        continue
                    track_id = int(items[0])
                    raw = items[5:]

                    if len(raw) % 3 == 1:
                        try:
                            instance_id = int(raw[-1])
                            kpt_data = list(map(float, raw[:-1]))
                            has_instance_id = True
                        except:
                            continue
                    else:
                        instance_id = None
                        kpt_data = list(map(float, raw))

                    remapped_id = instance_id if instance_id is not None else ""
                    detections.append((track_id, remapped_id, kpt_data))
                # Use actual frame number from filename instead of sequential index
                frame_num = extract_frame_number(os.path.basename(txt_path))
                if frame_num < 0:
                    frame_num = idx + 1

                # Merge duplicates by (track_id, instance_id) using per-kpt max confidence
                track_data = {}
                for track_id, remapped_id, kpt_data in detections:
                    key = (track_id, remapped_id if remapped_id != "" else None)
                    if key not in track_data:
                        track_data[key] = (kpt_data, remapped_id)
                    else:
                        prev, rid = track_data[key]
                        prev_np = np.array(prev)
                        curr_np = np.array(kpt_data)
                        for kp in range(len(self.kpt_names)):
                            if curr_np[kp*3 + 2] > prev_np[kp*3 + 2]:
                                prev_np[kp*3:kp*3+3] = curr_np[kp*3:kp*3+3]
                        track_data[key] = (prev_np.tolist(), rid)

                for (track_id, _), (kpt_data, remapped_id) in track_data.items():
                    row = [f"track_{track_id}", frame_num, 0.9]
                    for kp in range(len(self.kpt_names)):
                        x = kpt_data[kp*3] * width
                        y = kpt_data[kp*3+1] * height
                        conf = kpt_data[kp*3+2]
                        row.extend([x, y, conf])
                    if has_instance_id:
                        row.append(remapped_id)
                    rows.append(row)

            columns = ["track", "frame_idx", "instance.score"]
            for name in self.kpt_names:
                columns += [f"{name}.x", f"{name}.y", f"{name}.score"]
            if has_instance_id:
                columns.append("instance.id")

            df = pd.DataFrame(rows, columns=columns)
            save_path = os.path.join(output_dir, f"{video_name}.csv")
            df.to_csv(save_path, index=False)
            print(f"Saved: {save_path}")


