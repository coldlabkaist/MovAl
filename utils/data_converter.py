from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel,
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
import numpy as np
import pandas as pd
from pathlib import Path


class DataConverterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prepare Data")
        self.setFixedSize(300, 200)

        main_layout = QVBoxLayout()

        step1_group = QGroupBox("Convert to JSON")
        step1_layout = QVBoxLayout()
        self.make_json_btn = QPushButton("SLP to JSON")
        self.make_json_btn.setFixedHeight(40)
        self.make_json_btn.clicked.connect(self.run_slp_to_coco)

        self.dlc_to_coco_btn = QPushButton("DLC to JSON")
        self.dlc_to_coco_btn.setFixedHeight(40)
        self.dlc_to_coco_btn.clicked.connect(self.open_dlc_to_coco)

        step1_layout.addWidget(self.make_json_btn)
        step1_layout.addWidget(self.dlc_to_coco_btn)
        step1_group.setLayout(step1_layout)
        main_layout.addWidget(step1_group)

        step2_label = QLabel("Convert to TXT")
        self.json_to_txt_btn = QPushButton("JSON to TXT")
        self.json_to_txt_btn.setFixedHeight(40)
        self.json_to_txt_btn.clicked.connect(self.open_json_to_txt)
        main_layout.addWidget(step2_label)
        main_layout.addWidget(self.json_to_txt_btn)

        self.setLayout(main_layout)
        
    def run_slp_to_coco(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        bat_path = os.path.join(project_root, "utils", "sleap_converter", "slp_to_coco.bat")

        if not os.path.exists(bat_path):
            QMessageBox.warning(self, "Error", f"{bat_path} not found")
            return

        subprocess.Popen(bat_path, shell=True)
        QMessageBox.information(self, "Launching", "Launching SLP to COCO GUI...")
        
    def open_dlc_to_coco(self):
        dialog= DlcToCocoDialog(self)
        dialog.exec()

    def open_json_to_txt(self):
        dialog = JsonToTxtDialog(self)
        dialog.exec()
        
class DlcToCocoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DLC to COCO option select")
        self.setMinimumSize(600, 700)

        self.folder_path = ""
        self.config_path = ""
        self.video_checkboxes = []
        self.track_spinboxes = {}
        self.keypoint_checkboxes = []

        main_layout = QVBoxLayout(self)

        file_layout = QHBoxLayout()
        self.folder_path_field = QLineEdit()
        self.folder_path_field.setReadOnly(True)
        browse_btn = QPushButton("Select labeled-data Folder")
        browse_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(self.folder_path_field)
        file_layout.addWidget(browse_btn)
        main_layout.addLayout(file_layout)

        self.config_path_field = QLineEdit()
        self.config_path_field.setReadOnly(True)
        config_btn = QPushButton("Select config.yaml")
        config_btn.clicked.connect(self.select_config)

        config_layout = QHBoxLayout()
        config_layout.addWidget(self.config_path_field)
        config_layout.addWidget(config_btn)
        main_layout.addLayout(config_layout)

        self.video_group = self.create_groupbox("1. Select folders (video sources)")
        self.track_group = self.create_groupbox("2. Track names with order")
        self.keypoint_group = self.create_groupbox("3. Select keypoints for training")

        main_layout.addWidget(self.video_group)
        main_layout.addWidget(self.track_group)
        main_layout.addWidget(self.keypoint_group)

        button_layout = QHBoxLayout()
        extract_img_btn = QPushButton("Extract Images")
        extract_img_btn.clicked.connect(self.extract_images)
        extract_json_btn = QPushButton("Extract JSON")
        extract_json_btn.clicked.connect(self.extract_json)
        button_layout.addWidget(extract_img_btn)
        button_layout.addWidget(extract_json_btn)
        main_layout.addLayout(button_layout)

        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        main_layout.addWidget(self.status_label)

    def create_groupbox(self, title):
        group = QGroupBox(title)
        layout = QVBoxLayout()
        dummy = QTextEdit()
        dummy.setReadOnly(True)
        layout.addWidget(dummy)
        group.setLayout(layout)
        return group

    def set_groupbox_layout(self, groupbox, layout):
        old_widget = groupbox.layout().itemAt(0).widget()
        old_widget.deleteLater()
        container = QWidget()
        container.setLayout(layout)
        groupbox.layout().addWidget(container)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select labeled-data Folder")
        if not folder:
            return
        self.folder_path = folder
        self.folder_path_field.setText(folder)
        self.try_initialize()
        
    def select_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select config.yaml", "", "YAML files (*.yaml *.yml)")
        if not file_path:
            return
        self.config_path = file_path
        self.config_path_field.setText(file_path)
        self.try_initialize()
        
    def try_initialize(self):
        if not self.folder_path or not self.config_path:
            return 

        video_list = [
            f.name for f in os.scandir(self.folder_path)
            if f.is_dir() and "_labeled" not in f.name
        ]

        with open(self.config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        track_list = cfg.get("individuals", [])
        keypoints = cfg.get("multianimalbodyparts", [])
        self.skeleton = cfg.get("skeleton", [])

        self.update_video_list(video_list)
        self.update_track_list(track_list)
        self.update_keypoint_list(keypoints)

        self.status_label.setText("✔️ Config + Folder loaded successfully.")

    def find_valid_video_folders(self):
        return [
            f.name for f in os.scandir(self.folder_path)
            if f.is_dir() and "_labeled" not in f.name
        ]

    def update_video_list(self, video_list):
        layout = QVBoxLayout()
        self.video_checkboxes = []
        for name in video_list:
            cb = QCheckBox(name)
            cb.setChecked(True)
            layout.addWidget(cb)
            self.video_checkboxes.append(cb)
        self.set_groupbox_layout(self.video_group, layout)

    def update_track_list(self, track_list):
        outer_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Track name"))
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Input ID"))
        self.track_spinboxes = {}
        for track in track_list:
            left_layout.addWidget(QLabel(track))
            le = QLineEdit()
            le.setFixedWidth(50)
            self.track_spinboxes[track] = le
            right_layout.addWidget(le)
        outer_layout.addLayout(left_layout)
        outer_layout.addLayout(right_layout)
        self.set_groupbox_layout(self.track_group, outer_layout)

    def update_keypoint_list(self, kp_list):
        layout = QVBoxLayout()
        self.keypoint_checkboxes = []
        for kp in kp_list:
            cb = QCheckBox(kp)
            cb.setChecked(True)
            layout.addWidget(cb)
            self.keypoint_checkboxes.append(cb)
        self.set_groupbox_layout(self.keypoint_group, layout)

    def extract_images(self):
        if not self.folder_path:
            QMessageBox.warning(self, "Error", "select labeled-data folder first.")
            return

        selected_videos = [cb.text() for cb in self.video_checkboxes if cb.isChecked()]
        if not selected_videos:
            QMessageBox.warning(self, "Error", "Select at least one folder.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select output folder for extracted images")
        if not output_dir:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for video in selected_videos:
            folder = Path(self.folder_path) / video
            h5_files = list(folder.glob("*CollectedData*.h5"))
            if not h5_files:
                print(f"❌ No .h5 file: {folder}")
                continue
            df = pd.read_hdf(h5_files[0])
            for img_index in df.index:
                image_file = img_index if isinstance(img_index, str) else img_index[-1]
                img_path = folder / image_file
                if not img_path.exists():
                    print(f"⚠️ image omission: {img_path}")
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"⚠️ OpenCV failed: {img_path}")
                    continue
                unique_name = f"{video}_{Path(image_file).stem}.jpg"
                dst_path = output_dir / unique_name
                cv2.imwrite(str(dst_path), img)

        QMessageBox.information(self, "Success", f"✅ {len(selected_videos)} extract complete!")

    def extract_json(self):
        if not self.folder_path:
            QMessageBox.warning(self, "Error", "Select labeled-data folder first.")
            return

        config_path = Path(self.folder_path).parent / "config.yaml"
        if not config_path.exists():
            QMessageBox.warning(self, "Error", f"Cannot search config.yaml:\n{config_path}")
            return

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        skeleton_edges = cfg.get('skeleton', [])

        selected_bodyparts = [cb.text() for cb in self.keypoint_checkboxes if cb.isChecked()]
        selected_tracks = {name: int(box.text()) for name, box in self.track_spinboxes.items() if box.text().isdigit()}
        selected_videos = [cb.text() for cb in self.video_checkboxes if cb.isChecked()]

        categories = [
            {
                "id": track_id,
                "name": track_name,
                "supercategory": "animal",
                "keypoints": selected_bodyparts,
                "skeleton": [
                    [selected_bodyparts.index(a), selected_bodyparts.index(b)]
                    for a, b in self.skeleton if a in selected_bodyparts and b in selected_bodyparts
                ]
            }
            for track_name, track_id in selected_tracks.items()
        ]

        images, annotations = [], []
        image_id = 1
        annotation_id = 1

        for video in selected_videos:
            folder = Path(self.folder_path) / video
            h5_files = list(folder.glob("*CollectedData*.h5"))
            if not h5_files:
                continue
            df = pd.read_hdf(h5_files[0])
            scorer = df.columns.levels[0][0]

            for img_index, row in df.iterrows():
                image_file = img_index if isinstance(img_index, str) else img_index[-1]
                unique_name = f"{video}_{Path(image_file).stem}.jpg"
                image_path = folder / image_file

                if not image_path.exists():
                    continue

                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                height, width = img.shape[:2]

                images.append({
                    "file_name": unique_name,
                    "height": height,
                    "width": width,
                    "id": image_id
                })

                for indiv, cid in selected_tracks.items():
                    keypoints_list, xs, ys = [], [], []
                    for bp in selected_bodyparts:
                        x = row.get((scorer, indiv, bp, 'x'), np.nan)
                        y = row.get((scorer, indiv, bp, 'y'), np.nan)
                        v = 2 if not np.isnan(x) and not np.isnan(y) else 0
                        keypoints_list.extend([
                            float(x) if not np.isnan(x) else 0.0,
                            float(y) if not np.isnan(y) else 0.0,
                            v
                        ])
                        if v > 0:
                            xs.append(x)
                            ys.append(y)

                    if not xs:
                        continue

                    x_min, y_min = float(min(xs)), float(min(ys))
                    x_max, y_max = float(max(xs)), float(max(ys))
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": cid,
                        "keypoints": keypoints_list,
                        "num_keypoints": sum([1 for v in keypoints_list[2::3] if v > 0]),
                        "bbox": bbox,
                        "iscrowd": 0,
                        "area": bbox[2] * bbox[3]
                    })
                    annotation_id += 1

                image_id += 1

        output_path, _ = QFileDialog.getSaveFileName(self, "Save COCO JSON", "dlc_to_coco.json", "JSON (*.json)")
        if not output_path:
            return

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "images": images,
                "annotations": annotations,
                "categories": categories
            }, f, indent=4)

        QMessageBox.information(self, "Success", f"✅ JSON save complete: {output_path}")
        
class JsonToTxtDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("JSON to TXT Converter")
        self.setFixedSize(500, 600)

        layout = QVBoxLayout(self)

        json_layout = QHBoxLayout()
        self.json_path_label = QLabel("No file selected")
        self.json_path_label.setMinimumWidth(300)
        self.load_json_btn = QPushButton("Load JSON")
        self.load_json_btn.setFixedWidth(100)
        self.load_json_btn.clicked.connect(self.load_json)

        json_layout.addWidget(self.json_path_label)
        json_layout.addWidget(self.load_json_btn)
        layout.addLayout(json_layout)

        self.frame_label = QLabel("Total Frames: 0 / Total Labels: 0")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.frame_label)

        self.category_group = QGroupBox("Labels per Category")
        self.category_text = QTextEdit()
        self.category_text.setReadOnly(True)
        category_layout = QVBoxLayout()
        category_layout.addWidget(self.category_text)
        self.category_group.setLayout(category_layout)
        layout.addWidget(self.category_group)

        self.kp_group = QGroupBox("Keypoints")
        self.kp_text = QTextEdit()
        self.kp_text.setReadOnly(True)
        kp_layout = QVBoxLayout()
        kp_layout.addWidget(self.kp_text)
        self.kp_group.setLayout(kp_layout)
        layout.addWidget(self.kp_group)

        self.extract_btn = QPushButton("Extract TXT")
        self.extract_btn.setFixedHeight(30)
        layout.addWidget(self.extract_btn)
        self.extract_btn.clicked.connect(self.extract_txt)

    def load_json(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select JSON File", "", "JSON Files (*.json)")
        if not file_path:
            return

        self.json_path_label.setText(file_path)
        self.loaded_json_path = file_path

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        num_frames = len({img["file_name"] for img in data["images"]})
        num_labels = len(data["annotations"])

        self.frame_label.setText(f"Total Frames: {num_frames} / Total Labels: {num_labels}")

        counts = {}
        for ann in data["annotations"]:
            cid = ann["category_id"]
            counts[cid] = counts.get(cid, 0) + 1

        txt = ""
        font = QFont()
        font.setPointSize(11)
        self.category_text.setFont(font)

        for cat in data["categories"]:
            origin_id = cat['id']
            new_id = origin_id - 1
            count = counts.get(origin_id, 0)
            txt += f"{cat['name']} (id:{origin_id} --> {new_id}) : {count}s\n\n"

        self.category_text.setText(txt)

        kp_txt = ""
        self.kp_text.setFont(font)

        for idx, kp in enumerate(data["categories"][0]["keypoints"]):
            kp_txt += f"{idx}: {kp}\n\n"

        self.kp_text.setText(kp_txt)
        
    def extract_txt(self):
        if not hasattr(self, 'loaded_json_path'):
            QMessageBox.warning(self, "Error", "Please load a JSON file first.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return

        with open(self.loaded_json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        images_info = {img['id']: img for img in coco_data['images']}
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        cat_to_class = {}
        for cat in coco_data['categories']:
            cat_id = cat['id']       
            class_id = cat_id - 1    
            cat_to_class[cat_id] = class_id

        num_keypoints = len(coco_data['categories'][0]['keypoints'])
        label_output_dir = os.path.join(output_dir)

        for img_id, img_info in images_info.items():
            file_name = img_info['file_name']
            width = img_info['width']
            height = img_info['height']

            if img_id not in annotations_by_image:
                continue

            label_path = os.path.join(label_output_dir, os.path.splitext(file_name)[0] + ".txt")

            lines = []
            for ann in annotations_by_image[img_id]:
                category_id = ann['category_id']
                class_id = cat_to_class[category_id]

                x_min, y_min, w_box, h_box = ann['bbox']
                x_center = x_min + w_box / 2
                y_center = y_min + h_box / 2

                x_center_norm = x_center / width
                y_center_norm = y_center / height
                width_norm = w_box / width
                height_norm = h_box / height

                kp = ann['keypoints']
                kp_xy = []
                for i in range(num_keypoints):
                    x = kp[i*3]
                    y = kp[i*3+1]
                    v = kp[i*3+2]
                    if v == 0:
                        x_norm, y_norm = 0.0, 0.0
                    else:
                        x_norm, y_norm = x / width, y / height
                    kp_xy.extend([x_norm, y_norm, v])

                line = f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm} " + " ".join(map(str, kp_xy))
                lines.append(line)

            with open(label_path, 'w') as f:
                for line in lines:
                    f.write(line + "\n")

        QMessageBox.information(self, "Success", f"TXT files saved to:\n{label_output_dir}")
