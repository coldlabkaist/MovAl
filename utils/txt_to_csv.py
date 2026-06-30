from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDialog, QLineEdit, QMessageBox, QFileDialog, QScrollArea,
    QListView, QTreeView, QAbstractItemView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIntValidator
import os
from pathlib import Path
import pandas as pd
import yaml
import re
from utils.skeleton.skeleton_model import SkeletonModel
from utils.txt_conversion import parse_txt_pose_detections, resolve_frame_track_data

def extract_frame_number(filename):
    match = re.search(r'_(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)\.txt$', filename)
    return int(match.group(1)) if match else -1

class TxtToCsvDialog(QDialog):
    def __init__(self, current_project=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TXT to CSV Convert")
        self.setFixedSize(680, 460)

        self.current_project = current_project
        self.kpt_names = []
        self.txt_folders = []
        self.video_to_txts = {}
        self.video_widget_map = {}
        self._pixel_value_validator = QIntValidator(1, 100000, self)

        main_layout = QVBoxLayout(self)

        txt_btn = QPushButton("Load TXT Folders")
        txt_btn.clicked.connect(self.load_txt_folders)
        main_layout.addWidget(txt_btn)

        yaml_btn = QPushButton("Load Keypoints from YAML (Optional / from skeleton/(you_skeleton).yaml)")
        yaml_btn.clicked.connect(self.load_yaml)
        main_layout.addWidget(yaml_btn)

        self.kpt_names_label = QLabel("Kpt Names: ")
        self.kpt_names_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.kpt_names_label.setWordWrap(True)
        main_layout.addWidget(self.kpt_names_label)

        bulk_label = QLabel("Pixel size batch apply")
        bulk_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        main_layout.addWidget(bulk_label)

        bulk_layout = QHBoxLayout()
        self.bulk_width_edit = QLineEdit()
        self.bulk_width_edit.setPlaceholderText("width")
        self.bulk_width_edit.setValidator(self._pixel_value_validator)
        self.bulk_height_edit = QLineEdit()
        self.bulk_height_edit.setPlaceholderText("height")
        self.bulk_height_edit.setValidator(self._pixel_value_validator)
        self.apply_all_pixels_btn = QPushButton("Apply to all")
        self.apply_all_pixels_btn.clicked.connect(self.apply_pixel_size_to_all)
        self.fill_empty_pixels_btn = QPushButton("Fill empty only")
        self.fill_empty_pixels_btn.clicked.connect(
            lambda: self.apply_pixel_size_to_all(fill_empty_only=True)
        )
        bulk_layout.addWidget(self.bulk_width_edit)
        bulk_layout.addWidget(self.bulk_height_edit)
        bulk_layout.addWidget(self.apply_all_pixels_btn)
        bulk_layout.addWidget(self.fill_empty_pixels_btn)
        main_layout.addLayout(bulk_layout)

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

        if self.current_project is not None:
            self.load_keypoints_from_project(show_dialog=False)

    def load_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YAML File", "", "YAML Files (*.yaml *.yml)")
        if not file_path:
            return

        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        self._set_kpt_names(data.get('kpt_names', []), source=file_path)

    def load_keypoints_from_project(self, *, show_dialog: bool = True) -> bool:
        if self.current_project is None:
            return False

        try:
            skeleton_model = SkeletonModel()
            skeleton_model.load_from_dict(self.current_project.skeleton_data)
            _, _, kpt_names = skeleton_model.create_training_config()
            if not kpt_names:
                raise ValueError("No keypoints found in project skeleton.")
            self._set_kpt_names(
                kpt_names,
                source=f"{Path(self.current_project.project_file).name} / skeleton_data",
            )
            return True
        except Exception as err:
            if show_dialog:
                QMessageBox.warning(
                    self,
                    "Project keypoints load failed",
                    f"Could not read keypoints from current project:\n{err}",
                )
            return False

    def _set_kpt_names(self, names, *, source: str) -> None:
        self.kpt_names = list(names or [])

        kpt_text = f"Kpt Names (source: {source}):\n"
        for idx, name in enumerate(self.kpt_names):
            kpt_text += f"{idx} : {name}\n"
        self.kpt_names_label.setText(kpt_text)

    def _create_pixel_line_edit(self, placeholder: str) -> QLineEdit:
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setValidator(self._pixel_value_validator)
        return line_edit

    def _bulk_pixel_size(self):
        width_text = self.bulk_width_edit.text().strip()
        height_text = self.bulk_height_edit.text().strip()
        if not width_text or not height_text:
            QMessageBox.warning(
                self,
                "Bulk pixel size missing",
                "Enter both width and height before applying them in bulk.",
            )
            return None
        return width_text, height_text

    def apply_pixel_size_to_all(self, fill_empty_only: bool = False) -> None:
        if not self.video_widget_map:
            QMessageBox.warning(
                self,
                "No videos loaded",
                "Load TXT folders first.",
            )
            return

        pixel_size = self._bulk_pixel_size()
        if pixel_size is None:
            return
        width_text, height_text = pixel_size

        for width_edit, height_edit in self.video_widget_map.values():
            if fill_empty_only:
                if not width_edit.text().strip():
                    width_edit.setText(width_text)
                if not height_edit.text().strip():
                    height_edit.setText(height_text)
                continue
            width_edit.setText(width_text)
            height_edit.setText(height_text)

    def load_txt_folders(self):
        self.txt_folders = []

        dialog = QFileDialog(self, "Select TXT Folders")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
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
            width_edit = self._create_pixel_line_edit("width")
            height_edit = self._create_pixel_line_edit("height")
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
            if self.load_keypoints_from_project(show_dialog=False):
                pass
            else:
                QMessageBox.warning(self, "Error", "Load keypoints from current project or YAML first.")
                return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_dir:
            return

        for video_name in self.video_widget_map:
            all_txts = sorted(self.video_to_txts.get(video_name, []), key=lambda x: extract_frame_number(os.path.basename(x)))

            rows = []
            per_id_limit = (
                self.current_project.get_max_instances_per_id()
                if self.current_project is not None and hasattr(self.current_project, "get_max_instances_per_id")
                else 1
            )
            include_instance_id = per_id_limit > 1
            for idx, txt_path in enumerate(all_txts):
                with open(txt_path, "r") as f:
                    lines = f.readlines()

                detections, frame_has_instance_id = parse_txt_pose_detections(lines)
                include_instance_id = include_instance_id or frame_has_instance_id

                # Use actual frame number from filename instead of sequential index
                frame_num = extract_frame_number(os.path.basename(txt_path))
                if frame_num < 0:
                    frame_num = idx + 1

                track_data, frame_uses_instance_id = resolve_frame_track_data(
                    detections,
                    keypoint_count=len(self.kpt_names),
                    per_id_limit=per_id_limit,
                )
                include_instance_id = include_instance_id or frame_uses_instance_id

                for track_id, kpt_data, remapped_id in track_data:
                    row = {
                        "track": f"track_{track_id}",
                        "frame_idx": frame_num,
                        "instance.score": 0.9,
                    }
                    for kp in range(len(self.kpt_names)):
                        x, y, conf = kpt_data[kp*3:kp*3+3]
                        kp_name = self.kpt_names[kp]
                        row[f"{kp_name}.x"] = x
                        row[f"{kp_name}.y"] = y
                        row[f"{kp_name}.score"] = conf
                    if remapped_id is not None:
                        row["instance.id"] = remapped_id
                    rows.append(row)

            columns = ["track", "frame_idx", "instance.score"]
            for name in self.kpt_names:
                columns += [f"{name}.x", f"{name}.y", f"{name}.score"]
            if include_instance_id:
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
            if self.load_keypoints_from_project(show_dialog=False):
                pass
            else:
                QMessageBox.warning(self, "Error", "Load keypoints from current project or YAML first.")
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
            per_id_limit = (
                self.current_project.get_max_instances_per_id()
                if self.current_project is not None and hasattr(self.current_project, "get_max_instances_per_id")
                else 1
            )
            include_instance_id = per_id_limit > 1

            for idx, txt_path in enumerate(all_txts):
                with open(txt_path, "r") as f:
                    lines = f.readlines()

                detections, frame_has_instance_id = parse_txt_pose_detections(lines)
                include_instance_id = include_instance_id or frame_has_instance_id
                # Use actual frame number from filename instead of sequential index
                frame_num = extract_frame_number(os.path.basename(txt_path))
                if frame_num < 0:
                    frame_num = idx + 1

                track_data, frame_uses_instance_id = resolve_frame_track_data(
                    detections,
                    keypoint_count=len(self.kpt_names),
                    per_id_limit=per_id_limit,
                )
                include_instance_id = include_instance_id or frame_uses_instance_id

                for track_id, kpt_data, remapped_id in track_data:
                    row = {
                        "track": f"track_{track_id}",
                        "frame_idx": frame_num,
                        "instance.score": 0.9,
                    }
                    for kp in range(len(self.kpt_names)):
                        kp_name = self.kpt_names[kp]
                        row[f"{kp_name}.x"] = kpt_data[kp * 3] * width
                        row[f"{kp_name}.y"] = kpt_data[kp * 3 + 1] * height
                        row[f"{kp_name}.score"] = kpt_data[kp * 3 + 2]
                    if remapped_id is not None:
                        row["instance.id"] = remapped_id
                    rows.append(row)

            columns = ["track", "frame_idx", "instance.score"]
            for name in self.kpt_names:
                columns += [f"{name}.x", f"{name}.y", f"{name}.score"]
            if include_instance_id:
                columns.append("instance.id")

            df = pd.DataFrame(rows, columns=columns)
            save_path = os.path.join(output_dir, f"{video_name}.csv")
            df.to_csv(save_path, index=False)
            print(f"Saved: {save_path}")
