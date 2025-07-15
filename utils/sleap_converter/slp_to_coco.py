import sys
import sleap
from qtpy.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QCheckBox, QTextEdit, QGroupBox, QMessageBox,
    QFormLayout, QScrollArea
)
from qtpy.QtCore import Qt
import json
import cv2
import os
from sleap import load_file

class SlpToCocoOptionsGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SLP to COCO option select")
        self.setMinimumSize(600, 700)

        self.video_checkboxes = []
        self.track_spinboxes = {}
        self.keypoint_checkboxes = []
        self.slp_path = None

        layout = QVBoxLayout(self)

        file_layout = QHBoxLayout()
        self.slp_path_field = QLineEdit()
        self.slp_path_field.setReadOnly(True)
        browse_btn = QPushButton("Select SLP File")
        browse_btn.clicked.connect(self.select_slp_file)
        file_layout.addWidget(self.slp_path_field)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)

        self.video_group = self.create_groupbox("1. Select videos")
        layout.addWidget(self.video_group)

        self.track_group = self.create_groupbox("2. Track names with order")
        layout.addWidget(self.track_group)

        self.keypoint_group = self.create_groupbox("3. Select keypoint for training")
        layout.addWidget(self.keypoint_group)

        button_layout = QHBoxLayout()
        extract_img_btn = QPushButton("Extract Images")
        extract_img_btn.clicked.connect(self.extract_images)

        extract_json_btn = QPushButton("Extract JSON")
        extract_json_btn.clicked.connect(self.extract_json)

        button_layout.addWidget(extract_img_btn)
        button_layout.addWidget(extract_json_btn)
        layout.addLayout(button_layout)

        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)

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

    def select_slp_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select SLP File", "", "SLP Files (*.slp)")
        if not file:
            return
        self.slp_path = file

        self.slp_path_field.setText(file)
        self.status_label.setText("SLP parsing...")

        result = self.parse_slp_info(file)

        self.update_video_list(result["videos"])
        self.update_track_list(result["tracks"])
        self.update_keypoint_list(result["keypoints"])

        self.status_label.setText("SLP parsing complete.")

    def parse_slp_info(self, slp_path):
        labels = sleap.load_file(slp_path)

        video_paths = sorted({lf.video.backend.filename for lf in labels.labeled_frames})
        track_names = set()
        for lf in labels.labeled_frames:
            for inst in lf.instances:
                if hasattr(inst, "track_id"):
                    track_name = str(inst.track_id)
                elif hasattr(inst, "track") and hasattr(inst.track, "name"):
                    track_name = inst.track.name
                else:
                    track_name = f"untracked_{hash(inst)%100000}"
                track_names.add(track_name)

        track_names = sorted(track_names)

        skeleton = labels.skeletons[0]
        keypoint_names = [node.name for node in skeleton.nodes]

        return {
            "videos": video_paths,
            "tracks": track_names,
            "keypoints": keypoint_names
        }

    def update_video_list(self, video_list):
        layout = QVBoxLayout()
        self.video_checkboxes = []

        for video in video_list:
            checkbox = QCheckBox(video)
            checkbox.setChecked(True)
            layout.addWidget(checkbox)
            self.video_checkboxes.append(checkbox)

        self.set_groupbox_layout(self.video_group, layout)

    def update_track_list(self, track_list):
        outer_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Track name"))

        for track in track_list:
            label = QLabel(track)
            left_layout.addWidget(label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("input number"))

        self.track_spinboxes = {}

        for track in track_list:
            line_edit = QLineEdit()
            line_edit.setPlaceholderText("")
            line_edit.setFixedWidth(50) 
            right_layout.addWidget(line_edit)
            self.track_spinboxes[track] = line_edit

        outer_layout.addLayout(left_layout)
        outer_layout.addLayout(right_layout)

        self.set_groupbox_layout(self.track_group, outer_layout)

    def update_keypoint_list(self, keypoint_list):
        layout = QVBoxLayout()
        self.keypoint_checkboxes = []

        for kp in keypoint_list:
            checkbox = QCheckBox(kp)
            checkbox.setChecked(True)
            layout.addWidget(checkbox)
            self.keypoint_checkboxes.append(checkbox)

        self.set_groupbox_layout(self.keypoint_group, layout)

    def extract_json(self):

        if not self.slp_path:
            QMessageBox.warning(self, "Warning", "Select SLP file first.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save COCO JSON", "", "JSON Files (*.json)")
        if not save_path:
            return

        labels = load_file(self.slp_path)
        skeleton = labels.skeletons[0]
        selected_kps = [cb.text() for cb in self.keypoint_checkboxes if cb.isChecked()]
        selected_videos = [cb.text() for cb in self.video_checkboxes if cb.isChecked()]
        track_id_map = {k: v.text() for k, v in self.track_spinboxes.items()}
        track_to_category = {name: int(id_str) for name, id_str in track_id_map.items() if id_str.strip().isdigit()}
        track_name_map = {name: name for name in track_to_category}

        categories = [
            {
                "id": cat_id,
                "name": name,
                "supercategory": "animal",
                "keypoints": selected_kps,
                "skeleton": []
            }
            for name, cat_id in track_to_category.items()
        ]

        images, annotations = [], []
        image_id, annotation_id = 1, 1
        video_name_to_id = {}

        for video in labels.videos:
            if video.backend.filename not in selected_videos:
                continue
            video_path = os.path.basename(video.backend.filename)
            video_name = os.path.splitext(video_path)[0]          
            video_name_to_id[video] = video_name

        for lf in labels.labeled_frames:
            if lf.video not in video_name_to_id:
                continue
            img_name = f"{video_name_to_id[lf.video]}_frame_{lf.frame_idx:05d}.jpg"
            images.append({
                "file_name": img_name,
                "height": lf.video.height,
                "width": lf.video.width,
                "id": image_id
            })

            for inst in lf.instances:
                if hasattr(inst, "track_id"):
                    tn = str(inst.track_id)
                elif hasattr(inst, "track") and hasattr(inst.track, "name"):
                    tn = inst.track.name
                else:
                    tn = f"untracked_{hash(inst)%100000}"

                mapped_name = track_name_map.get(tn)
                if mapped_name is None:
                    continue
                cat_id = track_to_category[mapped_name]

                kp_list, xs, ys = [], [], []
                for idx, node in enumerate(skeleton.nodes):
                    if node.name not in selected_kps:
                        continue
                    if idx >= len(inst.points) or inst.points[idx] is None:
                        kp_list.extend([0.0, 0.0, 0])
                        continue
                    pt = inst.points[idx]
                    x, y = pt.x, pt.y
                    v = 2 if getattr(pt, "visible", True) else 1
                    kp_list.extend([x, y, v])
                    xs.append(x)
                    ys.append(y)

                if not xs or not ys:
                    continue
                x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "keypoints": kp_list,
                    "num_keypoints": sum([1 for v in kp_list[2::3] if v > 0]),
                    "bbox": bbox,
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3]
                })
                annotation_id += 1

            image_id += 1

        data = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        QMessageBox.information(self, "Done", f"JSON save complete: {save_path}")
    
    def extract_images(self):
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return

        labels = sleap.load_file(self.slp_path_field.text())

        selected_videos = [cb.text() for cb in self.video_checkboxes if cb.isChecked()]

        video_caps = {}
        for lf in labels.labeled_frames:
            video_path = lf.video.backend.filename
            if video_path not in selected_videos:
                continue

            if video_path not in video_caps:
                video_caps[video_path] = cv2.VideoCapture(video_path)

            cap = video_caps[video_path]
            cap.set(cv2.CAP_PROP_POS_FRAMES, lf.frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            video_name = os.path.splitext(os.path.basename(video_path))[0]
            img_filename = f"{video_name}_frame_{lf.frame_idx:05d}.jpg"
            img_path = os.path.join(output_dir, img_filename)
            cv2.imwrite(img_path, frame)

        for cap in video_caps.values():
            cap.release()

        QMessageBox.information(self, "Success", f"Images saved to:\n{output_dir}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SlpToCocoOptionsGUI()
    window.show()
    sys.exit(app.exec())
