from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap, QColor, QIcon, QPainter, QPen
from PyQt6.QtWidgets import (
    QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QSlider, QListWidget, QFrame, QApplication, QDialog, QListWidgetItem, QTreeWidget,
    QColorDialog, QGridLayout, QTreeWidgetItem, QComboBox, QHeaderView, QStyledItemDelegate,
)
from .gui import UI_LabelaryDialog
from .IO.video_loader import VideoLoader
from .widget.image_label import ClickableImageLabel
from .IO.data_loader import DataLoader
from .IO.save_files import save_modified_data
from .controller.keyboard_controller import KeyboardController
from utils.skeleton import SkeletonModel

from typing import Union, Optional, List
from pathlib import Path
import sys

class LabelaryDialog(QDialog, UI_LabelaryDialog):
    def __init__(self, project, parent= None):
        super().__init__(parent)
        self.setupUi(self)

        self.project = project
        self.load_skeleton_model()
        self.load_video_combo()
        self.load_mode_combo()

        self.video_loader = VideoLoader(self, self.skeleton_video_viewer, self.frame_slider, self.frame_number_label)
        DataLoader.parent = self

        orig = self.kpt_list
        self.gridLayout.replaceWidget(orig, self.kpt_list)
        orig.deleteLater()
        self.skeleton_video_viewer.node_selected.connect(self.kpt_list.highlight)

        self.play_button.clicked.connect(self.video_loader.toggle_playback)
        self.frame_slider.valueChanged.connect(self.video_loader.move_to_frame)
        self.load_data_button.clicked.connect(self.on_show_clicked)

        self.video_combo.currentIndexChanged.connect(self.update_label_combo)
        self.file_entry_idx = 0
        self.update_label_combo(video_index = self.file_entry_idx)

        self.edit_radio.clicked.connect(self.enableCorrectionMode)
        self.show_radio.clicked.connect(self.disableCorrectionMode)
        self.save_button.clicked.connect(lambda: save_modified_data(self))

    def load_skeleton_model(self):
        self.skeleton = SkeletonModel()
        try:
            self.skeleton.load_from_yaml(self.project.skeleton_yaml)
            DataLoader.load_skeleton_info(self.skeleton)
            self.skeleton_video_viewer.load_skeleton_model(self.skeleton)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Skeleton Load Error",
                f"Skeleton settings file not loaded:\n{e}"
            )
            self.accept()

    def load_video_combo(self):
        for video in self.project.get_video_list():
            p = Path(video)
            self.video_combo.addItem(p.name, p)

    def load_mode_combo(self):
        for display_mode in ["images", "davis", "contour"]:
            self.mode_combo.addItem(display_mode)
        self.mode_combo.setCurrentIndex(1)

    def update_label_combo(self, video_index = None, set_text = None):
        if video_index == None:
            video_index = self.file_entry_idx
        target = Path(set_text).stem if set_text else None
        selected_idx = -1

        file_entry = self.project.files[video_index]
        self.label_combo.clear()
        for csv_path in file_entry.csv:
            p = Path(csv_path)
            if target and p.stem == target:
                selected_idx = self.label_combo.count()
            self.label_combo.addItem(p.name, p) 
        for txt_path in file_entry.txt:
            p = Path(txt_path)
            if target and p.stem == target:
                selected_idx = self.label_combo.count()
            self.label_combo.addItem(p.name, p) 
        self.label_combo.addItem("Create new label")

        if selected_idx != -1:
            self.label_combo.setCurrentIndex(selected_idx)
        else:
            self.label_combo.setEditable(True)
            self.label_combo.setPlaceholderText("Select label file")
            self.label_combo.setEditable(False)

    def on_show_clicked(self):
        video_path = self.video_combo.currentData(Qt.ItemDataRole.UserRole)
        display_mode = self.mode_combo.currentText()
        self.video_loader.load_video(video_path, display_mode)

        label_name = self.label_combo.currentText()
        if label_name == "Create new label":
            self.create_new_label()
        else:
            label_path = Path(self.label_combo.currentData(Qt.ItemDataRole.UserRole))
            if label_path.is_dir():
                self.load_txt(label_path)
            elif label_path.suffix.lower() == ".csv":
                self.load_csv(label_path)
            else:
                QMessageBox.warning(
                    self,
                    "Unsupported Format",
                    f"지원하지 않는 파일/폴더입니다:\n{label_path}"
                )
                return

        self.update_keypoint_list()
        self.update_csv_points_on_image()

    def load_csv(self, path):
        DataLoader.load_csv_data(path)
        DataLoader

    def load_txt(self, path):
        DataLoader.load_txt_data(path)

    def create_new_label(self):
        DataLoader.create_new_data()

    def update_csv_points_on_image(self):
        current_frame = self.video_loader.current_frame
        coords_dict = DataLoader.get_keypoint_coordinates_by_frame(current_frame + 1)
        self.skeleton_video_viewer.setCSVPoints(coords_dict)
        
    def update_keypoint_list(self):
        self.kpt_list.clear()
        if DataLoader.loaded_data is None:
            return
        tracks = list(DataLoader.loaded_data["track"].unique())
        self.kpt_list.build(tracks, DataLoader.kp_order, self.skeleton)

    def enableCorrectionMode(self):
        self.skeleton_video_viewer.click_enabled = True
        print("Correction mode enabled: Click events are active.")

    def disableCorrectionMode(self):
        self.skeleton_video_viewer.click_enabled = False
        print("Show mode enabled: Click events are disabled.")

def run_labelary_with_project(current_project, parent=None):
    app = QApplication.instance() or QApplication(sys.argv)

    dlg = LabelaryDialog(current_project, parent) 
    keyboard_controller = KeyboardController(dlg.video_loader)
    app.installEventFilter(keyboard_controller)
    
    dlg.exec()  
    return 