from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap, QColor, QIcon, QPainter, QPen
from PyQt6.QtWidgets import (
    QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog,
    QSlider, QListWidget, QFrame, QApplication, QDialog, QListWidgetItem, QTreeWidget, QMessageBox,
    QColorDialog, QGridLayout, QTreeWidgetItem, QComboBox, QHeaderView, QStyledItemDelegate,
)
from .gui import UI_LabelaryDialog
from .IO.video_loader import VideoLoader
from .widget.image_label import ClickableImageLabel
from .IO.data_loader import DataLoader
from .IO.save_files import save_modified_data
from .controller.keyboard_controller import KeyboardController
from .controller.mouse_controller import MouseController
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

        self.video_loader = VideoLoader(self, 
                                        self.skeleton_video_viewer, 
                                        self.kpt_list, 
                                        self.frame_slider, 
                                        self.frame_number_label)
        DataLoader.parent = self
        DataLoader.max_animals = self.project.num_animals
        DataLoader.animals_name = self.project.animals_name
        self.skeleton_video_viewer.current_project = project

        self.install_controller()

        self.play_button.clicked.connect(self.play_or_pause)
        self.frame_slider.valueChanged.connect(self.video_loader.move_to_frame)
        self.frame_slider.sliderPressed.connect(self.on_frame_slider_pressed)
        self.frame_slider.sliderReleased.connect(self.on_frame_slider_released)
        self.load_data_button.clicked.connect(self.on_show_clicked)

        self.video_combo.currentIndexChanged.connect(self.update_label_combo)
        self.file_entry_idx = 0
        self.update_label_combo(video_index = self.file_entry_idx)

        self.set_color_combo()
        self.color_combo.currentIndexChanged.connect(self.set_color_mode)

        self.save_button.clicked.connect(self.open_save_dialog)

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
    
    def install_controller(self):
        mouse_controller = MouseController(self.skeleton_video_viewer, self.kpt_list)
        self.mouse_controller = mouse_controller
        self.skeleton_video_viewer.mouse_controller = mouse_controller
        self.skeleton_video_viewer.installEventFilter(mouse_controller)
        self.kpt_list.mouse_controller = mouse_controller

        keyboard_controller = KeyboardController(self, self.video_loader, mouse_controller=mouse_controller)
        self.keyboard_controller = keyboard_controller
        QApplication.instance().installEventFilter(keyboard_controller)

    def load_video_combo(self):
        for video in self.project.get_video_list():
            p = Path(video)
            self.video_combo.addItem(p.name, p)

    def load_mode_combo(self):
        for display_mode in ["images", "davis", "contour"]:
            self.mode_combo.addItem(display_mode)
        self.mode_combo.setCurrentIndex(1)

    def update_label_combo(self, video_index = None, set_text = None):
        if video_index is None:
            video_index = self.file_entry_idx

        file_entry = self.project.files[video_index]
        self.label_combo.clear()

        for csv_path in file_entry.csv:
            p = Path(csv_path)
            self.label_combo.addItem(p.name, p)
        num_csv = len(file_entry.csv)
        for txt_path in file_entry.txt:
            p = Path(txt_path)
            self.label_combo.addItem(p.name, p)
        self.label_combo.addItem("Load inference result", "Load inference result")
        self.label_combo.addItem("Create new label", "Create new label")

        if set_text:
            target_stem = Path(set_text).stem
            default_idx = next(
                (i for i in range(self.label_combo.count())
                if isinstance(self.label_combo.itemData(i), Path)
                    and self.label_combo.itemData(i).stem == target_stem),
                self.label_combo.count() - 1
            )
        elif num_csv > 0:
            default_idx = num_csv - 1
        else:
            default_idx = self.label_combo.count() - 1

        self.label_combo.setCurrentIndex(default_idx)

    def on_show_clicked(self):
        video_path = self.video_combo.currentData(Qt.ItemDataRole.UserRole)
        display_mode = self.mode_combo.currentText()
        if not self.video_loader.load_video(video_path, display_mode):
            return
        self.skeleton_video_viewer.video_loaded = True

        label_name = self.label_combo.currentText()
        if label_name == "Create new label":
            self.create_new_label()
        elif label_name == "Load inference result":
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "Select inference result directory",
                str(self.project.project_dir)
            )
            if not dir_path:
                return
            self.load_txt(dir_path)
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
                    f"Unsupported file/folder:\n{label_path}"
                )
                return

        self.mouse_controller.enable_control = True
        self.is_video_paused = True
        self.update_keypoint_list()
        self.update_csv_points_on_image()

    def load_csv(self, path):
        DataLoader.load_csv_data(path)

    def load_txt(self, path):
        DataLoader.load_txt_data(path)

    def create_new_label(self):
        DataLoader.create_new_data()

    def play_or_pause(self):
        self.is_video_paused = self.video_loader.toggle_playback()
        self.mouse_controller.enable_control = self.is_video_paused

    def on_frame_slider_pressed(self):
        if not self.is_video_paused:
            self.video_loader.toggle_playback()
        self.video_loader.move_to_frame(self.frame_slider.value())

    def on_frame_slider_released(self):
        if not self.is_video_paused:
            self.video_loader.toggle_playback()

    def update_csv_points_on_image(self):
        current_frame = self.video_loader.current_frame
        coords_dict = DataLoader.get_keypoint_coordinates_by_frame(current_frame + 1)
        self.skeleton_video_viewer.setCSVPoints(coords_dict)
        self.kpt_list.update_list_visibility(coords_dict)
        
    def update_keypoint_list(self):
        self.kpt_list.clear()
        if DataLoader.loaded_data is None:
            return
        tracks = list(self.project.animals_name)
        self.kpt_list.build(tracks, DataLoader.kp_order, self.skeleton)

    def set_color_combo(self):
        self.color_combo.addItem("cutie_light")
        self.color_combo.addItem("cutie_dark")
        self.color_combo.addItem("white")
        self.color_combo.addItem("black")
        self.color_combo.setCurrentIndex(0)

    def set_color_mode(self):
        color_mode = self.color_combo.currentText()
        self.skeleton_video_viewer.set_skeleton_color_mode(color_mode)

    def open_save_dialog(self):
        save_modified_data(self)

def run_labelary_with_project(current_project, parent=None):
    app = QApplication.instance() or QApplication(sys.argv)
    dlg = LabelaryDialog(current_project, parent) 
    dlg.exec()  
    return 