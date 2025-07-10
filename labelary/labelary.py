from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap, QColor, QIcon, QPainter, QPen
from PyQt6.QtWidgets import (
    QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QSlider, QListWidget, QFrame, QApplication, QDialog, QListWidgetItem, QTreeWidget,
    QColorDialog, QGridLayout, QTreeWidgetItem, QComboBox, QHeaderView, QStyledItemDelegate,
)
from .gui import UI_LabelaryDialog
from .video_loader import VideoLoader
from .widget.image_label import ClickableImageLabel
from .data_loader import DataLoader
from .file_saver import save_modified_csv
from .controller.keyboard_controller import KeyboardController
from utils.skeleton import SkeletonModel

from typing import Union, Optional, List
from pathlib import Path
import sys

def _make_chip(col: QColor, size: int = 12) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)

    p = QPainter(pm)
    p.fillRect(0, 0, size, size, col)
    pen = QPen(Qt.GlobalColor.black, 1)
    p.setPen(pen)
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return QIcon(pm)

class LabelaryDialog(QDialog, UI_LabelaryDialog):
    def __init__(self, project, parent= None):
        super().__init__(parent)
        self.project = project
        self.setupUi(self)
        self.load_skeleton()

        self.video_player = VideoLoader(self.video_label, self.frame_slider, self.frame_label)
        DataLoader.parent = self

        self.play_button.clicked.connect(self.video_player.toggle_playback)
        self.frame_slider.valueChanged.connect(self.video_player.move_to_frame)
        self.load_data_button.clicked.connect(self.on_show_clicked)
        '''self.load_video_button.clicked.connect(self.load_video)
        self.load_data_button.clicked.connect(self.open_load_data_dialog)'''
        
        self.video_combo.addItems(self.project.get_video_list())
        self.video_combo.currentIndexChanged.connect(self.on_video_selected)
        self.on_video_selected(0)

        self.edit_radio.clicked.connect(self.enableCorrectionMode)
        self.show_radio.clicked.connect(self.disableCorrectionMode)
        """self.add_skeleton_button.clicked.connect(self.open_skeleton_dialog)
        self.color_button.clicked.connect(self.open_color_dialog)"""
        self.save_button.clicked.connect(lambda: save_modified_csv(self))
        #self.reset_button.clicked.connect(self.reset_loaded_data)

    """def open_load_data_dialog(self):
        dialog = LoadDataDialog(self)
        dialog.exec()"""

    '''def load_data(self):
        self.video_player.load_video()
        self.update_keypoint_list()
        self.update_csv_points_on_image()'''

    def load_skeleton(self):
        self.skeleton = SkeletonModel()
        try:
            self.skeleton.load_from_yaml(self.project.skeleton_yaml)
            DataLoader.load_skeleton_info(self.skeleton)
            self.video_label.load_skeleton_model(self.skeleton)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Skeleton Load Error",
                f"스켈레톤 설정 파일을 불러오지 못했습니다:\n{e}"
            )
            self.accept()

    def on_video_selected(self, index):
        file_entry = self.project.files[index]
        self.label_combo.clear()
        for csv_path in file_entry.csv:
            self.label_combo.addItem(csv_path)
        for txt_path in file_entry.txt:
            self.label_combo.addItem(txt_path)
        self.label_combo.addItem("Create new label")

        self.label_combo.setEditable(True)
        self.label_combo.setPlaceholderText("Select label file")
        self.label_combo.setEditable(False)

    def on_show_clicked(self):
        video_path = Path(self.project.project_dir) / self.video_combo.currentText()
        self.video_player.load_video(video_path)

        label_name = self.label_combo.currentText()

        if label_name == "Create new label":
            self.create_new_label()

        else:
            label_path = Path(self.project.project_dir) / label_name
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
        current_frame = self.video_player.current_frame
        coords_dict = DataLoader.get_keypoint_coordinates_by_frame(current_frame + 1)
        self.video_label.setCSVPoints(coords_dict)
        
    def update_keypoint_list(self):
        """QListWidget을 색상 칩(검은 1-px 테두리 포함)과 함께 재구성""" 
        self.kpt_list.clear()
        if DataLoader.data is None:
            return

        tracks = list(DataLoader.data["track"].unique())
        nodes = DataLoader.skeleton_model.nodes
        for t_idx, tr in enumerate(tracks):
            # ─ Track 헤더 ─────────────────
            col = QColor("#FFFFFF") # 0708 TODO
            hdr = QListWidgetItem(f"Animal {t_idx+1} ({tr})")
            fnt = hdr.font(); fnt.setBold(True); hdr.setFont(fnt)
            hdr.setIcon(_make_chip(col))                 # ← 아이콘으로 설정# 트랙 헤더
            hdr.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.kpt_list.addItem(hdr)

            # ─ Key-points ────────────────
            for k_idx, kp in enumerate(DataLoader.kp_order):
                c = nodes[kp].color
                it = QListWidgetItem(f"    {kp}")
                it.setIcon(_make_chip(c))
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.kpt_list.addItem(it)

    def enableCorrectionMode(self):
        self.video_label.click_enabled = True
        print("Correction mode enabled: Click events are active.")

    def disableCorrectionMode(self):
        self.video_label.click_enabled = False
        print("Show mode enabled: Click events are disabled.")

    def open_skeleton_dialog(self):
        pass

    def open_color_dialog(self):
        pass

    def reset_loaded_data(self):
        """영상·프레임·UI를 초기 상태로 되돌린다. (Skeleton 은 유지)"""

        # 1) 비디오 정지 및 VideoPlayer 내부 상태 초기화
        if self.video_player.timer.isActive():
            self.video_player.timer.stop()
        self.video_player.video_path = None
        self.video_player.total_frames = 0
        self.video_player.current_frame = 0
        self.video_player.frame_dir = None
        self.video_player.slider.setMaximum(0)

        # 2) 영상 영역 클리어
        self.video_label.clear_all()        # 이미지·좌표 초기화

        # 3) 부수 UI 초기화
        self.frame_label.setText("0 / 0")
        self.kpt_list.clear()

        print("🔁 영상·UI 초기화")

def run_labelary_with_project(current_project, parent=None):
    app = QApplication.instance() or QApplication(sys.argv)

    dlg = LabelaryDialog(current_project, parent) 
    keyboard_controller = KeyboardController(dlg.video_player)
    app.installEventFilter(keyboard_controller)
    

    dlg.exec()  

    return 