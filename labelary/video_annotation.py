from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap, QColor, QIcon, QPainter, QPen
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QSlider, QListWidget, QFrame, QApplication, QDialog, QListWidgetItem, QTreeWidget,
    QColorDialog, QGridLayout, QTreeWidgetItem, QComboBox, QHeaderView, QStyledItemDelegate,
)
from .main_ui import LabelaryUI
from .video_loader import VideoPlayer
from .widget.image_label import ClickableImageLabel
from .data_loader import LoadDataDialog, DataLoader
from .file_saver import save_modified_csv
from .skeleton import SkeletonManager, SkeletonDialog
from .color import ColorManager, ColorDialog

from typing import Union, Optional, List

def _make_chip(col: QColor, size: int = 12) -> QIcon:
    """단색 사각형 + 얇은 검은 테두리를 아이콘으로 반환."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)

    p = QPainter(pm)
    p.fillRect(0, 0, size, size, col)        # 내부 채움
    pen = QPen(Qt.GlobalColor.black, 1)                  # 1-px black border
    p.setPen(pen)
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return QIcon(pm)

class VideoAnnotationGUI(QWidget, LabelaryUI):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.video_player = VideoPlayer(self.video_label, self.frame_slider, self.frame_label)

        self.play_button.clicked.connect(self.video_player.toggle_playback)
        self.frame_slider.valueChanged.connect(self.video_player.move_to_frame)
        self.load_video_button.clicked.connect(self.load_video)
        self.load_data_button.clicked.connect(self.open_load_data_dialog)
        self.load_data_button.clicked.connect(self.load_data)
        self.edit_radio.clicked.connect(self.enableCorrectionMode)
        self.show_radio.clicked.connect(self.disableCorrectionMode)
        self.add_skeleton_button.clicked.connect(self.open_skeleton_dialog)
        self.color_button.clicked.connect(self.open_color_dialog)
        self.save_button.clicked.connect(lambda: save_modified_csv(self))
        self.reset_button.clicked.connect(self.reset_loaded_data)

    def open_load_data_dialog(self):
        dialog = LoadDataDialog(self)
        dialog.exec()

    def load_video(self):
        self.video_player.load_video()

    def load_data(self):
        self.update_keypoint_list()
        self.update_csv_points_on_image()

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
        for t_idx, tr in enumerate(tracks):
            # ─ Track 헤더 ─────────────────
            col = ColorManager.track_color(tr, t_idx)
            hdr = QListWidgetItem(f"Animal {t_idx+1} ({tr})")
            fnt = hdr.font(); fnt.setBold(True); hdr.setFont(fnt)
            hdr.setIcon(_make_chip(col))                 # ← 아이콘으로 설정# 트랙 헤더
            hdr.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.kpt_list.addItem(hdr)

            # ─ Key-points ────────────────
            for k_idx, kp in enumerate(DataLoader.kp_order):
                c = ColorManager.kp_color(kp)
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
        self.video_player.pause()
        dialog = SkeletonDialog(self)
        dialog.exec()
        self.video_label.update()

    def open_color_dialog(self):
        self.video_player.pause()
        dialog = ColorDialog(self)
        dialog.exec()
        self.update_keypoint_list()
        self.video_label.update()

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
