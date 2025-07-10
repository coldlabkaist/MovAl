import cv2
import os
from pathlib import Path
from tqdm import tqdm
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QFileDialog, QGraphicsOpacityEffect, QApplication
from .data_loader import DataLoader

class VideoLoader:
    def __init__(self, video_label, slider, frame_label):
        self.video_label = video_label
        self.slider = slider
        self.frame_label = frame_label

        self.video_path = None
        self.total_frames = 0
        self.current_frame = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.frame_dir = None

    def load_video(self, path):
        new_parent = Path(
            str(path.parent).replace(
                f"{os.sep}raw_videos", f"{os.sep}frames"
            )
        )
        path = new_parent / path.stem / "images" #TODO mask 씌워진 데이터로 변경할 것, 예외처리도 고려할 것.
        self.frame_dir = path
        frame_list = [f for f in os.listdir(path) if f.endswith(".jpg")]
        first_frame_path = os.path.join(path, frame_list[0])

        if frame_list[0] and os.path.exists(first_frame_path):
            frame = cv2.imread(first_frame_path)
            if frame is not None:
                h, w = frame.shape[:2]
                DataLoader.set_image_dims(w = w, h = h)

                self.total_frames = len(frame_list)
                self.current_frame = 0
                self.display_frame(frame, reset = True)
                self.slider.setMaximum(self.total_frames - 1)
                self.slider.setValue(0)
                self.frame_label.setText(f"0 / {self.total_frames}")
                print(f"✅ 첫 프레임 표시 완료: {first_frame_path}")
            else:
                print("❌ 첫 프레임 이미지를 불러올 수 없습니다.")
        else:
            print("❌ 첫 프레임 이미지를 불러올 수 없습니다.")

    def pause(self):
        if self.timer.isActive():
            self.timer.stop()

    def display_frame(self, frame, reset = False):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setImage(pixmap, reset = reset)
        self.video_label.current_frame = self.current_frame

        self.slider.setValue(self.current_frame)
        self.frame_label.setText(f"{self.current_frame} / {self.total_frames}")

        csv_points = DataLoader.get_keypoint_coordinates_by_frame(self.current_frame + 1)
        self.video_label.setCSVPoints(csv_points)

    def toggle_playback(self):
        if self.timer.isActive():
            print("stop")
            self.timer.stop()
        else:
            print("start")
            self.timer.start(33)  # 약 30FPS

    def next_frame(self):
        if self.current_frame + 1 < self.total_frames:
            self.current_frame += 1
            frame_path = os.path.join(self.frame_dir, sorted(os.listdir(self.frame_dir))[self.current_frame])
            frame = cv2.imread(frame_path)
            self.display_frame(frame)
        else:
            self.timer.stop()

    def move_to_frame(self, frame_idx):
        if 0 <= frame_idx < self.total_frames:
            self.current_frame = frame_idx
            frame_path = os.path.join(self.frame_dir, sorted(os.listdir(self.frame_dir))[frame_idx])
            frame = cv2.imread(frame_path)
            self.display_frame(frame)
