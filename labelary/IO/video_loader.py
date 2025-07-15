import cv2
import os
from pathlib import Path
from tqdm import tqdm
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QFileDialog, QGraphicsOpacityEffect, QApplication, QMessageBox
from .data_loader import DataLoader
import warnings

class VideoLoader:
    def __init__(self, 
                parent, 
                skeleton_video_viewer, 
                kpt_list, 
                frame_slider, 
                frame_number_label, 
                frame_display_mode = "davis"):

        self.parent = parent
        self.skeleton_video_viewer = skeleton_video_viewer
        self.kpt_list = kpt_list
        self.frame_slider = frame_slider
        self.frame_number_label = frame_number_label
        self.frame_display_mode = frame_display_mode

        self.frame_dir = None
        self.current_frame = 0
        self.total_frames = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next_frame)

        self.fps = 30
        self.play_rate = 1.0

    def load_video(self, path, frame_display_mode):
        try:
            cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        except Exception:
            warnings.warn(f"Unable to load video from project: {path}. Video playback fps is fixed to 30.", UserWarning)
            self.fps = 30

        new_parent = Path(
            str(path.parent).replace(
                f"{os.sep}raw_videos", f"{os.sep}frames"
            )
        )
        self.frame_display_mode = frame_display_mode
        path = new_parent / path.stem / self._ensure_display_mode(frame_display_mode)
        self.frame_dir = path
        try:
            frame_list = [f for f in os.listdir(path) if f.endswith(".jpg")]
            first_frame_path = os.path.join(path, frame_list[0])

            if frame_list[0] and os.path.exists(first_frame_path):
                frame = cv2.imread(first_frame_path)
                if frame is not None:
                    self.display_video(frame, len(frame_list))
                    print(f"First frame displayed: {first_frame_path}")
                else:
                    print("Could not load first frame.")
                    return False
            else:
                print("Could not load first frame.")
                return False
        except FileNotFoundError as e:
            QMessageBox.warning( 
                self.parent,
                "File Not Found",
                f"Video file not loaded:\n{e}"
            )
            return False
        except Exception as e:
            QMessageBox.critical(
                self.parent,
                "Error",
                f"An error occurred while loading the video:\n{e}"
            )
            return False
        return True

    def _ensure_display_mode(self, display_mode):
        if display_mode not in ["images", "davis", "contour"]:
            raise RuntimeError("wrong display mode")
            return False
        if display_mode in ["davis", "contour"]:
            return "visualization/" + display_mode
        return display_mode

    def display_video(self, frame, total_frames):
        h, w = frame.shape[:2]
        DataLoader.set_image_dims(w = w, h = h)

        self.total_frames = total_frames
        self.current_frame = 0
        self.display_video_on_viewer(frame, reset = True)
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.frame_slider.setValue(0)

    def display_video_on_viewer(self, frame, reset = False):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.skeleton_video_viewer.setImage(pixmap, reset = reset)
        self.skeleton_video_viewer.current_frame = self.current_frame

        self.frame_slider.setValue(self.current_frame)
        self.frame_number_label.setText(f"{self.current_frame} / {self.total_frames}")

        csv_points = DataLoader.get_keypoint_coordinates_by_frame(self.current_frame + 1)
        self.skeleton_video_viewer.setCSVPoints(csv_points)
        self.kpt_list.update_list_visibility(csv_points)

    def toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            return True
        else:
            self.timer.start(self.fps*self.play_rate)
            return False

    def play_next_frame(self):
        if self.current_frame + 1 < self.total_frames:
            self.current_frame += 1
            frame_path = os.path.join(self.frame_dir, sorted(os.listdir(self.frame_dir))[self.current_frame])
            frame = cv2.imread(frame_path)
            self.display_video_on_viewer(frame)
        else:
            self.timer.stop()

    def move_to_frame(self, frame_idx, force = False):
        if self.timer.isActive() and not force:
            return
        if 0 <= frame_idx < self.total_frames:
            self.current_frame = frame_idx
            frame_path = os.path.join(self.frame_dir, sorted(os.listdir(self.frame_dir))[frame_idx])
            frame = cv2.imread(frame_path)
            self.display_video_on_viewer(frame)
        else:
            self.timer.stop()
