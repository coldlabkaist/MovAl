# video_loader.py (최종 통합 버전)
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QFileDialog, QGraphicsOpacityEffect, QApplication
from .data_loader import DataLoader

class VideoPlayer:
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

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if not file_path:
            print("❌ 비디오 파일을 선택하지 않았습니다.")
            return
        save_dir, first_frame_path = self.extract_and_save_frames(file_path)
        self.frame_dir = save_dir

        if first_frame_path and os.path.exists(first_frame_path):
            frame = cv2.imread(first_frame_path)
            if frame is not None:
                h, w = frame.shape[:2]
                DataLoader.set_image_dims(w, h)

                self.total_frames = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
                self.current_frame = 0
                self.display_frame(frame, reset = True)
                self.slider.setMaximum(self.total_frames - 1)
                self.slider.setValue(0)
                self.frame_label.setText(f"0 / {self.total_frames}")
                print(f"✅ 첫 프레임 표시 완료: {first_frame_path}")
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
            self.timer.stop()
        else:
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

    def extract_and_save_frames(self, file_path):
        video_name = Path(file_path).stem 
        video_dir  = Path(file_path).parent
        base_dir   = video_dir / video_name
        save_dir = base_dir / "images"
        save_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(save_dir.glob(f"{video_name}_*.jpg"))
        if existing:
            first_frame_path = os.path.join(save_dir, existing[0])
            print(f"📁 기존 프레임 사용: {first_frame_path}")
            return str(save_dir), str(existing[0])

        # ③ 총 프레임 수 → 패딩 자릿수 계산 (최소 2자리)
        print(f"📦 프레임 저장 시작: {save_dir}")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("❌ 비디오 열기에 실패했습니다. 경로가 적절히 설정되었는지 확인하세요.")
            return None, None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pad  = max(2, len(str(total))) # INDEX 시스템 변경시 : TODO

        print("📦 해당 작업에 시간이 소요될 수 있습니다.") # TODO : GPU acceleration
        frame_idx = 0
        first_frame_path = None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=frame_count, desc="Extracting frames", unit="frame", dynamic_ncols=True) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_filename = os.path.join(save_dir, f"{video_name}_{frame_idx+1:0{pad}d}.jpg") # INDEX 시스템 변경시 : TODO
                success = cv2.imwrite(str(frame_filename), frame)
                if not success:
                    print(f"❌ 저장 실패: {str(frame_filename)}")
                    break

                if frame_idx == 0:
                    first_frame_path = frame_filename

                frame_idx += 1
                pbar.update(1)

        cap.release()
        print(f"🎉 총 {frame_idx}개 프레임 저장 완료")

        return str(save_dir), str(first_frame_path)

