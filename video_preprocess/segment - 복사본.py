from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDialog, QLineEdit, QApplication, QMessageBox, QSpinBox, QFileDialog, 
    QListWidget, QListWidgetItem, QProgressBar, 
)
from PyQt6.QtGui import QClipboard
from PyQt6.QtCore import Qt
from utils.video import FrameExtractWorker
from typing import Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
import time 
import cv2
import shutil
import os
        
class CutieDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_project = parent.current_project
        self.frame_dir = os.path.join(self.current_project.project_dir, "frames")

        self.setWindowTitle("Start Cutie Inference")
        self.setFixedSize(400, 400)

        layout = QVBoxLayout()

        info_lbl = QLabel(
            f"<b>Project:</b> {self.current_project.title}  |  "
            f"<b>Animals:</b> {self.current_project.num_animals}  |  "
            f"<b>Skeleton:</b> {self.current_project.skeleton_name}"
        )
        layout.addWidget(info_lbl)

        self.video_list = QListWidget()
        self.video_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        for fe in self.current_project.files:
            fname = os.path.basename(fe.video)
            item = QListWidgetItem(fname)
            item.setData(Qt.ItemDataRole.UserRole, fe.video)
            self.video_list.addItem(item)
        layout.addWidget(self.video_list)
        
        self.progress_bar   = QProgressBar()
        self.progress_label = QLabel("대기 중…")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)

        layout.addStretch()
        self.frame_button = QPushButton("Create image frames")
        self.frame_button.clicked.connect(self.run_create_images)
        layout.addWidget(self.frame_button)
        self.run_button = QPushButton("Run Segmentation")
        self.run_button.clicked.connect(self.run_cutie)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def run_create_images(self):
        if self.video_list.count() == 0:
            QMessageBox.warning(self, "No videos", "프로젝트에 등록된 영상이 없습니다.")
            return

        overwrite_policy = None
        for idx_vid in range(self.video_list.count()):
            video_path  = self.video_list.item(idx_vid).data(Qt.ItemDataRole.UserRole)
            vname       = Path(video_path).stem
            v_frame_dir = os.path.join(self.frame_dir, vname)
            image_dir     = os.path.join(v_frame_dir, "images")
            mask_dir     = os.path.join(v_frame_dir, "masks")

            if os.path.exists(v_frame_dir):
                if overwrite_policy is None:
                    ans = QMessageBox.question(
                        self, "Folder exists",
                        f"{v_frame_dir} 가 이미 존재합니다.\n"
                        "내부 파일을 지우고 다시 생성할까요?",
                        QMessageBox.StandardButton.Yes |
                        QMessageBox.StandardButton.No |
                        QMessageBox.StandardButton.Cancel,
                        QMessageBox.StandardButton.Yes
                    )
                    if ans == QMessageBox.StandardButton.Cancel:
                        return
                    overwrite_policy = (ans == QMessageBox.StandardButton.Yes)

                if overwrite_policy:
                    shutil.rmtree(v_frame_dir)

            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_label.setText(f"{vname} 준비 중…")
            QApplication.processEvents()

            worker = FrameExtractWorker(video_path, image_dir, self)
            worker.prog.connect(
                lambda cur, total, elap, eta, vn=vname: self._on_progress(
                    vn, cur, total, elap, eta
                )
            )
            worker.done.connect(self._on_worker_done)
            worker.start()

            while worker.isRunning():
                QApplication.processEvents()
                time.sleep(0.01)

        self.progress_label.setText("완료")
        self.progress_bar.setValue(self.progress_bar.maximum())
        QMessageBox.information(self, "Done", "프레임 추출이 완료되었습니다.")

    def _on_progress(self, vname: str, cur: int, total: int, elap: float, eta: float):
        if total:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(cur)
            self.progress_label.setText(
                f"{vname}  {cur}/{total}  "
                f"경과 {elap:0.1f}s  남음 {eta:0.1f}s"
            )
        else:  # total=0 → ETA 계산 불가
            self.progress_bar.setRange(0, 0)  # 무한 모드
            self.progress_label.setText(
                f"{vname}  {cur} frames  경과 {elap:0.1f}s"
            )

    def _on_worker_done(self, success: bool, msg: str):
        if not success:
            QMessageBox.critical(self, "추출 실패", msg)
            self.progress_label.setText("오류로 건너뜀")

    def run_cutie(self):
        if not self.video_list.currentItem():
            QMessageBox.warning(self, "Select video", "먼저 세그먼트할 영상을 선택해 주세요.")
            return

        num_objects = self.current_project.num_animals
        video_path = self.video_list.currentItem().data(Qt.ItemDataRole.UserRole)
        vname = Path(video_path).stem
        workspace_dir = os.path.join(self.frame_dir, vname)

        if not os.path.exists(workspace_dir):
            QMessageBox.warning(
                self, "Frames not found",
                "이미지 프레임 폴더가 없습니다.\n먼저 [Create image frames] 를 실행해 주세요."
            )
            return

        project_root = os.path.dirname(os.path.abspath(__file__))
        cutie_dir = os.path.join(os.path.dirname(project_root), "Cutie")

        cmd = [
            "python", "interactive_demo.py",
            "--workspace", workspace_dir,
            "--num_objects", str(num_objects)
        ]

        try:
            subprocess.run(cmd, cwd=cutie_dir, check=True)
            QMessageBox.information(self, "Done", "Cutie segmentation 이 완료되었습니다.")
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Execution Failed", str(e))