from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDialog, QLineEdit, QApplication, QMessageBox, QSpinBox, 
    QListWidget, QListWidgetItem, QTextEdit 
)
from PyQt6.QtGui import QTextCursor, QTextOption
from PyQt6.QtCore import Qt
from pathlib import Path
import subprocess
import time 
import cv2
import shutil
import os
import sys 
        
class CutieDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_project = parent.current_project
        self.frame_dir = os.path.join(self.current_project.project_dir, "frames")
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.cutie_dir = os.path.join(os.path.dirname(self.project_root), "Cutie")

        self.setWindowTitle("Start Cutie Inference")
        self.setFixedSize(500, 600)

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
        self.video_list.setCurrentRow(-1) 
        self.video_list.clearSelection()
        layout.addWidget(self.video_list)

        self.log_label = QLabel("<b>Log</b>")
        layout.addWidget(self.log_label)
        self.log = QTextEdit(readOnly=True)
        self.log.setFixedHeight(400)
        self.log.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.log.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
        self.log.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.log)

        self._progress_block = None

        layout.addStretch()
        self.frame_button = QPushButton("Create image frames (Recommend)")
        self.frame_button.clicked.connect(self.run_create_images)
        layout.addWidget(self.frame_button)
        self.run_button = QPushButton("Run Segmentation")
        self.run_button.clicked.connect(self.run_cutie)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def run_create_images(self):
        if self.video_list.count() == 0:
            QMessageBox.warning(self, "No videos", "No videos in project.")
            return

        self.log.append("Frame extraction started. This task may take some time. Check terminal for detailed information.")

        overwrite_policy = False
        num_objects = self.current_project.num_animals
        for idx_vid in range(self.video_list.count()):
            video_path    = self.video_list.item(idx_vid).data(Qt.ItemDataRole.UserRole)
            video_name    = Path(video_path).stem
            workspace_dir = os.path.join(self.frame_dir, video_name)

            if os.path.exists(workspace_dir):
                if not overwrite_policy:
                    ans = QMessageBox.question(
                        self, "Folder exists",
                        f"Do you want to overwrite frames?",
                        QMessageBox.StandardButton.Yes |
                        QMessageBox.StandardButton.No
                    )
                    if ans == QMessageBox.StandardButton.Yes:
                        overwrite_policy = True
                        self.log.append("Overwrite: True")

                if overwrite_policy:
                    shutil.rmtree(workspace_dir)
                else:
                    continue

            cmd = [
                "python", "interactive_demo.py",
                "--video", video_path,
                "--workspace", workspace_dir,
                "--num_objects", str(num_objects),
                "--workspace_init_only",
            ]
            try:
                subprocess.run(cmd, cwd=self.cutie_dir, check=True)
                self.log.append(f"[OK] {video_name} - frames created.")
            except subprocess.CalledProcessError as e:
                self.log.append(f"[FAIL] {video_name} - {e}")

        QMessageBox.information(self, "Done", "Frame extraction completed.")

    def run_cutie(self):
        items = self.video_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "Select video", "Please select a video to segment first.")
            return

        num_objects = self.current_project.num_animals
        video_path = items[0].data(Qt.ItemDataRole.UserRole)
        video_name    = Path(video_path).stem
        workspace_dir = os.path.join(self.frame_dir, video_name)

        cmd = [
            "python", "interactive_demo.py",
            "--video", video_path,
            "--workspace", workspace_dir,
            "--num_objects", str(num_objects)
        ]

        self.log.append("The cutie dialog will open shortly.")
        try:
            subprocess.run(cmd, cwd=self.cutie_dir, check=True)
            self.log.append(f"[Done] Segmentation for {video_name} completed.")
        except subprocess.CalledProcessError as e:
            self.log.append(f"[FAIL] Cutie execution failed for {video_name} - {e}")
