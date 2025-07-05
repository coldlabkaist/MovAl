from PyQt6.QtWidgets import (
    QVBoxLayout, QPushButton,
    QTextEdit, QProgressBar, QLabel,
    QDialog, QLineEdit, QMessageBox, QSpinBox, QFileDialog, QProgressBar
)
from PyQt6.QtCore import Qt
import os
import glob
from .thread import ContourWorker

class ContourDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CCM: Contour Video from Segments")
        self.setFixedSize(800, 500)

        layout = QVBoxLayout()

        self.video_label = QLabel("Input Video:")
        self.video_path = QTextEdit()
        self.video_path.setReadOnly(True)
        self.video_path.setMaximumHeight(120)
        self.video_btn = QPushButton("Load Video")
        self.video_btn.clicked.connect(self.select_video)

        layout.addWidget(self.video_label)
        layout.addWidget(self.video_path)
        layout.addWidget(self.video_btn)

        self.output_label = QLabel("Output Directory:")
        self.output_dir = QLineEdit()
        self.output_dir.setReadOnly(True)
        self.output_btn = QPushButton("Select Output Directory")
        self.output_btn.clicked.connect(self.select_output_directory)

        layout.addWidget(self.output_label)
        layout.addWidget(self.output_dir)
        layout.addWidget(self.output_btn)

        self.fps_label = QLabel("FPS:")
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 120)
        self.fps_spinbox.setValue(30) 

        layout.addWidget(self.fps_label)
        layout.addWidget(self.fps_spinbox)

        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process_video)
        layout.addWidget(self.process_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet("""
        QProgressBar {
            border: 1px solid #bbb;
            border-radius: 5px;
            background-color: #f0f0f0;
            text-align: center;
            height: 20px;
            font-weight: bold;
        }
        QProgressBar::chunk {
            background-color: #3b9cff;
            width: 10px;
            margin: 0.5px;
        }
        """)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.monitor_log = QTextEdit()
        self.monitor_log.setReadOnly(True)
        self.monitor_log.setMinimumHeight(100)
        layout.addWidget(self.monitor_log)

        self.setLayout(layout)

    def select_video(self):
        possible_paths = []

        user_home = os.path.expanduser("~")
        default_workspace = os.path.join(user_home, "Cutie", "workspace")
        possible_paths.append(default_workspace)

        script_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cutie", "workspace"))
        possible_paths.append(script_base)

        for path in possible_paths:
            if os.path.isdir(path):
                workspace_path = path
                break
        else:
            QMessageBox.warning(self, "Workspace Not Found", "Cannot locate Cutie/workspace folder.\nPlease select it manually.")
            workspace_path = QFileDialog.getExistingDirectory(self, "Select Cutie/workspace Folder")
            if not workspace_path:
                return

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Input Videos",
            workspace_path,
            "Video Files (*.mp4 *.avi *.mov)"
        )

        if file_paths:
            existing_paths = self.video_path.toPlainText().strip().split("\n")
            all_paths = existing_paths + file_paths
            unique_paths = list(dict.fromkeys(path for path in all_paths if path.strip()))
            self.video_path.setText("\n".join(unique_paths))

    def select_output_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir.setText(folder)

    def process_video(self):
        video_paths = self.video_path.toPlainText().strip().split("\n")
        output_dir = self.output_dir.text()

        if not video_paths or not all(os.path.exists(v) for v in video_paths):
            QMessageBox.warning(self, "Error", "Please select valid video file(s).")
            return

        if not output_dir or not os.path.isdir(output_dir):
            QMessageBox.warning(self, "Error", "Please select a valid output directory.")
            return

        fps = self.fps_spinbox.value()

        if len(video_paths) == 1:
            video_path = video_paths[0]
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            cutie_workspace = self.detect_workspace(video_path)
            video_folder_path = os.path.join(cutie_workspace, video_name)
            masks_path = os.path.join(video_folder_path, "masks")
            seg_path = os.path.join(video_folder_path, "visualization", "davis")

            masks = sorted(glob.glob(os.path.join(masks_path, "*.png")))
            seg_frames = sorted(glob.glob(os.path.join(seg_path, "*.jpg")))

            if not masks or not seg_frames:
                QMessageBox.critical(self, "Error", f"No masks or segmented frames found in workspace:\n{video_folder_path}")
                return

            if len(masks) != len(seg_frames):
                QMessageBox.warning(self, "Mismatch", f"Warning: Number of masks and segmented frames differ ({len(masks)} vs {len(seg_frames)})")

            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(len(seg_frames))
            self.progress_bar.setValue(0)

            self.worker = ContourWorker(video_name, seg_frames, masks, fps, output_dir)
            self.worker.progress.connect(self.update_progress)
            self.worker.finished.connect(lambda: QMessageBox.information(self, "Done", f"{video_name} saved."))
            self.worker.start()

        else:
            self.multi_video_queue = video_paths
            self.fps = fps
            self.output_dir = output_dir
            self.run_next_video()

    def update_progress(self, value: int):
        self.progress_bar.setValue(value)

    def detect_workspace(self, video_path: str):
        user_home = os.path.expanduser("~")
        return os.path.join(user_home, "Cutie", "workspace")

    def run_next_video(self):
        if not self.multi_video_queue:
            self.status_label.setText("✅ All videos processed.")
            return

        video_path = self.multi_video_queue.pop(0)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.status_label.setText(f"🔄 Now Processing: {video_name}")

        cutie_workspace = self.detect_workspace(video_path)
        video_folder_path = os.path.join(cutie_workspace, video_name)
        masks_path = os.path.join(video_folder_path, "masks")
        seg_path = os.path.join(video_folder_path, "visualization", "davis")

        masks = sorted(glob.glob(os.path.join(masks_path, "*.png")))
        seg_frames = sorted(glob.glob(os.path.join(seg_path, "*.jpg")))

        if not masks or not seg_frames:
            self.monitor_log.append(f"⚠️ Skipping {video_name}: missing masks or frames.")
            self.run_next_video()
            return

        if len(masks) != len(seg_frames):
            self.monitor_log.append(f"⚠️ {video_name}: mask/frame count mismatch.")

        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(len(seg_frames))
        self.progress_bar.setValue(0)

        self.worker = ContourWorker(video_name, seg_frames, masks, self.fps, self.output_dir)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(lambda: self.on_video_finished(video_name))
        self.worker.start()

    def on_video_finished(self, video_name):
        self.monitor_log.append(f"✅ Done: {video_name}")
        self.run_next_video()