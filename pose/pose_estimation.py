from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QMessageBox
from pose.prepare_data import DataSplitDialog
from pose.yolo_use import YOLODialog, YoloInferenceDialog
from pose.task_state import pose_execution_state

class PoseEstimationDialog(QDialog):
    def __init__(self, current_project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pose Estimation Options")
        self.setFixedSize(350, 220)
        self.current_project = current_project
        self._train_dialog = None
        self._inference_dialog = None

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Step 1"))
        self.data_btn = QPushButton("Prepare datasets")
        self.data_btn.setFixedHeight(40)
        self.data_btn.clicked.connect(self.open_prepare_data)
        layout.addWidget(self.data_btn)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Step 2"))
        self.train_btn = QPushButton("Train Model")
        self.train_btn.setFixedHeight(40)
        self.train_btn.clicked.connect(self.train_yolo)
        layout.addWidget(self.train_btn)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Step 3"))
        self.inf_btn = QPushButton("Pose estimation")
        self.inf_btn.setFixedHeight(40)
        self.inf_btn.clicked.connect(self.pose_estimation)
        layout.addWidget(self.inf_btn)

        self.setLayout(layout)
        pose_execution_state.busy_changed.connect(self._on_pose_task_busy_changed)
        self._on_pose_task_busy_changed(
            pose_execution_state.is_busy(),
            pose_execution_state.active_task() or "",
        )

    def open_prepare_data(self):
        dialog = DataSplitDialog(self.current_project, self)
        dialog.exec()

    def train_yolo(self):
        if pose_execution_state.is_busy():
            running = pose_execution_state.active_task() or "pose task"
            QMessageBox.information(
                self,
                "Pose task already running",
                f"Another pose task is running ({running}).\n"
                "Please wait until it finishes.",
            )
            return

        if self._train_dialog is not None and self._train_dialog.isVisible():
            self._train_dialog.raise_()
            self._train_dialog.activateWindow()
            return

        dialog = YOLODialog(self.current_project, self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.destroyed.connect(self._on_train_dialog_destroyed)
        self._train_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        
    def pose_estimation(self):
        if pose_execution_state.is_busy():
            running = pose_execution_state.active_task() or "pose task"
            QMessageBox.information(
                self,
                "Pose task already running",
                f"Another pose task is running ({running}).\n"
                "Please wait until it finishes.",
            )
            return

        if self._inference_dialog is not None and self._inference_dialog.isVisible():
            self._inference_dialog.raise_()
            self._inference_dialog.activateWindow()
            return

        dialog = YoloInferenceDialog(self.current_project, self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.destroyed.connect(self._on_inference_dialog_destroyed)
        self._inference_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_pose_task_busy_changed(self, busy: bool, task_name: str):
        self.train_btn.setEnabled(not busy)
        self.inf_btn.setEnabled(not busy)
        if busy:
            self.train_btn.setText(f"Train Model (Busy: {task_name})")
            self.inf_btn.setText(f"Pose estimation (Busy: {task_name})")
        else:
            self.train_btn.setText("Train Model")
            self.inf_btn.setText("Pose estimation")

    def _on_train_dialog_destroyed(self, *args):
        self._train_dialog = None

    def _on_inference_dialog_destroyed(self, *args):
        self._inference_dialog = None
