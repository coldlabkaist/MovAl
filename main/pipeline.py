from PyQt6.QtWidgets import (
    QMessageBox,
)
from PyQt6.QtCore import Qt
from installation_manager import MainInstallDialog
from project_manager import ProjectManagerDialog
from video_preprocess import PreprocessDialog
from labelary import run_labelary_with_project
from pose import PoseEstimationDialog
from pose.task_state import pose_execution_state
from utils import TxtToCsvDialog, DataConverterDialog

class PipelineController:
    def __init__(self):    
        self.current_project = None
        self.main_window_load_project = None
        self.parent = None
        self._pose_dialog = None
        self._labelary_dialog = None

    def run_installation(self):
        dialog = MainInstallDialog(self.parent) 
        dialog.exec()

    def run_project_manager(self):
        active_task = (pose_execution_state.active_task() or "").lower()
        if pose_execution_state.is_busy() and active_task == "inference":
            QMessageBox.information(
                self.parent,
                "Inference running",
                "Project Manager cannot be opened while inference is running.",
            )
            return
        dialog = ProjectManagerDialog(
            self.main_window_load_project,
            self.parent,
            self.current_project,
        )
        dialog.exec()

    def run_video_preprocess(self):
        if self.current_project == None:
            QMessageBox.warning(None, "Project not found", "Please Select a Project")
            return
        dialog = PreprocessDialog(self.parent, self.current_project)
        dialog.exec()

    def run_labelary(self):
        if self.current_project == None:
            QMessageBox.warning(None, "Project not found", "Please Select a Project")
            return
        if self._labelary_dialog is not None and self._labelary_dialog.isVisible():
            self._labelary_dialog.raise_()
            self._labelary_dialog.activateWindow()
            return

        dialog = run_labelary_with_project(self.current_project, self.parent)
        self._labelary_dialog = dialog
        if self._labelary_dialog is not None:
            self._labelary_dialog.destroyed.connect(self._on_labelary_dialog_destroyed)

    def run_pose_estimation(self):
        if self.current_project == None:
            QMessageBox.warning(None, "Project not found", "Please Select a Project")
            return
        if self._pose_dialog is not None and self._pose_dialog.isVisible():
            self._pose_dialog.raise_()
            self._pose_dialog.activateWindow()
            return

        dialog = PoseEstimationDialog(self.current_project, self.parent)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.destroyed.connect(self._on_pose_dialog_destroyed)
        self._pose_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def data_convert(self):
        dialog = DataConverterDialog()
        dialog.exec()

    def data_extract(self):
        dialog = TxtToCsvDialog(current_project=self.current_project)
        dialog.exec()

    def _on_pose_dialog_destroyed(self, *args):
        self._pose_dialog = None

    def _on_labelary_dialog_destroyed(self, *args):
        self._labelary_dialog = None
