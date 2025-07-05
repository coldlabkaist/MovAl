from PyQt6.QtWidgets import (
    QMessageBox,
)
from installation_manager import MainInstallDialog
from project_manager import ProjectManagerDialog
from video_preprocess import PreprocessDialog
from labelary import run_labelary
from pose import PoseEstimationDialog
from utils import TxtToCsvDialog

class PipelineController:
    def __init__(self):    
        self.current_project = None
        self.set_main_window_project = None

    def run_installation(self):
        dialog = MainInstallDialog() 
        dialog.exec()

    def run_project_manager(self):
        dialog = ProjectManagerDialog(self.set_main_window_project)
        dialog.exec()

    def run_video_preprocess(self):
        if self.current_project == None:
            QMessageBox.warning(None, "Project not found", "Please Select Project")
            return
        dialog = PreprocessDialog(current_project = self.current_project)
        dialog.exec()

    def run_labelary(self):
        run_labelary()

    def run_pose_estimation(self):
        dialog = PoseEstimationDialog()
        dialog.exec()

    def data_extract(self):
        dialog = TxtToCsvDialog()
        dialog.exec()