from PyQt6.QtWidgets import (
    QMessageBox,
)
from installation_manager import MainInstallDialog
from project_manager import ProjectManagerDialog
from video_preprocess import PreprocessDialog
from labelary import run_labelary_with_project
from pose import PoseEstimationDialog
from utils import TxtToCsvDialog, DataConverterDialog

class PipelineController:
    def __init__(self):    
        self.current_project = None
        self.main_window_load_project = None
        self.parent = None

    def run_installation(self):
        dialog = MainInstallDialog(self.parent) 
        dialog.exec()

    def run_project_manager(self):
        dialog = ProjectManagerDialog(self.main_window_load_project, self.parent)
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
        run_labelary_with_project(self.current_project, self.parent)

    def run_pose_estimation(self):
        if self.current_project == None:
            QMessageBox.warning(None, "Project not found", "Please Select a Project")
            return
        dialog = PoseEstimationDialog(self.current_project, self.parent)
        dialog.exec()

    def data_convert(self):
        dialog = DataConverterDialog()
        dialog.exec()

    def data_extract(self):
        dialog = TxtToCsvDialog()
        dialog.exec()