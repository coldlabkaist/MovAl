from installation_manager import MainInstallDialog
from project_manager import ProjectManagerDialog
from utils import TxtToCsvDialog

class PipelineController:
    def run_installation(self):
        dialog = MainInstallDialog() 
        dialog.exec()

    def run_project_manager(self):
        dialog = ProjectManagerDialog() 
        dialog.exec()

    def run_video_preprocess(self):
        pass

    def run_pose_estimation(self):
        pass

    def data_extract(self):
        dialog = TxtToCsvDialog()
        dialog.exec()

    def export_video(self):
        pass