"""from preprocess.preprocess import PreprocessDialog
from data_extractor.txt_to_csv import TxtToCsvDialog
from pose.pose_estimation import PoseEstimationDialog"""
#TODO : Remove it
from installation_manager import MainInstallDialog

class PipelineController:
    def run_installation(self):
        dialog = MainInstallDialog() 
        dialog.exec()
        pass

    def run_project_manager(self):
        pass

    def run_video_preprocess(self):
        pass

    def run_pose_estimation(self):
        pass

    def data_extract(self):
        pass
