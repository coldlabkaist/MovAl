import sys
from PyQt6.QtWidgets import QApplication
from .video_annotation import VideoAnnotationGUI
from labelary.controller import KeyboardController

def run_labelary():
    app = QApplication.instance() or QApplication(sys.argv)

    window = VideoAnnotationGUI()

    keyboard_controller = KeyboardController(window.video_player)
    window.installEventFilter(keyboard_controller)

    window.show()

    if app is not QApplication.instance():
        app.exec()

    return
