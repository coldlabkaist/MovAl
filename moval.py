import sys
from PyQt6.QtWidgets import QApplication
from main.gui import MainWindow
from main.pipeline import PipelineController

def main():
    app = QApplication(sys.argv)
    controller = PipelineController()
    window = MainWindow(controller=controller)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()