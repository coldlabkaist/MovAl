import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from main.gui import MainWindow
from main.pipeline import PipelineController
import subprocess
from packaging import version 
from utils import __version__
from utils.ui_theme import build_app_stylesheet, get_theme_colors

def main():
    print(f"Move Altogether! MovAl version {__version__}")
    
    current_version = __version__
    latest_version = get_latest_git_tag()
    if latest_version and version.parse(latest_version) > version.parse(current_version):
        print(f"Updates available: Current version {current_version}, Latest version {latest_version}\n")
        print(f"Run 'python update_moval.py' to update to the latest release, or 'python update_moval.py {latest_version}' to install a specific tag.\n")

    app = QApplication(sys.argv)
    app.setStyleSheet(build_app_stylesheet())
    colors = get_theme_colors()
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["accent"]))
    palette.setColor(QPalette.ColorRole.Link, QColor(colors["accent"]))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(colors["accent_hover"]))
    app.setPalette(palette)
    controller = PipelineController()
    window = MainWindow(controller=controller)
    window.show()
    sys.exit(app.exec())

def get_latest_git_tag():
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--tags", "origin"],
            capture_output=True,
            text=True,
            check=True
        )
        tags = []
        for line in result.stdout.splitlines():
            if "refs/tags/" in line:
                tag = line.split("refs/tags/")[-1]
                if "^{}" not in tag:
                    tags.append(tag)
        if not tags:
            return None
        latest = sorted(tags, key=version.parse)[-1]
        return latest
    except Exception as e:
        print(f"[Unable to check update!] {e}")
        return None

if __name__ == "__main__":
    main()
