from __future__ import annotations
from pathlib import Path
import sys
import yaml
import os
import shutil 
import threading
from typing import Callable  
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeyEvent, QFont, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QCheckBox,
    QLineEdit,
    QComboBox
)

class _FileListWidget(QListWidget):

    def __init__(self, parent: QDialog | None = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)

        delete_action = QAction("Delete\t(Del)", self)
        delete_action.setShortcut("Delete")
        delete_action.triggered.connect(self._delete_selected)
        self.addAction(delete_action)

        self.setStyleSheet(
            """
            QListWidget::item:hover                 { background: lightgray; color: black;}
            QListWidget::item:selected              { background: lightgray; color: black;}
            QListWidget::item:selected:!active      { background: lightgray; color: black;}
            QListWidget::item:selected:disabled     { background: lightgray; color: black;}
            """
        )

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Delete:
            self._delete_selected()
        else:
            super().keyPressEvent(event)

    def _delete_selected(self) -> None:
        for item in list(self.selectedItems()):
            self.takeItem(self.row(item))

    def _style_item(self, item: QListWidgetItem, ftype: str) -> None:
        if ftype == "vid":
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            item.setBackground(QColor("#fff9c4"))

class ProjectManagerDialog(QDialog):
    def __init__(
        self, 
        set_main_window_project: Optional[Callable[[str], None]] = None,
        parent: QDialog | None = None,
    ) -> None:

        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.resize(800, 800)

        self._preset_dir = os.path.join(os.getcwd(), "preset", "skeleton")
        os.makedirs(self._preset_dir, exist_ok=True)
        self.set_main_window_project = set_main_window_project

        layout = QHBoxLayout(self)

        col1 = QVBoxLayout()
        col1.setSpacing(12)
        layout.addLayout(col1, 0)

        self.title_label = QLabel("<b>Project&nbsp;Title</b>")
        self.title_edit  = QLineEdit()
        self.title_edit.setPlaceholderText("e.g. CoLD_GH_250718")
        self.title_edit.setFixedWidth(250) 
        col1.addWidget(self.title_label)
        col1.addWidget(self.title_edit)
        col1.addSpacing(20)

        self.step1_label = QLabel("<b>Step&nbsp;1.</b> Set number of animal")
        self.step1_spin = QSpinBox()
        self.step1_spin.setRange(1, 100)
        self.step1_spin.setValue(2)
        col1.addWidget(self.step1_label)
        col1.addWidget(self.step1_spin)
        col1.addSpacing(20)

        self.step2_label = QLabel("<b>Step&nbsp;2.</b> Load videos")
        self.step2_button = QPushButton("Select videos …")
        self.step2_button.clicked.connect(self._on_select_videos)
        col1.addWidget(self.step2_label)
        col1.addWidget(self.step2_button)
        self.step2_check = QCheckBox("Create a copy of the file (Recommended)")
        self.step2_check.setChecked(True)
        col1.addWidget(self.step2_check)
        col1.addSpacing(20)

        self.step3_label = QLabel("<b>Step&nbsp;3.</b> Load txt/csv (Optional)")
        self.step3_buttons_row = QHBoxLayout()
        self.step3_button_csv = QPushButton("Load CSV files …")
        self.step3_button_txt = QPushButton("Load TXT folders …")
        self.step3_button_csv.clicked.connect(self._on_select_csv_files)
        self.step3_button_txt.clicked.connect(self._on_select_txt_folders)
        self.step3_buttons_row.addWidget(self.step3_button_csv)
        self.step3_buttons_row.addWidget(self.step3_button_txt)
        col1.addWidget(self.step3_label)
        col1.addLayout(self.step3_buttons_row)
        self.step3_check = QCheckBox("Create a copy of the file (Recommended)")
        self.step3_check.setChecked(True)
        col1.addWidget(self.step3_check)
        col1.addSpacing(20)

        self.step4_label = QLabel("<b>Step&nbsp;4.</b> Set skeleton")
        col1.addWidget(self.step4_label)
        self.step4_combo = QComboBox(self)
        self.load_combo_items()
        self.step4_button = QPushButton("Skeleton Setting …")
        self.step4_button.clicked.connect(self._on_select_skeleton)
        col1.addWidget(self.step4_combo)
        col1.addWidget(self.step4_button)
        col1.addSpacing(20)
        
        col1.addStretch()

        col2 = QVBoxLayout()
        col2.setSpacing(12)
        layout.addLayout(col2, 0)

        self.file_list = _FileListWidget()
        col2.addWidget(self.file_list)

        self.list_buttons_row = QHBoxLayout()
        self.list_button_sort = QPushButton("Auto Sort by Filename")
        self.list_button_reset = QPushButton("Reset List")
        self.list_button_sort.clicked.connect(self._on_list_sort)
        self.list_button_reset.clicked.connect(self._on_list_reset)
        self.list_buttons_row.addWidget(self.list_button_sort)
        self.list_buttons_row.addWidget(self.list_button_reset)
        col1.addLayout(self.list_buttons_row)

        self.create_label = QLabel(
            "<b>Note:</b> Before you click the button below,<br>"
            "make sure all videos are included and that<br>"
            "each CSV or TXT file is placed properly."
        )
        col1.addWidget(self.create_label)

        self.create_button = QPushButton("Create Project")
        self.create_button.clicked.connect(self._create_project)
        col1.addWidget(self.create_button)

    def _on_select_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "", # start dir
            "Videos (*.mp4 *.avi *.mov *.mkv)",
        )
        self._append_files(files, "vid")

    def _on_select_csv_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV Files",
            "", # start dir
            "CSV (*.csv)",
        )
        self._append_files(files, "csv")

    def _on_select_txt_folders(self):
        while True:
            folder = QFileDialog.getExistingDirectory(self, "Select TXT Folder")
            if not folder:
                break
            self._append_files([folder], "txt")
            if QMessageBox.question(
                self,
                "Add another folder?",
                "Do you want to add another TXT folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) == QMessageBox.StandardButton.No:
                break

    def _append_files(self, paths: list[str], filetype: str):
        for raw in paths:
            if not raw:
                continue

            label = f"[{filetype}] {raw}"

            if label in (self.file_list.item(i).text() for i in range(self.file_list.count())):
                continue

            item = QListWidgetItem(label, self.file_list)
            item.setData(Qt.ItemDataRole.UserRole, filetype)
            self.file_list._style_item(item, filetype)

    def load_combo_items(self, selected: str | None = None):
        self.step4_combo.clear()
        os.makedirs(self._preset_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(self._preset_dir) if f.endswith(".yaml"))
        self.step4_combo.addItems(files)

        if selected == None:
            self.step4_combo.setEditable(True)
            self.step4_combo.setPlaceholderText("Select config file")
            self.step4_combo.setEditable(False)
        else:
            idx = self.step4_combo.findText(selected, Qt.MatchFlag.MatchExactly)
            self.step4_combo.setCurrentIndex(idx)

    def _on_select_skeleton(self):
        from project_manager import SkeletonManagerDialog
        dialog = SkeletonManagerDialog(self) 
        dialog.exec()
            
    def _on_list_sort(self):
        entries: list[tuple[str, str]] = [
            (self.file_list.item(i).text(),
             self.file_list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self.file_list.count())
        ]

        def sort_key(entry: tuple[str, str]):
            label, ftype = entry 
            path_str = label.split("] ", 1)[1] if "] " in label else label
            p = Path(path_str)

            stem = p.stem.lower() 
            video_priority = 0 if ftype == "vid" else 1  
            ext = p.suffix.lower()

            return (stem, video_priority, ext)

        entries.sort(key=sort_key)

        self.file_list.clear()
        for p, ftype in entries:
            item = QListWidgetItem(p, self.file_list)
            item.setData(Qt.ItemDataRole.UserRole, ftype)
            self.file_list._style_item(item, ftype)

    def _on_list_reset(self):
        self.file_list.clear()

    def check_path_validity(self, path: str):
        # TODO
        ok = True
        err_msg = None
        return ok, err_msg

    def _create_project(self):
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "No files",
                                "Please add at least one video / txt / csv file.")
            return
        if not (title := self.title_edit.text().strip()):
            QMessageBox.warning(self, "No title", "Please enter a project title.")
            return

        root_dir = QFileDialog.getExistingDirectory(self, "Select project root folder")
        if not root_dir:
            return

        proj_dir = os.path.join(root_dir, title)
        if os.path.exists(proj_dir):
            QMessageBox.warning(
                self,
                "Duplicate project",
                f"A folder named “{title}” already exists in the selected directory.\n"
                "Please choose a different title or location.",
            )
            return

        _ensure_dir(proj_dir)

        subdirs = [
            "frames", "labels/csv", "labels/txt",
            "runs", "raw_videos", "outputs"
        ]   
        for sd in subdirs:
            _ensure_dir(os.path.join(proj_dir, sd))

        copy_videos = self.step2_check.isChecked()
        copy_labels = self.step3_check.isChecked() 
        project_files: list[dict] = []
        current_vid: dict | None = None
        errors: list[str] = []

        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            ftype = item.data(Qt.ItemDataRole.UserRole) 
            label = item.text()
            path_str = label[6:] if label.startswith("[") else label

            ok, err = self.check_path_validity(path_str)
            if not ok:
                errors.append(f"{path_str} → {err}")
                continue

            if ftype == "vid":
                if copy_videos:
                    path_str = _safe_copy(path_str, os.path.join(proj_dir, "raw_videos"))
                current_vid = {"video": path_str, "csv": [], "txt": []}
                project_files.append(current_vid)
            else:
                if current_vid is None:
                    errors.append(f"{path_str} → appears before any video")
                    continue
                if copy_labels:
                    sub = "csv" if ftype == "csv" else "txt"
                    path_str = _safe_copy(path_str, os.path.join(proj_dir, f"labels/{sub}"))
                current_vid[ftype].append(path_str)

        if not project_files:
            QMessageBox.critical(self, "Error",
                                "The list must start with at least one video file.")
            return

        config_path = os.path.join(proj_dir, "config.yaml")
        config = {
            "project_dir": proj_dir,
            "title": title,
            "num_animals": int(self.step1_spin.value()),
            "files": project_files,
            "skeleton": self.step4_combo.currentText(),
        }

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save YAML:\n{e}")
            return

        if errors:
            QMessageBox.warning(
                self,
                "Saved with warnings",
                "Project saved (config.yaml), but some paths were skipped:\n\n" + "\n".join(errors),
            )
        else:
            QMessageBox.information(self, "Done",
                                    f"Project folder created:\n{proj_dir}\n\nconfig.yaml saved.")

        self.set_main_window_project(path=config_path)
        self.accept()

def _ensure_dir(path: str | Path):
    os.makedirs(path, exist_ok=True)

def _safe_copy(src: str, dst_dir: str, workers: int = 8):
    if os.path.isdir(src):
        txt_files = [
            str(Path(src, f)) for f in os.listdir(src)
            if f.lower().endswith(".txt")
        ]
        _ensure_dir(dst_dir)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_copy_file_rename, f, dst_dir) for f in txt_files]
            for fu in as_completed(futures):
                fu.result()
        return dst_dir
    else:
        return _copy_file_rename(src, dst_dir)

_copy_lock = threading.Lock() 
def _copy_file_rename(src_file: str, dst_dir: str) :
    _ensure_dir(dst_dir)

    with _copy_lock:
        base = os.path.basename(src_file)
        name, ext = os.path.splitext(base)
        candidate = base
        idx = 2
        while os.path.exists(os.path.join(dst_dir, candidate)):
            candidate = f"{name} ({idx}){ext}"
            idx += 1
        dst_path = os.path.join(dst_dir, candidate)
        shutil.copy2(src_file, dst_path)

    return dst_path

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)  
        owns_app = True  
    dlg = ProjectManagerDialog()
    dlg.show() 

    if owns_app:
        sys.exit(app.exec())