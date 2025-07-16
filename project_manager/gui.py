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
    QComboBox,
    QScrollArea,
    QWidget,
)
from utils import __version__
from utils.skeleton import SkeletonModel

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
        col1.addSpacing(10)

        self.step1_label = QLabel("<b>Step&nbsp;1.</b> Set number of animal")
        self.step1_spin = QSpinBox()
        self.step1_spin.setRange(1, 16) # Note : Change the range to work on projects with more than 15 animals
        self.step1_spin.setValue(2)
        col1.addWidget(self.step1_label)
        col1.addWidget(self.step1_spin)

        self.instance_area = QScrollArea()
        self.instance_area.setFixedWidth(250) 
        row_h = self.step1_spin.sizeHint().height() + 4
        self.instance_area.setFixedHeight(row_h * 5 + 10)
        self.instance_area.setWidgetResizable(True)
        self.instance_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.instance_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.instance_container = QWidget()
        self.instance_layout = QVBoxLayout(self.instance_container)
        self.instance_layout.setContentsMargins(10, 10, 10, 10)
        self.instance_layout.setSpacing(4)
        self.instance_area.setWidget(self.instance_container)
        col1.addWidget(self.instance_area) 
        self._instance_fields: list[QLineEdit] = []
        self.step1_spin.valueChanged.connect(self._generate_instance_fields)
        self._generate_instance_fields(self.step1_spin.value())
        col1.addSpacing(10)

        self.step2_label = QLabel("<b>Step&nbsp;2.</b> Load videos")
        self.step2_button = QPushButton("Select videos …")
        self.step2_button.clicked.connect(self._on_select_videos)
        col1.addWidget(self.step2_label)
        col1.addWidget(self.step2_button)
        self.step2_check = QCheckBox("Create a copy of the file (Recommended)")
        self.step2_check.setChecked(True)
        col1.addWidget(self.step2_check)
        col1.addSpacing(10)

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
        col1.addSpacing(10)

        self.step4_label = QLabel("<b>Step&nbsp;4.</b> Set skeleton")
        col1.addWidget(self.step4_label)
        self.step4_combo = QComboBox(self)
        self.load_combo_items()
        self.step4_button = QPushButton("Skeleton Setting …")
        self.step4_button.clicked.connect(self._on_select_skeleton)
        col1.addWidget(self.step4_combo)
        col1.addWidget(self.step4_button)
        col1.addSpacing(10)
        
        col1.addSpacing(30)
        col1.addStretch()

        col2 = QVBoxLayout()
        col2.setSpacing(12)
        layout.addLayout(col2, 0)

        self.file_list = _FileListWidget()
        self.file_list.setMinimumWidth(250) 
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

    def _generate_instance_fields(self, count: int):
        old_texts = [e.text().strip() for e in self._instance_fields]
        while self.instance_layout.count():
            child = self.instance_layout.takeAt(0)
            if w := child.widget():
                w.setParent(None)
        self._instance_fields.clear()

        for idx in range(count):
            if idx < len(old_texts) and old_texts[idx]:
                default_name = old_texts[idx]
            else:
                default_name = f"track_{idx}"

            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            label = QLabel(f"Animal {idx+1}:")
            edit  = QLineEdit(default_name)
            row.addWidget(label)
            row.addWidget(edit, 1)

            self.instance_layout.addWidget(row_widget)
            self._instance_fields.append(edit)
        self.instance_layout.addStretch(1)

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
        ok = True
        err_msg = None
        return ok, err_msg

    def _create_project(self):
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "No files",
                                "Please add at least one video file.")
            return

        instance_names = [
            (e.text().strip() or f"track_{i}")
            for i, e in enumerate(self._instance_fields)
        ]
        if len(instance_names) != len(set(instance_names)):
            QMessageBox.warning(
                self,
                "Duplicate animal name",
                f"Set animal names without duplication.",
            )
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
            "frames", "labels", 
            "runs", "raw_videos", "outputs", "prediction"
        ]
        for sd in subdirs:
            _ensure_dir(os.path.join(proj_dir, sd))

        copy_videos = self.step2_check.isChecked()
        copy_labels = True
        project_files: list[dict] = []
        current_vid: dict | None = None
        errors: list[str] = []
        video_stems: set[str] = set()

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
                stem = Path(path_str).stem
                if stem in video_stems:
                    QMessageBox.warning(
                        self,
                        "Duplicate video name",
                        f"The video name “{stem}” is used more than once.\n"
                        "Please ensure all video filenames (without extension) are unique."
                    )
                    try:
                        shutil.rmtree(proj_dir)
                        QMessageBox.information(self, "Deleted",
                                                "Project folder and all contents have been removed.")
                    except Exception as e:
                        QMessageBox.critical(self, "Error",
                                            f"Failed to delete project folder:\n{e}")
                    return
                video_stems.add(stem)
                
                if copy_videos:
                    path_str = _safe_copy(path_str, os.path.join(proj_dir, "raw_videos"))
                current_vid = {"video": path_str, "csv": [], "txt": []}
                project_files.append(current_vid)
                _ensure_dir(os.path.join(proj_dir, "labels", Path(path_str).stem, "csv"))
                _ensure_dir(os.path.join(proj_dir, "labels", Path(path_str).stem, "txt"))
                current_video_name = Path(path_str).stem
            else:
                if current_vid is None:
                    errors.append(f"{path_str} → appears before any video")
                    continue
                if copy_labels:
                    sub = "csv" if ftype == "csv" else "txt"
                    path_str = _safe_copy(path_str, os.path.join(proj_dir, "labels", current_video_name, sub))
                current_vid[ftype].append(path_str)

        if not project_files:
            QMessageBox.critical(self, "Error",
                                "The list must start with at least one video file.")
            return

        project_config_path = os.path.join(proj_dir, "config.yaml")
        config = {
            "moval_version": __version__,
            "project_dir": proj_dir,
            "title": title,
            "num_animals": int(self.step1_spin.value()),
            "animals_name": instance_names,
            "files": project_files,
            "skeleton": self.step4_combo.currentText(),
        }
        try:
            with open(project_config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project YAML:\n{e}")
            return

        training_config_path = os.path.join(proj_dir, "runs", "training_config.yaml")
        training_base_dir = os.path.join(proj_dir, "runs", "dataset")
        _ensure_dir(training_base_dir)
        skeleton_model_dir = os.path.join(os.getcwd(), "preset", "skeleton", self.step4_combo.currentText())
        skeleton_model = SkeletonModel()
        skeleton_model.load_from_yaml(skeleton_model_dir)
        nkpt, flip_idx, kpt_names = skeleton_model.create_training_config()
        config = {
            "train": os.path.join(training_base_dir, "train"),
            "val": os.path.join(training_base_dir, "val"),
            "test": os.path.join(training_base_dir, "test"),
            "nc": len(instance_names),
            "names": {i: n for i, n in enumerate(instance_names)},
            "nkpt": nkpt,
            "kpt_shape": [nkpt, 3],
            "flip_idx" : flip_idx, 
            "kpt_names" : kpt_names
        }
        try:
            with open(training_config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save training YAML:\n{e}")
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

        self.set_main_window_project(path=project_config_path)
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
