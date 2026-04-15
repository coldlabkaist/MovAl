from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QKeyEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .skeleton import SkeletonManagerDialog
from utils import __version__
from utils.project import ProjectInformation
from utils.skeleton import SkeletonModel

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FileListWidget(QListWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self.setStyleSheet(
            """
            QListWidget::item:hover { background: lightgray; color: black; }
            QListWidget::item:selected { background: lightgray; color: black; }
            QListWidget::item:selected:!active { background: lightgray; color: black; }
            """
        )

        delete_action = QAction("Delete\t(Del)", self)
        delete_action.setShortcut("Delete")
        delete_action.triggered.connect(self._delete_selected)
        self.addAction(delete_action)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Delete:
            self._delete_selected()
            return
        super().keyPressEvent(event)

    def _delete_selected(self) -> None:
        for item in list(self.selectedItems()):
            self.takeItem(self.row(item))

    def style_item(self, item: QListWidgetItem, file_type: str) -> None:
        if file_type == "vid":
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            item.setBackground(QColor("#fff9c4"))


class _CreateProjectTab(QWidget):
    def __init__(self, dialog: "ProjectManagerDialog") -> None:
        super().__init__(dialog)
        self.dialog = dialog
        self._preset_dir = REPO_ROOT / "preset" / "skeleton"
        self._preset_dir.mkdir(parents=True, exist_ok=True)

        layout = QHBoxLayout(self)

        left_col = QVBoxLayout()
        left_col.setSpacing(12)
        layout.addLayout(left_col, 0)

        self.title_label = QLabel("<b>Project Title</b>")
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("e.g. CoLD_GH_250718")
        self.title_edit.setFixedWidth(260)
        left_col.addWidget(self.title_label)
        left_col.addWidget(self.title_edit)

        self.step1_label = QLabel("<b>Step 1.</b> Set number of animals")
        self.step1_spin = QSpinBox()
        self.step1_spin.setRange(1, 16)
        self.step1_spin.setValue(2)
        left_col.addWidget(self.step1_label)
        left_col.addWidget(self.step1_spin)

        self.instance_area = QScrollArea()
        self.instance_area.setFixedWidth(260)
        self.instance_area.setWidgetResizable(True)
        self.instance_container = QWidget()
        self.instance_layout = QVBoxLayout(self.instance_container)
        self.instance_layout.setContentsMargins(10, 10, 10, 10)
        self.instance_layout.setSpacing(4)
        self.instance_area.setWidget(self.instance_container)
        left_col.addWidget(self.instance_area)

        self._instance_fields: list[QLineEdit] = []
        self.step1_spin.valueChanged.connect(self._generate_instance_fields)
        self._generate_instance_fields(self.step1_spin.value())

        self.step2_label = QLabel("<b>Step 2.</b> Load videos")
        self.step2_button = QPushButton("Select Videos...")
        self.step2_button.clicked.connect(self._on_select_videos)
        self.step2_check = QCheckBox("Copy videos into project raw_videos (Recommended)")
        self.step2_check.setChecked(True)
        left_col.addWidget(self.step2_label)
        left_col.addWidget(self.step2_button)
        left_col.addWidget(self.step2_check)

        self.step3_label = QLabel("<b>Step 3.</b> Load labels (Optional)")
        self.step3_button_csv = QPushButton("Load CSV Files...")
        self.step3_button_txt = QPushButton("Load TXT Folders...")
        self.step3_button_csv.clicked.connect(self._on_select_csv_files)
        self.step3_button_txt.clicked.connect(self._on_select_txt_folders)
        step3_buttons = QHBoxLayout()
        step3_buttons.addWidget(self.step3_button_csv)
        step3_buttons.addWidget(self.step3_button_txt)
        left_col.addWidget(self.step3_label)
        left_col.addLayout(step3_buttons)

        self.step4_label = QLabel("<b>Step 4.</b> Set skeleton")
        self.step4_combo = QComboBox(self)
        self.step4_button = QPushButton("Skeleton Setting...")
        self.step4_button.clicked.connect(self._open_skeleton_manager)
        left_col.addWidget(self.step4_label)
        left_col.addWidget(self.step4_combo)
        left_col.addWidget(self.step4_button)
        self._load_skeleton_items()

        left_col.addStretch(1)

        note_label = QLabel(
            "<b>Note:</b> Put each CSV or TXT item immediately after its target video. "
            "CSV and TXT imports are copied into the standard labels folders."
        )
        note_label.setWordWrap(True)
        left_col.addWidget(note_label)

        self.create_button = QPushButton("Create Project")
        self.create_button.clicked.connect(self._create_project)
        left_col.addWidget(self.create_button)

        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        layout.addLayout(right_col, 1)

        self.file_list = _FileListWidget()
        self.file_list.setMinimumWidth(320)
        right_col.addWidget(self.file_list)

        list_buttons = QHBoxLayout()
        self.list_button_sort = QPushButton("Auto Sort by Filename")
        self.list_button_reset = QPushButton("Reset List")
        self.list_button_sort.clicked.connect(self._on_list_sort)
        self.list_button_reset.clicked.connect(self.file_list.clear)
        list_buttons.addWidget(self.list_button_sort)
        list_buttons.addWidget(self.list_button_reset)
        right_col.addLayout(list_buttons)

    def _generate_instance_fields(self, count: int) -> None:
        old_texts = [field.text().strip() for field in self._instance_fields]
        while self.instance_layout.count():
            child = self.instance_layout.takeAt(0)
            if widget := child.widget():
                widget.setParent(None)
        self._instance_fields.clear()

        for index in range(count):
            default_name = old_texts[index] if index < len(old_texts) and old_texts[index] else f"track_{index}"
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_layout.addWidget(QLabel(f"Animal {index + 1}:"))
            field = QLineEdit(default_name)
            row_layout.addWidget(field, 1)
            self.instance_layout.addWidget(row_widget)
            self._instance_fields.append(field)

        self.instance_layout.addStretch(1)

    def _append_files(self, paths: list[str], file_type: str) -> None:
        existing = {self.file_list.item(index).text() for index in range(self.file_list.count())}
        for raw_path in paths:
            if not raw_path:
                continue
            label = f"[{file_type}] {raw_path}"
            if label in existing:
                continue
            item = QListWidgetItem(label, self.file_list)
            item.setData(Qt.ItemDataRole.UserRole, file_type)
            self.file_list.style_item(item, file_type)

    def _on_select_videos(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv)",
        )
        self._append_files(files, "vid")

    def _on_select_csv_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV Files",
            "",
            "CSV Files (*.csv)",
        )
        self._append_files(files, "csv")

    def _on_select_txt_folders(self) -> None:
        while True:
            folder = QFileDialog.getExistingDirectory(self, "Select TXT Folder")
            if not folder:
                break
            self._append_files([folder], "txt")
            reply = QMessageBox.question(
                self,
                "Add another folder?",
                "Do you want to add another TXT folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                break

    def _load_skeleton_items(self, selected: Optional[str] = None) -> None:
        self.step4_combo.clear()
        skeleton_files = sorted(path.name for path in self._preset_dir.glob("*.yaml"))
        self.step4_combo.addItems(skeleton_files)
        if selected:
            index = self.step4_combo.findText(selected, Qt.MatchFlag.MatchExactly)
            if index >= 0:
                self.step4_combo.setCurrentIndex(index)

    def _open_skeleton_manager(self) -> None:
        dialog = SkeletonManagerDialog(self)
        dialog.exec()
        self._load_skeleton_items(self.step4_combo.currentText())

    def _on_list_sort(self) -> None:
        entries: list[tuple[str, str]] = [
            (
                self.file_list.item(index).text(),
                self.file_list.item(index).data(Qt.ItemDataRole.UserRole),
            )
            for index in range(self.file_list.count())
        ]

        def sort_key(entry: tuple[str, str]) -> tuple[str, int, str]:
            label, file_type = entry
            path_text = label.split("] ", 1)[1] if "] " in label else label
            path = Path(path_text)
            return (path.stem.lower(), 0 if file_type == "vid" else 1, path.suffix.lower())

        entries.sort(key=sort_key)
        self.file_list.clear()
        for label, file_type in entries:
            item = QListWidgetItem(label, self.file_list)
            item.setData(Qt.ItemDataRole.UserRole, file_type)
            self.file_list.style_item(item, file_type)

    def _validate_inputs(self) -> tuple[str, list[str], str] | None:
        title = self.title_edit.text().strip()
        if not title:
            QMessageBox.warning(self, "No title", "Please enter a project title.")
            return None

        instance_names = [
            field.text().strip() or f"track_{index}"
            for index, field in enumerate(self._instance_fields)
        ]
        if len(instance_names) != len(set(instance_names)):
            QMessageBox.warning(self, "Duplicate animal name", "Animal names must be unique.")
            return None

        skeleton_name = self.step4_combo.currentText().strip()
        if not skeleton_name:
            QMessageBox.warning(self, "No skeleton", "Please select a skeleton preset.")
            return None

        if self.file_list.count() == 0:
            QMessageBox.warning(self, "No files", "Please add at least one video file.")
            return None

        return title, instance_names, skeleton_name

    def _create_project(self) -> None:
        validated = self._validate_inputs()
        if validated is None:
            return

        title, instance_names, skeleton_name = validated
        root_dir = QFileDialog.getExistingDirectory(self, "Select project root folder")
        if not root_dir:
            return

        project_dir = Path(root_dir) / title
        if project_dir.exists():
            QMessageBox.warning(
                self,
                "Duplicate project",
                f"A folder named '{title}' already exists in the selected location.",
            )
            return

        project = ProjectInformation.create(
            project_dir,
            title=title,
            num_animals=int(self.step1_spin.value()),
            animals_name=instance_names,
            skeleton_name=skeleton_name,
            moval_version=__version__,
        )

        current_video_name: Optional[str] = None
        warnings: list[str] = []
        copy_videos = self.step2_check.isChecked()

        try:
            for index in range(self.file_list.count()):
                item = self.file_list.item(index)
                file_type = item.data(Qt.ItemDataRole.UserRole)
                text = item.text()
                raw_path = text.split("] ", 1)[1] if "] " in text else text

                if file_type == "vid":
                    record = project.add_video(raw_path, copy_into_project=copy_videos, save=False)
                    current_video_name = record.name
                    continue

                if current_video_name is None:
                    warnings.append(f"{raw_path} appears before any video and was skipped.")
                    continue

                if file_type == "csv":
                    project.import_csv_files(current_video_name, [raw_path])
                elif file_type == "txt":
                    project.import_txt_directory(current_video_name, raw_path)

            if not project.video_records:
                raise ValueError("The list must include at least one video.")

            project.save()
            self._write_training_config(project)
        except Exception as err:
            if project_dir.exists():
                shutil.rmtree(project_dir)
            QMessageBox.critical(self, "Create Project Failed", str(err))
            return

        message = (
            f"Project folder created:\n{project_dir}\n\n"
            f"Main project file:\n{project.project_file}"
        )
        if warnings:
            QMessageBox.warning(self, "Created with warnings", message + "\n\n" + "\n".join(warnings))
        else:
            QMessageBox.information(self, "Done", message)

        self.dialog.load_project(project.project_file)
        self.dialog.accept()

    def _write_training_config(self, project: ProjectInformation) -> None:
        training_base_dir = project.project_dir_path / "runs" / "dataset"
        training_base_dir.mkdir(parents=True, exist_ok=True)

        skeleton_model = SkeletonModel()
        skeleton_model.load_from_yaml(project.skeleton_yaml)
        nkpt, flip_idx, kpt_names = skeleton_model.create_training_config()

        config = {
            "train": (training_base_dir / "train").as_posix(),
            "val": (training_base_dir / "val").as_posix(),
            "test": (training_base_dir / "test").as_posix(),
            "nc": len(project.animals_name),
            "names": {index: name for index, name in enumerate(project.animals_name)},
            "nkpt": nkpt,
            "kpt_shape": [nkpt, 3],
            "flip_idx": flip_idx,
            "kpt_names": kpt_names,
        }

        import yaml

        target_path = project.project_dir_path / "runs" / "training_config.yaml"
        with target_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


class _ManageProjectTab(QWidget):
    def __init__(self, dialog: "ProjectManagerDialog", project: Optional[ProjectInformation]) -> None:
        super().__init__(dialog)
        self.dialog = dialog
        self.project = project

        layout = QVBoxLayout(self)

        project_row = QHBoxLayout()
        self.open_project_button = QPushButton("Open Project...")
        self.use_current_project_button = QPushButton("Use Current Project")
        self.compress_button = QPushButton("Compress Project")
        self.refresh_button = QPushButton("Refresh")
        self.open_project_button.clicked.connect(self.dialog.open_project_from_picker)
        self.use_current_project_button.clicked.connect(self.dialog.use_parent_current_project)
        self.compress_button.clicked.connect(self._compress_project)
        self.refresh_button.clicked.connect(self.refresh_views)
        project_row.addWidget(self.open_project_button)
        project_row.addWidget(self.use_current_project_button)
        project_row.addStretch(1)
        project_row.addWidget(self.compress_button)
        project_row.addWidget(self.refresh_button)
        layout.addLayout(project_row)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        video_controls = QHBoxLayout()
        self.copy_added_videos_check = QCheckBox("Copy added videos into raw_videos")
        self.copy_added_videos_check.setChecked(True)
        self.add_video_button = QPushButton("Add Videos...")
        self.remove_video_button = QPushButton("Remove Selected Videos")
        self.add_video_button.clicked.connect(self._add_videos)
        self.remove_video_button.clicked.connect(self._remove_selected_videos)
        video_controls.addWidget(self.copy_added_videos_check)
        video_controls.addStretch(1)
        video_controls.addWidget(self.add_video_button)
        video_controls.addWidget(self.remove_video_button)
        layout.addLayout(video_controls)

        self.video_list = QListWidget()
        self.video_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.video_list.itemSelectionChanged.connect(self._sync_video_selection_to_csv_combo)
        layout.addWidget(self.video_list)

        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("Manage CSVs for:"))
        self.csv_video_combo = QComboBox()
        self.csv_video_combo.currentIndexChanged.connect(self.refresh_csv_list)
        csv_row.addWidget(self.csv_video_combo, 1)
        self.add_csv_button = QPushButton("Add CSV...")
        self.remove_csv_button = QPushButton("Remove Selected CSVs")
        self.add_csv_button.clicked.connect(self._add_csvs)
        self.remove_csv_button.clicked.connect(self._remove_selected_csvs)
        csv_row.addWidget(self.add_csv_button)
        csv_row.addWidget(self.remove_csv_button)
        layout.addLayout(csv_row)

        self.csv_list = QListWidget()
        self.csv_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.csv_list)

        self.txt_label = QLabel()
        self.txt_label.setWordWrap(True)
        layout.addWidget(self.txt_label)

        self.note_label = QLabel(
            "Project JSON stores the video registry and UI state. "
            "CSV/TXT files are discovered from labels/<video>/ at runtime."
        )
        self.note_label.setWordWrap(True)
        layout.addWidget(self.note_label)

        self._managed_widgets = [
            self.copy_added_videos_check,
            self.add_video_button,
            self.remove_video_button,
            self.compress_button,
            self.refresh_button,
            self.video_list,
            self.csv_video_combo,
            self.add_csv_button,
            self.remove_csv_button,
            self.csv_list,
        ]
        self.set_project(project)

    def set_project(self, project: Optional[ProjectInformation]) -> None:
        self.project = project
        self.refresh_views()

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in self._managed_widgets:
            widget.setEnabled(enabled)

    def refresh_views(self) -> None:
        if self.project is None:
            self._set_controls_enabled(False)
            self.video_list.clear()
            self.csv_list.clear()
            self.csv_video_combo.clear()
            self.info_label.setText(
                "No project is connected yet.<br>"
                "Use <b>Open Project...</b> or <b>Use Current Project</b> to manage an existing project."
            )
            self.txt_label.setText("No project selected.")
            return

        self._set_controls_enabled(True)
        project_dir = self.project.project_dir_path
        self.info_label.setText(
            f"<b>{self.project.title}</b><br>"
            f"Project file: {self.project.project_file}<br>"
            f"Project folder: {project_dir}"
        )

        self.video_list.clear()
        self.csv_video_combo.clear()
        file_entries_by_name = {entry.name: entry for entry in self.project.files}

        for record in self.project.video_records:
            file_entry = file_entries_by_name.get(record.name)
            csv_count = len(file_entry.csv) if file_entry else 0
            txt_count = len(file_entry.txt) if file_entry else 0
            access_state = self.project.get_video_access_state(record)
            if record.relative_path:
                location_text = record.relative_path
            else:
                location_text = record.source_path or record.file_name
            storage_text = "project copy" if access_state["storage"] == "project_copy" else "external source"
            status_text = "OK" if access_state["exists"] else "MISSING"
            item = QListWidgetItem(
                f"{record.file_name} | {storage_text} | {status_text} | "
                f"{location_text} | {csv_count} CSV | {txt_count} TXT"
            )
            item.setData(Qt.ItemDataRole.UserRole, record.name)
            self.video_list.addItem(item)
            self.csv_video_combo.addItem(record.file_name, record.name)

        self.refresh_csv_list()

    def refresh_csv_list(self) -> None:
        self.csv_list.clear()
        video_name = self.current_video_name()
        if video_name is None:
            self.txt_label.setText("No video selected.")
            return

        entry = next((item for item in self.project.files if item.name == video_name), None)
        if entry is None:
            self.txt_label.setText("No data found for the selected video.")
            return

        for csv_path in entry.csv:
            item = QListWidgetItem(Path(csv_path).name)
            item.setData(Qt.ItemDataRole.UserRole, csv_path)
            self.csv_list.addItem(item)

        txt_dir = self.project.txt_dir(video_name)
        txt_count = sum(1 for _ in txt_dir.glob("*.txt")) if txt_dir.exists() else 0
        self.txt_label.setText(
            f"TXT folder for '{entry.file_name or video_name}': {txt_dir} "
            f"({txt_count} txt files detected)"
        )

    def current_video_name(self) -> Optional[str]:
        if self.csv_video_combo.count() == 0:
            return None
        return self.csv_video_combo.currentData(Qt.ItemDataRole.UserRole)

    def _sync_video_selection_to_csv_combo(self) -> None:
        selected = self.video_list.selectedItems()
        if not selected:
            return
        video_name = selected[0].data(Qt.ItemDataRole.UserRole)
        index = self.csv_video_combo.findData(video_name, Qt.ItemDataRole.UserRole)
        if index >= 0 and self.csv_video_combo.currentIndex() != index:
            self.csv_video_combo.setCurrentIndex(index)

    def _add_videos(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "No project selected", "Load a project first.")
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Video Files",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv)",
        )
        if not paths:
            return

        errors: list[str] = []
        added = 0
        copy_into_project = self.copy_added_videos_check.isChecked()
        for path in paths:
            try:
                self.project.add_video(path, copy_into_project=copy_into_project, save=False)
                added += 1
            except Exception as err:
                errors.append(f"{Path(path).name}: {err}")

        if added:
            self.project.save()
            self.dialog.load_project(self.project.project_file)
            self.project = self.dialog.current_project or self.project

        self.refresh_views()
        if errors:
            QMessageBox.warning(self, "Some videos were skipped", "\n".join(errors))
        elif added:
            QMessageBox.information(self, "Videos added", f"Added {added} video(s) to the project.")

    def _remove_selected_videos(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "No project selected", "Load a project first.")
            return

        items = self.video_list.selectedItems()
        if not items:
            QMessageBox.information(self, "No selection", "Select at least one video to remove.")
            return

        names = [item.data(Qt.ItemDataRole.UserRole) for item in items]
        reply = QMessageBox.question(
            self,
            "Remove videos",
            "Remove the selected videos from the project and delete their project-managed "
            "raw copies, labels, and frames? External source videos are not deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        for name in names:
            self.project.remove_video(name, delete_project_data=True, save=False)

        self.project.save()
        self.dialog.load_project(self.project.project_file)
        self.project = self.dialog.current_project or self.project
        self.refresh_views()

    def _add_csvs(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "No project selected", "Load a project first.")
            return

        video_name = self.current_video_name()
        if video_name is None:
            QMessageBox.warning(self, "No video selected", "Select a video first.")
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add CSV Files",
            "",
            "CSV Files (*.csv)",
        )
        if not paths:
            return

        try:
            imported = self.project.import_csv_files(video_name, paths)
        except Exception as err:
            QMessageBox.critical(self, "CSV import failed", str(err))
            return

        self.refresh_csv_list()
        QMessageBox.information(self, "CSV imported", f"Imported {len(imported)} CSV file(s).")

    def _remove_selected_csvs(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "No project selected", "Load a project first.")
            return

        video_name = self.current_video_name()
        if video_name is None:
            QMessageBox.warning(self, "No video selected", "Select a video first.")
            return
        items = self.csv_list.selectedItems()
        if not items:
            QMessageBox.information(self, "No selection", "Select at least one CSV to remove.")
            return

        reply = QMessageBox.question(
            self,
            "Remove CSV files",
            "Delete the selected CSV files from this project?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        values = [item.data(Qt.ItemDataRole.UserRole) for item in items]
        self.project.remove_csv_files(video_name, values)
        self.refresh_csv_list()

    def _compress_project(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "No project selected", "Load a project first.")
            return

        reply = QMessageBox.question(
            self,
            "Compress project",
            "Delete image-heavy assets to reduce project size?\n\n"
            "Keeps:\n"
            "- runs/dataset\n"
            "- frames/<video>/masks\n"
            "- videos, labels, configs, weights\n\n"
            "Deletes image files in other project folders such as frames/images, "
            "frames/visualization, predicts, and image outputs under runs.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            result = self.project.compress_project()
        except Exception as err:
            QMessageBox.critical(self, "Compression failed", str(err))
            return

        self.refresh_views()
        freed_mb = result["deleted_bytes"] / (1024 * 1024) if result["deleted_bytes"] else 0.0
        QMessageBox.information(
            self,
            "Project compressed",
            f"Deleted {result['deleted_images']} image file(s), "
            f"removed {result['deleted_dirs']} empty folder(s), "
            f"freed about {freed_mb:.2f} MB.",
        )


class ProjectManagerDialog(QDialog):
    def __init__(
        self,
        main_window_load_project: Optional[Callable[[str], None]] = None,
        parent: Optional[QDialog] = None,
        current_project: Optional[ProjectInformation] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project Manager")
        self.resize(980, 760)

        self.main_window_load_project = main_window_load_project
        self.current_project = current_project or getattr(parent, "current_project", None)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.create_tab = _CreateProjectTab(self)
        self.tabs.addTab(self.create_tab, "Create New Project")
        self.manage_tab = _ManageProjectTab(self, self.current_project)
        self.tabs.addTab(self.manage_tab, "Project Manager")
        if self.current_project is not None:
            self.tabs.setCurrentWidget(self.manage_tab)

    def load_project(self, path: str | Path) -> None:
        loaded_project: Optional[ProjectInformation] = None
        if self.main_window_load_project is not None:
            self.main_window_load_project(path=path)
            loaded_project = getattr(self.parent(), "current_project", None)
            if not isinstance(loaded_project, ProjectInformation):
                loaded_project = None

        if loaded_project is None:
            loaded_project = ProjectInformation.from_path(path)
            loaded_project.ensure_project_file()

        self.set_current_project(loaded_project)

    def set_current_project(self, project: Optional[ProjectInformation]) -> None:
        self.current_project = project
        self.manage_tab.set_project(project)

    def open_project_from_picker(self) -> None:
        start_dir = ""
        if self.current_project is not None:
            start_dir = str(self.current_project.project_dir_path)
        elif self.parent() is not None and hasattr(self.parent(), "last_searched_dir"):
            start_dir = getattr(self.parent(), "last_searched_dir") or ""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select project file",
            start_dir,
            "Project files (*.json *.yaml *.yml)",
        )
        if not path:
            return
        self.load_project(path)
        self.tabs.setCurrentWidget(self.manage_tab)

    def use_parent_current_project(self) -> None:
        parent_project = getattr(self.parent(), "current_project", None)
        if not isinstance(parent_project, ProjectInformation):
            QMessageBox.information(
                self,
                "No current project",
                "There is no project loaded in the main window right now.",
            )
            return
        self.set_current_project(parent_project)
        self.tabs.setCurrentWidget(self.manage_tab)
