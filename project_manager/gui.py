from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QBrush, QColor, QKeyEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .skeleton import SkeletonManagerDialog
from utils import __version__
from utils.project import ProjectInformation
from utils.skeleton import SkeletonModel

REPO_ROOT = Path(__file__).resolve().parents[1]


def _set_tooltip(widget: QWidget, text: str) -> None:
    widget.setToolTip(text)


def _make_separator(parent: QWidget) -> QFrame:
    line = QFrame(parent)
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


class _FileListWidget(QListWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self.setStyleSheet(
            """
            QListWidget::item:hover { background: #eaf1ff; color: #111827; }
            QListWidget::item:selected { background: #dbe8ff; color: #111827; }
            QListWidget::item:selected:!active { background: #dbe8ff; color: #111827; }
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
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        left_col = QVBoxLayout()
        left_col.setSpacing(10)
        layout.addLayout(left_col, 3)

        self.title_label = QLabel("<b>Project Title</b>")
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("e.g. CoLD_recording_250718")
        self.title_edit.setMinimumWidth(360)
        self.title_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        left_col.addWidget(self.title_label)
        left_col.addWidget(self.title_edit)
        _set_tooltip(self.title_edit, "Name of the project folder that will be created.")

        self.step1_label = QLabel("<b>Step 1.</b> Set number of animals")
        self.step1_spin = QSpinBox()
        self.step1_spin.setRange(1, 16)
        self.step1_spin.setValue(2)
        left_col.addWidget(self.step1_label)
        left_col.addWidget(self.step1_spin)
        _set_tooltip(self.step1_spin, "How many tracked animals the project will use.")

        self.instance_label = QLabel("<b>Animal Names</b>")
        self.instance_area = QScrollArea()
        self.instance_area.setWidgetResizable(True)
        self.instance_area.setMinimumHeight(240)
        self.instance_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.instance_container = QWidget()
        self.instance_layout = QVBoxLayout(self.instance_container)
        self.instance_layout.setContentsMargins(10, 10, 10, 10)
        self.instance_layout.setSpacing(6)
        self.instance_area.setWidget(self.instance_container)
        left_col.addWidget(self.instance_label)
        left_col.addWidget(self.instance_area, 1)
        _set_tooltip(
            self.instance_area,
            "Enter the track names that will appear throughout the project.",
        )

        self._instance_fields: list[QLineEdit] = []
        self.step1_spin.valueChanged.connect(self._generate_instance_fields)
        self._generate_instance_fields(self.step1_spin.value())

        self.step2_label = QLabel("<b>Step 2.</b> Load videos")
        self.step2_button = QPushButton("Select Videos")
        self.step2_button.clicked.connect(self._on_select_videos)
        self.step2_check = QCheckBox("Copy videos into project raw_videos (Recommended)")
        self.step2_check.setChecked(True)
        left_col.addWidget(self.step2_label)
        left_col.addWidget(self.step2_button)
        left_col.addWidget(self.step2_check)
        _set_tooltip(
            self.step2_check,
            "If checked, the new project stores its own local video copies under raw_videos.",
        )

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
        step4_row = QHBoxLayout()
        step4_row.addWidget(self.step4_combo, 1)
        step4_row.addWidget(self.step4_button)
        left_col.addLayout(step4_row)
        self._load_skeleton_items()
        _set_tooltip(
            self.step4_combo,
            "Preset skeleton that will be copied into the new project's project.json.",
        )

        note_label = QLabel(
            "<b>Note:</b> Put each CSV or TXT item immediately after its target video. "
            "CSV and TXT imports are copied into the standard labels folders."
        )
        note_label.setWordWrap(True)
        left_col.addWidget(note_label)

        self.create_button = QPushButton("Create Project")
        self.create_button.clicked.connect(self._create_project)
        left_col.addWidget(self.create_button)
        left_col.addStretch(1)

        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        layout.addLayout(right_col, 4)

        self.file_list = _FileListWidget()
        self.file_list.setMinimumWidth(420)
        right_col.addWidget(self.file_list, 1)
        _set_tooltip(
            self.file_list,
            "Ordered import list. Each CSV or TXT entry is attached to the nearest video above it.",
        )

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
            row_layout.setSpacing(8)
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


class _ManageProjectTab(QWidget):
    VIDEO_TREE_COLUMNS = ["Video", "Storage", "Status", "CSV", "TXT", "Path"]

    def __init__(self, dialog: "ProjectManagerDialog", project: Optional[ProjectInformation]) -> None:
        super().__init__(dialog)
        self.dialog = dialog
        self.project = project

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.description_label = QLabel(
            "Open a project to manage videos, label files, compression settings, "
            "and the project-local skeleton."
        )
        self.description_label.setWordWrap(True)

        project_row = QHBoxLayout()
        project_row.addWidget(self.description_label, 1)
        self.open_project_button = QPushButton("Open Project")
        self.open_project_button.clicked.connect(self.dialog.open_project_from_picker)
        project_row.addWidget(self.open_project_button)
        layout.addLayout(project_row)
        _set_tooltip(
            self.open_project_button,
            "Open an existing MovAl project.json or a legacy config.yaml file.",
        )

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        _set_tooltip(
            self.info_label,
            "Shows the currently opened project name, the project.json file name and path, and the project folder.",
        )

        layout.addWidget(_make_separator(self))

        self.project_section_label = QLabel("<b>Project Manager</b>")
        layout.addWidget(self.project_section_label)

        video_controls = QHBoxLayout()
        self.copy_added_videos_check = QCheckBox("Copy newly added videos into project raw_videos")
        self.copy_added_videos_check.setChecked(True)
        self.add_video_button = QPushButton("Add Videos")
        self.relink_video_button = QPushButton("Relink Source...")
        self.copy_existing_video_button = QPushButton("Copy Into Project")
        self.remove_video_button = QPushButton("Remove Selected Videos")
        self.add_video_button.clicked.connect(self._add_videos)
        self.relink_video_button.clicked.connect(self._relink_selected_video)
        self.copy_existing_video_button.clicked.connect(self._copy_selected_videos_into_project)
        self.remove_video_button.clicked.connect(self._remove_selected_videos)
        video_controls.addWidget(self.copy_added_videos_check)
        video_controls.addStretch(1)
        video_controls.addWidget(self.add_video_button)
        video_controls.addWidget(self.relink_video_button)
        video_controls.addWidget(self.copy_existing_video_button)
        video_controls.addWidget(self.remove_video_button)
        layout.addLayout(video_controls)
        _set_tooltip(
            self.copy_added_videos_check,
            "This applies only to videos added from this screen. Existing videos are not changed automatically.",
        )
        _set_tooltip(self.relink_video_button, "Update the absolute source path for one external video.")
        _set_tooltip(
            self.copy_existing_video_button,
            "Copy selected external videos into raw_videos and switch them to project-relative paths.",
        )

        self.video_tree = QTreeWidget()
        self.video_tree.setColumnCount(len(self.VIDEO_TREE_COLUMNS))
        self.video_tree.setHeaderLabels(self.VIDEO_TREE_COLUMNS)
        self.video_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.video_tree.itemSelectionChanged.connect(self._sync_video_selection_to_csv_combo)
        header = self.video_tree.header()
        for index in range(len(self.VIDEO_TREE_COLUMNS) - 1):
            header.setSectionResizeMode(index, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(self.VIDEO_TREE_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        header_item = self.video_tree.headerItem()
        header_item.setToolTip(0, "Video name inside this project.")
        header_item.setToolTip(1, "Project copy means raw_videos stores the file. External link means project.json stores an absolute source path.")
        header_item.setToolTip(2, "Available, Missing source, or Fallback copy.")
        header_item.setToolTip(3, "Number of CSV label files in labels/<video>/csv.")
        header_item.setToolTip(4, "Number of TXT label files in labels/<video>/txt.")
        header_item.setToolTip(5, "Resolved video path currently used by MovAl.")
        _set_tooltip(
            self.video_tree,
            "Select one or more videos to remove them, relink an external path, or copy them into the project.",
        )
        _set_tooltip(
            self.project_section_label,
            "Manage the current project's videos and label files here. Hover the table headers and buttons for details.",
        )

        self.video_csv_splitter = QSplitter(Qt.Orientation.Vertical, self)
        layout.addWidget(self.video_csv_splitter, 1)
        video_panel = QWidget(self)
        video_panel_layout = QVBoxLayout(video_panel)
        video_panel_layout.setContentsMargins(0, 0, 0, 0)
        video_panel_layout.addWidget(self.video_tree, 1)
        self.video_csv_splitter.addWidget(video_panel)

        csv_panel = QWidget(self)
        csv_panel_layout = QVBoxLayout(csv_panel)
        csv_panel_layout.setContentsMargins(0, 0, 0, 0)

        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("Manage CSVs for:"))
        self.csv_video_combo = QComboBox()
        self.csv_video_combo.currentIndexChanged.connect(self.refresh_csv_list)
        csv_row.addWidget(self.csv_video_combo, 1)
        self.add_csv_button = QPushButton("Add CSV")
        self.remove_csv_button = QPushButton("Remove Selected CSVs")
        self.add_csv_button.clicked.connect(self._add_csvs)
        self.remove_csv_button.clicked.connect(self._remove_selected_csvs)
        csv_row.addWidget(self.add_csv_button)
        csv_row.addWidget(self.remove_csv_button)
        csv_panel_layout.addLayout(csv_row)

        self.csv_list = QListWidget()
        self.csv_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        csv_panel_layout.addWidget(self.csv_list, 1)
        _set_tooltip(self.csv_video_combo, "Choose which video's labels/<video>/csv folder you want to manage.")
        _set_tooltip(self.csv_list, "CSV label files currently stored inside this project.")

        self.txt_label = QLabel()
        self.txt_label.setWordWrap(True)
        csv_panel_layout.addWidget(self.txt_label)
        _set_tooltip(
            self.txt_label,
            "Shows the TXT label folder currently linked to the selected video.",
        )
        self.video_csv_splitter.addWidget(csv_panel)
        self.video_csv_splitter.setStretchFactor(0, 3)
        self.video_csv_splitter.setStretchFactor(1, 2)
        self.video_csv_splitter.setSizes([420, 260])

        layout.addWidget(_make_separator(self))

        skeleton_header_row = QHBoxLayout()
        self.skeleton_section_label = QLabel("<b>Skeleton Manager</b>")
        skeleton_header_row.addWidget(self.skeleton_section_label)
        skeleton_header_row.addStretch(1)
        self.edit_skeleton_button = QPushButton("Edit Project Skeleton")
        self.edit_skeleton_button.clicked.connect(self._edit_project_skeleton)
        skeleton_header_row.addWidget(self.edit_skeleton_button)
        layout.addLayout(skeleton_header_row)
        self.skeleton_info_label = QLabel()
        self.skeleton_info_label.setWordWrap(True)
        layout.addWidget(self.skeleton_info_label)
        _set_tooltip(
            self.skeleton_section_label,
            "Edit the project-local skeleton. Visualization edits are always allowed; full structural edits require confirmation.",
        )
        _set_tooltip(
            self.skeleton_info_label,
            "Shows the base preset name and the current project skeleton summary stored in project.json.",
        )
        _set_tooltip(
            self.edit_skeleton_button,
            "Open the project-local skeleton editor. By default only node visualization can be changed.",
        )

        layout.addWidget(_make_separator(self))

        compress_row = QHBoxLayout()
        self.compress_section_label = QLabel("<b>Compress</b>")
        compress_row.addWidget(self.compress_section_label)
        self.delete_runs_check = QCheckBox("Also delete extra files under runs/")
        self.delete_predicts_check = QCheckBox("Also delete predicts/")
        compress_row.addWidget(self.delete_runs_check)
        compress_row.addWidget(self.delete_predicts_check)
        self.compress_button = QPushButton("Compress Project")
        self.compress_button.clicked.connect(self._compress_project)
        compress_row.addStretch(1)
        compress_row.addWidget(self.compress_button)
        layout.addLayout(compress_row)
        _set_tooltip(
            self.compress_section_label,
            "Removes large generated assets while keeping masks, videos, labels, and config files. runs/dataset is always deleted.",
        )
        _set_tooltip(
            self.delete_runs_check,
            "If checked, remove run outputs under runs/ except config files. runs/dataset is deleted either way.",
        )
        _set_tooltip(
            self.delete_predicts_check,
            "If checked, remove predict result folders under predicts/.",
        )
        _set_tooltip(
            self.compress_button,
            "Run project compression using the options on the same row.",
        )

        self._managed_widgets = [
            self.copy_added_videos_check,
            self.add_video_button,
            self.relink_video_button,
            self.copy_existing_video_button,
            self.remove_video_button,
            self.video_tree,
            self.edit_skeleton_button,
            self.delete_runs_check,
            self.delete_predicts_check,
            self.compress_button,
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

    def _format_status_text(self, record, access_state: dict) -> str:
        status = access_state.get("status")
        if status == "fallback_copy":
            return "Fallback copy"
        if status == "missing_source":
            return "Missing source"
        if record.relative_path:
            return "Available"
        return "Missing source"

    def _format_storage_text(self, record) -> str:
        return "Project copy" if record.relative_path else "External link"

    def _update_skeleton_summary(self) -> None:
        if self.project is None:
            self.skeleton_info_label.setText("No project skeleton loaded.")
            return

        data = self.project.skeleton_data or {}
        node_count = len(data.get("nodes", []))
        edge_count = len(data.get("connections", []))
        sym_count = len(data.get("symmetry", []))
        self.skeleton_info_label.setText(
            f"<b>Project Skeleton</b><br>"
            f"Base preset: {self.project.skeleton_name}<br>"
            f"Nodes: {node_count} | Connections: {edge_count} | Symmetry pairs: {sym_count}"
        )

    def refresh_views(self) -> None:
        if self.project is None:
            self._set_controls_enabled(False)
            self.video_tree.clear()
            self.csv_list.clear()
            self.csv_video_combo.clear()
            self.info_label.setText(
                "No project is connected yet.<br>"
                "Use <b>Open Project</b> to choose an existing project."
            )
            self.txt_label.setText("No project selected.")
            self.skeleton_info_label.setText("No project skeleton loaded.")
            return

        self._set_controls_enabled(True)
        project_dir = self.project.project_dir_path
        self.info_label.setText(
            f"<b>{self.project.title}</b><br>"
            f"Project file name: {self.project.project_file.name}<br>"
            f"Project file path: {self.project.project_file}<br>"
            f"Project folder: {project_dir}"
        )

        self.video_tree.clear()
        self.csv_video_combo.clear()
        file_entries_by_name = {entry.name: entry for entry in self.project.files}

        for record in self.project.video_records:
            file_entry = file_entries_by_name.get(record.name)
            csv_count = len(file_entry.csv) if file_entry else 0
            txt_dir = self.project.txt_dir(record.name)
            txt_count = sum(1 for _ in txt_dir.glob("*.txt")) if txt_dir.exists() else 0
            access_state = self.project.get_video_access_state(record)
            item = QTreeWidgetItem(
                [
                    record.file_name,
                    self._format_storage_text(record),
                    self._format_status_text(record, access_state),
                    str(csv_count),
                    str(txt_count),
                    str(access_state["path"]),
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, record.name)
            if access_state["status"] == "missing_source":
                for column in range(len(self.VIDEO_TREE_COLUMNS)):
                    item.setForeground(column, QBrush(QColor("#b3261e")))
            elif access_state["status"] == "fallback_copy":
                for column in range(len(self.VIDEO_TREE_COLUMNS)):
                    item.setForeground(column, QBrush(QColor("#8a5a00")))
            self.video_tree.addTopLevelItem(item)
            self.csv_video_combo.addItem(record.file_name, record.name)

        self.refresh_csv_list()
        self._update_skeleton_summary()

    def refresh_csv_list(self) -> None:
        self.csv_list.clear()
        if self.project is None:
            self.txt_label.setText("No project selected.")
            return

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
            f"TXT folder : {txt_dir} "
            f"({txt_count} txt files detected)"
        )

    def current_video_name(self) -> Optional[str]:
        if self.csv_video_combo.count() == 0:
            return None
        return self.csv_video_combo.currentData(Qt.ItemDataRole.UserRole)

    def _selected_video_names(self) -> list[str]:
        names: list[str] = []
        for item in self.video_tree.selectedItems():
            name = item.data(0, Qt.ItemDataRole.UserRole)
            if name:
                names.append(name)
        return names

    def _sync_video_selection_to_csv_combo(self) -> None:
        selected = self.video_tree.selectedItems()
        if not selected:
            return
        video_name = selected[0].data(0, Qt.ItemDataRole.UserRole)
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

    def _relink_selected_video(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "No project selected", "Load a project first.")
            return

        names = self._selected_video_names()
        if len(names) != 1:
            QMessageBox.information(self, "Select one video", "Choose exactly one video to relink.")
            return

        record = self.project.get_video_record(names[0])
        if record is None:
            return

        start_dir = ""
        access_state = self.project.get_video_access_state(record)
        try:
            start_dir = str(Path(access_state["path"]).parent)
        except Exception:
            start_dir = ""

        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Relink source for {record.file_name}",
            start_dir,
            "Videos (*.mp4 *.avi *.mov *.mkv)",
        )
        if not path:
            return

        try:
            self.project.relink_video_source(record.name, path)
        except Exception as err:
            QMessageBox.critical(self, "Relink failed", str(err))
            return

        self.dialog.load_project(self.project.project_file)
        self.project = self.dialog.current_project or self.project
        self.refresh_views()
        QMessageBox.information(
            self,
            "Video relinked",
            f"{record.file_name} now points to:\n{path}",
        )

    def _copy_selected_videos_into_project(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "No project selected", "Load a project first.")
            return

        names = self._selected_video_names()
        if not names:
            QMessageBox.information(
                self,
                "No selection",
                "Select one or more videos to copy into raw_videos.",
            )
            return

        copied = 0
        skipped = 0
        errors: list[str] = []
        for name in names:
            record = self.project.get_video_record(name)
            if record is None:
                continue
            if record.relative_path:
                skipped += 1
                continue
            try:
                self.project.copy_video_into_project(name, save=False)
                copied += 1
            except Exception as err:
                errors.append(f"{record.file_name}: {err}")

        if copied:
            self.project.save()
            self.dialog.load_project(self.project.project_file)
            self.project = self.dialog.current_project or self.project
        self.refresh_views()

        lines = [f"Copied {copied} video(s) into raw_videos."]
        if skipped:
            lines.append(f"Skipped {skipped} video(s) that were already project copies.")
        if errors:
            lines.append("")
            lines.extend(errors)
            QMessageBox.warning(self, "Copy finished with warnings", "\n".join(lines))
            return
        QMessageBox.information(self, "Copy finished", "\n".join(lines))

    def _remove_selected_videos(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "No project selected", "Load a project first.")
            return

        names = self._selected_video_names()
        if not names:
            QMessageBox.information(self, "No selection", "Select at least one video to remove.")
            return

        reply = QMessageBox.question(
            self,
            "Remove videos",
            "Remove the selected videos from the project and delete their project-managed "
            "raw copies, labels, and frames?\n\nExternal source videos are not deleted.",
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

    def _edit_project_skeleton(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "No project selected", "Load a project first.")
            return

        def save_callback(model: SkeletonModel, structure_edit_unlocked: bool) -> None:
            txt_count = sum(1 for _ in self.project.project_dir_path.glob("labels/*/txt/*.txt"))
            model_count = 0
            for suffix in ("*.pt", "*.pth", "*.onnx", "*.ckpt"):
                model_count += sum(1 for _ in (self.project.project_dir_path / "runs").rglob(suffix))

            self.project.set_skeleton_data(model.to_dict(), save=True)
            self.dialog.load_project(self.project.project_file)
            self.project = self.dialog.current_project or self.project
            self.refresh_views()

            if structure_edit_unlocked:
                detail = [
                    "Full skeleton editing was enabled for this save.",
                    "Existing TXT labels or trained models may no longer match the updated skeleton.",
                    "MovAl will warn you when an incompatible TXT folder is loaded.",
                ]
                if txt_count or model_count:
                    detail.append("")
                    detail.append(f"Detected TXT files: {txt_count}")
                    detail.append(f"Detected model files under runs/: {model_count}")
                QMessageBox.warning(self, "Compatibility warning", "\n".join(detail))

        dialog = SkeletonManagerDialog(
            self,
            project=self.project,
            allow_structure_edit=False,
            save_callback=save_callback,
        )
        dialog.exec()

    def _compress_project(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, "No project selected", "Load a project first.")
            return

        delete_runs = self.delete_runs_check.isChecked()
        delete_predicts = self.delete_predicts_check.isChecked()
        message = [
            "Delete large generated assets to reduce project size?",
            "",
            "Always keeps:",
            "- raw videos",
            "- labels",
            "- frames/<video>/masks",
            "- project config files",
            "",
            "Always deletes:",
            "- runs/dataset",
            "- non-mask image files under frames/",
        ]
        if delete_runs:
            message.append("- extra run outputs under runs/")
        if delete_predicts:
            message.append("- predicts/")

        reply = QMessageBox.question(
            self,
            "Compress project",
            "\n".join(message),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            result = self.project.compress_project(
                delete_runs=delete_runs,
                delete_predicts=delete_predicts,
            )
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
        self.resize(800, 800)

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
            previous_project = getattr(self.parent(), "current_project", None)
            self.main_window_load_project(path=path)
            loaded_project = getattr(self.parent(), "current_project", None)
            if (
                isinstance(loaded_project, ProjectInformation)
                and loaded_project is not previous_project
            ):
                self.set_current_project(loaded_project)
                return

        if loaded_project is None:
            loaded_project = ProjectInformation.from_path(path)
            if not self._handle_legacy_project_conversion(loaded_project):
                return
            parent_ensure = getattr(self.parent(), "_ensure_project_has_skeleton", None)
            if callable(parent_ensure):
                if not parent_ensure(loaded_project):
                    return
            elif not self._ensure_project_has_skeleton(loaded_project):
                return

        self.set_current_project(loaded_project)

    def _handle_legacy_project_conversion(self, project: ProjectInformation) -> bool:
        legacy_path = project.legacy_source_path
        if legacy_path is None:
            project.ensure_project_file()
            return True

        reply = QMessageBox.question(
            self,
            "Legacy YAML detected",
            "This project YAML file is from an old MovAl format and cannot be used directly in the current version.\n\n"
            "Convert it to project.json now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            QMessageBox.information(
                self,
                "Load canceled",
                "Project loading was canceled because conversion to project.json was not approved.",
            )
            return False

        json_path = project.ensure_project_file()
        remove_reply = QMessageBox.question(
            self,
            "Delete old YAML?",
            "Conversion is complete.\n\n"
            f"New JSON: {json_path}\n"
            f"Old YAML: {legacy_path}\n\n"
            "Delete the old YAML file?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if remove_reply == QMessageBox.StandardButton.Yes:
            try:
                legacy_path.unlink()
            except Exception as err:
                QMessageBox.warning(
                    self,
                    "Delete failed",
                    f"Could not delete legacy YAML:\n{legacy_path}\n\n{err}",
                )
        return True

    @staticmethod
    def _project_has_skeleton_nodes(project: ProjectInformation) -> bool:
        data = project.skeleton_data or {}
        nodes = data.get("nodes", []) if isinstance(data, dict) else []
        return isinstance(nodes, list) and len(nodes) > 0

    def _ensure_project_has_skeleton(self, project: ProjectInformation) -> bool:
        if self._project_has_skeleton_nodes(project):
            return True

        while not self._project_has_skeleton_nodes(project):
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Skeleton required")
            msg_box.setText(
                "This project has no valid skeleton.\n\n"
                "Please match a skeleton preset or draw a new skeleton before loading."
            )
            preset_btn = msg_box.addButton("Match From Preset", QMessageBox.ButtonRole.AcceptRole)
            draw_btn = msg_box.addButton("Draw/Edit Skeleton", QMessageBox.ButtonRole.ActionRole)
            cancel_btn = msg_box.addButton(QMessageBox.StandardButton.Cancel)
            msg_box.setDefaultButton(preset_btn)
            msg_box.exec()
            clicked = msg_box.clickedButton()

            if clicked == cancel_btn:
                return False
            if clicked == preset_btn:
                self._apply_skeleton_preset_to_project(project)
                continue
            if clicked == draw_btn:
                self._draw_project_skeleton(project)
                continue
            return False
        return True

    def _apply_skeleton_preset_to_project(self, project: ProjectInformation) -> None:
        preset_dir = REPO_ROOT / "preset" / "skeleton"
        preset_dir.mkdir(parents=True, exist_ok=True)
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select skeleton preset YAML",
            str(preset_dir),
            "YAML files (*.yaml *.yml);;All Files (*)",
        )
        if not selected_path:
            return

        try:
            model = SkeletonModel()
            model.load_from_yaml(selected_path)
            model_dict = model.to_dict()
            if not model_dict.get("nodes"):
                raise ValueError("Selected preset has no keypoints.")
            project.set_skeleton_data(
                model_dict,
                skeleton_name=Path(selected_path).name,
                save=True,
            )
        except Exception as err:
            QMessageBox.critical(
                self,
                "Skeleton preset failed",
                f"Failed to apply selected preset:\n{err}",
            )

    def _draw_project_skeleton(self, project: ProjectInformation) -> None:
        def save_callback(model: SkeletonModel, _structure_edit_unlocked: bool) -> None:
            model_dict = model.to_dict()
            if not model_dict.get("nodes"):
                raise ValueError("Add at least one keypoint before saving.")
            project.set_skeleton_data(model_dict, save=True)

        dialog = SkeletonManagerDialog(
            self,
            project=project,
            allow_structure_edit=True,
            save_callback=save_callback,
        )
        dialog.exec()

    def set_current_project(self, project: Optional[ProjectInformation]) -> None:
        self.current_project = project
        self.manage_tab.set_project(project)
        if project is not None:
            self.tabs.setCurrentWidget(self.manage_tab)

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
