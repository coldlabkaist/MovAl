from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Optional, Union

from PyQt6.QtCore import QStandardPaths, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from utils import __version__
from utils.project import ProjectInformation
from pose.task_state import pose_execution_state


class MainWindow(QMainWindow):
    def __init__(self, controller=None):
        super().__init__()
        self.setWindowTitle("Move Altogether: MovAl")
        self.setGeometry(100, 100, 650, 550)
        self.setFixedSize(self.size())

        self.controller = controller
        self.controller.parent = self
        self.current_project: Optional[ProjectInformation] = None
        self.last_searched_dir: Optional[str] = None
        self.desktop_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DesktopLocation
        )
        appdata_root = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
        self.appdata_dir = (
            Path(appdata_root)
            if appdata_root
            else (Path.home() / "AppData" / "Roaming" / "MovAl")
        )
        self.last_project_log_path = self.appdata_dir / "last_project.json"

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        outer_layout = QVBoxLayout()
        central_widget.setLayout(outer_layout)

        title_label = QLabel("Welcome to MovAl")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer_layout.addWidget(title_label)

        proj_bar = QHBoxLayout()
        proj_bar.addWidget(QLabel("Current project:", self))
        self.proj_name = QLineEdit(self)
        self.proj_name.setReadOnly(True)
        self.proj_name.setPlaceholderText("No project loaded")
        proj_bar.addWidget(self.proj_name, 1)

        self.btn_load_project = QPushButton("Load Project", self)
        self.btn_load_project.clicked.connect(self.on_load_project_clicked)
        proj_bar.addWidget(self.btn_load_project)
        outer_layout.addLayout(proj_bar)

        self.pose_progress_widget = QWidget(self)
        self.pose_progress_widget.setVisible(False)
        pose_progress_layout = QHBoxLayout(self.pose_progress_widget)
        pose_progress_layout.setContentsMargins(0, 0, 0, 0)
        pose_progress_layout.setSpacing(8)
        self.pose_progress_label = QLabel("Pose task running...", self.pose_progress_widget)
        self.pose_progress_label.setStyleSheet("font-size: 11px;")
        self.pose_progress_bar = QFrame(self.pose_progress_widget)
        self.pose_progress_bar_layout = QVBoxLayout(self.pose_progress_bar)
        self.pose_progress_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.pose_progress = QProgressBar(self.pose_progress_bar)
        self.pose_progress.setFixedHeight(12)
        self.pose_progress.setRange(0, 0)
        self.pose_progress.setTextVisible(False)
        self.pose_progress_bar_layout.addWidget(self.pose_progress)
        pose_progress_layout.addWidget(self.pose_progress_label)
        pose_progress_layout.addWidget(self.pose_progress_bar, 1)
        outer_layout.addWidget(self.pose_progress_widget)

        self.controller.main_window_load_project = self.on_load_project_clicked
        pose_execution_state.busy_changed.connect(self._on_pose_busy_changed)
        pose_execution_state.progress_changed.connect(self._on_pose_progress_changed)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setFixedHeight(2)
        outer_layout.addSpacing(5)
        outer_layout.addWidget(line)
        outer_layout.addSpacing(5)

        main_layout = QHBoxLayout()
        outer_layout.addLayout(main_layout)

        self.button_layout = QVBoxLayout()
        self.button_layout.setSpacing(5)
        self.button_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addLayout(self.button_layout)
        main_layout.insertStretch(0, 1)
        main_layout.addStretch(1)

        right_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        image_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "background_image.png")
        )
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(
            pixmap.scaledToWidth(450, Qt.TransformationMode.SmoothTransformation)
        )
        right_layout.addWidget(self.image_label)
        main_layout.addLayout(right_layout, 2)

        self.setup_buttons()
        self._restore_last_project()

    def _on_pose_busy_changed(self, busy: bool, task_name: str) -> None:
        if not busy:
            self.pose_progress_widget.setVisible(False)
            self.pose_progress_label.setText("Pose task running...")
            self.pose_progress.setRange(0, 0)
            self.pose_progress.setTextVisible(False)
            return

        self.pose_progress_widget.setVisible(True)
        self.pose_progress_label.setText(f"{task_name.capitalize()} running...")

    def _on_pose_progress_changed(self, task_name: str, done: int, total: int, message: str) -> None:
        if not task_name:
            return

        self.pose_progress_widget.setVisible(True)
        label_text = message if message else f"{task_name.capitalize()} running..."
        self.pose_progress_label.setText(label_text)

        if total > 0:
            self.pose_progress.setRange(0, total)
            self.pose_progress.setValue(min(done, total))
            self.pose_progress.setTextVisible(True)
            self.pose_progress.setFormat("%v/%m")
        else:
            self.pose_progress.setRange(0, 0)
            self.pose_progress.setTextVisible(False)

    def _restore_last_project(self) -> None:
        last_path = self._read_last_project_path()
        if last_path is None:
            return
        if not last_path.exists():
            self._clear_last_project_log()
            return
        self.on_load_project_clicked(path=last_path)

    def _read_last_project_path(self) -> Optional[Path]:
        try:
            if not self.last_project_log_path.exists():
                return None
            with self.last_project_log_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        raw_path = data.get("last_project_path")
        if not raw_path:
            return None
        return Path(raw_path).expanduser()

    def _write_last_project_path(self, path: Union[str, Path]) -> None:
        try:
            resolved = str(Path(path).expanduser().resolve())
            self.appdata_dir.mkdir(parents=True, exist_ok=True)
            payload = {"last_project_path": resolved}
            with self.last_project_log_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _clear_last_project_log(self) -> None:
        try:
            if self.last_project_log_path.exists():
                self.last_project_log_path.unlink()
        except Exception:
            pass

    def setup_buttons(self) -> None:
        installation_label = QLabel("Installation (Cutie / YOLO)")
        installation_label.setStyleSheet("margin-top: 10px;")
        self.button_layout.addWidget(installation_label)
        self.installation_btn = QPushButton("Installation")
        self.installation_btn.setFixedHeight(25)
        self.installation_btn.setMinimumWidth(180)
        self.installation_btn.clicked.connect(self.controller.run_installation)
        self.button_layout.addWidget(self.installation_btn)

        step1_label = QLabel("Step 1")
        step1_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.button_layout.addWidget(step1_label)
        self.create_project_btn = QPushButton("Create / Manage Project")
        self.create_project_btn.setFixedHeight(25)
        self.create_project_btn.setMinimumWidth(180)
        self.create_project_btn.clicked.connect(self.controller.run_project_manager)
        self.button_layout.addWidget(self.create_project_btn)

        step2_label = QLabel("Step 2")
        step2_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.button_layout.addWidget(step2_label)
        self.preprocess_btn = QPushButton("Preprocess")
        self.preprocess_btn.setFixedHeight(25)
        self.preprocess_btn.setMinimumWidth(180)
        self.preprocess_btn.clicked.connect(self.controller.run_video_preprocess)
        self.button_layout.addWidget(self.preprocess_btn)

        step3_label = QLabel("Step 3")
        step3_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.button_layout.addWidget(step3_label)
        self.label_btn = QPushButton("Labelary")
        self.label_btn.setFixedHeight(25)
        self.label_btn.setMinimumWidth(180)
        self.label_btn.clicked.connect(self.controller.run_labelary)
        self.button_layout.addWidget(self.label_btn)

        step4_label = QLabel("Step 4")
        step4_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.button_layout.addWidget(step4_label)
        self.pose_btn = QPushButton("Pose Estimation")
        self.pose_btn.setFixedHeight(25)
        self.pose_btn.setMinimumWidth(180)
        self.pose_btn.clicked.connect(self.controller.run_pose_estimation)
        self.button_layout.addWidget(self.pose_btn)

        optional_label = QLabel("Additional Tools")
        optional_label.setStyleSheet("margin-top: 10px;")
        self.button_layout.addWidget(optional_label)
        self.convert_btn = QPushButton("Data Convert\n(DLC/SLEAP to TXT)")
        self.convert_btn.setFixedHeight(40)
        self.convert_btn.setMinimumWidth(180)
        self.convert_btn.clicked.connect(self.controller.data_convert)
        self.button_layout.addWidget(self.convert_btn)
        self.extract_btn = QPushButton("Data Extract\n(TXT to CSV)")
        self.extract_btn.setFixedHeight(40)
        self.extract_btn.setMinimumWidth(180)
        self.extract_btn.clicked.connect(self.controller.data_extract)
        self.button_layout.addWidget(self.extract_btn)

    def on_load_project_clicked(
        self,
        checked: bool = False,
        path: Optional[Union[str, Path]] = None,
    ) -> None:
        start_dir = self.last_searched_dir if self.last_searched_dir else self.desktop_dir
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select project file",
                start_dir,
                "Project files (*.json *.yaml *.yml)",
            )
            if not path:
                return
            self.last_searched_dir = os.path.dirname(path)
        else:
            path = str(path)

        try:
            project = ProjectInformation.from_path(path)
            if not self._handle_legacy_project_conversion(project):
                return
            if not self._ensure_project_has_skeleton(project):
                return
            project.ensure_project_file()

            self.current_project = project
            self.controller.current_project = project
            self.proj_name.setText(project.title or project.project_file.stem)
            self.last_searched_dir = str(project.project_file.parent)
            self._write_last_project_path(project.project_file)
        except FileNotFoundError as err:
            if Path(path).expanduser() == self._read_last_project_path():
                self._clear_last_project_log()
            QMessageBox.warning(self, "File not found", str(err))
            return
        except Exception as err:
            QMessageBox.critical(self, "Load Error", str(err))
            return

        self._warn_if_project_version_differs()

    def _handle_legacy_project_conversion(self, project: ProjectInformation) -> bool:
        legacy_path = project.legacy_source_path
        if legacy_path is None:
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

    def _warn_if_project_version_differs(self) -> None:
        if self.current_project is None or not self.current_project.moval_version:
            return

        curr_major, curr_minor, *_ = (__version__.split(".") + ["0", "0"])
        proj_major, proj_minor, *_ = (
            self.current_project.moval_version.split(".") + ["0", "0"]
        )
        if (curr_major, curr_minor) != (proj_major, proj_minor):
            warnings.warn(
                "This project was created in a previous version of MovAl. "
                "Some files may not be fully compatible.",
                UserWarning,
            )

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
        from utils.skeleton import SkeletonModel

        preset_dir = Path(__file__).resolve().parents[1] / "preset" / "skeleton"
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
        from project_manager.skeleton import SkeletonManagerDialog
        from utils.skeleton import SkeletonModel

        saved = {"done": False}

        def save_callback(model: SkeletonModel, _structure_edit_unlocked: bool) -> None:
            model_dict = model.to_dict()
            if not model_dict.get("nodes"):
                raise ValueError("Add at least one keypoint before saving.")
            project.set_skeleton_data(model_dict, save=True)
            saved["done"] = True

        dialog = SkeletonManagerDialog(
            self,
            project=project,
            allow_structure_edit=True,
            save_callback=save_callback,
        )
        dialog.exec()
