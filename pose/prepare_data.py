from __future__ import annotations

import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pose.task_state import pose_execution_state

ONLINE_DATASET_ROOT = "online_datasets"


def _raise_if_cancelled(should_cancel) -> None:
    if callable(should_cancel) and should_cancel():
        raise InterruptedError("Operation cancelled by user.")


def _resolve_frame_dir(project_dir: Path, video_name: str, frame_type: str) -> Path:
    if frame_type == "video":
        return project_dir / "frames" / video_name / "images"
    if frame_type in ("davis", "contour"):
        return project_dir / "frames" / video_name / "visualization" / frame_type
    if frame_type == "images":
        return project_dir / "frames" / video_name / "images"
    raise ValueError(f"Unsupported frame type: {frame_type}")


def _extract_video_frames_to_images(video_path: Path, image_dir: Path, *, should_cancel=None) -> int:
    image_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video for frame extraction: {video_path}")

    frame_idx = 0
    while True:
        _raise_if_cancelled(should_cancel)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        out_path = image_dir / f"{frame_idx:07d}.jpg"
        if not cv2.imwrite(str(out_path), frame):
            cap.release()
            raise RuntimeError(f"Failed to write frame image: {out_path}")
        frame_idx += 1

    cap.release()

    if frame_idx == 0:
        raise ValueError(f"No decodable frames found in video: {video_path}")
    return frame_idx


def _count_video_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return max(0, count)


def _collect_label_image_pairs(
    current_project,
    selected_entries,
    frame_type: str,
    label_dirs: dict[str, Path] | None = None,
    *,
    should_cancel=None,
) -> list[tuple[Path, Path, str]]:
    project_dir = Path(current_project.project_dir)
    digit_re = re.compile(r"(\d+)$")
    pair_list: list[tuple[Path, Path, str]] = []
    label_dirs = label_dirs or {}

    for fe in selected_entries:
        _raise_if_cancelled(should_cancel)
        video_path = Path(fe.video)
        video_name = video_path.stem
        label_dir = Path(label_dirs.get(video_name, project_dir / "labels" / video_name / "txt"))
        if not label_dir.is_dir():
            continue

        img_dir = _resolve_frame_dir(project_dir, video_name, frame_type)
        if frame_type == "video" and not any(img_dir.glob("*.jpg")):
            _extract_video_frames_to_images(video_path, img_dir, should_cancel=should_cancel)

        for lbl_file in sorted(label_dir.glob("*.txt")):
            _raise_if_cancelled(should_cancel)
            match = digit_re.search(lbl_file.stem)
            if not match:
                continue

            orig_num_str = match.group(1)
            frame_idx = int(orig_num_str)
            frame_num = f"{frame_idx:07d}"
            base_name = f"{video_name}_{frame_idx:0{len(orig_num_str)}d}"
            img_path = img_dir / f"{frame_num}.jpg"
            if not img_path.exists():
                continue

            pair_list.append((lbl_file, img_path, base_name))

    return pair_list


def create_dataset_split(
    current_project,
    selected_entries,
    frame_type: str,
    dataset_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    *,
    clear_existing: bool = True,
    seed: int | None = None,
    label_dirs: dict[str, Path] | None = None,
    progress_callback=None,
    should_cancel=None,
) -> dict[str, int]:
    _raise_if_cancelled(should_cancel)
    if progress_callback is not None:
        progress_callback(0, 0, "Collecting label-image pairs...")

    dataset_dir = Path(dataset_dir)
    pair_list = _collect_label_image_pairs(
        current_project,
        selected_entries,
        frame_type,
        label_dirs=label_dirs,
        should_cancel=should_cancel,
    )
    if not pair_list:
        raise ValueError("Could not find label-image pair.")

    total_pairs = len(pair_list)
    if progress_callback is not None:
        progress_callback(0, total_pairs, "Copying split files...")

    _raise_if_cancelled(should_cancel)
    if dataset_dir.exists() and clear_existing:
        shutil.rmtree(dataset_dir)

    for split in ("train", "val", "test"):
        _raise_if_cancelled(should_cancel)
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    shuffled_pairs = list(pair_list)
    random.Random(seed).shuffle(shuffled_pairs)
    total = len(shuffled_pairs)

    if total == 1:
        # Ultralytics requires a non-empty validation set. Reuse the only sample.
        split_map = {
            "train": shuffled_pairs[:],
            "val": shuffled_pairs[:],
            "test": [],
        }
    else:
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        train_count = max(1, train_count)
        val_count = max(1, val_count)

        if train_count + val_count > total:
            overflow = train_count + val_count - total
            reducible_train = max(0, train_count - 1)
            reduce_train = min(reducible_train, overflow)
            train_count -= reduce_train
            overflow -= reduce_train
            if overflow > 0:
                val_count = max(1, val_count - overflow)

        if train_count + val_count > total:
            val_count = max(1, total - train_count)

        if train_count + val_count > total:
            train_count = max(1, total - val_count)

        train_end = train_count
        val_end = train_end + val_count
        split_map = {
            "train": shuffled_pairs[:train_end],
            "val": shuffled_pairs[train_end:val_end],
            "test": shuffled_pairs[val_end:],
        }

    copied_pairs = 0
    for split, pairs in split_map.items():
        _raise_if_cancelled(should_cancel)
        img_dst_root = dataset_dir / split / "images"
        lbl_dst_root = dataset_dir / split / "labels"
        for lbl_path, img_path, base in pairs:
            _raise_if_cancelled(should_cancel)
            shutil.copy(lbl_path, lbl_dst_root / f"{base}.txt")
            shutil.copy(img_path, img_dst_root / f"{base}{img_path.suffix.lower()}")
            copied_pairs += 1
            if progress_callback is not None:
                progress_callback(
                    copied_pairs,
                    total_pairs,
                    f"Copying {split} split ({copied_pairs}/{total_pairs})",
                )

    if progress_callback is not None:
        progress_callback(total_pairs, total_pairs, "Dataset split complete")
    return {split: len(pairs) for split, pairs in split_map.items()}


def create_online_training_dataset(
    current_project,
    frame_type: str = "video",
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    *,
    dataset_root: str | Path | None = None,
    seed: int | None = None,
    label_dirs: dict[str, Path] | None = None,
) -> tuple[Path, dict[str, int]]:
    project_dir = Path(current_project.project_dir)
    dataset_root = Path(dataset_root) if dataset_root is not None else project_dir / "runs" / ONLINE_DATASET_ROOT
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    dataset_dir = dataset_root / f"online_training_dataset_{stamp}"
    counts = create_dataset_split(
        current_project,
        list(current_project.files),
        frame_type,
        dataset_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        clear_existing=False,
        seed=seed,
        label_dirs=label_dirs,
        should_cancel=None,
    )
    return dataset_dir, counts


class DataSplitDialog(QDialog):
    def __init__(self, current_project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Split")
        self.setFixedSize(500, 400)

        self.current_project = current_project
        self.files = current_project.files
        self.split_worker = None

        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            """
            QScrollArea {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
            }
            QScrollArea > QWidget > QWidget {
                background: #ffffff;
            }
            """
        )
        inner_widget = QWidget()
        self.files_lay = QVBoxLayout(inner_widget)
        self.files_lay.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(inner_widget)
        layout.addWidget(scroll)

        layout.addSpacing(40)
        self.count_label = QLabel("0 files selected / 0 frames / 0 labels")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        count_font = QFont()
        count_font.setPointSize(11)
        self.count_label.setFont(count_font)
        layout.addWidget(self.count_label)

        layout.addSpacing(20)

        ratio_layout = QFormLayout()

        self.train_slider = QSlider(Qt.Orientation.Horizontal)
        self.train_slider.setRange(0, 100)
        self.train_slider.setValue(70)
        self.train_spin = QSpinBox()
        self.train_spin.setRange(0, 100)
        self.train_spin.setValue(70)
        self.train_slider.valueChanged.connect(self.train_spin.setValue)
        self.train_spin.valueChanged.connect(self.train_slider.setValue)
        ratio_layout.addRow("Train %", self.create_slider_spinbox_layout(self.train_slider, self.train_spin))

        self.valid_slider = QSlider(Qt.Orientation.Horizontal)
        self.valid_slider.setRange(0, 100)
        self.valid_slider.setValue(20)
        self.valid_spin = QSpinBox()
        self.valid_spin.setRange(0, 100)
        self.valid_spin.setValue(20)
        self.valid_slider.valueChanged.connect(self.valid_spin.setValue)
        self.valid_spin.valueChanged.connect(self.valid_slider.setValue)
        ratio_layout.addRow("Valid %", self.create_slider_spinbox_layout(self.valid_slider, self.valid_spin))

        self.test_slider = QSlider(Qt.Orientation.Horizontal)
        self.test_slider.setRange(0, 100)
        self.test_slider.setValue(10)
        self.test_spin = QSpinBox()
        self.test_spin.setRange(0, 100)
        self.test_spin.setValue(10)
        self.test_slider.valueChanged.connect(self.test_spin.setValue)
        self.test_spin.valueChanged.connect(self.test_slider.setValue)
        ratio_layout.addRow("Test %", self.create_slider_spinbox_layout(self.test_slider, self.test_spin))

        layout.addLayout(ratio_layout)

        layout.addSpacing(20)

        self.frame_type_combo = QComboBox()
        self.frame_type_combo.addItem("video")
        self.frame_type_combo.addItem("images")
        self.frame_type_combo.addItem("davis")
        self.frame_type_combo.addItem("contour")
        preferred_mode = self.current_project.get_preferred_frame_mode()
        preferred_index = self.frame_type_combo.findText(preferred_mode, Qt.MatchFlag.MatchExactly)
        if preferred_index >= 0:
            self.frame_type_combo.setCurrentIndex(preferred_index)
        layout.addWidget(self.frame_type_combo)

        self.run_btn = QPushButton("Run")
        layout.addWidget(self.run_btn)
        self.run_btn.clicked.connect(self.run_split)

        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        self._populate_file_items()
        self.frame_type_combo.currentTextChanged.connect(self._frame_type_changed)
        self.frame_type_combo.currentTextChanged.connect(self._save_frame_type)

    def _frame_type_changed(self):
        self._populate_file_items()
        self._update_selection_count()

    def _save_frame_type(self, frame_type: str) -> None:
        self.current_project.set_preferred_frame_mode(frame_type)

    def _populate_file_items(self) -> None:
        self._clear_file_items()
        for fe in self.files:
            current_project = self.current_project
            video_path = Path(fe.video)
            video_stem = video_path.stem
            frame_type = self.frame_type_combo.currentText()
            frame_dir = _resolve_frame_dir(Path(current_project.project_dir), video_stem, frame_type)
            label_dir = Path(current_project.project_dir) / "labels" / video_stem / "txt"
            frame_cnt = sum(1 for _ in frame_dir.glob("*.jpg"))
            if frame_type == "video" and frame_cnt == 0:
                frame_cnt = _count_video_frames(video_path)
            label_cnt = sum(1 for _ in label_dir.glob("*.txt"))

            row_lay = QHBoxLayout()
            chk = QCheckBox()
            chk.stateChanged.connect(self._update_selection_count)
            chk._frame_cnt = frame_cnt
            chk._label_cnt = label_cnt
            chk._file_entry = fe

            name_lbl = QLabel(video_path.name)
            count_lbl = QLabel(f"({frame_cnt:,} frames, {label_cnt:,} labels)")
            count_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            row_lay.addWidget(chk)
            row_lay.addWidget(name_lbl, 1)
            row_lay.addWidget(count_lbl)

            self.files_lay.addLayout(row_lay)

        self.files_lay.addStretch(1)

    def _clear_file_items(self) -> None:
        while self.files_lay.count():
            item = self.files_lay.takeAt(0)

            if widget := item.widget():
                widget.deleteLater()
            elif child_lay := item.layout():
                while child_lay.count():
                    sub_item = child_lay.takeAt(0)
                    if w := sub_item.widget():
                        w.deleteLater()

    def _update_selection_count(self):
        total_files = 0
        total_frames = 0
        total_labels = 0

        for i in range(self.files_lay.count() - 1):
            lay = self.files_lay.itemAt(i)
            if not isinstance(lay, QHBoxLayout):
                continue
            chk = lay.itemAt(0).widget()
            if isinstance(chk, QCheckBox) and chk.isChecked():
                total_files += 1
                total_frames += getattr(chk, "_frame_cnt", 0)
                total_labels += getattr(chk, "_label_cnt", 0)

        self.count_label.setText(
            f"{total_files} files selected / "
            f"{total_frames:,} frames / "
            f"{total_labels:,} labels"
        )

    def get_selected_entries(self):
        selected_entries = []
        for i in range(self.files_lay.count() - 1):
            lay = self.files_lay.itemAt(i)
            if not isinstance(lay, QHBoxLayout):
                continue
            chk = lay.itemAt(0).widget()
            if isinstance(chk, QCheckBox) and chk.isChecked():
                selected_entries.append(chk._file_entry)
        return selected_entries

    def create_slider_spinbox_layout(self, slider, spinbox):
        hlayout = QHBoxLayout()
        hlayout.addWidget(slider)
        hlayout.addWidget(spinbox)
        return hlayout

    def run_split(self):
        if self.split_worker is not None and self.split_worker.isRunning():
            return

        selected_entries = self.get_selected_entries()
        if not selected_entries:
            QMessageBox.warning(self, "Error", "First, select a video file.")
            return

        active_task = (pose_execution_state.active_task() or "").lower()
        if pose_execution_state.is_busy() and active_task == "training":
            QMessageBox.information(
                self,
                "Training in progress",
                "Dataset preparation is disabled while training is running.",
            )
            return

        self.run_btn.setEnabled(False)

        dataset_dir = Path(self.current_project.project_dir) / "runs" / "dataset"
        self.split_worker = DataSplitWorker(
            current_project=self.current_project,
            selected_entries=selected_entries,
            frame_type=self.frame_type_combo.currentText(),
            dataset_dir=dataset_dir,
            train_ratio=self.train_spin.value() / 100.0,
            val_ratio=self.valid_spin.value() / 100.0,
        )
        self.split_worker.progress.connect(self._on_split_progress)
        self.split_worker.success.connect(self._on_split_success)
        self.split_worker.cancelled.connect(self._on_split_cancelled)
        self.split_worker.failure.connect(self._on_split_failure)
        self.split_worker.finished.connect(self._on_split_finished)
        self.split_worker.start()

    def _on_split_progress(self, done: int, total: int, message: str):
        _ = (done, total, message)

    def _on_split_success(self, split_counts: dict):
        QMessageBox.information(
            self,
            "Success",
            (f"Data Split completed\n"
             f"Train: {split_counts['train']}\n"
             f"Val:   {split_counts['val']}\n"
             f"Test:  {split_counts['test']}")
        )

    def _on_split_failure(self, error_text: str):
        if error_text.startswith("ValueError: "):
            QMessageBox.warning(self, "Error", error_text[len("ValueError: "):])
        else:
            QMessageBox.critical(self, "Error", f"Failed to create dataset split:\n{error_text}")

    def _on_split_cancelled(self):
        QMessageBox.information(self, "Cancelled", "Data split was cancelled.")

    def _on_split_finished(self):
        self.run_btn.setEnabled(True)
        self.split_worker = None

    def closeEvent(self, event):
        if self.split_worker is not None and self.split_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Data preparation in progress",
                "Data preparation is running.\n\nStop this task and close the window?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            self.split_worker.request_cancel()
            self.run_btn.setEnabled(False)
            if not self.split_worker.wait(4000):
                QMessageBox.warning(
                    self,
                    "Still stopping",
                    "Cancellation is still in progress. Please try closing again in a moment.",
                )
                event.ignore()
                return
        super().closeEvent(event)


class DataSplitWorker(QThread):
    progress = pyqtSignal(int, int, str)
    success = pyqtSignal(dict)
    cancelled = pyqtSignal()
    failure = pyqtSignal(str)

    def __init__(
        self,
        current_project,
        selected_entries,
        frame_type: str,
        dataset_dir: Path,
        train_ratio: float,
        val_ratio: float,
    ):
        super().__init__()
        self.current_project = current_project
        self.selected_entries = list(selected_entries)
        self.frame_type = frame_type
        self.dataset_dir = Path(dataset_dir)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True
        self.requestInterruption()

    def _is_cancel_requested(self) -> bool:
        return self._cancel_requested or self.isInterruptionRequested()

    def run(self):
        try:
            split_counts = create_dataset_split(
                self.current_project,
                self.selected_entries,
                self.frame_type,
                self.dataset_dir,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                clear_existing=True,
                progress_callback=self._report_progress,
                should_cancel=self._is_cancel_requested,
            )
            self.success.emit(split_counts)
        except InterruptedError:
            self.cancelled.emit()
        except ValueError as err:
            self.failure.emit(f"ValueError: {err}")
        except Exception as err:
            self.failure.emit(str(err))

    def _report_progress(self, done: int, total: int, message: str):
        self.progress.emit(int(done), int(total), str(message))
