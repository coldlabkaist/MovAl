from __future__ import annotations

import random
import re
import shutil
from dataclasses import dataclass
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
from pose.split_state import is_data_split_running, set_data_split_running
from utils.runtime_locks import is_project_compression_running

ONLINE_DATASET_ROOT = "online_datasets"


@dataclass(frozen=True)
class DatasetSample:
    label_path: Path
    video_path: Path
    video_name: str
    frame_idx: int
    base_name: str
    image_path: Path | None = None


def _raise_if_cancelled(should_cancel) -> None:
    if callable(should_cancel) and should_cancel():
        raise InterruptedError("Operation cancelled by user.")


def _resolve_frame_dir(project_dir: Path, video_name: str, frame_type: str) -> Path:
    if frame_type in ("davis", "contour"):
        return project_dir / "frames" / video_name / "visualization" / frame_type
    if frame_type == "images":
        return project_dir / "frames" / video_name / "images"
    raise ValueError(f"Unsupported frame type: {frame_type}")


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


def _open_video_capture(video_path: Path):
    for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
        cap = cv2.VideoCapture(str(video_path), backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def _write_image_checked(frame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write frame image: {output_path}")
    try:
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"Frame image was not written correctly: {output_path}")
    except OSError as err:
        raise RuntimeError(f"Unable to verify written frame image: {output_path}") from err


def _collect_training_samples(
    current_project,
    selected_entries,
    frame_type: str,
    label_dirs: dict[str, Path] | None = None,
    *,
    should_cancel=None,
) -> list[DatasetSample]:
    project_dir = Path(current_project.project_dir)
    digit_re = re.compile(r"(\d+)$")
    samples: list[DatasetSample] = []
    missing_images: list[str] = []
    label_dirs = label_dirs or {}

    for fe in selected_entries:
        _raise_if_cancelled(should_cancel)
        video_path = Path(fe.video)
        video_name = fe.name
        label_dir = Path(label_dirs.get(video_name, project_dir / "labels" / video_name / "txt"))
        if not label_dir.is_dir():
            continue

        img_dir = None if frame_type == "video" else _resolve_frame_dir(project_dir, video_name, frame_type)

        for lbl_file in sorted(label_dir.glob("*.txt")):
            _raise_if_cancelled(should_cancel)
            match = digit_re.search(lbl_file.stem)
            if not match:
                continue

            orig_num_str = match.group(1)
            frame_idx = int(orig_num_str)
            base_name = f"{video_name}_{frame_idx:0{len(orig_num_str)}d}"

            if frame_type == "video":
                samples.append(
                    DatasetSample(
                        label_path=lbl_file,
                        video_path=video_path,
                        video_name=video_name,
                        frame_idx=frame_idx,
                        base_name=base_name,
                    )
                )
                continue

            frame_num = f"{frame_idx:07d}"
            img_path = img_dir / f"{frame_num}.jpg"
            if not img_path.exists():
                missing_images.append(f"{video_name}:{frame_idx} ({img_path})")
                continue

            samples.append(
                DatasetSample(
                    label_path=lbl_file,
                    video_path=video_path,
                    video_name=video_name,
                    frame_idx=frame_idx,
                    base_name=base_name,
                    image_path=img_path,
                )
            )

    if missing_images:
        preview = "\n".join(missing_images[:20])
        more = "" if len(missing_images) <= 20 else f"\n... and {len(missing_images) - 20} more"
        raise FileNotFoundError(f"Missing frame images for labels:\n{preview}{more}")

    return samples


def _materialize_image_samples(
    split_map: dict[str, list[DatasetSample]],
    dataset_dir: Path,
    *,
    progress_callback=None,
    should_cancel=None,
) -> None:
    copied = 0
    total = sum(len(samples) for samples in split_map.values())
    for split, samples in split_map.items():
        img_dst_root = dataset_dir / split / "images"
        lbl_dst_root = dataset_dir / split / "labels"
        for sample in samples:
            _raise_if_cancelled(should_cancel)
            if sample.image_path is None or not sample.image_path.exists():
                raise FileNotFoundError(
                    f"Missing source image for {sample.video_name}:{sample.frame_idx}"
                )
            shutil.copy(sample.label_path, lbl_dst_root / f"{sample.base_name}.txt")
            shutil.copy(sample.image_path, img_dst_root / f"{sample.base_name}{sample.image_path.suffix.lower()}")
            copied += 1
            if progress_callback is not None:
                progress_callback(copied, total, f"Copying {split} split ({copied}/{total})")


def _materialize_video_samples(
    split_map: dict[str, list[DatasetSample]],
    dataset_dir: Path,
    *,
    progress_callback=None,
    should_cancel=None,
) -> None:
    targets_by_video: dict[Path, dict[int, list[tuple[str, DatasetSample]]]] = {}
    for split, samples in split_map.items():
        for sample in samples:
            targets_by_video.setdefault(sample.video_path, {}).setdefault(sample.frame_idx, []).append((split, sample))

    written = 0
    total = sum(len(samples) for samples in split_map.values())
    for video_path, targets in targets_by_video.items():
        _raise_if_cancelled(should_cancel)
        cap = _open_video_capture(video_path)
        if cap is None:
            raise FileNotFoundError(f"Unable to open video for dataset split: {video_path}")

        target_frames = sorted(targets)
        max_target = target_frames[-1] if target_frames else -1
        current_idx = 0
        try:
            while current_idx <= max_target:
                _raise_if_cancelled(should_cancel)
                ok, frame = cap.read()
                if not ok or frame is None:
                    missing = [idx for idx in target_frames if idx >= current_idx]
                    preview = ", ".join(str(idx) for idx in missing[:20])
                    more = "" if len(missing) <= 20 else f", ... and {len(missing) - 20} more"
                    raise RuntimeError(
                        f"Video ended or failed before labeled frames could be decoded: "
                        f"{video_path}\nMissing frame indices: {preview}{more}"
                    )

                if current_idx in targets:
                    for split, sample in targets[current_idx]:
                        img_dst = dataset_dir / split / "images" / f"{sample.base_name}.jpg"
                        lbl_dst = dataset_dir / split / "labels" / f"{sample.base_name}.txt"
                        _write_image_checked(frame, img_dst)
                        shutil.copy(sample.label_path, lbl_dst)
                        written += 1
                        if progress_callback is not None:
                            progress_callback(
                                written,
                                total,
                                f"Writing video frames ({written}/{total})",
                            )
                current_idx += 1
        finally:
            cap.release()


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
        progress_callback(0, 0, "Collecting labeled frames...")

    dataset_dir = Path(dataset_dir)
    samples = _collect_training_samples(
        current_project,
        selected_entries,
        frame_type,
        label_dirs=label_dirs,
        should_cancel=should_cancel,
    )
    if not samples:
        raise ValueError("Could not find labeled training samples.")

    shuffled_samples = list(samples)
    random.Random(seed).shuffle(shuffled_samples)
    total = len(shuffled_samples)

    if total == 1:
        # Ultralytics requires a non-empty validation set. Reuse the only sample.
        split_map = {
            "train": shuffled_samples[:],
            "val": shuffled_samples[:],
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
            "train": shuffled_samples[:train_end],
            "val": shuffled_samples[train_end:val_end],
            "test": shuffled_samples[val_end:],
        }

    materialize_total = sum(len(items) for items in split_map.values())
    if progress_callback is not None:
        progress_callback(0, materialize_total, "Preparing dataset split files...")

    _raise_if_cancelled(should_cancel)
    build_dir = dataset_dir.with_name(f"{dataset_dir.name}.building")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    try:
        for split in ("train", "val", "test"):
            _raise_if_cancelled(should_cancel)
            (build_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (build_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        if frame_type == "video":
            _materialize_video_samples(
                split_map,
                build_dir,
                progress_callback=progress_callback,
                should_cancel=should_cancel,
            )
        else:
            _materialize_image_samples(
                split_map,
                build_dir,
                progress_callback=progress_callback,
                should_cancel=should_cancel,
            )

        _raise_if_cancelled(should_cancel)
        if dataset_dir.exists():
            if clear_existing:
                shutil.rmtree(dataset_dir)
            else:
                raise FileExistsError(f"Dataset directory already exists: {dataset_dir}")
        build_dir.rename(dataset_dir)
    except Exception:
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)
        raise

    if progress_callback is not None:
        progress_callback(materialize_total, materialize_total, "Dataset split complete")
    return {split: len(items) for split, items in split_map.items()}


def create_online_training_dataset(
    current_project,
    frame_type: str = "video",
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    *,
    dataset_root: str | Path | None = None,
    seed: int | None = None,
    label_dirs: dict[str, Path] | None = None,
    progress_callback=None,
    should_cancel=None,
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
        progress_callback=progress_callback,
        should_cancel=should_cancel,
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
        self._ratio_guard = False

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
        self.train_slider.setRange(40, 90)
        self.train_slider.setValue(70)
        self.train_spin = QSpinBox()
        self.train_spin.setRange(40, 90)
        self.train_spin.setValue(70)
        self.train_slider.valueChanged.connect(self.train_spin.setValue)
        self.train_spin.valueChanged.connect(self.train_slider.setValue)
        ratio_layout.addRow("Train %", self.create_slider_spinbox_layout(self.train_slider, self.train_spin))

        self.valid_slider = QSlider(Qt.Orientation.Horizontal)
        self.valid_slider.setRange(5, 55)
        self.valid_slider.setValue(20)
        self.valid_spin = QSpinBox()
        self.valid_spin.setRange(5, 55)
        self.valid_spin.setValue(20)
        self.valid_slider.valueChanged.connect(self.valid_spin.setValue)
        self.valid_spin.valueChanged.connect(self.valid_slider.setValue)
        ratio_layout.addRow("Valid %", self.create_slider_spinbox_layout(self.valid_slider, self.valid_spin))

        self.test_slider = QSlider(Qt.Orientation.Horizontal)
        self.test_slider.setRange(5, 55)
        self.test_slider.setValue(10)
        self.test_spin = QSpinBox()
        self.test_spin.setRange(5, 55)
        self.test_spin.setValue(10)
        self.test_slider.valueChanged.connect(self.test_spin.setValue)
        self.test_spin.valueChanged.connect(self.test_slider.setValue)
        self.test_slider.setEnabled(False)
        self.test_spin.setEnabled(False)
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
        self.run_btn.setProperty("primary", True)
        layout.addWidget(self.run_btn)
        self.run_btn.clicked.connect(self.run_split)

        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        self._populate_file_items()
        self.frame_type_combo.currentTextChanged.connect(self._frame_type_changed)
        self.frame_type_combo.currentTextChanged.connect(self._save_frame_type)
        self.train_spin.valueChanged.connect(lambda v: self._on_ratio_input_changed("train", v))
        self.valid_spin.valueChanged.connect(lambda v: self._on_ratio_input_changed("val", v))
        self._apply_ratio_constraints(self.train_spin.value(), self.valid_spin.value(), source="train")

    def _frame_type_changed(self):
        selected_names = self._selected_video_names()
        self._populate_file_items(selected_names=selected_names)
        self._update_selection_count()

    def _save_frame_type(self, frame_type: str) -> None:
        self.current_project.set_preferred_frame_mode(frame_type)

    def _populate_file_items(self, selected_names: set[str] | None = None) -> None:
        selected_names = selected_names or set()
        self._clear_file_items()
        for fe in self.files:
            current_project = self.current_project
            video_path = Path(fe.video)
            video_name = fe.name
            frame_type = self.frame_type_combo.currentText()
            label_dir = Path(current_project.project_dir) / "labels" / video_name / "txt"
            if frame_type == "video":
                frame_cnt = _count_video_frames(video_path)
            else:
                frame_dir = _resolve_frame_dir(Path(current_project.project_dir), video_name, frame_type)
                frame_cnt = sum(1 for _ in frame_dir.glob("*.jpg"))
            label_cnt = sum(1 for _ in label_dir.glob("*.txt"))

            row_lay = QHBoxLayout()
            chk = QCheckBox()
            chk.setChecked(video_name in selected_names)
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

    def _selected_video_names(self) -> set[str]:
        selected: set[str] = set()
        for i in range(self.files_lay.count() - 1):
            lay = self.files_lay.itemAt(i)
            if not isinstance(lay, QHBoxLayout):
                continue
            chk = lay.itemAt(0).widget()
            if isinstance(chk, QCheckBox) and chk.isChecked():
                fe = getattr(chk, "_file_entry", None)
                if fe is not None:
                    selected.add(fe.name)
        return selected

    def _on_ratio_input_changed(self, source: str, value: int) -> None:
        if self._ratio_guard:
            return
        train = self.train_spin.value()
        valid = self.valid_spin.value()
        if source == "train":
            train = int(value)
        else:
            valid = int(value)
        self._apply_ratio_constraints(train, valid, source=source)

    def _apply_ratio_constraints(self, train: int, valid: int, *, source: str) -> None:
        self._ratio_guard = True
        try:
            current_train = int(self.train_spin.value())
            current_valid = int(self.valid_spin.value())

            if source == "train":
                # Train changes should not force-adjust Valid.
                valid = max(5, min(current_valid, 55))
                train = max(40, min(int(train), 90))
                train = min(train, 95 - valid)
            elif source == "val":
                # Valid changes should not force-adjust Train.
                train = max(40, min(current_train, 90))
                valid = max(5, min(int(valid), 55))
                valid = min(valid, 95 - train)
            else:
                # Defensive fallback for non-standard callers.
                train = max(40, min(int(train), 90))
                valid = max(5, min(int(valid), 55))
                if train + valid > 95:
                    valid = max(5, 95 - train)
                    if train + valid > 95:
                        train = max(40, 95 - valid)

            test = 100 - train - valid

            # Update only the source side to avoid any opposite-side slider motion.
            if source == "train":
                if self.train_spin.value() != train:
                    self.train_spin.setValue(train)
            elif source == "val":
                if self.valid_spin.value() != valid:
                    self.valid_spin.setValue(valid)
            else:
                if self.train_spin.value() != train:
                    self.train_spin.setValue(train)
                if self.valid_spin.value() != valid:
                    self.valid_spin.setValue(valid)

            if self.test_spin.value() != test:
                self.test_spin.setValue(test)
        finally:
            self._ratio_guard = False

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

        if is_data_split_running():
            QMessageBox.information(self, "Data split in progress", "Data split is already running.")
            return

        if is_project_compression_running():
            QMessageBox.information(
                self,
                "Compression in progress",
                "Project compression is running. Please wait until it finishes.",
            )
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
        set_data_split_running(True)
        try:
            self.split_worker.start()
        except Exception:
            set_data_split_running(False)
            self.run_btn.setEnabled(True)
            self.split_worker = None
            raise

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
        set_data_split_running(False)
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
