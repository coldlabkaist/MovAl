from __future__ import annotations

import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pose.progress_ui import TaskProgressPanel
from pose.task_state import pose_execution_state
from pose.split_state import (
    is_data_split_running,
    set_data_split_running,
    update_data_split_progress,
)
from utils.runtime_locks import is_project_compression_running

ONLINE_DATASET_ROOT = "online_datasets"
LABEL_SOURCE_TXT = "txt"
LABEL_SOURCE_RECENT_CSV = "recent_csv"


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


def _most_recent_csv_path(project_dir: Path, video_name: str) -> Path | None:
    csv_dir = project_dir / "labels" / video_name / "csv"
    candidates = [path for path in csv_dir.glob("*.csv") if path.is_file()]
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, str]:
        try:
            modified_ns = path.stat().st_mtime_ns
        except OSError:
            modified_ns = 0
        return modified_ns, path.name.casefold()

    return max(candidates, key=sort_key)


def _count_csv_labeled_frames(csv_path: Path) -> int:
    frame_data = pd.read_csv(csv_path, usecols=["frame_idx"])
    frame_numbers = pd.to_numeric(frame_data["frame_idx"], errors="coerce")
    return int(frame_numbers.dropna().nunique())


def _project_keypoint_names(current_project) -> list[str]:
    skeleton_data = getattr(current_project, "skeleton_data", {}) or {}
    names = [
        str(node.get("name"))
        for node in skeleton_data.get("nodes", [])
        if isinstance(node, dict) and node.get("name")
    ]
    if not names:
        raise ValueError("The project skeleton does not contain any keypoints.")
    return names


def _resolve_csv_track_mapping(
    raw_tracks: list[str],
    animal_names: list[str],
    csv_path: Path,
) -> dict[str, str]:
    project_names = [str(name) for name in animal_names]
    if len(raw_tracks) > len(project_names):
        raise ValueError(
            f"{csv_path.name} contains {len(raw_tracks)} tracks, but the project "
            f"contains only {len(project_names)} IDs."
        )

    mapping: dict[str, str] = {}
    used_names: set[str] = set()
    unresolved: list[str] = []
    for raw_track in raw_tracks:
        if raw_track in project_names and raw_track not in used_names:
            mapping[raw_track] = raw_track
            used_names.add(raw_track)
        else:
            unresolved.append(raw_track)

    still_unresolved: list[str] = []
    for raw_track in unresolved:
        match = re.fullmatch(r"(?:track_)?(\d+)", raw_track, flags=re.IGNORECASE)
        track_index = int(match.group(1)) if match else -1
        if 0 <= track_index < len(project_names):
            mapped_name = project_names[track_index]
            if mapped_name not in used_names:
                mapping[raw_track] = mapped_name
                used_names.add(mapped_name)
                continue
        still_unresolved.append(raw_track)

    remaining_names = [name for name in project_names if name not in used_names]
    if len(still_unresolved) == 1 and len(remaining_names) == 1:
        mapping[still_unresolved[0]] = remaining_names[0]
        still_unresolved.clear()

    if still_unresolved:
        unresolved_text = ", ".join(still_unresolved)
        raise ValueError(
            f"Track mapping is ambiguous in {csv_path.name}: {unresolved_text}. "
            "Open this CSV in Labelary, confirm the track mapping, and save it before splitting."
        )
    return mapping


def _video_dimensions(video_path: Path) -> tuple[int, int]:
    capture = _open_video_capture(video_path)
    if capture is None:
        raise RuntimeError(f"Unable to open video for coordinate normalization: {video_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Unable to read video dimensions: {video_path}")
    return width, height


def _load_csv_pose_dataframe(current_project, file_entry, csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)
    required_base = {"track", "frame_idx"}
    missing_base = sorted(required_base.difference(df.columns))
    if missing_base:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {', '.join(missing_base)}"
        )
    if df.empty:
        raise ValueError(f"{csv_path.name} does not contain any labels.")
    if df["track"].isna().any():
        raise ValueError(f"{csv_path.name} contains an empty track name.")

    keypoint_names = _project_keypoint_names(current_project)
    missing_coordinates = [
        column
        for keypoint in keypoint_names
        for column in (f"{keypoint}.x", f"{keypoint}.y")
        if column not in df.columns
    ]
    if missing_coordinates:
        raise ValueError(
            f"{csv_path.name} does not match the project skeleton. Missing columns: "
            f"{', '.join(missing_coordinates)}"
        )

    raw_tracks = df["track"].astype(str).drop_duplicates().tolist()
    track_mapping = _resolve_csv_track_mapping(
        raw_tracks,
        list(current_project.animals_name),
        csv_path,
    )
    df["track"] = df["track"].astype(str).map(track_mapping)
    if df["track"].isna().any():
        raise ValueError(f"Failed to map every track in {csv_path.name}.")

    frame_numbers = pd.to_numeric(df["frame_idx"], errors="coerce")
    if (
        frame_numbers.isna().any()
        or (frame_numbers < 0).any()
        or (frame_numbers % 1 != 0).any()
    ):
        raise ValueError(f"{csv_path.name} contains invalid frame indices.")
    df["frame_idx"] = frame_numbers.astype(int)

    x_columns = [f"{keypoint}.x" for keypoint in keypoint_names]
    y_columns = [f"{keypoint}.y" for keypoint in keypoint_names]
    visibility_columns = [f"{keypoint}.visibility" for keypoint in keypoint_names]
    coordinate_columns = [*x_columns, *y_columns]
    for column in coordinate_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    coordinates = df[coordinate_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{csv_path.name} contains missing or non-numeric coordinates.")

    if coordinates.size:
        coordinate_max = float(np.max(coordinates))
        if 1.01 < coordinate_max <= 2.0:
            raise ValueError(
                f"{csv_path.name} contains coordinates just outside the normalized 0-1 range."
            )
        if coordinate_max > 2.0:
            width, height = _video_dimensions(Path(file_entry.video))
            df[x_columns] = df[x_columns] / width
            df[y_columns] = df[y_columns] / height
            coordinates = df[coordinate_columns].to_numpy(dtype=np.float64)

    if coordinates.size and (
        float(np.min(coordinates)) < -0.01 or float(np.max(coordinates)) > 1.01
    ):
        raise ValueError(
            f"{csv_path.name} contains coordinates outside the normalized 0-1 range."
        )

    for column in visibility_columns:
        if column not in df.columns:
            df[column] = 2
        df[column] = (
            pd.to_numeric(df[column], errors="coerce").fillna(0).clip(0, 2).astype(int)
        )

    score_values = (
        df["instance.score"]
        if "instance.score" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    df["_instance_score_sort"] = pd.to_numeric(
        score_values, errors="coerce"
    ).fillna(0.0)
    sort_columns = ["frame_idx", "track", "_instance_score_sort"]
    ascending = [True, True, False]
    if "instance.id" in df.columns:
        df["instance.id"] = pd.to_numeric(df["instance.id"], errors="coerce")
        sort_columns.append("instance.id")
        ascending.append(True)
        df = (
            df.sort_values(sort_columns, ascending=ascending, kind="stable")
            .groupby(
                ["frame_idx", "track", "instance.id"],
                dropna=False,
                sort=False,
                group_keys=False,
            )
            .head(1)
        )

    instance_limit = current_project.get_max_instances_per_id()
    df = (
        df.sort_values(sort_columns, ascending=ascending, kind="stable")
        .groupby(["frame_idx", "track"], sort=False, group_keys=False)
        .head(instance_limit)
        .drop(columns=["_instance_score_sort"], errors="ignore")
        .reset_index(drop=True)
    )
    return df, keypoint_names


def _write_pose_txt_files(
    target_dir: Path,
    df: pd.DataFrame,
    animal_names: list[str],
    keypoint_names: list[str],
    *,
    should_cancel=None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    frame_numbers = df["frame_idx"].to_numpy(dtype=np.int64)
    max_frame = int(frame_numbers.max())
    padding = max(2, len(str(max_frame)))
    class_ids = df["track"].map(
        {str(name): index for index, name in enumerate(animal_names)}
    )
    if class_ids.isna().any():
        raise ValueError("CSV contains a track that is not part of the project.")

    x_values = df[[f"{name}.x" for name in keypoint_names]].to_numpy(dtype=np.float64)
    y_values = df[[f"{name}.y" for name in keypoint_names]].to_numpy(dtype=np.float64)
    visibility_values = df[
        [f"{name}.visibility" for name in keypoint_names]
    ].to_numpy(dtype=np.int8)
    class_values = class_ids.to_numpy(dtype=np.int32)

    for frame_idx in np.unique(frame_numbers):
        _raise_if_cancelled(should_cancel)
        lines: list[str] = []
        for row_index in np.flatnonzero(frame_numbers == frame_idx):
            xs = x_values[row_index]
            ys = y_values[row_index]
            values = [
                float((xs.min() + xs.max()) / 2),
                float((ys.min() + ys.max()) / 2),
                float(xs.max() - xs.min()),
                float(ys.max() - ys.min()),
            ]
            parts = [str(int(class_values[row_index]))]
            parts.extend(f"{value:.6f}" for value in values)
            for keypoint_index in range(len(keypoint_names)):
                parts.extend(
                    (
                        f"{float(xs[keypoint_index]):.6f}",
                        f"{float(ys[keypoint_index]):.6f}",
                        str(int(visibility_values[row_index, keypoint_index])),
                    )
                )
            lines.append(" ".join(parts))

        (target_dir / f"{int(frame_idx):0{padding}d}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )


def _prepare_recent_csv_label_dirs(
    current_project,
    selected_entries,
    target_root: Path,
    *,
    progress_callback=None,
    should_cancel=None,
) -> dict[str, Path]:
    project_dir = Path(current_project.project_dir)
    label_dirs: dict[str, Path] = {}
    total = len(selected_entries)
    for index, file_entry in enumerate(selected_entries, start=1):
        _raise_if_cancelled(should_cancel)
        csv_path = _most_recent_csv_path(project_dir, file_entry.name)
        if csv_path is None:
            raise ValueError(f"No CSV file was found for {file_entry.name}.")
        if progress_callback is not None:
            progress_callback(
                0,
                0,
                f"Converting most recent CSV ({index}/{total}): {csv_path.name}",
            )

        dataframe, keypoint_names = _load_csv_pose_dataframe(
            current_project,
            file_entry,
            csv_path,
        )
        output_dir = target_root / file_entry.name
        _write_pose_txt_files(
            output_dir,
            dataframe,
            list(current_project.animals_name),
            keypoint_names,
            should_cancel=should_cancel,
        )
        label_dirs[file_entry.name] = output_dir
    return label_dirs


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

        def write_target_frame(frame_idx: int, frame) -> None:
            nonlocal written
            for split, sample in targets[frame_idx]:
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

        try:
            while current_idx <= max_target:
                _raise_if_cancelled(should_cancel)
                ok, frame = cap.read()
                if not ok or frame is None:
                    missing = []
                    for target_idx in target_frames:
                        if target_idx < current_idx:
                            continue
                        _raise_if_cancelled(should_cancel)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                        recovered, recovered_frame = cap.read()
                        if not recovered or recovered_frame is None:
                            missing.append(target_idx)
                            continue
                        write_target_frame(target_idx, recovered_frame)

                    if not missing:
                        break
                    preview = ", ".join(str(idx) for idx in missing[:20])
                    more = "" if len(missing) <= 20 else f", ... and {len(missing) - 20} more"
                    raise RuntimeError(
                        f"Video ended or failed before labeled frames could be decoded: "
                        f"{video_path}\nMissing frame indices: {preview}{more}"
                    )

                if current_idx in targets:
                    write_target_frame(current_idx, frame)
                current_idx += 1
        finally:
            cap.release()


def create_dataset_split(
    current_project,
    selected_entries,
    frame_type: str,
    dataset_dir: str | Path,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
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
    FRAME_COUNT_ROLE = int(Qt.ItemDataRole.UserRole) + 1
    LABEL_COUNT_ROLE = int(Qt.ItemDataRole.UserRole) + 2

    def __init__(self, current_project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Split")
        self.setMinimumSize(700, 580)
        self.resize(760, 640)

        self.current_project = current_project
        self.files = current_project.files
        self._entries_by_name = {entry.name: entry for entry in self.files}
        self.split_worker = None
        self._ratio_guard = False
        self._csv_label_count_cache: dict[tuple[str, int, int], int] = {}

        layout = QVBoxLayout(self)
        selection_hint = QLabel("Select videos for the dataset. Use Ctrl or Shift to select multiple rows.")
        selection_hint.setObjectName("SubtleText")
        selection_hint.setWordWrap(True)
        layout.addWidget(selection_hint)

        self.file_tree = QTreeWidget()
        self.file_tree.setColumnCount(3)
        self.file_tree.setHeaderLabels(["Video", "Frames", "Labels"])
        self.file_tree.setRootIsDecorated(False)
        self.file_tree.setAlternatingRowColors(True)
        self.file_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.file_tree.setColumnWidth(0, 390)
        self.file_tree.setColumnWidth(1, 100)
        self.file_tree.itemSelectionChanged.connect(self._update_selection_count)
        layout.addWidget(self.file_tree, 1)

        selection_buttons = QHBoxLayout()
        self.select_all_button = QPushButton("Select All")
        self.clear_selection_button = QPushButton("Clear Selection")
        self.invert_selection_button = QPushButton("Invert Selection")
        self.select_all_button.clicked.connect(self.file_tree.selectAll)
        self.clear_selection_button.clicked.connect(self.file_tree.clearSelection)
        self.invert_selection_button.clicked.connect(self._invert_selection)
        selection_buttons.addWidget(self.select_all_button)
        selection_buttons.addWidget(self.clear_selection_button)
        selection_buttons.addWidget(self.invert_selection_button)
        selection_buttons.addStretch(1)
        layout.addLayout(selection_buttons)
        self.count_label = QLabel("0 files selected / 0 labels")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        count_font = QFont()
        count_font.setPointSize(11)
        self.count_label.setFont(count_font)
        layout.addWidget(self.count_label)

        layout.addSpacing(20)

        ratio_layout = QFormLayout()

        self.train_slider = QSlider(Qt.Orientation.Horizontal)
        self.train_slider.setRange(40, 90)
        self.train_slider.setValue(85)
        self.train_spin = QSpinBox()
        self.train_spin.setRange(40, 90)
        self.train_spin.setValue(85)
        self.train_slider.valueChanged.connect(self.train_spin.setValue)
        self.train_spin.valueChanged.connect(self.train_slider.setValue)
        ratio_layout.addRow("Train %", self.create_slider_spinbox_layout(self.train_slider, self.train_spin))

        self.valid_slider = QSlider(Qt.Orientation.Horizontal)
        self.valid_slider.setRange(5, 55)
        self.valid_slider.setValue(10)
        self.valid_spin = QSpinBox()
        self.valid_spin.setRange(5, 55)
        self.valid_spin.setValue(10)
        self.valid_slider.valueChanged.connect(self.valid_spin.setValue)
        self.valid_spin.valueChanged.connect(self.valid_slider.setValue)
        ratio_layout.addRow("Valid %", self.create_slider_spinbox_layout(self.valid_slider, self.valid_spin))

        self.test_slider = QSlider(Qt.Orientation.Horizontal)
        self.test_slider.setRange(5, 55)
        self.test_slider.setValue(5)
        self.test_spin = QSpinBox()
        self.test_spin.setRange(5, 55)
        self.test_spin.setValue(5)
        self.test_slider.valueChanged.connect(self.test_spin.setValue)
        self.test_spin.valueChanged.connect(self.test_slider.setValue)
        self.test_slider.setToolTip("Calculated automatically from Train and Valid percentages")
        self.test_spin.setToolTip("Calculated automatically from Train and Valid percentages")
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

        self.label_source_combo = QComboBox()
        self.label_source_combo.addItem("TXT", LABEL_SOURCE_TXT)
        self.label_source_combo.addItem("Most recent CSV", LABEL_SOURCE_RECENT_CSV)
        self.label_source_combo.setToolTip(
            "TXT uses the project's existing TXT labels. Most recent CSV selects "
            "the newest CSV for each video and converts it to temporary YOLO TXT labels."
        )

        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Frame source"))
        source_layout.addWidget(self.frame_type_combo, 1)
        source_layout.addWidget(QLabel("Label source"))
        source_layout.addWidget(self.label_source_combo, 1)
        layout.addLayout(source_layout)

        self.progress_panel = TaskProgressPanel(self)
        layout.addWidget(self.progress_panel)

        self.run_btn = QPushButton("Run")
        self.run_btn.setProperty("primary", True)
        layout.addWidget(self.run_btn)
        self.run_btn.clicked.connect(self._on_run_button_clicked)

        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        self._populate_file_items()
        self.frame_type_combo.currentTextChanged.connect(self._frame_type_changed)
        self.frame_type_combo.currentTextChanged.connect(self._save_frame_type)
        self.label_source_combo.currentIndexChanged.connect(self._label_source_changed)
        self.train_spin.valueChanged.connect(lambda v: self._on_ratio_input_changed("train", v))
        self.valid_spin.valueChanged.connect(lambda v: self._on_ratio_input_changed("val", v))
        self._apply_ratio_constraints(self.train_spin.value(), self.valid_spin.value(), source="train")

    def _frame_type_changed(self):
        selected_names = self._selected_video_names()
        self._populate_file_items(selected_names=selected_names)
        self._update_selection_count()

    def _label_source_changed(self):
        selected_names = self._selected_video_names()
        self._populate_file_items(selected_names=selected_names)
        self._update_selection_count()

    def _current_label_source(self) -> str:
        return str(self.label_source_combo.currentData() or LABEL_SOURCE_TXT)

    def _save_frame_type(self, frame_type: str) -> None:
        self.current_project.set_preferred_frame_mode(frame_type)

    def _populate_file_items(self, selected_names: set[str] | None = None) -> None:
        selected_names = selected_names or set()
        self.file_tree.clear()
        current_project = self.current_project
        frame_type = self.frame_type_combo.currentText()
        label_source = self._current_label_source()
        project_dir = Path(current_project.project_dir)
        for file_entry in self.files:
            video_path = Path(file_entry.video)
            video_name = file_entry.name
            label_dir = project_dir / "labels" / video_name / "txt"
            if frame_type == "video":
                frame_count = 0
                frame_text = "—"
            else:
                frame_dir = _resolve_frame_dir(project_dir, video_name, frame_type)
                frame_count = sum(1 for _ in frame_dir.glob("*.jpg"))
                frame_text = f"{frame_count:,}"

            label_tooltip = str(label_dir)
            if label_source == LABEL_SOURCE_RECENT_CSV:
                csv_path = _most_recent_csv_path(project_dir, video_name)
                label_count = 0
                if csv_path is not None:
                    try:
                        stat = csv_path.stat()
                        cache_key = (str(csv_path), stat.st_mtime_ns, stat.st_size)
                        if cache_key not in self._csv_label_count_cache:
                            self._csv_label_count_cache[cache_key] = _count_csv_labeled_frames(
                                csv_path
                            )
                        label_count = self._csv_label_count_cache[cache_key]
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        UnicodeError,
                        pd.errors.ParserError,
                    ):
                        label_count = 0
                    label_tooltip = f"Most recent CSV: {csv_path}"
                else:
                    label_tooltip = "No CSV file found"
            else:
                label_count = sum(1 for _ in label_dir.glob("*.txt"))

            item = QTreeWidgetItem([video_path.name, frame_text, f"{label_count:,}"])
            item.setData(0, Qt.ItemDataRole.UserRole, video_name)
            item.setData(0, self.FRAME_COUNT_ROLE, frame_count)
            item.setData(0, self.LABEL_COUNT_ROLE, label_count)
            item.setToolTip(0, str(video_path))
            item.setToolTip(2, label_tooltip)
            self.file_tree.addTopLevelItem(item)
            item.setSelected(video_name in selected_names)
        self._update_selection_count()

    def _selected_video_names(self) -> set[str]:
        return {
            str(item.data(0, Qt.ItemDataRole.UserRole))
            for item in self.file_tree.selectedItems()
        }

    def _invert_selection(self) -> None:
        for index in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(index)
            item.setSelected(not item.isSelected())

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

    def _update_selection_count(self):
        selected_items = self.file_tree.selectedItems()
        total_files = len(selected_items)
        total_frames = sum(int(item.data(0, self.FRAME_COUNT_ROLE) or 0) for item in selected_items)
        total_labels = sum(int(item.data(0, self.LABEL_COUNT_ROLE) or 0) for item in selected_items)
        frame_type = self.frame_type_combo.currentText()

        if frame_type == "video":
            self.count_label.setText(f"{total_files} files selected / {total_labels:,} labels")
        else:
            self.count_label.setText(
                f"{total_files} files selected / {total_frames:,} frames / {total_labels:,} labels"
            )

    def get_selected_entries(self):
        selected_names = self._selected_video_names()
        return [
            self._entries_by_name[name]
            for name in self._entries_by_name
            if name in selected_names
        ]

    def _set_split_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.file_tree,
            self.select_all_button,
            self.clear_selection_button,
            self.invert_selection_button,
            self.train_slider,
            self.train_spin,
            self.valid_slider,
            self.valid_spin,
            self.frame_type_combo,
            self.label_source_combo,
        ):
            widget.setEnabled(enabled)

    def _on_run_button_clicked(self) -> None:
        if self.split_worker is not None and self.split_worker.isRunning():
            self.split_worker.request_cancel()
            self.progress_panel.set_stopping("Stopping data split...")
            self.run_btn.setEnabled(False)
            self.run_btn.setText("Stopping...")
            return
        self.run_split()

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

        label_source = self._current_label_source()
        if label_source == LABEL_SOURCE_RECENT_CSV:
            project_dir = Path(self.current_project.project_dir)
            missing_csv = [
                entry.name
                for entry in selected_entries
                if _most_recent_csv_path(project_dir, entry.name) is None
            ]
            if missing_csv:
                QMessageBox.warning(
                    self,
                    "CSV labels not found",
                    "No CSV file was found for:\n" + "\n".join(missing_csv),
                )
                return

        active_task = (pose_execution_state.active_task() or "").lower()
        if pose_execution_state.is_busy() and active_task == "training":
            QMessageBox.information(
                self,
                "Training in progress",
                "Dataset preparation is disabled while training is running.",
            )
            return

        self._set_split_controls_enabled(False)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Stop")
        initial_message = (
            "Converting most recent CSV labels..."
            if label_source == LABEL_SOURCE_RECENT_CSV
            else "Collecting labeled frames..."
        )
        self.progress_panel.reset(initial_message)

        dataset_dir = Path(self.current_project.project_dir) / "runs" / "dataset"
        self.split_worker = DataSplitWorker(
            current_project=self.current_project,
            selected_entries=selected_entries,
            frame_type=self.frame_type_combo.currentText(),
            label_source=label_source,
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
            self._set_split_controls_enabled(True)
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Run")
            self.split_worker = None
            raise

    def _on_split_progress(self, done: int, total: int, message: str):
        detail = f"{done:,} / {total:,} samples" if total > 0 else ""
        self.progress_panel.update_progress(done, total, message, detail)
        update_data_split_progress(done, total, message)

    def _on_split_success(self, split_counts: dict):
        detail = (
            f"Train {split_counts['train']:,} · Val {split_counts['val']:,} · "
            f"Test {split_counts['test']:,}"
        )
        self.progress_panel.set_result("Dataset split complete", success=True, detail=detail)
        QMessageBox.information(
            self,
            "Success",
            (f"Data Split completed\n"
             f"Train: {split_counts['train']}\n"
             f"Val:   {split_counts['val']}\n"
             f"Test:  {split_counts['test']}")
        )

    def _on_split_failure(self, error_text: str):
        self.progress_panel.set_result("Data split failed", success=False, detail=error_text)
        if error_text.startswith("ValueError: "):
            QMessageBox.warning(self, "Error", error_text[len("ValueError: "):])
        else:
            QMessageBox.critical(self, "Error", f"Failed to create dataset split:\n{error_text}")

    def _on_split_cancelled(self):
        self.progress_panel.set_result(
            "Data split cancelled",
            success=False,
            cancelled=True,
        )
        QMessageBox.information(self, "Cancelled", "Data split was cancelled.")

    def _on_split_finished(self):
        set_data_split_running(False)
        self._set_split_controls_enabled(True)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run")
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
        label_source: str,
        dataset_dir: Path,
        train_ratio: float,
        val_ratio: float,
    ):
        super().__init__()
        self.current_project = current_project
        self.selected_entries = list(selected_entries)
        self.frame_type = frame_type
        self.label_source = str(label_source)
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
        temporary_labels: TemporaryDirectory | None = None
        try:
            label_dirs = None
            if self.label_source == LABEL_SOURCE_RECENT_CSV:
                runs_dir = Path(self.current_project.project_dir) / "runs"
                runs_dir.mkdir(parents=True, exist_ok=True)
                temporary_labels = TemporaryDirectory(
                    prefix=".csv_split_labels_",
                    dir=runs_dir,
                )
                label_dirs = _prepare_recent_csv_label_dirs(
                    self.current_project,
                    self.selected_entries,
                    Path(temporary_labels.name),
                    progress_callback=self._report_progress,
                    should_cancel=self._is_cancel_requested,
                )

            split_counts = create_dataset_split(
                self.current_project,
                self.selected_entries,
                self.frame_type,
                self.dataset_dir,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                clear_existing=True,
                label_dirs=label_dirs,
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
        finally:
            if temporary_labels is not None:
                temporary_labels.cleanup()

    def _report_progress(self, done: int, total: int, message: str):
        self.progress.emit(int(done), int(total), str(message))
