from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFontMetrics, QImage, QPainter, QPen
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog, QWidget

from utils.skeleton import SkeletonModel

from ..widget.image_label import CUTIE_COLOR_BASE, SKELETON_COLOR_SET
from .data_loader import DataLoader
from .save_files import _find_project, _sanitize_index


def _qimage_to_rgb_array(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    byte_count = image.height() * image.bytesPerLine()

    ptr = image.bits()
    if hasattr(ptr, "setsize"):
        ptr.setsize(byte_count)
        buffer = np.frombuffer(ptr, dtype=np.uint8, count=byte_count)
    elif hasattr(ptr, "asstring"):
        buffer = np.frombuffer(ptr.asstring(byte_count), dtype=np.uint8, count=byte_count)
    else:
        buffer = np.frombuffer(bytes(ptr), dtype=np.uint8, count=byte_count)

    rgb = buffer.reshape((image.height(), image.bytesPerLine() // 3, 3))
    return rgb[:, : image.width(), :].copy()


def _mixed_track_color(track: str, animals_name: list[str], color_mode: str) -> QColor:
    try:
        idx = animals_name.index(track)
    except ValueError:
        idx = 0
    base = QColor(CUTIE_COLOR_BASE[idx % len(CUTIE_COLOR_BASE)])
    other, mix_ratio = SKELETON_COLOR_SET.get(color_mode, SKELETON_COLOR_SET["cutie_light"])
    return QColor(
        round(base.red() * (1 - mix_ratio) + other.red() * mix_ratio),
        round(base.green() * (1 - mix_ratio) + other.green() * mix_ratio),
        round(base.blue() * (1 - mix_ratio) + other.blue() * mix_ratio),
    )


def _render_overlay_on_rgb_frame(
    frame_rgb: np.ndarray,
    *,
    skeleton_model: SkeletonModel,
    animals_name: list[str],
    color_mode: str,
    csv_points: dict[str, dict[str, tuple[float, float, int]]],
) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    qimg = QImage(
        frame_rgb.data,
        width,
        height,
        frame_rgb.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()
    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    for track in animals_name:
        pts = csv_points.get(track, {})
        if not pts:
            continue

        track_color = _mixed_track_color(track, animals_name, color_mode)
        edge_pen = QPen(track_color, 2)
        edge_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        edge_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(edge_pen)

        for edge in skeleton_model.edges:
            a, b = tuple(edge)
            if a not in pts or b not in pts:
                continue
            p1 = pts[a]
            p2 = pts[b]
            painter.drawLine(
                int(round(p1[0] * width)),
                int(round(p1[1] * height)),
                int(round(p2[0] * width)),
                int(round(p2[1] * height)),
            )

        for node_name, node in skeleton_model.nodes.items():
            if node_name not in pts:
                continue

            px, py, vis = pts[node_name]
            cx = px * width
            cy = py * height
            r = 5.0

            base_color = node.color
            pen = QPen(base_color, node.thickness)
            brush = QBrush(base_color if node.filled else Qt.BrushStyle.NoBrush)
            painter.setPen(pen)
            painter.setBrush(brush)

            shape = node.shape.lower()
            if vis == 1:
                d = r
                painter.drawLine(int(round(cx - d)), int(round(cy - d)), int(round(cx + d)), int(round(cy + d)))
                painter.drawLine(int(round(cx - d)), int(round(cy + d)), int(round(cx + d)), int(round(cy - d)))
            elif shape == "circle":
                painter.drawEllipse(int(round(cx - r)), int(round(cy - r)), int(round(2 * r)), int(round(2 * r)))
            elif shape == "square":
                painter.drawRect(int(round(cx - r)), int(round(cy - r)), int(round(2 * r)), int(round(2 * r)))
            elif shape == "text":
                txt = node.text or node.name
                painter.save()
                font = painter.font()
                font.setPixelSize(int(max(r * 3, 8)))
                painter.setFont(font)
                fm = QFontMetrics(font)
                text_w = fm.horizontalAdvance(txt)
                text_h = fm.height()
                painter.drawText(int(round(cx - text_w / 2)), int(round(cy + text_h / 4)), txt)
                painter.restore()
            else:
                painter.drawEllipse(int(round(cx - r)), int(round(cy - r)), int(round(2 * r)), int(round(2 * r)))

    painter.end()
    return _qimage_to_rgb_array(qimg)


class _VideoExportThread(QThread):
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        *,
        out_path: str,
        image_files: list[Path],
        frame_groups: dict[int, pd.DataFrame],
        skeleton_data: dict,
        animals_name: list[str],
        color_mode: str,
        coords_are_normalized: bool,
        width: int,
        height: int,
        fps: float,
        fourcc: int,
    ) -> None:
        super().__init__()
        self.out_path = out_path
        self.image_files = list(image_files)
        self.frame_groups = frame_groups
        self.skeleton_data = skeleton_data
        self.animals_name = list(animals_name)
        self.color_mode = color_mode
        self.coords_are_normalized = coords_are_normalized
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc

    def run(self) -> None:
        skeleton_model = SkeletonModel()
        skeleton_model.load_from_dict(self.skeleton_data)

        writer = cv2.VideoWriter(self.out_path, self.fourcc, self.fps, (self.width, self.height))
        if not writer.isOpened():
            self.error_signal.emit("Could not open video writer for output file.")
            return

        try:
            total = len(self.image_files)
            first_frame_is_one = bool(self.frame_groups and min(self.frame_groups) == 1)
            for index, img_path in enumerate(self.image_files):
                frame_img = cv2.imread(str(img_path))
                if frame_img is None:
                    self.progress_signal.emit(index + 1, total)
                    continue

                frame_num = index + 1 if first_frame_is_one else index
                frame_df = self.frame_groups.get(frame_num)
                if frame_df is not None:
                    frame_coords: dict[str, dict[str, tuple[float, float, int]]] = {}
                    for _, row in frame_df.iterrows():
                        track = str(row.get("track"))
                        kp_map: dict[str, tuple[float, float, int]] = {}
                        for node_name in skeleton_model.nodes:
                            x = row.get(f"{node_name}.x")
                            y = row.get(f"{node_name}.y")
                            vis = row.get(f"{node_name}.visibility")
                            if pd.isna(x) or pd.isna(y):
                                continue
                            px = float(x)
                            py = float(y)
                            if not self.coords_are_normalized:
                                px /= self.width
                                py /= self.height
                            kp_map[node_name] = (px, py, int(vis) if not pd.isna(vis) else 2)
                        if kp_map:
                            frame_coords[track] = kp_map

                    if frame_coords:
                        rgb_frame = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                        rendered_rgb = _render_overlay_on_rgb_frame(
                            rgb_frame,
                            skeleton_model=skeleton_model,
                            animals_name=self.animals_name,
                            color_mode=self.color_mode,
                            csv_points=frame_coords,
                        )
                        frame_img = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)

                writer.write(frame_img)
                self.progress_signal.emit(index + 1, total)
        except Exception as exc:
            writer.release()
            self.error_signal.emit(str(exc))
            return

        writer.release()
        self.finished_signal.emit(self.out_path)


def _export_video_stub(parent: QWidget) -> None:
    existing_thread = getattr(parent, "_video_export_thread", None)
    if existing_thread is not None and existing_thread.isRunning():
        QMessageBox.information(parent, "Video export", "A video export is already running.")
        return

    if DataLoader.loaded_data is None:
        QMessageBox.warning(parent, "Warning", "Load CSV/TXT first")
        return

    df = _sanitize_index(DataLoader.loaded_data.copy())

    project = _find_project(parent)
    if project is None or not hasattr(project, "project_dir"):
        QMessageBox.critical(parent, "Error", "Project information not found.")
        return
    project_dir = Path(project.project_dir)
    video_name = Path(parent.video_combo.currentText()).stem
    mode_text = parent.mode_combo.currentText() if hasattr(parent, "mode_combo") else "images"

    if mode_text == "video":
        mode_subdir = "images"
    elif mode_text == "images":
        mode_subdir = "images"
    elif mode_text == "davis":
        mode_subdir = "visualization/davis"
    elif mode_text == "contour":
        mode_subdir = "visualization/contour"
    else:
        mode_subdir = mode_text
    frames_dir = project_dir / "frames" / video_name / mode_subdir
    if not frames_dir.exists():
        QMessageBox.critical(parent, "Error", f"Frames directory not found:\n{frames_dir}")
        return

    image_files = sorted(frames_dir.glob("*.png"))
    if not image_files:
        image_files = sorted(frames_dir.glob("*.jpg"))
    if not image_files:
        image_files = sorted(frames_dir.glob("*.jpeg"))
    if not image_files:
        QMessageBox.critical(parent, "Error", f"No frame images found in {frames_dir}")
        return

    total_images = len(image_files)
    frame_indices = sorted(df["frame_idx"].unique().astype(int))
    total_frames = len(frame_indices)
    if total_images != total_frames:
        resp = QMessageBox.question(
            parent,
            "Frame count mismatch",
            (
                f"Number of frame images: {total_images}\n"
                f"Number of frames in skeleton data: {total_frames}\n\n"
                "Continue exporting the video with available frames?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

    now_str = datetime.now().strftime("%y%m%d%H%M")
    default_name = f"{video_name}_{now_str}.mp4"
    out_path, _ = QFileDialog.getSaveFileName(
        parent,
        "Export Video",
        str(project_dir / "outputs" / default_name),
        "MP4 Video (*.mp4);;AVI Video (*.avi);;All Files (*)",
    )
    if not out_path:
        return

    out_path = str(Path(out_path))
    if Path(out_path).suffix == "":
        out_path += ".mp4"

    ext = Path(out_path).suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*("XVID" if ext == ".avi" else "mp4v"))

    sample_img = cv2.imread(str(image_files[0]))
    if sample_img is None:
        QMessageBox.critical(parent, "Error", f"Failed to read frame image: {image_files[0].name}")
        return
    height, width = sample_img.shape[0:2]

    fps = 0.0
    video_path = None
    if hasattr(parent, "project") and parent.project:
        video_entries = [
            f for f in parent.project.files
            if Path(f.video) == parent.video_combo.currentData(Qt.ItemDataRole.UserRole)
        ]
        if video_entries:
            video_path = Path(video_entries[0].video)
    if video_path is None or not video_path.exists():
        vid_path = parent.video_combo.currentData(Qt.ItemDataRole.UserRole) if hasattr(parent, "video_combo") else None
        if vid_path is None:
            vid_path = Path(parent.video_combo.currentText()) if hasattr(parent, "video_combo") else None
        if vid_path and vid_path.exists():
            video_path = vid_path

    if video_path and video_path.exists():
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        except Exception:
            fps = 0.0
    if fps is None or fps <= 0:
        warnings.warn(
            f"Unable to read original fps from project file: {video_path}. Video playback fps is fixed to 30.",
            UserWarning,
        )
        fps = 30.0

    xy_cols = [c for c in df.columns if c.endswith((".x", ".y"))]
    coords_are_normalized = df[xy_cols].max().max() <= 1.0 + 1e-6
    frame_groups = {int(f): df[df["frame_idx"] == f].copy() for f in frame_indices}
    skeleton_data = parent.skeleton.to_dict()
    animals_name = list(parent.project.animals_name) if getattr(parent, "project", None) else list(DataLoader.animals_name or [])
    color_mode = getattr(parent.skeleton_video_viewer, "skeleton_color_mode", "cutie_light")

    progress = QProgressDialog("Exporting video...", "", 0, len(image_files), parent)
    progress.setWindowTitle("Video Export")
    progress.setWindowModality(Qt.WindowModality.NonModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(True)
    progress.setAutoReset(True)
    progress.setCancelButton(None)
    progress.setValue(0)
    progress.show()

    thread = _VideoExportThread(
        out_path=out_path,
        image_files=image_files,
        frame_groups=frame_groups,
        skeleton_data=skeleton_data,
        animals_name=animals_name,
        color_mode=color_mode,
        coords_are_normalized=coords_are_normalized,
        width=width,
        height=height,
        fps=fps,
        fourcc=fourcc,
    )

    def _on_progress(done: int, total: int) -> None:
        progress.setMaximum(total)
        progress.setValue(done)
        progress.setLabelText(f"Exporting video... {done}/{total}")

    def _cleanup_thread() -> None:
        parent._video_export_thread = None
        parent._video_export_progress = None

    def _on_finished(saved_path: str) -> None:
        progress.setValue(progress.maximum())
        progress.close()
        _cleanup_thread()
        QMessageBox.information(parent, "Success", f"Video Exported:\n{saved_path}")

    def _on_error(message: str) -> None:
        progress.close()
        _cleanup_thread()
        QMessageBox.critical(parent, "Error", f"Failed during video export:\n{message}")

    thread.progress_signal.connect(_on_progress)
    thread.finished_signal.connect(_on_finished)
    thread.error_signal.connect(_on_error)
    thread.finished.connect(thread.deleteLater)

    parent._video_export_thread = thread
    parent._video_export_progress = progress
    thread.start()
