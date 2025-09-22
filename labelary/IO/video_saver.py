from PyQt6.QtWidgets import (
    QMessageBox, QWidget, QFileDialog
)
from PyQt6.QtCore import Qt
import pandas as pd
from pathlib import Path
import cv2
from .data_loader import DataLoader
from .save_files import _sanitize_index, _find_project
import re
from datetime import datetime
from tqdm import tqdm
import warnings

def _export_video_stub(parent: QWidget) -> None:
    if DataLoader.loaded_data is None:
        QMessageBox.warning(parent, "Warning", "Load CSV/TXT first")
        return

    from .save_files import _sanitize_index
    df = _sanitize_index(DataLoader.loaded_data.copy())

    project = _find_project(parent)
    if project is None or not hasattr(project, "project_dir"):
        QMessageBox.critical(parent, "Error", "Project information not found.")
        return
    project_dir = Path(project.project_dir)
    video_name = Path(parent.video_combo.currentText()).stem
    mode_text = parent.mode_combo.currentText() if hasattr(parent, "mode_combo") else "images"

    if mode_text == "images":
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
    if not image_files or len(image_files) == 0:
        QMessageBox.critical(parent, "Error", f"No frame images found in {frames_dir}")
        return

    total_images = len(image_files)
    frame_indices = sorted(df["frame_idx"].unique().astype(int))
    total_frames = len(frame_indices)

    if total_images != total_frames:
        resp = QMessageBox.question(
            parent, "Frame count mismatch",
            (f"Number of frame images: {total_images}\n"
             f"Number of frames in skeleton data: {total_frames}\n\n"
             "Continue exporting the video with available frames?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

    now_str = datetime.now().strftime("%y%m%d%H%M")
    default_name = f"{video_name}_{now_str}.mp4"
    out_path, filter_sel = QFileDialog.getSaveFileName(
        parent, "Export Video", str(project_dir / "outputs" / default_name),
        "MP4 Video (*.mp4);;AVI Video (*.avi);;All Files (*)"
    )
    if not out_path:
        return 

    out_path = str(Path(out_path)) 
    if Path(out_path).suffix == "":
        out_path += ".mp4"

    ext = Path(out_path).suffix.lower()
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") 

    sample_img = cv2.imread(str(image_files[0]))
    if sample_img is None:
        QMessageBox.critical(parent, "Error", f"Failed to read frame image: {image_files[0].name}")
        return
    height, width = sample_img.shape[0:2]

    fps = 0.0
    video_path = None
    if hasattr(parent, "project") and parent.project:
        video_entries = [f for f in parent.project.files if Path(f.video) == parent.video_combo.currentData(Qt.ItemDataRole.UserRole)]
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
        warnings.warn(f"Unable to read original fps from project file: {video_path}. It's possible that the video directory specified in the project's config file wasn't read."
                    "Check the project's config.py file and make sure the directory is set properly. Video playback fps is fixed to 30.", UserWarning)
        fps = 30.0 

    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        QMessageBox.critical(parent, "Error", "Could not open video writer for output file.")
        return

    height, width = sample_img.shape[:2]
    xy_cols = [c for c in df.columns if c.endswith((".x", ".y"))]
    coords_are_normalized = df[xy_cols].max().max() <= 1.0 + 1e-6

    try:
        frame_groups = {int(f): df[df["frame_idx"] == f] for f in frame_indices}
        pad_width = len(image_files[0].stem)

        for i, img_path in tqdm(enumerate(image_files),
                        total=len(image_files),
                        desc="Exporting video",
                        unit="frame"):
            frame_img = cv2.imread(str(img_path))
            if frame_img is None:
                continue

            frame_num = i
            if frame_indices and frame_indices[0] == 1:
                frame_num = i + 1

            if frame_num in frame_groups:
                f_df = frame_groups[frame_num]
                for _, row in f_df.iterrows():
                    track_label = row.get("track")
                    if isinstance(track_label, str):
                        match = re.search(r"(\d+)$", track_label)
                        track_id = int(match.group(1)) if match else 0
                    else:
                        track_id = int(track_label) if track_label is not None else 0
                    qcol = parent.skeleton_video_viewer._skeleton_color(track_id)
                    color = (int(qcol.blue()), int(qcol.green()), int(qcol.red()))

                    for edge in parent.skeleton.edges:
                        nodes = list(edge)
                        if len(nodes) != 2:
                            continue
                        name1, name2 = nodes[0], nodes[1]
                        x1, y1 = row.get(f"{name1}.x"), row.get(f"{name1}.y")
                        x2, y2 = row.get(f"{name2}.x"), row.get(f"{name2}.y")
                        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                            pt1 = (int(round(x1)), int(round(y1)))
                            pt2 = (int(round(x2)), int(round(y2)))
                            cv2.line(frame_img, pt1, pt2, color, thickness=2)

                    for node_name, node in parent.skeleton.nodes.items():
                        x = row.get(f"{node_name}.x")
                        y = row.get(f"{node_name}.y")
                        vis = row.get(f"{node_name}.visibility")
                        if x is None or y is None:
                            continue 
                        if coords_are_normalized:
                            cx, cy = int(round(x * width)), int(round(y * height))
                        else:
                            cx, cy = int(round(x)), int(round(y))
                        if node.shape == "circle":
                            radius = 3
                            if node.filled:
                                cv2.circle(frame_img, (cx, cy), radius, color, thickness=-1)
                            else:
                                cv2.circle(frame_img, (cx, cy), radius, color, thickness=node.thickness or 1)
                        elif node.shape == "square":
                            half = 3
                            if node.filled:
                                cv2.rectangle(frame_img, (cx - half, cy - half), (cx + half, cy + half), color, thickness=-1)
                            else:
                                cv2.rectangle(frame_img, (cx - half, cy - half), (cx + half, cy + half), color, thickness=node.thickness or 1)
                        elif node.shape == "text":
                            text = node.text if node.text is not None else str(track_id)
                            cv2.putText(frame_img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            writer.write(frame_img)

    except Exception as e:
        writer.release()
        QMessageBox.critical(parent, "Error", f"Failed during video export:\n{e}")
        return

    writer.release()
    QMessageBox.information(parent, "Success", f"✅ Video Exported:\n{out_path}")
