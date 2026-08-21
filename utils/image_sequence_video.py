from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union
import os
import re
import shutil
import time

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".m4v", ".mkv"}
ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]
ValidationMode = Union[str, bool]


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    fourcc: Optional[str]


@dataclass(frozen=True)
class VideoValidationReport:
    frames_compared: int
    max_mean_abs_diff: float
    max_frame_abs_diff: int


@dataclass(frozen=True)
class ImageSequenceVideoResult:
    output_path: Path
    fps: float
    codec: str
    width: int
    height: int
    padded: bool
    frame_count: int
    validation_mode: str
    validation: Optional[VideoValidationReport]


class VideoEncodingError(RuntimeError):
    pass


class VideoValidationError(RuntimeError):
    pass


class VideoEncodingCancelled(RuntimeError):
    pass


def natural_image_files(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    return sorted(
        (item for item in root.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: _natural_key(path.name),
    )


def read_video_metadata(video_path: str | Path) -> Optional[VideoMetadata]:
    path = Path(video_path)
    if not path.is_file():
        return None

    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc = _fourcc_int_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        return VideoMetadata(
            path=path,
            fps=fps if fps > 0 else 30.0,
            frame_count=max(0, frame_count),
            width=max(0, width),
            height=max(0, height),
            fourcc=fourcc,
        )
    finally:
        cap.release()


def default_video_path(output_dir: str | Path, source_name: str, reference_video: str | Path | None = None) -> Path:
    suffix = ".mp4"
    if reference_video is not None:
        ref_suffix = Path(reference_video).suffix.lower()
        if ref_suffix == ".avi":
            suffix = ".avi"
    return Path(output_dir) / f"{source_name}{suffix}"


def encode_frame_sequence_to_video(
    frames: Iterable[np.ndarray],
    output_path: str | Path,
    *,
    fps: float | None = None,
    reference_video: str | Path | None = None,
    fourcc: str | None = None,
    frame_count: int | None = None,
    validate: ValidationMode = "container",
    min_free_bytes: int | None = None,
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
    progress_label: str = "Encoding video",
) -> ImageSequenceVideoResult:
    iterator = iter(frames)
    try:
        _raise_if_cancelled(should_cancel)
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("No video frames were provided.") from exc

    output = Path(output_path)
    if output.suffix == "":
        output = default_video_path(output.parent, output.name, reference_video)
    if output.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video output extension: {output.suffix}")

    metadata = read_video_metadata(reference_video) if reference_video is not None else None
    output_fps = _safe_fps(fps if fps is not None else (metadata.fps if metadata else None))
    first = _ensure_bgr_frame(first, source="first frame")
    src_height, src_width = first.shape[:2]
    width = _even_dimension(src_width)
    height = _even_dimension(src_height)
    padded = width != src_width or height != src_height

    output.parent.mkdir(parents=True, exist_ok=True)
    _ensure_free_space_for_bytes(output.parent, min_free_bytes)
    temp_path = _temporary_video_path(output)
    codec_candidates = _codec_candidates(output, reference_metadata=metadata, requested_fourcc=fourcc)

    writer = None
    selected_codec = ""
    for candidate in codec_candidates:
        _raise_if_cancelled(should_cancel)
        writer = cv2.VideoWriter(
            str(temp_path),
            cv2.VideoWriter_fourcc(*candidate),
            output_fps,
            (width, height),
        )
        if writer.isOpened():
            selected_codec = candidate
            break
        writer.release()
        writer = None

    if writer is None:
        raise VideoEncodingError(
            f"Could not open a video writer for {output.suffix}. Tried codecs: {', '.join(codec_candidates)}"
        )

    written = 0
    total = int(frame_count or 0)
    try:
        for frame in _prepend(first, iterator):
            _raise_if_cancelled(should_cancel)
            frame = _prepare_frame(
                _ensure_bgr_frame(frame, source=f"frame {written}"),
                width=width,
                height=height,
                source_path=Path(f"frame {written}"),
            )
            writer.write(frame)
            written += 1
            if progress_callback is not None:
                shown_total = total if total > 0 else written
                progress_callback(written, shown_total, f"{progress_label} ({written}/{shown_total})")
    except Exception:
        writer.release()
        _unlink_quietly(temp_path)
        raise

    writer.release()
    if total > 0 and written != total:
        _unlink_quietly(temp_path)
        raise VideoEncodingError(f"Encoded {written} frames, expected {total} frames.")

    validation_mode = _normalize_validation_mode(validate)
    try:
        if validation_mode == "container":
            validate_video_container(temp_path, expected_frame_count=written, width=width, height=height)
        elif validation_mode not in {"none", "container"}:
            raise ValueError("Frame stream validation only supports none or container mode.")
        temp_path.replace(output)
    except Exception:
        _unlink_quietly(temp_path)
        raise

    return ImageSequenceVideoResult(
        output_path=output,
        fps=output_fps,
        codec=selected_codec,
        width=width,
        height=height,
        padded=padded,
        frame_count=written,
        validation_mode=validation_mode,
        validation=None,
    )


def encode_image_sequence_to_video(
    image_files: Sequence[str | Path] | Iterable[str | Path],
    output_path: str | Path,
    *,
    fps: float | None = None,
    reference_video: str | Path | None = None,
    fourcc: str | None = None,
    validate: ValidationMode = "container",
    max_mean_abs_diff: float = 18.0,
    sample_frames: int = 5,
    min_free_bytes: int | None = None,
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> ImageSequenceVideoResult:
    paths = [Path(item) for item in image_files]
    if not paths:
        raise ValueError("No image files were provided.")

    output = Path(output_path)
    if output.suffix == "":
        output = default_video_path(output.parent, output.name, reference_video)
    if output.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video output extension: {output.suffix}")

    metadata = read_video_metadata(reference_video) if reference_video is not None else None
    output_fps = _safe_fps(fps if fps is not None else (metadata.fps if metadata else None))

    first = _read_image(paths[0])
    src_height, src_width = first.shape[:2]
    width = _even_dimension(src_width)
    height = _even_dimension(src_height)
    padded = width != src_width or height != src_height

    output.parent.mkdir(parents=True, exist_ok=True)
    _ensure_free_space(output.parent, paths, min_free_bytes=min_free_bytes)
    temp_path = _temporary_video_path(output)
    codec_candidates = _codec_candidates(output, reference_metadata=metadata, requested_fourcc=fourcc)

    writer = None
    selected_codec = ""
    for candidate in codec_candidates:
        _raise_if_cancelled(should_cancel)
        writer = cv2.VideoWriter(
            str(temp_path),
            cv2.VideoWriter_fourcc(*candidate),
            output_fps,
            (width, height),
        )
        if writer.isOpened():
            selected_codec = candidate
            break
        writer.release()
        writer = None

    if writer is None:
        raise VideoEncodingError(
            f"Could not open a video writer for {output.suffix}. Tried codecs: {', '.join(codec_candidates)}"
        )

    try:
        total = len(paths)
        for idx, image_path in enumerate(paths):
            _raise_if_cancelled(should_cancel)
            frame = _prepare_frame(_read_image(image_path), width=width, height=height, source_path=image_path)
            writer.write(frame)
            if progress_callback is not None:
                progress_callback(idx + 1, total, f"Encoding input video ({idx + 1}/{total})")
    except Exception:
        writer.release()
        _unlink_quietly(temp_path)
        raise

    writer.release()

    validation_mode = _normalize_validation_mode(validate)
    validation_report = None
    try:
        if validation_mode == "container":
            validate_video_container(temp_path, expected_frame_count=len(paths), width=width, height=height)
        elif validation_mode == "sample":
            validation_report = validate_video_against_images(
                temp_path,
                paths,
                width=width,
                height=height,
                max_mean_abs_diff=max_mean_abs_diff,
                sample_frames=sample_frames,
                progress_callback=progress_callback,
                should_cancel=should_cancel,
            )
        elif validation_mode == "full":
            validation_report = validate_video_against_images(
                temp_path,
                paths,
                width=width,
                height=height,
                max_mean_abs_diff=max_mean_abs_diff,
                sample_frames=None,
                progress_callback=progress_callback,
                should_cancel=should_cancel,
            )
        temp_path.replace(output)
    except Exception:
        _unlink_quietly(temp_path)
        raise

    return ImageSequenceVideoResult(
        output_path=output,
        fps=output_fps,
        codec=selected_codec,
        width=width,
        height=height,
        padded=padded,
        frame_count=len(paths),
        validation_mode=validation_mode,
        validation=validation_report,
    )


def validate_video_container(
    video_path: str | Path,
    *,
    expected_frame_count: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> VideoMetadata:
    metadata = read_video_metadata(video_path)
    if metadata is None:
        raise VideoValidationError(f"Could not open encoded video: {video_path}")

    if width is not None and metadata.width and metadata.width != int(width):
        raise VideoValidationError(f"Encoded video width mismatch: {metadata.width} != {width}")
    if height is not None and metadata.height and metadata.height != int(height):
        raise VideoValidationError(f"Encoded video height mismatch: {metadata.height} != {height}")
    if expected_frame_count is not None and metadata.frame_count:
        if abs(metadata.frame_count - int(expected_frame_count)) > 1:
            raise VideoValidationError(
                f"Encoded video frame count mismatch: {metadata.frame_count} != {expected_frame_count}"
            )

    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoValidationError(f"Could not open encoded video: {video_path}")
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoValidationError(f"Could not decode the first frame: {video_path}")
        if width is not None and frame.shape[1] != int(width):
            raise VideoValidationError(f"Decoded frame width mismatch: {frame.shape[1]} != {width}")
        if height is not None and frame.shape[0] != int(height):
            raise VideoValidationError(f"Decoded frame height mismatch: {frame.shape[0]} != {height}")
    finally:
        cap.release()
    return metadata


def validate_video_against_images(
    video_path: str | Path,
    image_files: Sequence[str | Path] | Iterable[str | Path],
    *,
    width: int | None = None,
    height: int | None = None,
    max_mean_abs_diff: float = 18.0,
    sample_frames: int | None = None,
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> VideoValidationReport:
    paths = [Path(item) for item in image_files]
    if not paths:
        raise ValueError("No image files were provided for validation.")

    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoValidationError(f"Could not open encoded video: {video_path}")

    frame_indices = _validation_frame_indices(len(paths), sample_frames)
    max_mean = 0.0
    max_abs = 0
    mismatches: list[str] = []
    compared = 0
    total = len(frame_indices)

    try:
        previous_idx = -1
        for progress_idx, frame_idx in enumerate(frame_indices):
            _raise_if_cancelled(should_cancel)
            if frame_idx != previous_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, video_frame = cap.read()
            if not ok or video_frame is None:
                raise VideoValidationError(f"Encoded video ended before frame {frame_idx}: {video_path}")
            previous_idx = frame_idx

            source_frame = _read_image(paths[frame_idx])
            target_width = width if width is not None else video_frame.shape[1]
            target_height = height if height is not None else video_frame.shape[0]
            source_frame = _prepare_frame(
                source_frame,
                width=target_width,
                height=target_height,
                source_path=paths[frame_idx],
            )

            if video_frame.shape[:2] != source_frame.shape[:2]:
                raise VideoValidationError(
                    f"Frame size mismatch at {frame_idx}: video={video_frame.shape[1]}x{video_frame.shape[0]}, "
                    f"image={source_frame.shape[1]}x{source_frame.shape[0]}"
                )

            diff = cv2.absdiff(video_frame, source_frame)
            mean_diff = float(np.mean(diff))
            frame_max_abs = int(np.max(diff))
            max_mean = max(max_mean, mean_diff)
            max_abs = max(max_abs, frame_max_abs)
            compared += 1

            if mean_diff > max_mean_abs_diff and len(mismatches) < 5:
                mismatches.append(f"frame {frame_idx}: mean abs diff {mean_diff:.2f}")

            if progress_callback is not None:
                progress_callback(progress_idx + 1, total, f"Validating input video ({progress_idx + 1}/{total})")
    finally:
        cap.release()

    if mismatches:
        detail = "; ".join(mismatches)
        raise VideoValidationError(
            f"Encoded video differs from source images beyond tolerance {max_mean_abs_diff:.2f}: {detail}"
        )

    return VideoValidationReport(
        frames_compared=compared,
        max_mean_abs_diff=max_mean,
        max_frame_abs_diff=max_abs,
    )


def _normalize_validation_mode(validate: ValidationMode) -> str:
    if validate is True:
        return "full"
    if validate is False or validate is None:
        return "none"
    mode = str(validate).strip().lower()
    if mode in {"none", "container", "sample", "full"}:
        return mode
    raise ValueError(f"Unsupported validation mode: {validate}")


def _validation_frame_indices(total: int, sample_frames: int | None) -> list[int]:
    if total <= 0:
        return []
    if sample_frames is None or sample_frames >= total:
        return list(range(total))
    sample_frames = max(1, int(sample_frames))
    if sample_frames == 1:
        return [0]
    return sorted({round(i * (total - 1) / (sample_frames - 1)) for i in range(sample_frames)})


def _prepend(first: np.ndarray, rest: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
    yield first
    yield from rest


def _ensure_bgr_frame(frame: np.ndarray, *, source: str) -> np.ndarray:
    if frame is None:
        raise VideoEncodingError(f"Missing video frame: {source}")
    if not isinstance(frame, np.ndarray):
        raise VideoEncodingError(f"Video frame is not a numpy array: {source}")
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3:
        raise VideoEncodingError(f"Video frame must have 2 or 3 dimensions: {source}")
    channels = frame.shape[2]
    if channels == 3:
        return frame
    if channels == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    raise VideoEncodingError(f"Video frame has unsupported channel count {channels}: {source}")


def _natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", text)]


def _safe_fps(value: float | int | None) -> float:
    try:
        fps = float(value or 0.0)
    except (TypeError, ValueError):
        fps = 0.0
    if fps <= 0 or fps > 1000:
        return 30.0
    return fps


def _even_dimension(value: int) -> int:
    return int(value) if int(value) % 2 == 0 else int(value) + 1


def _read_image(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image frame: {path}")
    return frame


def _prepare_frame(frame: np.ndarray, *, width: int, height: int, source_path: Path) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    if frame_width > width or frame_height > height:
        raise VideoEncodingError(
            f"Image frame is larger than the target video size: {source_path} "
            f"({frame_width}x{frame_height} > {width}x{height})"
        )
    if frame_width == width and frame_height == height:
        return frame

    padded = np.zeros((height, width, 3), dtype=frame.dtype)
    padded[:frame_height, :frame_width] = frame
    if frame_width < width:
        padded[:frame_height, frame_width:] = frame[:, frame_width - 1:frame_width]
    if frame_height < height:
        padded[frame_height:, :] = padded[frame_height - 1:frame_height, :]
    return padded


def _temporary_video_path(output_path: Path) -> Path:
    stamp = f"{os.getpid()}_{time.monotonic_ns()}"
    return output_path.with_name(f".{output_path.stem}.building_{stamp}{output_path.suffix}")


def _ensure_free_space_for_bytes(output_dir: Path, min_free_bytes: int | None) -> None:
    required = max(64 * 1024 * 1024, int(min_free_bytes or 0))
    free = shutil.disk_usage(output_dir).free
    if free < required:
        raise VideoEncodingError(
            f"Not enough free disk space for video encoding: {output_dir} "
            f"(free={free:,} bytes, required={required:,} bytes)"
        )


def _ensure_free_space(output_dir: Path, image_paths: Sequence[Path], *, min_free_bytes: int | None) -> None:
    required = int(min_free_bytes) if min_free_bytes is not None else _estimate_required_free_space(image_paths)
    if required <= 0:
        return
    free = shutil.disk_usage(output_dir).free
    if free < required:
        raise VideoEncodingError(
            f"Not enough free disk space for video encoding: {output_dir} "
            f"(free={free:,} bytes, required={required:,} bytes)"
        )


def _estimate_required_free_space(image_paths: Sequence[Path]) -> int:
    total = 0
    for path in image_paths:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return max(64 * 1024 * 1024, int(total * 0.5))


def _codec_candidates(
    output_path: Path,
    *,
    reference_metadata: VideoMetadata | None,
    requested_fourcc: str | None,
) -> list[str]:
    candidates: list[str] = []
    if requested_fourcc:
        candidates.append(requested_fourcc)

    suffix = output_path.suffix.lower()
    if suffix == ".avi":
        candidates.extend(["MJPG", "XVID", "mp4v"])
    elif suffix in {".mp4", ".mov", ".m4v", ".mkv"}:
        candidates.extend(["mp4v", "avc1", "MJPG"])
    else:
        candidates.append("mp4v")

    if reference_metadata and reference_metadata.fourcc:
        if output_path.suffix.lower() == reference_metadata.path.suffix.lower():
            candidates.append(reference_metadata.fourcc)

    valid: list[str] = []
    for codec in candidates:
        codec = str(codec or "")[:4]
        if len(codec) != 4:
            continue
        if codec not in valid:
            valid.append(codec)
    return valid


def _fourcc_int_to_str(value: float | int | None) -> Optional[str]:
    try:
        code = int(value or 0)
    except (TypeError, ValueError):
        return None
    if code <= 0:
        return None
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    if all(32 <= ord(char) <= 126 for char in chars):
        return chars
    return None


def _raise_if_cancelled(should_cancel: CancelCallback | None) -> None:
    if should_cancel is not None and should_cancel():
        raise VideoEncodingCancelled("Video encoding was cancelled.")


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass