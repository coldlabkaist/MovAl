from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import unittest

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    cv2 = None
    np = None

if cv2 is not None:
    module_path = Path(__file__).resolve().parents[1] / "utils" / "image_sequence_video.py"
    spec = importlib.util.spec_from_file_location("image_sequence_video_under_test", module_path)
    image_sequence_video = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = image_sequence_video
    spec.loader.exec_module(image_sequence_video)

    default_video_path = image_sequence_video.default_video_path
    encode_image_sequence_to_video = image_sequence_video.encode_image_sequence_to_video
    encode_frame_sequence_to_video = image_sequence_video.encode_frame_sequence_to_video
    natural_image_files = image_sequence_video.natural_image_files
    read_video_metadata = image_sequence_video.read_video_metadata
    validate_video_against_images = image_sequence_video.validate_video_against_images


@unittest.skipIf(cv2 is None, "OpenCV is not installed in this Python environment")
class ImageSequenceVideoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(__file__).resolve().parent / "_tmp_image_sequence_video"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        for path in sorted(self.tmp_dir.glob("*"), reverse=True):
            if path.is_file():
                path.unlink()
        self.tmp_dir.rmdir()

    def _write_frames(self, count: int = 6) -> list[Path]:
        paths: list[Path] = []
        x_grad = np.tile(np.arange(64, dtype=np.uint8), (48, 1))
        y_grad = np.tile(np.arange(48, dtype=np.uint8).reshape(48, 1), (1, 64))
        for idx in range(count):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = (x_grad + idx * 7) % 255
            frame[:, :, 1] = (y_grad * 3 + idx * 5) % 255
            frame[:, :, 2] = 80 + idx * 10
            path = self.tmp_dir / f"frame_{idx:04d}.jpg"
            self.assertTrue(cv2.imwrite(str(path), frame))
            paths.append(path)
        return paths

    def test_default_video_path_keeps_avi_for_avi_reference(self) -> None:
        self.assertEqual(
            default_video_path(self.tmp_dir, "sample", self.tmp_dir / "source.avi").suffix,
            ".avi",
        )
        self.assertEqual(
            default_video_path(self.tmp_dir, "sample", self.tmp_dir / "source.mp4").suffix,
            ".mp4",
        )

    def test_natural_image_files_sorts_frame_numbers(self) -> None:
        for name in ["frame_10.jpg", "frame_2.jpg", "frame_1.jpg"]:
            self.assertTrue(cv2.imwrite(str(self.tmp_dir / name), np.zeros((8, 8, 3), dtype=np.uint8)))
        self.assertEqual([path.name for path in natural_image_files(self.tmp_dir)], ["frame_1.jpg", "frame_2.jpg", "frame_10.jpg"])

    def test_encode_avi_sequence_uses_fast_container_validation_by_default(self) -> None:
        frames = self._write_frames()
        output = self.tmp_dir / "encoded.avi"

        result = encode_image_sequence_to_video(
            frames,
            output,
            fps=12.5,
            fourcc="MJPG",
            max_mean_abs_diff=20.0,
        )

        self.assertEqual(result.output_path, output)
        self.assertEqual(result.frame_count, len(frames))
        self.assertEqual(result.codec, "MJPG")
        self.assertEqual(result.validation_mode, "container")
        self.assertIsNone(result.validation)

        metadata = read_video_metadata(output)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.frame_count, len(frames))
        self.assertEqual(metadata.width, 64)
        self.assertEqual(metadata.height, 48)
        self.assertAlmostEqual(metadata.fps, 12.5, delta=1.0)

    def test_encode_frame_stream_uses_shared_container_validation(self) -> None:
        frames = self._write_frames()
        output = self.tmp_dir / "stream_encoded.avi"

        def frame_stream():
            for path in frames:
                frame = cv2.imread(str(path))
                self.assertIsNotNone(frame)
                yield frame

        progress_values: list[tuple[int, int]] = []
        result = encode_frame_sequence_to_video(
            frame_stream(),
            output,
            fps=24.0,
            fourcc="MJPG",
            frame_count=len(frames),
            validate="container",
            progress_callback=lambda done, total, _message: progress_values.append((done, total)),
        )

        self.assertEqual(result.output_path, output)
        self.assertEqual(result.frame_count, len(frames))
        self.assertEqual(result.codec, "MJPG")
        self.assertEqual(result.validation_mode, "container")
        self.assertEqual(progress_values[-1], (len(frames), len(frames)))

        metadata = read_video_metadata(output)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.frame_count, len(frames))
        self.assertEqual(metadata.width, 64)
        self.assertEqual(metadata.height, 48)

    def test_full_validation_compares_each_decoded_frame(self) -> None:
        frames = self._write_frames()
        output = self.tmp_dir / "encoded_full.avi"

        result = encode_image_sequence_to_video(
            frames,
            output,
            fps=12.5,
            fourcc="MJPG",
            validate="full",
            max_mean_abs_diff=20.0,
        )

        self.assertEqual(result.validation_mode, "full")
        self.assertIsNotNone(result.validation)
        self.assertEqual(result.validation.frames_compared, len(frames))

        report = validate_video_against_images(output, frames, max_mean_abs_diff=20.0)
        self.assertEqual(report.frames_compared, len(frames))


if __name__ == "__main__":
    unittest.main()
