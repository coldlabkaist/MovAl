from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "pose" / "progress_parsing.py"
SPEC = importlib.util.spec_from_file_location("pose_progress_parsing", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PoseProgressParsingTests(unittest.TestCase):
    def test_parses_training_epoch_with_total_hint(self) -> None:
        self.assertEqual(MODULE.parse_training_epoch("  12/400  7.2G  0.01", 400), (12, 400))
        self.assertIsNone(MODULE.parse_training_epoch("batch 12/50", 400))

    def test_parses_inference_frame_only(self) -> None:
        self.assertEqual(
            MODULE.parse_inference_frame("video 1/1 (frame 32/120) 640x640"),
            (32, 120),
        )
        self.assertEqual(MODULE.parse_inference_frame("image 7/25 sample.jpg"), (7, 25))
        self.assertIsNone(MODULE.parse_inference_frame("video 1/4 complete"))

    def test_reads_last_training_results_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.csv"
            path.write_text(
                "epoch,time,train/pose_loss,metrics/mAP50(P)\n"
                "1,10.5,0.8,0.2\n"
                "2,20.5,0.6,0.4\n",
                encoding="utf-8",
            )
            result = MODULE.read_training_results(path)

        self.assertIsNotNone(result)
        self.assertEqual(result["epoch"], 2)
        self.assertEqual(result["metrics"]["pose_loss"], 0.6)
        self.assertEqual(result["metrics"]["mAP50"], 0.4)


if __name__ == "__main__":
    unittest.main()
