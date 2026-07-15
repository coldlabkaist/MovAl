from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "project_manager" / "import_tools.py"
SPEC = importlib.util.spec_from_file_location("project_import_tools_edge", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
IMPORT_TOOLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORT_TOOLS)
auto_sort_import_entries = IMPORT_TOOLS.auto_sort_import_entries


class AutoSortEdgeCaseTests(unittest.TestCase):
    def test_sorts_by_video_filename_before_parent_folder(self) -> None:
        entries = [
            ("C:/a/video10.mp4", "vid"),
            ("C:/z/video2.mp4", "vid"),
        ]

        result, unmatched = auto_sort_import_entries(entries)

        self.assertEqual(unmatched, [])
        self.assertEqual([Path(path).stem for path, _ in result], ["video2", "video10"])

    def test_exact_filename_match_wins_over_misleading_parent_folder(self) -> None:
        entries = [
            ("C:/labels/video1/video2.csv", "csv"),
            ("C:/videos/video1.mp4", "vid"),
            ("C:/videos/video2.mp4", "vid"),
        ]

        result, unmatched = auto_sort_import_entries(entries)

        self.assertEqual(unmatched, [])
        video2_index = result.index(("C:/videos/video2.mp4", "vid"))
        self.assertEqual(result[video2_index + 1], ("C:/labels/video1/video2.csv", "csv"))

    def test_matches_deeplabcut_direct_suffix(self) -> None:
        entries = [
            ("C:/labels/mouse2DLC_resnet.csv", "csv"),
            ("C:/videos/mouse1.mp4", "vid"),
            ("C:/videos/mouse2.mp4", "vid"),
        ]

        result, unmatched = auto_sort_import_entries(entries)

        self.assertEqual(unmatched, [])
        self.assertEqual(result[-1], ("C:/labels/mouse2DLC_resnet.csv", "csv"))


if __name__ == "__main__":
    unittest.main()
