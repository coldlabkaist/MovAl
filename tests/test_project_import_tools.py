from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "project_manager" / "import_tools.py"
SPEC = importlib.util.spec_from_file_location("project_import_tools", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
IMPORT_TOOLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORT_TOOLS)
auto_sort_import_entries = IMPORT_TOOLS.auto_sort_import_entries
find_csv_files = IMPORT_TOOLS.find_csv_files


class FindCsvFilesTests(unittest.TestCase):
    def test_recursively_finds_csv_files_in_natural_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "nested").mkdir()
            (root / "video10.csv").touch()
            (root / "video2.CSV").touch()
            (root / "nested" / "video1.csv").touch()
            (root / "nested" / "ignore.txt").touch()

            result = find_csv_files(root)

            self.assertEqual(
                [path.relative_to(root).as_posix() for path in result],
                ["nested/video1.csv", "video2.CSV", "video10.csv"],
            )


class AutoSortImportEntriesTests(unittest.TestCase):
    def test_naturally_sorts_and_groups_labels_by_video(self) -> None:
        entries = [
            ("C:/labels/video10_interpolated.csv", "csv"),
            ("C:/videos/video10.mp4", "vid"),
            ("C:/labels/video2.csv", "csv"),
            ("C:/videos/video2.mp4", "vid"),
        ]

        result, unmatched = auto_sort_import_entries(entries)

        self.assertEqual(unmatched, [])
        self.assertEqual(
            result,
            [
                ("C:/videos/video2.mp4", "vid"),
                ("C:/labels/video2.csv", "csv"),
                ("C:/videos/video10.mp4", "vid"),
                ("C:/labels/video10_interpolated.csv", "csv"),
            ],
        )

    def test_uses_parent_folder_for_generic_recursive_csv_name(self) -> None:
        entries = [
            ("C:/labels/video2/results/pose.csv", "csv"),
            ("C:/videos/video1.mp4", "vid"),
            ("C:/videos/video2.mp4", "vid"),
        ]

        result, unmatched = auto_sort_import_entries(entries)

        self.assertEqual(unmatched, [])
        self.assertEqual(result[-1], ("C:/labels/video2/results/pose.csv", "csv"))

    def test_places_unmatched_labels_before_videos(self) -> None:
        entries = [
            ("C:/labels/unknown.csv", "csv"),
            ("C:/videos/video1.mp4", "vid"),
            ("C:/videos/video2.mp4", "vid"),
        ]

        result, unmatched = auto_sort_import_entries(entries)

        self.assertEqual(unmatched, [("C:/labels/unknown.csv", "csv")])
        self.assertEqual(result[0], ("C:/labels/unknown.csv", "csv"))

    def test_one_video_receives_all_labels(self) -> None:
        entries = [
            ("C:/labels/arbitrary.csv", "csv"),
            ("C:/videos/video1.mp4", "vid"),
        ]

        result, unmatched = auto_sort_import_entries(entries)

        self.assertEqual(unmatched, [])
        self.assertEqual(result[0][1], "vid")


if __name__ == "__main__":
    unittest.main()
