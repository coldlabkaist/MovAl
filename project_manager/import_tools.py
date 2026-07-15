from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


ImportEntry = tuple[str, str]


def natural_sort_key(value: str | Path) -> tuple[tuple[int, int | str], ...]:
    """Return a case-insensitive key that orders embedded numbers numerically."""
    parts = re.split(r"(\d+)", Path(value).as_posix().casefold())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    )


def find_csv_files(folder: str | Path) -> list[Path]:
    """Find every CSV below *folder*, including CSVs in nested directories."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"CSV folder not found: {root}")

    csv_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".csv"
    ]
    return sorted(csv_files, key=lambda path: natural_sort_key(path.relative_to(root)))


def _prefix_match_kind(candidate: str, video_stem: str) -> int:
    """Score filename conventions that confidently identify a video's labels."""
    if candidate == video_stem:
        return 4
    if not candidate.startswith(video_stem):
        return 0

    remainder = candidate[len(video_stem) :]
    if remainder and remainder[0] in " ._-([{":
        return 3
    # DeepLabCut commonly appends DLC directly to the source video's stem.
    if remainder.startswith("dlc"):
        return 2
    return 0


def _label_match_score(label_path: Path, video_path: Path) -> tuple[int, int, int]:
    video_stem = video_path.stem.casefold()
    label_stem = label_path.stem.casefold()

    filename_kind = _prefix_match_kind(label_stem, video_stem)
    component_kind = 0
    for component in label_path.parent.parts:
        component_stem = Path(component).stem.casefold()
        component_kind = max(component_kind, _prefix_match_kind(component_stem, video_stem))

    # A matching folder is especially useful for recursively loaded generic names
    # such as <video>/results/pose.csv.
    match_kind = max(filename_kind * 100, component_kind * 10)
    if not match_kind:
        return (0, 0, 0)

    common_parent_parts = 0
    label_parent = label_path.parent.resolve(strict=False).parts
    video_parent = video_path.parent.resolve(strict=False).parts
    for label_part, video_part in zip(label_parent, video_parent):
        if label_part.casefold() != video_part.casefold():
            break
        common_parent_parts += 1
    return (match_kind, len(video_stem), common_parent_parts)


def auto_sort_import_entries(
    entries: Iterable[ImportEntry],
) -> tuple[list[ImportEntry], list[ImportEntry]]:
    """Naturally sort videos and place confidently matched labels after each video.

    Entries use ``(path, file_type)``. Labels that cannot be matched safely are
    returned first, so project creation cannot silently attach them to the wrong
    video. The second return value contains those unmatched entries for UI feedback.
    """
    materialized = list(entries)
    videos = [entry for entry in materialized if entry[1] == "vid"]
    labels = [entry for entry in materialized if entry[1] != "vid"]

    videos.sort(
        key=lambda entry: (
            natural_sort_key(Path(entry[0]).name),
            natural_sort_key(entry[0]),
        )
    )
    labels.sort(
        key=lambda entry: (
            natural_sort_key(Path(entry[0]).name),
            natural_sort_key(entry[0]),
            entry[1],
        )
    )

    if not videos:
        return labels, []

    grouped: list[list[ImportEntry]] = [[] for _ in videos]
    unmatched: list[ImportEntry] = []

    if len(videos) == 1:
        grouped[0].extend(labels)
    else:
        video_paths = [Path(path) for path, _ in videos]
        for label in labels:
            label_path = Path(label[0])
            scores = [_label_match_score(label_path, video_path) for video_path in video_paths]
            best_score = max(scores)
            if best_score == (0, 0, 0) or scores.count(best_score) != 1:
                unmatched.append(label)
                continue
            grouped[scores.index(best_score)].append(label)

    sorted_entries = list(unmatched)
    for video, video_labels in zip(videos, grouped):
        sorted_entries.append(video)
        sorted_entries.extend(video_labels)
    return sorted_entries, unmatched
