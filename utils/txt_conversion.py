from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional


Detection = tuple[int, Optional[int], list[float]]
ResolvedDetection = tuple[int, list[float], Optional[int]]


def _merge_keypoints(
    existing: list[float],
    incoming: list[float],
    keypoint_count: int,
) -> list[float]:
    merged = list(existing)
    limit = min(keypoint_count, len(merged) // 3, len(incoming) // 3)
    for kp_idx in range(limit):
        base = kp_idx * 3
        if incoming[base + 2] > merged[base + 2]:
            merged[base:base + 3] = incoming[base:base + 3]
    return merged


def parse_txt_pose_detections(lines: Iterable[str]) -> tuple[list[Detection], bool]:
    detections: list[Detection] = []
    has_explicit_instance_id = False

    for line in lines:
        items = line.strip().split()
        if len(items) < 6:
            continue

        try:
            track_id = int(float(items[0]))
        except Exception:
            continue

        raw = items[5:]
        try:
            if len(raw) % 3 == 1:
                instance_id = int(float(raw[-1]))
                kpt_data = list(map(float, raw[:-1]))
                has_explicit_instance_id = True
            elif len(raw) % 3 == 0:
                instance_id = None
                kpt_data = list(map(float, raw))
            else:
                continue
        except Exception:
            continue

        detections.append((track_id, instance_id, kpt_data))

    return detections, has_explicit_instance_id


def resolve_frame_track_data(
    detections: list[Detection],
    *,
    keypoint_count: int,
    per_id_limit: int,
) -> tuple[list[ResolvedDetection], bool]:
    if not detections:
        return [], False

    uses_instance_id = per_id_limit > 1 or any(
        instance_id is not None for _, instance_id, _ in detections
    )
    if not uses_instance_id:
        seen_tracks: set[int] = set()
        track_data: list[ResolvedDetection] = []
        for track_id, _instance_id, kpt_data in detections:
            if track_id in seen_tracks:
                continue
            seen_tracks.add(track_id)
            track_data.append((track_id, list(kpt_data), None))
        return track_data, False

    grouped: dict[int, list[tuple[Optional[int], list[float]]]] = defaultdict(list)
    track_order: list[int] = []
    for track_id, instance_id, kpt_data in detections:
        if track_id not in grouped:
            track_order.append(track_id)
        grouped[track_id].append((instance_id, kpt_data))

    limit = max(1, int(per_id_limit or 1))
    track_data: list[ResolvedDetection] = []

    for track_id in track_order:
        entries = grouped[track_id]
        by_instance: dict[int, list[float]] = {}
        ordered_ids: list[int] = []
        unlabeled: list[list[float]] = []

        for instance_id, kpt_data in entries:
            if instance_id is None:
                unlabeled.append(list(kpt_data))
                continue
            if instance_id not in by_instance:
                by_instance[instance_id] = list(kpt_data)
                ordered_ids.append(instance_id)
            else:
                by_instance[instance_id] = _merge_keypoints(
                    by_instance[instance_id],
                    kpt_data,
                    keypoint_count,
                )

        next_instance_id = 1
        for kpt_data in unlabeled:
            while next_instance_id in by_instance:
                next_instance_id += 1
            by_instance[next_instance_id] = kpt_data
            ordered_ids.append(next_instance_id)
            next_instance_id += 1

        for instance_id in ordered_ids[:limit]:
            track_data.append((track_id, by_instance[instance_id], instance_id))

    return track_data, True
