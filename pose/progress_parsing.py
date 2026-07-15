from __future__ import annotations

import csv
import re
from pathlib import Path


_FRACTION_RE = re.compile(r"(?<![\d.])(\d+)\s*/\s*(\d+)(?!\d)")
_FRAME_PATTERNS = (
    re.compile(r"\bframe\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE),
    re.compile(r"\bimage\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE),
)


def parse_training_epoch(line: str, total_hint: int = 0) -> tuple[int, int] | None:
    """Extract an epoch fraction from an Ultralytics training log line."""
    matches = [(int(done), int(total)) for done, total in _FRACTION_RE.findall(line or "")]
    if total_hint > 0:
        for done, total in matches:
            if total == total_hint and 0 <= done <= total:
                return done, total
        return None
    for done, total in matches:
        if total > 1 and 0 <= done <= total:
            return done, total
    return None


def parse_inference_frame(line: str) -> tuple[int, int] | None:
    """Extract a frame fraction without confusing it with source counters."""
    for pattern in _FRAME_PATTERNS:
        match = pattern.search(line or "")
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            if total > 0 and 0 <= done <= total:
                return done, total
    return None


def read_training_results(results_path: str | Path) -> dict[str, object] | None:
    """Read the last complete row of an Ultralytics results.csv file."""
    path = Path(results_path)
    if not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
            rows = list(csv.DictReader(file_obj))
    except (OSError, UnicodeError, csv.Error):
        return None
    if not rows:
        return None

    row = {str(key or "").strip(): str(value or "").strip() for key, value in rows[-1].items()}
    try:
        epoch = int(float(row.get("epoch", "0")))
    except (TypeError, ValueError):
        epoch = 0

    metrics: dict[str, float] = {}
    metric_candidates = {
        "pose_loss": ("train/pose_loss", "val/pose_loss"),
        "mAP50": ("metrics/mAP50(P)", "metrics/mAP50(B)"),
        "mAP50-95": ("metrics/mAP50-95(P)", "metrics/mAP50-95(B)"),
        "time": ("time",),
    }
    for display_name, candidates in metric_candidates.items():
        for candidate in candidates:
            raw_value = row.get(candidate)
            if raw_value in (None, ""):
                continue
            try:
                metrics[display_name] = float(raw_value)
            except ValueError:
                pass
            break
    return {"epoch": max(0, epoch), "metrics": metrics, "row": row}


def format_training_metrics(metrics: dict[str, float]) -> str:
    parts: list[str] = []
    for key in ("pose_loss", "mAP50", "mAP50-95"):
        if key in metrics:
            parts.append(f"{key}: {metrics[key]:.4g}")
    return " · ".join(parts)
