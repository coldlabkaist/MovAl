from __future__ import annotations

import json
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - import guard
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except ImportError:  # pragma: no cover - import guard
    tk = None
    filedialog = None
    messagebox = None


SCHEMA_VERSION = 2
PROJECT_JSON_NAME = "project.json"
LEGACY_SUFFIXES = (".yaml", ".yml")
PROJECT_SKELETON_DIRNAME = "skeleton"
PROJECT_SKELETON_FILENAME = "project_skeleton.yaml"
REPO_ROOT = Path(__file__).resolve().parent
PRESET_SKELETON_DIR = REPO_ROOT / "preset" / "skeleton"
STANDARD_DIRS = ("frames", "labels", "runs", "raw_videos", "outputs", "predicts", PROJECT_SKELETON_DIRNAME)


class ConversionError(RuntimeError):
    pass


@dataclass
class ConversionReport:
    yaml_path: Path
    json_path: Path
    backup_path: Path | None = None
    video_count: int = 0
    copied_csv: int = 0
    copied_txt: int = 0
    warnings: list[str] = field(default_factory=list)


def ensure_runtime_ready() -> None:
    if yaml is None:
        raise ConversionError(
            "PyYAML is required to read legacy project files.\n"
            "Install it with 'pip install pyyaml' or run this script in your MovAl environment."
        ) from YAML_IMPORT_ERROR
    if tk is None or filedialog is None or messagebox is None:
        raise ConversionError("tkinter is not available in this Python environment.")


def repair_video_path(video_path: str, project_dir: Path) -> Path:
    stored_path = Path(video_path).expanduser()
    if stored_path.exists():
        return stored_path.resolve()

    fallback = project_dir / "raw_videos" / stored_path.name
    if stored_path.name and fallback.exists():
        return fallback.resolve()

    if stored_path.is_absolute():
        return stored_path
    return (project_dir / stored_path).resolve()


def repair_label_path(label_path: str, project_dir: Path, video_name: str, kind: str) -> Path:
    stored_path = Path(label_path).expanduser()
    if stored_path.exists():
        return stored_path.resolve()

    label_root = project_dir / "labels" / video_name / kind
    fallback = label_root if kind == "txt" else label_root / stored_path.name
    if video_name and fallback.exists():
        return fallback.resolve()

    if stored_path.is_absolute():
        return stored_path
    return (project_dir / stored_path).resolve()


def is_inside(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def next_backup_path(target_path: Path) -> Path:
    candidate = target_path.with_name(target_path.name + ".bak")
    index = 2
    while candidate.exists():
        candidate = target_path.with_name(f"{target_path.name}.bak{index}")
        index += 1
    return candidate


def next_unique_path(target_dir: Path, file_name: str) -> Path:
    candidate = target_dir / file_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        trial = target_dir / f"{stem} ({index}){suffix}"
        if not trial.exists():
            return trial
        index += 1


def copy_file_if_needed(source_path: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        if source_path.resolve().parent == target_dir.resolve():
            return 0
    except OSError:
        pass

    target_path = next_unique_path(target_dir, source_path.name)
    shutil.copy2(source_path, target_path)
    return 1


def normalize_ui_state(raw_state: Any) -> dict[str, Any]:
    state = raw_state if isinstance(raw_state, dict) else {}
    preferred_mode = state.get("preferred_frame_mode")
    if preferred_mode not in {"images", "davis", "contour"}:
        preferred_mode = "davis"

    labelary_state = state.get("labelary")
    if not isinstance(labelary_state, dict):
        labelary_state = {}

    label_type = labelary_state.get("label_type")
    if label_type not in {"csv", "txt"}:
        label_type = None

    frame_index_raw = labelary_state.get("frame_index", 0)
    try:
        frame_index = max(0, int(frame_index_raw))
    except (TypeError, ValueError):
        frame_index = 0

    return {
        "preferred_frame_mode": preferred_mode,
        "labelary": {
            "frame_index": frame_index,
            "video_name": labelary_state.get("video_name"),
            "label_name": labelary_state.get("label_name"),
            "label_type": label_type,
            "color_mode": labelary_state.get("color_mode"),
        },
    }


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def list_or_empty(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def read_yaml_file(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser()
    if not yaml_path.is_file():
        raise ConversionError(f"Skeleton YAML not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConversionError(f"Invalid YAML structure: {yaml_path}")
    return data


def normalize_skeleton_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"nodes": [], "connections": [], "symmetry": []}
    return {
        "nodes": list(data.get("nodes", []) or []),
        "connections": list(data.get("connections", []) or []),
        "symmetry": list(data.get("symmetry", []) or []),
    }


def load_skeleton_data_from_name(skeleton_name: str, report: ConversionReport) -> dict[str, Any]:
    clean_name = Path(str(skeleton_name or "")).name
    if not clean_name:
        report.warnings.append("No skeleton preset was recorded in the legacy YAML.")
        return normalize_skeleton_data({})
    try:
        return normalize_skeleton_data(read_yaml_file(PRESET_SKELETON_DIR / clean_name))
    except Exception as exc:
        report.warnings.append(f"Failed to load preset skeleton '{clean_name}': {exc}")
        return normalize_skeleton_data({})


def write_project_skeleton_file(project_dir: Path, skeleton_data: dict[str, Any]) -> Path:
    skeleton_dir = project_dir / PROJECT_SKELETON_DIRNAME
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    skeleton_path = skeleton_dir / PROJECT_SKELETON_FILENAME
    with skeleton_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(normalize_skeleton_data(skeleton_data), f, sort_keys=False, allow_unicode=True)
    return skeleton_path


def standard_video_record(raw_video: str, project_dir: Path) -> dict[str, Any]:
    resolved_path = repair_video_path(raw_video, project_dir)
    file_name = Path(raw_video).name or resolved_path.name
    video_name = Path(file_name).stem or Path(raw_video).stem

    if resolved_path.exists() and is_inside(project_dir, resolved_path):
        try:
            relative_path = resolved_path.resolve().relative_to(project_dir.resolve()).as_posix()
        except ValueError:
            relative_path = resolved_path.name
        return {
            "name": video_name,
            "file_name": file_name,
            "relative_path": relative_path,
        }

    fallback = project_dir / "raw_videos" / file_name
    if fallback.exists():
        return {
            "name": video_name,
            "file_name": file_name,
            "relative_path": fallback.relative_to(project_dir).as_posix(),
        }

    return {
        "name": video_name,
        "file_name": file_name,
        "source_path": resolved_path.as_posix(),
    }


def migrate_csvs(csv_values: list[Any], project_dir: Path, video_name: str, report: ConversionReport) -> int:
    copied = 0
    target_dir = project_dir / "labels" / video_name / "csv"
    target_dir.mkdir(parents=True, exist_ok=True)

    for raw_value in csv_values or []:
        resolved_path = repair_label_path(str(raw_value or ""), project_dir, video_name, "csv")
        if not resolved_path.exists():
            report.warnings.append(f"[{video_name}] Missing CSV: {raw_value}")
            continue

        if resolved_path.is_dir():
            csv_files = sorted(resolved_path.glob("*.csv"))
            if not csv_files:
                report.warnings.append(f"[{video_name}] CSV directory has no csv files: {resolved_path}")
                continue
            for csv_file in csv_files:
                copied += copy_file_if_needed(csv_file, target_dir)
            continue

        copied += copy_file_if_needed(resolved_path, target_dir)

    return copied


def migrate_txts(txt_values: list[Any], project_dir: Path, video_name: str, report: ConversionReport) -> int:
    copied = 0
    target_dir = project_dir / "labels" / video_name / "txt"
    target_dir.mkdir(parents=True, exist_ok=True)

    for raw_value in txt_values or []:
        resolved_path = repair_label_path(str(raw_value or ""), project_dir, video_name, "txt")
        if not resolved_path.exists():
            report.warnings.append(f"[{video_name}] Missing TXT source: {raw_value}")
            continue

        if resolved_path.is_file():
            if resolved_path.suffix.lower() != ".txt":
                report.warnings.append(f"[{video_name}] Skipped non-txt file: {resolved_path}")
                continue
            copied += copy_file_if_needed(resolved_path, target_dir)
            continue

        txt_files = sorted(resolved_path.glob("*.txt"))
        if not txt_files:
            report.warnings.append(f"[{video_name}] TXT directory has no txt files: {resolved_path}")
            continue
        for txt_file in txt_files:
            copied += copy_file_if_needed(txt_file, target_dir)

    return copied


def build_project_payload(data: dict[str, Any], project_dir: Path, report: ConversionReport) -> dict[str, Any]:
    for rel_path in STANDARD_DIRS:
        (project_dir / rel_path).mkdir(parents=True, exist_ok=True)

    raw_files = data.get("files")
    if not isinstance(raw_files, list):
        raise ConversionError(f"'{report.yaml_path.name}' does not look like a MovAl project YAML.")

    seen_video_names: set[str] = set()
    videos: list[dict[str, Any]] = []

    for item in raw_files:
        if not isinstance(item, dict):
            continue

        raw_video = str(item.get("video", "") or "").strip()
        if not raw_video:
            report.warnings.append("Skipped a file entry with an empty video path.")
            continue

        video_record = standard_video_record(raw_video, project_dir)
        video_name = video_record["name"]
        if video_name in seen_video_names:
            report.warnings.append(f"Skipped duplicate video name: {video_name}")
            continue

        seen_video_names.add(video_name)
        videos.append(video_record)
        report.copied_csv += migrate_csvs(item.get("csv", []), project_dir, video_name, report)
        report.copied_txt += migrate_txts(item.get("txt", []), project_dir, video_name, report)

    if not videos:
        raise ConversionError(f"No valid video entries were found in '{report.yaml_path.name}'.")

    report.video_count = len(videos)
    skeleton_name = Path(str(data.get("skeleton", "") or "")).name
    skeleton_data = load_skeleton_data_from_name(skeleton_name, report)
    return {
        "schema_version": SCHEMA_VERSION,
        "moval_version": str(data.get("moval_version", "")),
        "title": str(data.get("title", report.yaml_path.stem)),
        "num_animals": safe_int(data.get("num_animals", 0), default=0),
        "animals_name": list_or_empty(data.get("animals_name", [])),
        "skeleton": skeleton_name,
        "skeleton_data": skeleton_data,
        "videos": videos,
        "ui_state": normalize_ui_state(data.get("ui_state")),
    }


def convert_project_file(yaml_path: str | Path) -> ConversionReport:
    yaml_path = Path(yaml_path).expanduser().resolve()
    if not yaml_path.is_file():
        raise ConversionError(f"YAML file not found: {yaml_path}")
    if yaml_path.suffix.lower() not in LEGACY_SUFFIXES:
        raise ConversionError(f"Select a .yaml or .yml file: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConversionError(f"Invalid YAML structure: {yaml_path}")

    project_dir = yaml_path.parent
    json_path = project_dir / PROJECT_JSON_NAME
    report = ConversionReport(yaml_path=yaml_path, json_path=json_path)

    payload = build_project_payload(data, project_dir, report)
    write_project_skeleton_file(project_dir, payload.get("skeleton_data", {}))

    if json_path.exists():
        backup_path = next_backup_path(json_path)
        shutil.copy2(json_path, backup_path)
        report.backup_path = backup_path

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return report


def create_hidden_root() -> tk.Tk:
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    return root


def format_report(report: ConversionReport) -> str:
    lines = [
        f"- {report.yaml_path.name} -> {report.json_path.name}",
        f"  videos: {report.video_count}, copied csv: {report.copied_csv}, copied txt: {report.copied_txt}",
    ]
    if report.backup_path is not None:
        lines.append(f"  backup: {report.backup_path.name}")
    if report.warnings:
        lines.append(f"  warnings: {len(report.warnings)}")
        lines.extend(f"    {warning}" for warning in report.warnings[:5])
        if len(report.warnings) > 5:
            lines.append(f"    ... {len(report.warnings) - 5} more")
    return "\n".join(lines)


def show_summary(reports: list[ConversionReport], failures: list[str]) -> None:
    success_count = len(reports)
    lines = [f"Converted {success_count} project(s)."]
    if reports:
        lines.append("")
        lines.append("Results:")
        lines.extend(format_report(report) for report in reports)
    if failures:
        lines.append("")
        lines.append("Failures:")
        lines.extend(failures[:10])
        if len(failures) > 10:
            lines.append(f"... {len(failures) - 10} more failure(s)")

    title = "Conversion Complete" if not failures else "Conversion Finished with Errors"
    messagebox.showinfo(title, "\n".join(lines))


def main() -> int:
    try:
        ensure_runtime_ready()
        root = create_hidden_root()
    except ConversionError as exc:
        if messagebox is not None:
            try:
                temp_root = create_hidden_root()
                messagebox.showerror("Converter Error", str(exc))
                temp_root.destroy()
            except Exception:
                print(str(exc), file=sys.stderr)
        else:
            print(str(exc), file=sys.stderr)
        return 1

    try:
        yaml_paths = filedialog.askopenfilenames(
            title="Select legacy MovAl project YAML files",
            filetypes=[("YAML files", "*.yaml *.yml")],
        )
        if not yaml_paths:
            return 0

        reports: list[ConversionReport] = []
        failures: list[str] = []
        for raw_path in yaml_paths:
            try:
                reports.append(convert_project_file(raw_path))
            except Exception as exc:
                detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                failures.append(f"- {Path(raw_path).name}: {detail}")

        show_summary(reports, failures)
        return 0 if not failures else 1
    finally:
        root.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
