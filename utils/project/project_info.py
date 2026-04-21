from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from utils.skeleton.skeleton_model import SkeletonModel

PROJECT_FILENAME = "project.json"
LEGACY_PROJECT_FILENAMES = ("config.yaml", "config.yml")
FRAME_MODES = {"images", "davis", "contour"}
IMAGE_FILE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROJECT_SKELETON_DIRNAME = "skeleton"
PROJECT_SKELETON_FILENAME = "project_skeleton.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]
PRESET_SKELETON_DIR = REPO_ROOT / "preset" / "skeleton"


def _project_path_from_input(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        json_path = candidate / PROJECT_FILENAME
        if json_path.is_file():
            return json_path.resolve()
        for legacy_name in LEGACY_PROJECT_FILENAMES:
            legacy_path = candidate / legacy_name
            if legacy_path.is_file():
                return legacy_path.resolve()
        raise FileNotFoundError(f"Project file not found in: {candidate}")
    return candidate.resolve()


def _load_document(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        raise ValueError(f"Unsupported project file format: {path}")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid project file: {path}")
    return data


def _normalize_frame_mode(mode: str | None) -> str:
    if mode in FRAME_MODES:
        return mode
    return "davis"


def _is_inside(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _unique_copy_path(dst_dir: Path, name: str) -> Path:
    candidate = dst_dir / name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        trial = dst_dir / f"{stem} ({index}){suffix}"
        if not trial.exists():
            return trial
        index += 1


def _copy_file_with_unique_name(src: str | Path, dst_dir: str | Path) -> Path:
    src_path = Path(src).expanduser()
    if not src_path.is_file():
        raise FileNotFoundError(f"File not found: {src_path}")

    dst_dir_path = Path(dst_dir)
    dst_dir_path.mkdir(parents=True, exist_ok=True)
    dst_path = _unique_copy_path(dst_dir_path, src_path.name)
    shutil.copy2(src_path, dst_path)
    return dst_path


def _copy_txt_directory(src_dir: str | Path, dst_dir: str | Path) -> Path:
    src_dir_path = Path(src_dir).expanduser()
    if not src_dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {src_dir_path}")

    dst_dir_path = Path(dst_dir)
    dst_dir_path.mkdir(parents=True, exist_ok=True)
    for txt_file in sorted(src_dir_path.glob("*.txt")):
        shutil.copy2(txt_file, dst_dir_path / txt_file.name)
    return dst_dir_path


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def _count_dirs(path: Path) -> int:
    if not path.exists() or path.is_file():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_dir())


def _count_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return total


def _read_yaml_file(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure: {yaml_path}")
    return data


def _load_skeleton_data_from_name(skeleton_name: str) -> dict[str, Any]:
    if not skeleton_name:
        return {"nodes": [], "connections": [], "symmetry": []}
    path = PRESET_SKELETON_DIR / Path(skeleton_name).name
    return _read_yaml_file(path)


def _normalize_skeleton_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"nodes": [], "connections": [], "symmetry": []}
    return {
        "nodes": list(data.get("nodes", []) or []),
        "connections": list(data.get("connections", []) or []),
        "symmetry": list(data.get("symmetry", []) or []),
    }


@dataclass
class FileEntry:
    name: str
    video: str
    csv: list[str] = field(default_factory=list)
    txt: list[str] = field(default_factory=list)
    file_name: str = ""
    relative_path: Optional[str] = None
    source_path: Optional[str] = None


@dataclass
class VideoRecord:
    name: str
    file_name: str
    relative_path: Optional[str] = None
    source_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoRecord":
        relative_path = data.get("relative_path")
        source_path = data.get("source_path")
        file_name = data.get("file_name")
        name = data.get("name")

        if not file_name:
            raw_hint = relative_path or source_path or data.get("video") or ""
            file_name = Path(raw_hint).name
        if not name:
            name = Path(file_name).stem

        return cls(
            name=name,
            file_name=file_name,
            relative_path=Path(relative_path).as_posix() if relative_path else None,
            source_path=str(Path(source_path).expanduser()) if source_path else None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "file_name": self.file_name,
        }
        if self.relative_path:
            payload["relative_path"] = Path(self.relative_path).as_posix()
        if self.source_path:
            payload["source_path"] = str(Path(self.source_path).expanduser())
        return payload


@dataclass
class ProjectInformation:
    moval_version: str
    project_dir: str
    project_file: Path
    title: str
    num_animals: int
    animals_name: list[str]
    skeleton_name: str
    skeleton_yaml: Path
    skeleton_data: dict[str, Any] = field(default_factory=dict)
    video_records: list[VideoRecord] = field(default_factory=list)
    ui_state: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2
    legacy_source_path: Optional[Path] = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        project_dir: str | Path,
        *,
        title: str,
        num_animals: int,
        animals_name: list[str],
        skeleton_name: str,
        moval_version: str,
        ui_state: Optional[dict[str, Any]] = None,
    ) -> "ProjectInformation":
        project_dir_path = Path(project_dir).expanduser().resolve()
        project = cls(
            moval_version=moval_version,
            project_dir=project_dir_path.as_posix(),
            project_file=project_dir_path / PROJECT_FILENAME,
            title=title,
            num_animals=int(num_animals),
            animals_name=list(animals_name),
            skeleton_name=Path(skeleton_name).name,
            skeleton_yaml=project_dir_path / PROJECT_SKELETON_DIRNAME / PROJECT_SKELETON_FILENAME,
            skeleton_data=_load_skeleton_data_from_name(skeleton_name),
            video_records=[],
            ui_state=ui_state or {},
        )
        project.ensure_standard_structure()
        project._ensure_ui_defaults()
        project.ensure_project_skeleton_file()
        return project

    @classmethod
    def from_path(cls, path: str | Path) -> "ProjectInformation":
        config_path = _project_path_from_input(path)
        data = _load_document(config_path)
        if config_path.suffix.lower() == ".json":
            return cls._from_json(config_path, data)
        return cls._from_legacy_yaml(config_path, data)

    @property
    def files(self) -> list[FileEntry]:
        project_dir_path = self.project_dir_path
        file_entries: list[FileEntry] = []
        for record in self.video_records:
            csv_dir = project_dir_path / "labels" / record.name / "csv"
            txt_dir = project_dir_path / "labels" / record.name / "txt"
            csv_files = [
                csv_path.resolve().as_posix()
                for csv_path in sorted(csv_dir.glob("*.csv"))
                if csv_path.is_file()
            ]
            txt_entries = [txt_dir.resolve().as_posix()] if any(txt_dir.glob("*.txt")) else []
            file_entries.append(
                FileEntry(
                    name=record.name,
                    video=self.resolve_video_path(record).as_posix(),
                    csv=csv_files,
                    txt=txt_entries,
                    file_name=record.file_name,
                    relative_path=record.relative_path,
                    source_path=record.source_path,
                )
            )
        return file_entries

    @property
    def project_dir_path(self) -> Path:
        return Path(self.project_dir)

    @property
    def video_names(self) -> list[str]:
        return [record.name for record in self.video_records]

    @classmethod
    def _from_json(cls, config_path: Path, data: dict[str, Any]) -> "ProjectInformation":
        project_dir = config_path.parent.resolve()
        skeleton_name = Path(data.get("skeleton", "")).name
        project = cls(
            moval_version=str(data.get("moval_version", "")),
            project_dir=project_dir.as_posix(),
            project_file=config_path,
            title=str(data.get("title", "")),
            num_animals=int(data.get("num_animals", 0)),
            animals_name=list(data.get("animals_name", [])),
            skeleton_name=skeleton_name,
            skeleton_yaml=project_dir / PROJECT_SKELETON_DIRNAME / PROJECT_SKELETON_FILENAME,
            skeleton_data=_normalize_skeleton_data(
                data.get("skeleton_data", _load_skeleton_data_from_name(skeleton_name))
            ),
            video_records=[
                VideoRecord.from_dict(item)
                for item in data.get("videos", [])
                if isinstance(item, dict)
            ],
            ui_state=data.get("ui_state", {}) or {},
            schema_version=int(data.get("schema_version", 2)),
        )
        project._ensure_ui_defaults()
        project.ensure_standard_structure()
        project.ensure_project_skeleton_file()
        return project

    @classmethod
    def _from_legacy_yaml(cls, config_path: Path, data: dict[str, Any]) -> "ProjectInformation":
        project_dir = config_path.parent.resolve()
        skeleton_name = Path(data.get("skeleton", "")).name
        video_records: list[VideoRecord] = []
        for item in data.get("files", []):
            if not isinstance(item, dict):
                continue
            record = _legacy_video_record(item, project_dir)
            if record.name and record.name not in {video.name for video in video_records}:
                video_records.append(record)

        project = cls(
            moval_version=str(data.get("moval_version", "")),
            project_dir=project_dir.as_posix(),
            project_file=project_dir / PROJECT_FILENAME,
            title=str(data.get("title", "")),
            num_animals=int(data.get("num_animals", 0)),
            animals_name=list(data.get("animals_name", [])),
            skeleton_name=skeleton_name,
            skeleton_yaml=project_dir / PROJECT_SKELETON_DIRNAME / PROJECT_SKELETON_FILENAME,
            skeleton_data=_load_skeleton_data_from_name(skeleton_name),
            video_records=video_records,
            ui_state=data.get("ui_state", {}) or {},
            schema_version=2,
            legacy_source_path=config_path,
        )
        project._ensure_ui_defaults()
        project.ensure_standard_structure()
        project.ensure_project_skeleton_file()
        return project

    def _ensure_ui_defaults(self) -> None:
        if not isinstance(self.ui_state, dict):
            self.ui_state = {}
        self.ui_state["preferred_frame_mode"] = _normalize_frame_mode(
            self.ui_state.get("preferred_frame_mode")
        )
        labelary_state = self.ui_state.get("labelary")
        if not isinstance(labelary_state, dict):
            labelary_state = {}
        labelary_state["frame_index"] = max(0, int(labelary_state.get("frame_index", 0) or 0))
        if labelary_state.get("label_type") not in {"csv", "txt", None}:
            labelary_state["label_type"] = None
        self.ui_state["labelary"] = labelary_state

    def ensure_standard_structure(self) -> None:
        self.project_dir_path.mkdir(parents=True, exist_ok=True)
        for rel_path in ("frames", "labels", "runs", "raw_videos", "outputs", "predicts", PROJECT_SKELETON_DIRNAME):
            (self.project_dir_path / rel_path).mkdir(parents=True, exist_ok=True)

    def ensure_project_file(self) -> Path:
        if not self.project_file.exists() or self.legacy_source_path is not None:
            self.save()
        return self.project_file

    def build_training_config(self, dataset_dir: str | Path | None = None) -> dict[str, Any]:
        if dataset_dir is None:
            training_base_dir = self.project_dir_path / "runs" / "dataset"
        else:
            training_base_dir = Path(dataset_dir).expanduser()

        skeleton_model = SkeletonModel()
        skeleton_model.load_from_dict(self.skeleton_data)
        nkpt, flip_idx, kpt_names = skeleton_model.create_training_config()

        return {
            "train": (training_base_dir / "train").as_posix(),
            "val": (training_base_dir / "val").as_posix(),
            "test": (training_base_dir / "test").as_posix(),
            "nc": len(self.animals_name),
            "names": {index: name for index, name in enumerate(self.animals_name)},
            "nkpt": nkpt,
            "kpt_shape": [nkpt, 3],
            "flip_idx": flip_idx,
            "kpt_names": kpt_names,
        }

    def write_training_config_yaml(
        self,
        *,
        dataset_dir: str | Path | None = None,
        target_path: str | Path | None = None,
    ) -> Path:
        config = self.build_training_config(dataset_dir=dataset_dir)
        output_path = (
            Path(target_path).expanduser()
            if target_path is not None
            else self.project_dir_path / "runs" / "training_config.yaml"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        return output_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "moval_version": self.moval_version,
            "title": self.title,
            "num_animals": self.num_animals,
            "animals_name": list(self.animals_name),
            "skeleton": self.skeleton_name,
            "skeleton_data": self.skeleton_data,
            "videos": [record.to_dict() for record in self.video_records],
            "ui_state": self.ui_state,
        }

    def save(self) -> Path:
        self._ensure_ui_defaults()
        self.ensure_project_skeleton_file()
        self.project_file.parent.mkdir(parents=True, exist_ok=True)
        with self.project_file.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        self.legacy_source_path = None
        return self.project_file

    def ensure_project_skeleton_file(self) -> Path:
        self.ensure_standard_structure()
        self.skeleton_yaml.parent.mkdir(parents=True, exist_ok=True)
        with self.skeleton_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(_normalize_skeleton_data(self.skeleton_data), f, sort_keys=False, allow_unicode=True)
        return self.skeleton_yaml

    def set_skeleton_data(
        self,
        skeleton_data: dict[str, Any],
        *,
        skeleton_name: Optional[str] = None,
        save: bool = True,
    ) -> None:
        self.skeleton_data = _normalize_skeleton_data(skeleton_data)
        if skeleton_name is not None:
            self.skeleton_name = Path(skeleton_name).name
        if save:
            self.save()

    def get_video_list(self) -> list[str]:
        return [file_entry.video for file_entry in self.files]

    def get_video_record(self, video_name_or_path: str | Path) -> Optional[VideoRecord]:
        raw = str(video_name_or_path)
        key = Path(raw).stem if Path(raw).suffix or Path(raw).parts else raw
        for record in self.video_records:
            if record.name == key:
                return record
        return None

    def resolve_video_path(self, record_or_name: VideoRecord | str | Path) -> Path:
        record = (
            record_or_name
            if isinstance(record_or_name, VideoRecord)
            else self.get_video_record(record_or_name)
        )
        if record is None:
            raise KeyError(f"Video not found: {record_or_name}")

        if record.relative_path:
            return (self.project_dir_path / record.relative_path).expanduser()
        if record.source_path:
            source_path = Path(record.source_path).expanduser()
            if source_path.exists():
                return source_path
            fallback = self.project_dir_path / "raw_videos" / record.file_name
            if fallback.exists():
                return fallback
            return source_path
        return self.project_dir_path / "raw_videos" / record.file_name

    def get_video_access_state(self, record_or_name: VideoRecord | str | Path) -> dict[str, Any]:
        record = (
            record_or_name
            if isinstance(record_or_name, VideoRecord)
            else self.get_video_record(record_or_name)
        )
        if record is None:
            raise KeyError(f"Video not found: {record_or_name}")

        resolved_path = self.resolve_video_path(record)
        source_exists = bool(record.source_path and Path(record.source_path).expanduser().exists())
        raw_fallback_exists = (self.project_dir_path / "raw_videos" / record.file_name).exists()
        if record.relative_path:
            storage = "project_copy"
            status = "available" if resolved_path.exists() else "missing_source"
            using_raw_fallback = False
        else:
            storage = "external_source"
            using_raw_fallback = bool(record.source_path and not source_exists and raw_fallback_exists)
            status = "fallback_copy" if using_raw_fallback else ("available" if resolved_path.exists() else "missing_source")

        return {
            "name": record.name,
            "storage": storage,
            "path": resolved_path,
            "exists": resolved_path.exists(),
            "source_exists": source_exists,
            "raw_fallback_exists": raw_fallback_exists,
            "using_raw_fallback": using_raw_fallback,
            "status": status,
        }

    def add_video(self, source_path: str | Path, *, copy_into_project: bool, save: bool = True) -> VideoRecord:
        src_path = Path(source_path).expanduser()
        if not src_path.is_file():
            raise FileNotFoundError(f"Video not found: {src_path}")

        video_name = src_path.stem
        if video_name in self.video_names:
            raise ValueError(
                f"The video name '{video_name}' is already in this project. "
                "Video filenames without extensions must be unique."
            )

        self.ensure_standard_structure()
        if copy_into_project:
            raw_dir = self.project_dir_path / "raw_videos"
            raw_dir.mkdir(parents=True, exist_ok=True)
            dst_path = raw_dir / src_path.name
            if dst_path.exists():
                raise FileExistsError(f"Destination already exists: {dst_path}")
            shutil.copy2(src_path, dst_path)
            record = VideoRecord(
                name=video_name,
                file_name=src_path.name,
                relative_path=dst_path.relative_to(self.project_dir_path).as_posix(),
                source_path=None,
            )
        else:
            record = VideoRecord(
                name=video_name,
                file_name=src_path.name,
                relative_path=None,
                source_path=src_path.resolve().as_posix(),
            )

        self.video_records.append(record)
        self._ensure_video_asset_dirs(video_name)
        if save:
            self.save()
        return record

    def relink_video_source(
        self,
        video_name_or_path: str | Path,
        source_path: str | Path,
        *,
        save: bool = True,
    ) -> VideoRecord:
        record = self.get_video_record(video_name_or_path)
        if record is None:
            raise KeyError(f"Video not found: {video_name_or_path}")

        src_path = Path(source_path).expanduser()
        if not src_path.is_file():
            raise FileNotFoundError(f"Video not found: {src_path}")
        if src_path.stem != record.name:
            raise ValueError(
                "The selected file must have the same filename stem as the project video "
                f"('{record.name}')."
            )

        record.file_name = src_path.name
        record.relative_path = None
        record.source_path = src_path.resolve().as_posix()
        if save:
            self.save()
        return record

    def copy_video_into_project(self, video_name_or_path: str | Path, *, save: bool = True) -> VideoRecord:
        record = self.get_video_record(video_name_or_path)
        if record is None:
            raise KeyError(f"Video not found: {video_name_or_path}")
        if record.relative_path:
            return record

        source_path = self.resolve_video_path(record)
        if not source_path.exists():
            raise FileNotFoundError(
                "The external source video could not be found. "
                "Relink it first or place the same file under raw_videos."
            )

        raw_dir = self.project_dir_path / "raw_videos"
        raw_dir.mkdir(parents=True, exist_ok=True)
        dst_path = raw_dir / source_path.name
        if not dst_path.exists():
            shutil.copy2(source_path, dst_path)

        record.file_name = dst_path.name
        record.relative_path = dst_path.relative_to(self.project_dir_path).as_posix()
        record.source_path = None
        if save:
            self.save()
        return record

    def remove_video(self, video_name_or_path: str | Path, *, delete_project_data: bool = True, save: bool = True) -> None:
        record = self.get_video_record(video_name_or_path)
        if record is None:
            raise KeyError(f"Video not found: {video_name_or_path}")

        self.video_records = [item for item in self.video_records if item.name != record.name]
        if delete_project_data:
            if record.relative_path:
                _remove_path(self.project_dir_path / record.relative_path)
            _remove_path(self.project_dir_path / "labels" / record.name)
            _remove_path(self.project_dir_path / "frames" / record.name)

        labelary_state = self.ui_state.get("labelary", {})
        if isinstance(labelary_state, dict) and labelary_state.get("video_name") == record.name:
            self.ui_state["labelary"] = {
                "frame_index": 0,
                "video_name": None,
                "label_name": None,
                "label_type": None,
                "color_mode": labelary_state.get("color_mode"),
            }

        if save:
            self.save()

    def import_csv_files(self, video_name_or_path: str | Path, csv_paths: list[str | Path]) -> list[Path]:
        record = self.get_video_record(video_name_or_path)
        if record is None:
            raise KeyError(f"Video not found: {video_name_or_path}")

        target_dir = self.project_dir_path / "labels" / record.name / "csv"
        imported: list[Path] = []
        for csv_path in csv_paths:
            imported.append(_copy_file_with_unique_name(csv_path, target_dir))
        return imported

    def remove_csv_files(self, video_name_or_path: str | Path, csv_names_or_paths: list[str | Path]) -> list[Path]:
        record = self.get_video_record(video_name_or_path)
        if record is None:
            raise KeyError(f"Video not found: {video_name_or_path}")

        removed: list[Path] = []
        target_dir = self.project_dir_path / "labels" / record.name / "csv"
        for value in csv_names_or_paths:
            path = Path(value)
            csv_path = path if path.is_absolute() else target_dir / path.name
            if csv_path.exists():
                csv_path.unlink()
                removed.append(csv_path)
        return removed

    def import_txt_directory(self, video_name_or_path: str | Path, txt_dir: str | Path) -> Path:
        record = self.get_video_record(video_name_or_path)
        if record is None:
            raise KeyError(f"Video not found: {video_name_or_path}")
        target_dir = self.project_dir_path / "labels" / record.name / "txt"
        return _copy_txt_directory(txt_dir, target_dir)

    def csv_dir(self, video_name_or_path: str | Path) -> Path:
        record = self.get_video_record(video_name_or_path)
        if record is None:
            raise KeyError(f"Video not found: {video_name_or_path}")
        return self.project_dir_path / "labels" / record.name / "csv"

    def txt_dir(self, video_name_or_path: str | Path) -> Path:
        record = self.get_video_record(video_name_or_path)
        if record is None:
            raise KeyError(f"Video not found: {video_name_or_path}")
        return self.project_dir_path / "labels" / record.name / "txt"

    def get_preferred_frame_mode(self) -> str:
        self._ensure_ui_defaults()
        return _normalize_frame_mode(self.ui_state.get("preferred_frame_mode"))

    def set_preferred_frame_mode(self, mode: str, *, save: bool = True) -> None:
        self.ui_state["preferred_frame_mode"] = _normalize_frame_mode(mode)
        if save:
            self.save()

    def get_labelary_state(self) -> dict[str, Any]:
        self._ensure_ui_defaults()
        labelary_state = self.ui_state.get("labelary", {})
        return dict(labelary_state)

    def save_labelary_state(
        self,
        *,
        video_name: Optional[str] = None,
        label_name: Optional[str] = None,
        label_type: Optional[str] = None,
        frame_index: Optional[int] = None,
        color_mode: Optional[str] = None,
        mode: Optional[str] = None,
        save: bool = True,
    ) -> None:
        self._ensure_ui_defaults()
        labelary_state = self.ui_state.setdefault("labelary", {})
        if video_name is not None:
            labelary_state["video_name"] = video_name
        if label_name is not None:
            labelary_state["label_name"] = label_name
        if label_type in {"csv", "txt", None}:
            labelary_state["label_type"] = label_type
        if frame_index is not None:
            labelary_state["frame_index"] = max(0, int(frame_index))
        if color_mode is not None:
            labelary_state["color_mode"] = color_mode
        if mode is not None:
            self.ui_state["preferred_frame_mode"] = _normalize_frame_mode(mode)
        if save:
            self.save()

    def _ensure_video_asset_dirs(self, video_name: str) -> None:
        for rel_path in (
            Path("labels") / video_name / "csv",
            Path("labels") / video_name / "txt",
        ):
            (self.project_dir_path / rel_path).mkdir(parents=True, exist_ok=True)

    def compress_project(
        self,
        *,
        delete_runs: bool = False,
        delete_predicts: bool = False,
    ) -> dict[str, int]:
        deleted_images = 0
        deleted_files = 0
        deleted_dirs = 0
        deleted_bytes = 0

        frames_root = self.project_dir_path / "frames"
        if frames_root.exists():
            for image_path in frames_root.rglob("*"):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in IMAGE_FILE_SUFFIXES:
                    continue
                if self._should_keep_frame_image(image_path):
                    continue
                deleted_images += 1
                deleted_files += 1
                try:
                    deleted_bytes += image_path.stat().st_size
                except OSError:
                    pass
                image_path.unlink()

        dataset_root = self.project_dir_path / "runs" / "dataset"
        if dataset_root.exists():
            deleted_files += _count_files(dataset_root)
            deleted_dirs += _count_dirs(dataset_root) + 1
            deleted_bytes += _count_bytes(dataset_root)
            shutil.rmtree(dataset_root)

        if delete_runs:
            runs_root = self.project_dir_path / "runs"
            if runs_root.exists():
                for child in list(runs_root.iterdir()):
                    if child.name == "dataset":
                        continue
                    if child.is_file() and child.suffix.lower() in {".yaml", ".yml", ".json"}:
                        continue
                    deleted_files += _count_files(child) if child.is_dir() else 1
                    deleted_dirs += _count_dirs(child) + (1 if child.is_dir() else 0)
                    deleted_bytes += _count_bytes(child)
                    _remove_path(child)

        if delete_predicts:
            predicts_root = self.project_dir_path / "predicts"
            if predicts_root.exists():
                deleted_files += _count_files(predicts_root)
                deleted_dirs += _count_dirs(predicts_root) + 1
                deleted_bytes += _count_bytes(predicts_root)
                shutil.rmtree(predicts_root)

        deleted_dirs += self._cleanup_empty_dirs(self.project_dir_path / "frames")
        deleted_dirs += self._cleanup_empty_dirs(self.project_dir_path / "runs")
        if delete_predicts:
            deleted_dirs += self._cleanup_empty_dirs(self.project_dir_path / "predicts")

        return {
            "deleted_images": deleted_images,
            "deleted_files": deleted_files,
            "deleted_dirs": deleted_dirs,
            "deleted_bytes": deleted_bytes,
        }

    def _should_keep_frame_image(self, image_path: Path) -> bool:
        frames_root = (self.project_dir_path / "frames").resolve()
        image_path = image_path.resolve()
        if frames_root.exists() and _is_inside(frames_root, image_path):
            try:
                relative_parts = image_path.relative_to(frames_root).parts
            except ValueError:
                relative_parts = ()
            if len(relative_parts) >= 2 and relative_parts[1] == "masks":
                return True
        return False

    def _cleanup_empty_dirs(self, root_path: Path) -> int:
        if not root_path.exists():
            return 0
        removed = 0
        for directory in sorted(
            (path for path in root_path.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                continue
            removed += 1
        return removed


def _legacy_video_record(item: dict[str, Any], project_dir: Path) -> VideoRecord:
    raw_video = str(item.get("video", "") or "")
    repaired_path = _repair_video_path(raw_video, project_dir)
    repaired_path = repaired_path.expanduser()

    if repaired_path.exists() and _is_inside(project_dir, repaired_path):
        relative_path = repaired_path.resolve().relative_to(project_dir.resolve()).as_posix()
        source_path = None
    else:
        relative_path = None
        source_path = repaired_path.as_posix()

    file_name = Path(raw_video).name or repaired_path.name
    name = Path(file_name).stem or Path(raw_video).stem
    return VideoRecord(
        name=name,
        file_name=file_name,
        relative_path=relative_path,
        source_path=source_path,
    )


def _repair_video_path(video_path: str, project_dir: Path) -> Path:
    stored_path = Path(video_path).expanduser()
    if stored_path.exists():
        return stored_path.resolve()

    fallback = project_dir / "raw_videos" / stored_path.name
    if stored_path.name and fallback.exists():
        return fallback.resolve()

    if stored_path.is_absolute():
        return stored_path
    return (project_dir / stored_path).resolve(strict=False)


__all__ = [
    "FRAME_MODES",
    "FileEntry",
    "LEGACY_PROJECT_FILENAMES",
    "PROJECT_FILENAME",
    "ProjectInformation",
    "VideoRecord",
]
