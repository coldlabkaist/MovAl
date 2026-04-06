from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import yaml

def load_yaml(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"YAML not found: {path}")

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass
class FileEntry:
    video: str
    csv: List[str] = field(default_factory=list)
    txt: List[str] = field(default_factory=list)

@dataclass
class ProjectInformation:
    moval_version: str
    project_dir: "Path"
    title: str
    num_animals: int
    animals_name: List[str]
    skeleton_name: str
    skeleton_yaml: "Path"
    files: List[FileEntry] = field(default_factory=list)

    @staticmethod
    def from_yaml(path: str | Path) -> "ProjectInformation":
        config_path = Path(path).expanduser().resolve()
        data = load_yaml(config_path)
        moval_version      = data.get("moval_version") 
        # Treat the config file location as the source of truth so moved projects
        # still work even if the stored absolute paths are stale.
        project_dir = config_path.parent.resolve()
        title        = data.get("title", "")
        num_animals  = int(data.get("num_animals", 0))
        animals_name = data.get("animals_name", []) 
        skeleton_name = Path(data.get("skeleton", "")).name
        skeleton_yaml  = Path.cwd() / "preset" / "skeleton" / skeleton_name

        file_entries: list[FileEntry] = []
        for item in data.get("files", []):
            video_path = _repair_video_path(item.get("video", ""), project_dir)
            video_stem = Path(video_path).stem or Path(item.get("video", "")).stem
            file_entries.append(
                FileEntry(
                    video=video_path,
                    csv=[
                        _repair_label_path(csv_path, project_dir, video_stem, "csv")
                        for csv_path in item.get("csv", [])
                    ],
                    txt=[
                        _repair_label_path(txt_path, project_dir, video_stem, "txt")
                        for txt_path in item.get("txt", [])
                    ],
                )
            )

        return ProjectInformation(
            moval_version=moval_version,
            project_dir=project_dir.as_posix(),
            title=title,
            num_animals=num_animals,
            animals_name=animals_name,
            skeleton_name=skeleton_name,
            skeleton_yaml=skeleton_yaml,
            files=file_entries,
        )

    def get_video_list(self) -> List[str]:
        return [f.video for f in self.files]


def _repair_video_path(video_path: str, project_dir: Path) -> str:
    stored_path = Path(video_path).expanduser()
    if stored_path.exists():
        return stored_path.resolve().as_posix()

    fallback = project_dir / "raw_videos" / stored_path.name
    if stored_path.name and fallback.exists():
        return fallback.resolve().as_posix()

    return stored_path.as_posix()


def _repair_label_path(label_path: str, project_dir: Path, video_stem: str, kind: str) -> str:
    stored_path = Path(label_path).expanduser()
    if stored_path.exists():
        return stored_path.resolve().as_posix()

    label_root = project_dir / "labels" / video_stem / kind
    fallback = label_root if kind == "txt" else label_root / stored_path.name
    if video_stem and fallback.exists():
        return fallback.resolve().as_posix()

    return stored_path.as_posix()
