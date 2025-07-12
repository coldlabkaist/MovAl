from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
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
    project_dir: "Path"
    title: str
    num_animals: int
    skeleton_name: str
    skeleton_yaml: "Path"
    files: List[FileEntry] = field(default_factory=list)

    @staticmethod
    def from_yaml(path: str | Path) -> "ProjectInformation":
        data = load_yaml(path)

        project_dir  = data.get("project_dir", "")
        title        = data.get("title", "")
        num_animals  = int(data.get("num_animals", 0))
        skeleton_name = Path(data.get("skeleton", "")).name
        skeleton_yaml  = Path.cwd() / "preset" / "skeleton" / skeleton_name

        file_entries: list[FileEntry] = []
        for item in data.get("files", []):
            file_entries.append(
                FileEntry(
                    video=item.get("video", ""),
                    csv=list(item.get("csv", [])),
                    txt=list(item.get("txt", [])),
                )
            )

        return ProjectInformation(
            project_dir=project_dir,
            title=title,
            num_animals=num_animals,
            skeleton_name=skeleton_name,
            skeleton_yaml=skeleton_yaml,
            files=file_entries,
        )

    def get_video_list(self) -> List[str]:
        return [f.video for f in self.files]