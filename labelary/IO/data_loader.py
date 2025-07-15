import pandas as pd
from pathlib import Path
from typing import Union, Optional, List
from PyQt6.QtWidgets import (
    QFileDialog, QMessageBox, QDialog, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox
)
from utils.skeleton import SkeletonModel
from typing import Optional

class DataLoader:
    parent: Optional[QDialog] = None

    loaded_data: Optional[pd.DataFrame] = None
    csv_path: Optional[str] = None
    skeleton_model: "SkeletonModel" = None
    kp_order: list = None
    _skeleton_loaded: bool = False

    img_width: Optional[int] = None 
    img_height: Optional[int] = None
    _coords_normalized: bool = False

    max_animals = 0
    animals_name = None
    track_mapping: dict[str, str] = {}

    ### Skeleton ###

    @classmethod
    def load_skeleton_info(cls, skeleton_model: "SkeletonModel") -> None:
        if cls._skeleton_loaded:
            return
        cls.skeleton_model = skeleton_model
        cls.kp_order = list(skeleton_model.nodes)
        cls._skeleton_loaded = True

    @classmethod
    def _ensure_skeleton(cls) -> None:
        if not cls._skeleton_loaded:
            raise RuntimeError("Skeleton information has not been loaded.")

    @classmethod
    def _check_skeleton_compat(cls, df: pd.DataFrame, parent=None) -> None:
        if not cls.kp_order:
            return

        file_kps = list()
        for c in df.columns:
            if isinstance(c, str):
                if c.endswith(".x"):
                    file_kps.append(c.split(".")[0])
        skel_kps = list(cls.skeleton_model.nodes)

        if file_kps != skel_kps:
            QMessageBox.critical(
                cls.parent,
                "Skeleton Mismatch",
                "The skeleton information in this data does not match the project configuration.\n"
                "Please select a different label or review the project file."
            )

    ### Coord Normalization ###

    @classmethod
    def set_image_dims(cls, w: int, h: int) -> None:
        cls.img_width, cls.img_height = w, h
        if cls.loaded_data is None:
            return
        if not cls._coords_normalized:
            cls._normalize_df()

    @staticmethod
    def _first_frame_row(df):
        try:
            first_idx = df["frame.idx"].min() 
            return df[df["frame.idx"] == first_idx].iloc[0]
        except Exception:
            return None
            
    @classmethod
    def _needs_normalize(cls, row):
        for col, val in row.items():
            if col.endswith(".x") and val > 1:
                y = row.get(col.replace(".x", ".y"), 0)
                if y > 1:
                    return True
        return False

    @classmethod
    def _normalize_df(cls):
        if cls.img_width is None or cls.img_height is None:
            print("⚠️ 해상도 정보 없음 → 정규화 건너뜀")
            return
        if not cls.kp_order:
            cls.kp_order = [c[:-2] for c in cls.loaded_data.columns if c.endswith(".x")]

        for kp in cls.kp_order:
            cls.loaded_data[f"{kp}.x"] /= cls.img_width
            cls.loaded_data[f"{kp}.y"] /= cls.img_height
        cls._coords_normalized = True

    ### Get Coords ###

    @classmethod
    def get_keypoint_coordinates_by_frame(cls, frame_idx):
        if cls.loaded_data is None or cls.loaded_data.empty:
            return {}
        coords = {t: {} for t in cls.animals_name}
        try:
            frame_df = cls.loaded_data.xs(frame_idx, level="frame.idx")
        except KeyError:
            return coords

        coords = {t: {} for t in cls.loaded_data["track"].unique()}
        for track, group in frame_df.groupby(level="track"): 
            if track not in coords:
                continue
            row = group.iloc[0]
            for kp in cls.kp_order:
                xcol, ycol, scol = f"{kp}.x", f"{kp}.y", f"{kp}.visibility"
                if xcol in row and ycol in row and scol in row:
                    coords[track][kp] = (row[xcol], row[ycol], row[scol])
        return coords

    @staticmethod
    def is_empty(obj) -> bool:
        if obj is None:
            return True
        if isinstance(obj, pd.DataFrame):
            return obj.empty
        if isinstance(obj, Sized):
            return len(obj) == 0
        return False

    ### Update ###

    @classmethod
    def update_kpt_visibility(cls, track, frame_idx, keypoint, visibility):
        if cls.loaded_data is None:
            print("DataLoader.update_kpt_visibility: No data loaded.")
            return
        mask = (cls.loaded_data["track"] == track) & (cls.loaded_data["frame.idx"] == frame_idx)
        if mask.sum() == 0:
            print(f"DataLoader.update_kpt_visibility: No row for track={track}, frame={frame_idx}")
            return
        col_v = f"{keypoint}.visibility"
        if col_v not in cls.loaded_data.columns:
            print(f"DataLoader.update_kpt_visibility: Column {col_v} not found.")
            return
        cls.loaded_data.loc[mask, col_v] = visibility
        return cls.loaded_data.loc[mask]

    @classmethod
    def update_point(cls, track, frame_idx, keypoint, norm_x, norm_y):
        if cls.loaded_data is None:
            print("DataLoader.update_point: No data loaded.")
            return
        mask = (cls.loaded_data["track"] == track) & (cls.loaded_data["frame.idx"] == frame_idx)
        if mask.sum() == 0:
            print(f"DataLoader.update_point: No row for track={track}, frame={frame_idx}")
            return
        col_x, col_y = f"{keypoint}.x", f"{keypoint}.y"
        if col_x not in cls.loaded_data.columns or col_y not in cls.loaded_data.columns:
            print(f"DataLoader.update_point: Columns {col_x} or {col_y} not found.")
            return
        cls.loaded_data.loc[mask, [col_x, col_y]] = [norm_x, norm_y]
        return cls.loaded_data.loc[mask]

    ### Modify Label ###

    @classmethod
    def create_new_data(cls, n_tracks: int = 1) -> bool:
        cls._ensure_skeleton()
        cols = ["track", "frame.idx", "instance.visibility"]
        for kp in cls.kp_order:
            cols += [f"{kp}.x", f"{kp}.y", f"{kp}.visibility"]
        df = pd.DataFrame(columns=cols)
        cls.loaded_data = df
        cls.csv_path = None
        cls._coords_normalized = True
        return True

    @classmethod
    def _to_project_name(cls, raw_track: str) -> str:
        return cls.track_mapping.get(raw_track, raw_track)

    @classmethod
    def add_skeleton_instance(cls,
                              frame_idx: int,
                              track_name: str,
                              anchor_xy: "tuple[float, float] | None" = None,
                              nearby_range: int = 300) -> bool:
        cls._ensure_skeleton()
        if cls.loaded_data is None:
            return False
        track_name = cls._to_project_name(track_name)

        frame_df = cls.loaded_data[cls.loaded_data["frame.idx"] == frame_idx]
        if frame_df["track"].nunique() >= getattr(cls, "max_animals", 1):
            print("Cannot add new skeleton: maximum instances reached for this frame.")
            return False

        if ((cls.loaded_data["track"] == track_name) &
            (cls.loaded_data["frame.idx"] == frame_idx)).any():
            return False

        new_row = {"track": track_name, "frame.idx": frame_idx}
        if "instance.visibility" in cls.loaded_data.columns:
            new_row["instance.visibility"] = 2

        init_coords: dict[str, tuple[float, float, int]] = {}
        mask = ((cls.loaded_data["track"] == track_name) &
                (cls.loaded_data["frame.idx"].between(frame_idx - nearby_range,
                                                      frame_idx + nearby_range)))
        if not cls.loaded_data[mask].empty:
            near_df = cls.loaded_data[mask]
            idx_nearest = (near_df["frame.idx"] - frame_idx).abs().idxmin()
            src = near_df.loc[idx_nearest]
            for kp in cls.kp_order:
                xcol, ycol, vcol = f"{kp}.x", f"{kp}.y", f"{kp}.visibility"
                init_coords[kp] = (src[xcol], src[ycol], src.get(vcol, 2))

        if not init_coords:
            ax, ay = anchor_xy if anchor_xy is not None else (0.5, 0.5)
            xs = [n.x for n in cls.skeleton_model.nodes.values()]
            ys = [n.y for n in cls.skeleton_model.nodes.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            w0, h0       = max_x - min_x, max_y - min_y
            cx, cy       = (min_x + max_x) / 2, (min_y + max_y) / 2

            target_size  = 0.125
            scale        = target_size / max(w0, h0) if max(w0, h0) else 1.0

            for kp, node in cls.skeleton_model.nodes.items():
                nx = ax + (node.x - cx) * scale
                ny = ay + (node.y - cy) * scale
                nx = max(0.0, min(nx, 1.0))
                ny = max(0.0, min(ny, 1.0))
                init_coords[kp] = (nx, ny, 2)

        for kp in cls.kp_order:
            xcol, ycol, vcol = f"{kp}.x", f"{kp}.y", f"{kp}.visibility"
            nx, ny, vis = init_coords.get(kp, (0.5, 0.5, 1))
            new_row[xcol], new_row[ycol], new_row[vcol] = nx, ny, vis

        try:
            cls.loaded_data = pd.concat([cls.loaded_data,
                                         pd.DataFrame([new_row])],
                                        ignore_index=True)
        except Exception as e:
            print(f"❌ Failed to add new skeleton row: {e}")
            return False
        if not cls.loaded_data.empty:
            cls.loaded_data = (cls.loaded_data
                               .set_index(["frame.idx", "track"], drop=False)
                               .sort_index())
        return True

    @classmethod
    def swap_or_rename_instance(cls,
                                frame_idx: int,
                                old_track: str,
                                new_track: str) -> bool:
        if cls.loaded_data is None or cls.loaded_data.empty:
            return False
        if old_track == new_track:
            return True

        m_old = (cls.loaded_data["frame.idx"] == frame_idx) & (cls.loaded_data["track"] == old_track)
        m_new = (cls.loaded_data["frame.idx"] == frame_idx) & (cls.loaded_data["track"] == new_track)

        if m_old.sum() == 0:
            return False 
        if m_new.sum() > 0:
            tmp_name = "__tmp_track__"
            cls.loaded_data.loc[m_new, "track"] = tmp_name
            cls.loaded_data.loc[m_old, "track"] = new_track
            cls.loaded_data.loc[cls.loaded_data["track"] == tmp_name, "track"] = old_track
        else:
            cls.loaded_data.loc[m_old, "track"] = new_track
        cls.loaded_data = (
            cls.loaded_data
            .set_index(["frame.idx", "track"], drop=False)
            .sort_index()
        )
        return True

    @classmethod
    def delete_instance(cls, frame_idx: int, track: str) -> bool:
        if cls.loaded_data is None or cls.loaded_data.empty:
            return False

        before = len(cls.loaded_data)
        cls.loaded_data = cls.loaded_data[
            ~((cls.loaded_data["frame.idx"] == frame_idx) &
              (cls.loaded_data["track"] == track))
        ]
        after = len(cls.loaded_data)
        if after == before:
            print(f"DeleteInstance: nothing to delete ({track}@{frame_idx})")
            return False

        if not cls.loaded_data.empty:
            cls.loaded_data = (
                cls.loaded_data
                .set_index(["frame.idx", "track"], drop=False)
                .sort_index()
            )
        print(f"Deleted {track} @ frame {frame_idx}")
        return True

    ### Load Label ###

    @classmethod
    def load_csv_data(cls, file_path: Union[str, Path]) -> bool:
        cls._ensure_skeleton()
        return cls._load_generic(file_path, read_func=pd.read_csv)

    @classmethod
    def load_txt_data(cls, path: Union[str, Path], sep: str = "\s+") -> bool:
        print("This may take some time.")
        cls._ensure_skeleton()
        path = Path(path)

        if path.is_dir():
            txt_files = sorted(path.glob("*.txt"))      # test_1.txt, test_2.txt, …
            if not txt_files:
                print("There is no txt file in the directory.")
                return False

            frames = []
            for single_frame_txt in txt_files:
                try:
                    f_idx = int(single_frame_txt.stem.split("_")[-1])
                except ValueError:
                    print(f"'{single_frame_txt.name}' → Failed to parse frame number, skipped")
                    continue

                single_frame_df = cls._txt_to_df(single_frame_txt, sep=sep, frame_idx=f_idx)
                if not single_frame_df.empty:
                    frames.append(single_frame_df)

            if not frames:
                print("There is no readable txt.")
                return False

            df_total = pd.concat(frames, ignore_index=True)
            return cls._load_generic(df_total, from_dataframe=True) 

        print("Attempting to read incorrect txt directory")
        return False

    @classmethod
    def _txt_to_df(cls, fp: Path, sep: str, frame_idx: int) -> pd.DataFrame:
        raw = pd.read_table(fp, header=None, sep=sep, engine="python")
        if raw.empty:
            return pd.DataFrame()

        kp_n   = (raw.shape[1] - 5) // 3
        if cls.kp_order and kp_n != len(cls.kp_order):
            raise ValueError(
                f"TXT file '{fp.name}' contains {kp_n} key-points, "
                f"but skeleton expects {len(cls.kp_order)}."
            )
        if not cls.kp_order:
            cls.kp_order = [f"kp{i+1}" for i in range(kp_n)]

        cols = (
            ["track.num", "bbox.x", "bbox.y", "bbox.w", "bbox.h"] +
            sum([[f"{kp}.x", f"{kp}.y", f"{kp}.visibility"] for kp in cls.kp_order], [])
        )
        raw.columns = cols[: raw.shape[1]]

        raw["track"] = raw["track.num"].apply(lambda n: f"track_{int(n)}")
        raw["frame.idx"] = frame_idx
        raw.drop(columns=["track.num", "bbox.x", "bbox.y", "bbox.w", "bbox.h"], inplace=True)

        for kp in cls.kp_order:
            vis_col = f"{kp}.visibility"
            if vis_col in raw.columns:
                raw.loc[~raw[vis_col].isin([1, 2]), vis_col] = 2
        return raw
        
    @classmethod
    def _load_generic(cls, src, read_func=None, *, from_dataframe: bool = False) -> bool:
        try:
            if from_dataframe:                  # txt
                df = src
                origin = "<DataFrame>"
                cls.csv_path = None
            else:                               # csv
                df = read_func(src)
                origin = str(src)
                cls.csv_path = origin

            cls._check_skeleton_compat(df)

            unique_tracks = df["track"].unique().tolist()
            if len(unique_tracks) > cls.max_animals:
                QMessageBox.critical(
                    None,
                    "Load Error",
                    f"The total number of tracks ({len(unique_tracks)}) exceeds the maximum allowed ({cls.max_animals})."
                )
                return False
            if set(unique_tracks) != set(cls.animals_name):
                mapping = cls._match_tracks(unique_tracks, cls.animals_name)
                if mapping is None:
                    return False
                df["track"] = df["track"].map(mapping)
                cls.track_mapping = mapping


            for col in list(df.columns):
                if col.endswith(".score"):
                    vis = col.replace(".score", ".visibility")
                    if vis not in df.columns:
                        df[vis] = 2
            df = df.drop(columns=[c for c in df.columns if c.endswith(".score")])

            kp_order: List[str] = []
            for c in df.columns:
                if c.endswith(".x"):
                    base = c[:-2]
                    if base not in kp_order:
                        kp_order.append(base)
            cls.kp_order = kp_order  

            base_cols = ["track", "frame.idx"]
            if "instance.visibility" in df.columns:
                base_cols.append("instance.visibility")

            new_order = base_cols.copy()
            for kp in kp_order:
                for suf in (".x", ".y", ".visibility"):
                    col = f"{kp}{suf}"
                    if col in df.columns:
                        new_order.append(col)
            new_order += [c for c in df.columns if c not in new_order]

            df = df[new_order] 
            df = (
                df.set_index(["frame.idx", "track"], drop=False)
                .sort_index()
            )

            first = cls._first_frame_row(df)
            if first is not None and cls._needs_normalize(first):
                cls.loaded_data = df
                cls._coords_normalized = False
                cls._normalize_df()
            else:
                cls.loaded_data = df
                cls._coords_normalized = True

            print(f"Loaded: {origin}")
            return True

        except Exception as e:
            print(f"Failed to load data: {e}")
            return False
            
    @classmethod
    def _match_tracks(cls, tracks: list[str], animal_names: list[str]):
        dlg = TrackMatchDialog(tracks, animal_names)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.get_mapping()
        return None

class TrackMatchDialog(QDialog):
    def __init__(self, tracks: list[str], animal_names: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Track mapping")
        self.comboboxes: dict[str, QComboBox] = {}

        layout = QVBoxLayout(self)
        for t in tracks:
            row = QHBoxLayout()
            row.addWidget(QLabel(t))

            cb = QComboBox()
            cb.addItems(animal_names)
            if t in animal_names:
                cb.setCurrentIndex(animal_names.index(t))
            else:
                cb.setEditable(True)
                cb.setPlaceholderText("select name")
                cb.setEditable(False)
                cb.setCurrentIndex(-1)
            row.addWidget(cb, 1)

            layout.addLayout(row)
            self.comboboxes[t] = cb

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Okay")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self._validate_and_accept)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)

        note = QLabel("The label file and the project config animal name do not match.\n"
                        "Match the names using the dialog below.\n"
                        "Or, press Cancel to cancel loading.")
        layout.addWidget(note)
        layout.addLayout(btn_row)

    def _validate_and_accept(self):
        mapping = {track: cb.currentText() for track, cb in self.comboboxes.items()}
        names = list(mapping.values())
        if len(names) != len(set(names)):
            QMessageBox.warning(self,
                                "Warning",
                                "Please select without duplication")
            return
        self.accept()

    def get_mapping(self) -> dict[str,str]:
        return {
            track: combo.currentText()
            for track, combo in self.comboboxes.items()
        }