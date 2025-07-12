import pandas as pd
from pathlib import Path
from typing import Union, Optional, List
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QDialog, QPushButton, QVBoxLayout
from utils.skeleton import SkeletonModel

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

        # kp_order 가 비어 있으면 즉석에서 추출
        if not cls.kp_order:
            cls.kp_order = [c[:-2] for c in cls.loaded_data.columns if c.endswith(".x")]

        for kp in cls.kp_order:
            cls.loaded_data[f"{kp}.x"] /= cls.img_width
            cls.loaded_data[f"{kp}.y"] /= cls.img_height
        cls._coords_normalized = True

    ### Get Coords ###

    @classmethod
    def get_keypoint_coordinates_by_frame(cls, frame_idx):
        if cls.is_empty(cls.loaded_data):
            return {}
        try:
            frame_df = cls.loaded_data.xs(frame_idx, level="frame.idx")
        except KeyError:
            return {t: {} for t in cls.loaded_data["track"].unique()}

        coords = {t: {} for t in cls.loaded_data["track"].unique()}
        for track, group in frame_df.groupby(level="track"): 
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
    def update_visibility(cls, track, frame_idx, keypoint, visibility):
        if cls.loaded_data is None:
            print("DataLoader.update_visibility: No data loaded.")
            return
        mask = (cls.loaded_data["track"] == track) & (cls.loaded_data["frame.idx"] == frame_idx)
        if mask.sum() == 0:
            print(f"DataLoader.update_visibility: No row for track={track}, frame={frame_idx}")
            return
        col_v = f"{keypoint}.visibility"
        if col_v not in cls.loaded_data.columns:
            print(f"DataLoader.update_visibility: Column {col_v} not found.")
            return
        cls.loaded_data.loc[mask, col_v] = visibility
        print(f"✅ Updated {col_v} to {cls.loaded_data.loc[mask, col_v]} (track={track}, frame={frame_idx})")
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
        print(f"✅ Updated {col_x}, {col_y} to ({cls.loaded_data.loc[mask, [col_x, col_y]]})")
        return cls.loaded_data.loc[mask]

    ### Create Label ###

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

            print(f"✅ Loaded: {origin}")
            return True

        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False
