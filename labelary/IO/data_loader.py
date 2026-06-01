from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import List, Optional, Union

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from tqdm import tqdm

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

    max_animals = 0
    max_instances_per_id = 1
    animals_name = None
    track_mapping: dict[str, str] = {}
    _expected_cols: Optional[int] = None
    _col_names: Optional[List[str]] = None
    _BATCH_ROWS: int = 20000
    records_tmp: List[dict] = []

    _label_version: int = 0
    _label_frames_cache: Optional[list] = None
    _label_cache_version: int = -1
    _inference_mode: bool = False

    INSTANCE_ID_COL = "instance.id"
    INSTANCE_SCORE_COL = "instance.score"
    INSTANCE_KEY_SEPARATOR = "::instance::"

    @classmethod
    def _bump_label_version(cls) -> None:
        cls._label_version += 1

    @classmethod
    def is_multi_instance_enabled(cls) -> bool:
        return max(1, int(cls.max_instances_per_id or 1)) > 1

    @classmethod
    def get_max_instances_per_id(cls) -> int:
        return max(1, int(cls.max_instances_per_id or 1))

    @classmethod
    def total_instance_capacity_per_frame(cls) -> int:
        return max(1, int(cls.max_animals or 0)) * cls.get_max_instances_per_id()

    @classmethod
    def get_labeled_frames(cls) -> list[int]:
        df = cls.loaded_data
        if df is None or df.empty:
            return []
        if (
            cls._label_frames_cache is not None
            and cls._label_cache_version == cls._label_version
        ):
            return cls._label_frames_cache

        try:
            labeled = sorted(int(v) for v in df["frame_idx"].dropna().unique().tolist())
        except Exception:
            labeled = []

        cls._label_frames_cache = labeled
        cls._label_cache_version = cls._label_version
        return labeled

    @classmethod
    def _init_txt_schema(cls, sample_fp: Path, sep: str) -> None:
        if cls._expected_cols is not None:
            return

        tmp = pd.read_csv(sample_fp, header=None, sep=sep, engine="python")
        cls._expected_cols = tmp.shape[1]
        cls._col_names = [f"c{i}" for i in range(cls._expected_cols)]

    @classmethod
    def load_skeleton_info(cls, skeleton_model: "SkeletonModel") -> None:
        cls.skeleton_model = skeleton_model
        cls.kp_order = list(skeleton_model.nodes)
        cls._skeleton_loaded = True
        cls._expected_cols = None
        cls._col_names = None
        cls.track_mapping = {}
        cls.loaded_data = None
        cls.csv_path = None
        cls._coords_normalized = False

    @classmethod
    def _ensure_skeleton(cls) -> None:
        if not cls._skeleton_loaded:
            raise RuntimeError("Skeleton information has not been loaded.")

    @classmethod
    def _check_skeleton_compat(cls, df: pd.DataFrame, parent=None) -> bool:
        if not cls.kp_order:
            return True

        file_kps = []
        for col in df.columns:
            if isinstance(col, str) and col.endswith(".x"):
                file_kps.append(col.split(".")[0])
        skel_kps = list(cls.skeleton_model.nodes)

        if len(file_kps) != len(skel_kps) or set(file_kps) != set(skel_kps):
            QMessageBox.warning(
                cls.parent or parent,
                "Skeleton Mismatch",
                "The skeleton information in this data does not match the project configuration.\n"
                "Please select a different label or review the project file.",
            )
            return False
        return True

    @classmethod
    def _check_txt_skeleton_compat(cls, sample_fp: Path) -> bool:
        cls._ensure_skeleton()
        if cls._expected_cols is None:
            return True

        extra_cols = cls._expected_cols - 5
        if extra_cols < 0:
            QMessageBox.warning(
                cls.parent,
                "TXT Format Error",
                f"The TXT layout in '{sample_fp.name}' does not match the expected YOLO pose format.",
            )
            return False

        if extra_cols % 3 == 0:
            txt_kpt_count = extra_cols // 3
        elif extra_cols > 0 and (extra_cols - 1) % 3 == 0:
            txt_kpt_count = (extra_cols - 1) // 3
        else:
            QMessageBox.warning(
                cls.parent,
                "TXT Format Error",
                f"The TXT layout in '{sample_fp.name}' does not match the expected YOLO pose format.",
            )
            return False

        expected_count = len(cls.kp_order or [])
        if expected_count and txt_kpt_count != expected_count:
            QMessageBox.warning(
                cls.parent,
                "Skeleton Mismatch",
                "This TXT folder is not compatible with the current project skeleton.\n"
                f"TXT keypoints: {txt_kpt_count}\n"
                f"Project skeleton keypoints: {expected_count}",
            )
            return False
        return True

    @classmethod
    def set_image_dims(cls, w: int, h: int) -> None:
        cls.img_width, cls.img_height = w, h
        if cls.loaded_data is None:
            return
        if not cls._coords_normalized:
            cls._normalize_df()

    @staticmethod
    def _first_frame_row(df: pd.DataFrame):
        try:
            first_idx = df["frame_idx"].min()
            return df[df["frame_idx"] == first_idx].iloc[0]
        except Exception:
            return None

    @classmethod
    def _needs_normalize(cls, row) -> bool:
        for col, val in row.items():
            if col.endswith(".x") and val > 1:
                y = row.get(col.replace(".x", ".y"), 0)
                if y > 1:
                    return True
        return False

    @classmethod
    def _normalize_df(cls) -> None:
        if cls.img_width is None or cls.img_height is None:
            print("No resolution information, Skip normalization")
            return
        if not cls.kp_order:
            cls.kp_order = [c[:-2] for c in cls.loaded_data.columns if c.endswith(".x")]

        for kp in cls.kp_order:
            cls.loaded_data[f"{kp}.x"] /= cls.img_width
            cls.loaded_data[f"{kp}.y"] /= cls.img_height
        cls._coords_normalized = True

    @classmethod
    def _requires_instance_ids(cls, df: pd.DataFrame) -> bool:
        return (
            cls.INSTANCE_ID_COL in df.columns
            or cls.is_multi_instance_enabled()
            or df.duplicated(["frame_idx", "track"]).any()
        )

    @classmethod
    def _sort_df(cls, df: pd.DataFrame, *, preserve_index: bool = False) -> pd.DataFrame:
        sort_cols = ["frame_idx", "track"]
        if cls.INSTANCE_ID_COL in df.columns:
            sort_cols.append(cls.INSTANCE_ID_COL)
        sorted_df = df.sort_values(sort_cols, kind="stable")
        return sorted_df if preserve_index else sorted_df.reset_index(drop=True)

    @classmethod
    def _assign_instance_ids_inplace(cls, df: pd.DataFrame) -> None:
        if cls.INSTANCE_ID_COL not in df.columns:
            df[cls.INSTANCE_ID_COL] = pd.Series([pd.NA] * len(df), dtype="Int64")
        df[cls.INSTANCE_ID_COL] = pd.to_numeric(
            df[cls.INSTANCE_ID_COL], errors="coerce"
        ).astype("Int64")

        for _, indices in df.groupby(["frame_idx", "track"], sort=False).groups.items():
            used: set[int] = set()
            for idx in indices:
                raw = df.at[idx, cls.INSTANCE_ID_COL]
                value = None
                if pd.notna(raw):
                    try:
                        parsed = int(raw)
                        if parsed > 0 and parsed not in used:
                            value = parsed
                    except Exception:
                        value = None
                if value is None:
                    value = 1
                    while value in used:
                        value += 1
                used.add(value)
                df.at[idx, cls.INSTANCE_ID_COL] = value

        df[cls.INSTANCE_ID_COL] = pd.to_numeric(
            df[cls.INSTANCE_ID_COL], errors="coerce"
        ).astype("Int64")

    @classmethod
    def _normalize_loaded_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.reset_index(drop=True)
        df = df.loc[:, ~df.columns.duplicated()]
        df["track"] = df["track"].astype(str)
        df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce").fillna(0).astype(int)

        if "instance.visibility" not in df.columns:
            df["instance.visibility"] = 2

        if cls._requires_instance_ids(df):
            cls._assign_instance_ids_inplace(df)

        return cls._sort_df(df)

    @classmethod
    def _row_score_series(cls, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)

        score = pd.Series(np.zeros(len(df), dtype=float), index=df.index, dtype=float)
        if cls.INSTANCE_SCORE_COL in df.columns:
            score = pd.to_numeric(df[cls.INSTANCE_SCORE_COL], errors="coerce").fillna(score)

        kp_score_cols = [
            col for col in df.columns
            if isinstance(col, str)
            and col.endswith(".score")
            and col != cls.INSTANCE_SCORE_COL
        ]
        if kp_score_cols:
            kp_score = (
                df[kp_score_cols]
                .apply(pd.to_numeric, errors="coerce")
                .max(axis=1)
                .fillna(score)
            )
            score = np.maximum(score.to_numpy(dtype=float), kp_score.to_numpy(dtype=float))
            score = pd.Series(score, index=df.index, dtype=float)

        return score.fillna(0.0)

    @classmethod
    def _prune_loaded_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "frame_idx" not in df.columns or "track" not in df.columns:
            return df

        limit = cls.get_max_instances_per_id()
        pruned = df.copy()
        pruned["_instance_score_sort"] = cls._row_score_series(pruned)

        sort_cols = ["frame_idx", "track", "_instance_score_sort"]
        ascending = [True, True, False]
        if cls.INSTANCE_ID_COL in pruned.columns:
            pruned[cls.INSTANCE_ID_COL] = pd.to_numeric(pruned[cls.INSTANCE_ID_COL], errors="coerce")
            sort_cols.append(cls.INSTANCE_ID_COL)
            ascending.append(True)
            pruned = (
                pruned.sort_values(sort_cols, ascending=ascending, kind="stable")
                .groupby(["frame_idx", "track", cls.INSTANCE_ID_COL], dropna=False, sort=False, group_keys=False)
                .head(1)
            )
            sort_cols = ["frame_idx", "track", "_instance_score_sort", cls.INSTANCE_ID_COL]
            ascending = [True, True, False, True]
        pruned = (
            pruned.sort_values(sort_cols, ascending=ascending, kind="stable")
            .groupby(["frame_idx", "track"], sort=False, group_keys=False)
            .head(limit)
            .drop(columns=["_instance_score_sort"], errors="ignore")
            .reset_index(drop=True)
        )
        return pruned

    @classmethod
    def visible_dataframe(cls, df: pd.DataFrame | None = None) -> pd.DataFrame:
        source = cls.loaded_data if df is None else df
        if source is None or source.empty:
            return pd.DataFrame(columns=(source.columns if source is not None else []))

        visible = cls._sort_df(source, preserve_index=True)
        if cls.is_multi_instance_enabled():
            return visible.reset_index(drop=True)

        return (
            visible.groupby(["frame_idx", "track"], sort=False, group_keys=False)
            .head(1)
            .reset_index(drop=True)
        )

    @classmethod
    def _frame_df(cls, frame_idx: int) -> pd.DataFrame:
        if cls.loaded_data is None or cls.loaded_data.empty:
            return pd.DataFrame(columns=(cls.loaded_data.columns if cls.loaded_data is not None else []))
        return cls.loaded_data[cls.loaded_data["frame_idx"] == int(frame_idx)].copy()

    @classmethod
    def _visible_frame_df(cls, frame_idx: int) -> pd.DataFrame:
        frame_df = cls._frame_df(frame_idx)
        if frame_df.empty:
            return frame_df

        visible = cls._sort_df(frame_df, preserve_index=True)
        if cls.is_multi_instance_enabled():
            return visible
        return visible.groupby(["frame_idx", "track"], sort=False, group_keys=False).head(1)

    @classmethod
    def make_instance_key(cls, track_name: str, instance_id: Optional[int] = None) -> str:
        base_track = str(track_name)
        if not cls.is_multi_instance_enabled():
            return base_track
        resolved_id = int(instance_id or 1)
        return f"{base_track}{cls.INSTANCE_KEY_SEPARATOR}{resolved_id}"

    @classmethod
    def split_instance_key(cls, key: str) -> tuple[str, Optional[int]]:
        raw = str(key)
        if cls.INSTANCE_KEY_SEPARATOR not in raw:
            return raw, None
        base_track, raw_id = raw.split(cls.INSTANCE_KEY_SEPARATOR, 1)
        try:
            return base_track, int(raw_id)
        except Exception:
            return base_track, None

    @classmethod
    def get_base_track_name(cls, key: str) -> str:
        return cls.split_instance_key(key)[0]

    @classmethod
    def get_instance_id_from_key(cls, key: str) -> Optional[int]:
        return cls.split_instance_key(key)[1]

    @classmethod
    def display_label_for_key(cls, key: str) -> str:
        base_track, instance_id = cls.split_instance_key(key)
        if not cls.is_multi_instance_enabled() or instance_id is None:
            return base_track
        return f"{base_track} [{instance_id}]"

    @classmethod
    def row_instance_key(cls, row: pd.Series) -> str:
        track = str(row.get("track", ""))
        instance_id = row.get(cls.INSTANCE_ID_COL)
        if pd.isna(instance_id):
            instance_id = None
        return cls.make_instance_key(track, int(instance_id) if instance_id is not None else None)

    @classmethod
    def get_track_keys_for_frame(cls, frame_idx: int) -> list[str]:
        frame_df = cls._visible_frame_df(frame_idx)
        if frame_df.empty:
            return []
        return [cls.row_instance_key(row) for _, row in frame_df.iterrows()]

    @classmethod
    def get_keypoint_coordinates_by_frame(cls, frame_idx: int) -> dict[str, dict[str, tuple[float, float, int]]]:
        if cls.loaded_data is None or cls.loaded_data.empty:
            return {}

        frame_df = cls._visible_frame_df(frame_idx)
        if frame_df.empty:
            return {}

        coords: dict[str, dict[str, tuple[float, float, int]]] = {}
        for _, row in frame_df.iterrows():
            instance_key = cls.row_instance_key(row)
            kp_map: dict[str, tuple[float, float, int]] = {}
            for kp in cls.kp_order:
                xcol, ycol, scol = f"{kp}.x", f"{kp}.y", f"{kp}.visibility"
                if xcol in row and ycol in row and scol in row:
                    kp_map[kp] = (row[xcol], row[ycol], int(row[scol]))
            coords[instance_key] = kp_map
        return coords

    @classmethod
    def frame_has_labels(cls, frame_idx: int) -> bool:
        df = cls.loaded_data
        if df is None or df.empty:
            return False
        try:
            return bool((df["frame_idx"] == int(frame_idx)).any())
        except Exception:
            return False

    @staticmethod
    def is_empty(obj) -> bool:
        if obj is None:
            return True
        if isinstance(obj, pd.DataFrame):
            return obj.empty
        if isinstance(obj, Sized):
            return len(obj) == 0
        return False

    @classmethod
    def frame_track_instance_count(cls, frame_idx: int, track_name: str) -> int:
        frame_df = cls._frame_df(frame_idx)
        if frame_df.empty:
            return 0
        base_track = cls._to_project_name(track_name)
        return int((frame_df["track"] == base_track).sum())

    @classmethod
    def frame_has_capacity_for_new_instance(cls, frame_idx: int) -> bool:
        return len(cls.available_track_names_for_new_instance(frame_idx)) > 0

    @classmethod
    def available_track_names_for_new_instance(cls, frame_idx: int) -> list[str]:
        names = list(cls.animals_name or [])
        limit = cls.get_max_instances_per_id()
        available = []
        for name in names:
            if cls.frame_track_instance_count(frame_idx, name) < limit:
                available.append(name)
        return available

    @classmethod
    def _track_priority_order(cls, preferred_track: Optional[str] = None) -> list[str]:
        names = [str(name) for name in (cls.animals_name or [])]
        if not names:
            return []
        if not preferred_track:
            return names

        preferred_track = cls._to_project_name(preferred_track)
        if preferred_track not in names:
            return names

        start_idx = names.index(preferred_track)
        return names[start_idx + 1 :] + names[: start_idx + 1]

    @classmethod
    def find_reference_instance_row(
        cls,
        frame_idx: int,
        track_name: str,
        instance_id: Optional[int],
        nearby_range: int = 300,
    ) -> Optional[pd.Series]:
        if cls.loaded_data is None or cls.loaded_data.empty:
            return None

        track_name = cls._to_project_name(track_name)
        mask = (
            (cls.loaded_data["track"] == track_name)
            & (cls.loaded_data["frame_idx"] != int(frame_idx))
            & (cls.loaded_data["frame_idx"].between(frame_idx - nearby_range, frame_idx + nearby_range))
        )

        if instance_id is not None:
            if cls.INSTANCE_ID_COL in cls.loaded_data.columns:
                instance_series = pd.to_numeric(
                    cls.loaded_data[cls.INSTANCE_ID_COL], errors="coerce"
                )
                mask &= instance_series == int(instance_id)
            elif int(instance_id) != 1:
                return None

        ref_df = cls.loaded_data.loc[mask].copy()
        if ref_df.empty:
            return None

        ref_df["_frame_distance"] = (ref_df["frame_idx"] - int(frame_idx)).abs()
        ref_df = ref_df.sort_values(["_frame_distance"], kind="stable")
        return ref_df.iloc[0]

    @classmethod
    def _has_reference_for_instance_slot(
        cls,
        frame_idx: int,
        track_name: str,
        instance_id: Optional[int],
        nearby_range: int = 300,
    ) -> bool:
        return cls.find_reference_instance_row(
            frame_idx,
            track_name,
            instance_id,
            nearby_range=nearby_range,
        ) is not None

    @classmethod
    def resolve_new_instance_track(
        cls,
        frame_idx: int,
        preferred_track: Optional[str] = None,
        nearby_range: int = 300,
    ) -> Optional[str]:
        cls._ensure_skeleton()
        names = [str(name) for name in (cls.animals_name or [])]
        if not names:
            return None

        limit = cls.get_max_instances_per_id()
        preferred_track = cls._to_project_name(preferred_track) if preferred_track else None

        if preferred_track and cls.frame_track_instance_count(frame_idx, preferred_track) < limit:
            preferred_slot = cls._next_free_instance_id(frame_idx, preferred_track)
            if cls._has_reference_for_instance_slot(
                frame_idx,
                preferred_track,
                preferred_slot,
                nearby_range=nearby_range,
            ):
                return preferred_track

        ordered_tracks = cls._track_priority_order(preferred_track)
        for slot in range(1, limit + 1):
            for track_name in ordered_tracks:
                if cls.frame_track_instance_count(frame_idx, track_name) >= limit:
                    continue
                if cls._next_free_instance_id(frame_idx, track_name) != slot:
                    continue
                return track_name

        return None

    @classmethod
    def _resolve_visible_row_index(cls, frame_idx: int, instance_key: str) -> Optional[int]:
        frame_df = cls._visible_frame_df(frame_idx)
        if frame_df.empty:
            return None

        base_track, instance_id = cls.split_instance_key(instance_key)
        if cls.is_multi_instance_enabled():
            if cls.INSTANCE_ID_COL in frame_df.columns and instance_id is not None:
                matched = frame_df[
                    (frame_df["track"] == base_track)
                    & (frame_df[cls.INSTANCE_ID_COL] == int(instance_id))
                ]
            else:
                matched = frame_df[frame_df["track"] == base_track]
        else:
            matched = frame_df[frame_df["track"] == base_track]

        if matched.empty:
            return None
        return int(matched.index[0])

    @classmethod
    def update_kpt_visibility(cls, instance_key, frame_idx, keypoint, visibility):
        if cls.loaded_data is None:
            print("DataLoader.update_kpt_visibility: No data loaded.")
            return

        row_index = cls._resolve_visible_row_index(frame_idx, instance_key)
        if row_index is None:
            print(f"DataLoader.update_kpt_visibility: No row for key={instance_key}, frame={frame_idx}")
            return

        col_v = f"{keypoint}.visibility"
        if col_v not in cls.loaded_data.columns:
            print(f"DataLoader.update_kpt_visibility: Column {col_v} not found.")
            return

        cls.loaded_data.loc[row_index, col_v] = visibility
        return cls.loaded_data.loc[[row_index]]

    @classmethod
    def update_point(cls, instance_key, frame_idx, keypoint, norm_x, norm_y):
        if cls.loaded_data is None:
            print("DataLoader.update_point: No data loaded.")
            return

        row_index = cls._resolve_visible_row_index(frame_idx, instance_key)
        if row_index is None:
            print(f"DataLoader.update_point: No row for key={instance_key}, frame={frame_idx}")
            return

        col_x, col_y = f"{keypoint}.x", f"{keypoint}.y"
        if col_x not in cls.loaded_data.columns or col_y not in cls.loaded_data.columns:
            print(f"DataLoader.update_point: Columns {col_x} or {col_y} not found.")
            return

        cls.loaded_data.loc[row_index, [col_x, col_y]] = [norm_x, norm_y]
        return cls.loaded_data.loc[[row_index]]

    @classmethod
    def create_new_data(cls, n_tracks: int = 1) -> bool:
        _ = n_tracks
        cls._ensure_skeleton()
        cols = ["track", "frame_idx", "instance.visibility"]
        if cls.is_multi_instance_enabled():
            cols.append(cls.INSTANCE_ID_COL)
        for kp in cls.kp_order:
            cols += [f"{kp}.x", f"{kp}.y", f"{kp}.visibility"]
        cls.loaded_data = pd.DataFrame(columns=cols)
        cls.csv_path = None
        cls._coords_normalized = True
        cls._bump_label_version()
        return True

    @classmethod
    def add_auto_labeled_frame(cls, frame_idx: int, instances: list[dict]) -> bool:
        cls._ensure_skeleton()
        if not instances:
            return False

        if cls.loaded_data is None:
            cls.create_new_data()

        if cls.frame_has_labels(frame_idx):
            return False

        rows: list[dict] = []
        per_track_counts: dict[str, int] = {}
        per_track_limit = cls.get_max_instances_per_id()
        include_instance_id = cls.is_multi_instance_enabled() or (
            cls.loaded_data is not None and cls.INSTANCE_ID_COL in cls.loaded_data.columns
        )

        for instance in instances:
            track_name = instance.get("track")
            if track_name is None:
                continue
            track_name = cls._to_project_name(str(track_name))
            count = per_track_counts.get(track_name, 0)
            if count >= per_track_limit:
                continue

            keypoints = instance.get("keypoints", {})
            row = {
                "track": track_name,
                "frame_idx": int(frame_idx),
                "instance.visibility": 2,
            }
            raw_instance_id = instance.get("instance_id")
            resolved_instance_id = (
                int(raw_instance_id)
                if raw_instance_id is not None and str(raw_instance_id).strip() != ""
                else count + 1
            )
            if include_instance_id:
                row[cls.INSTANCE_ID_COL] = resolved_instance_id

            for kp in cls.kp_order:
                x, y, vis = keypoints.get(kp, (0.0, 0.0, 1))
                row[f"{kp}.x"] = float(x)
                row[f"{kp}.y"] = float(y)
                row[f"{kp}.visibility"] = int(vis)
            rows.append(row)
            per_track_counts[track_name] = count + 1

        if not rows:
            return False

        new_rows = pd.DataFrame.from_records(rows)
        if cls.loaded_data is None or cls.loaded_data.empty:
            cls.loaded_data = new_rows
        else:
            cls.loaded_data = pd.concat(
                [cls.loaded_data.reset_index(drop=True), new_rows],
                ignore_index=True,
                sort=False,
            )

        cls.loaded_data = cls._normalize_loaded_df(cls.loaded_data)
        cls._coords_normalized = True
        cls._bump_label_version()
        return True

    @classmethod
    def _to_project_name(cls, raw_track: str) -> str:
        return cls.track_mapping.get(raw_track, raw_track)

    @classmethod
    def _next_free_instance_id(
        cls,
        frame_idx: int,
        track_name: str,
        *,
        exclude_index: Optional[int] = None,
    ) -> int:
        frame_df = cls._frame_df(frame_idx)
        if frame_df.empty or cls.INSTANCE_ID_COL not in frame_df.columns:
            return 1

        mask = frame_df["track"] == track_name
        if exclude_index is not None:
            mask &= frame_df.index != exclude_index
        used = {
            int(v)
            for v in frame_df.loc[mask, cls.INSTANCE_ID_COL].dropna().tolist()
            if int(v) > 0
        }
        candidate = 1
        while candidate in used:
            candidate += 1
        return candidate

    @classmethod
    def add_skeleton_instance(
        cls,
        frame_idx: int,
        track_name: str,
        anchor_xy: "tuple[float, float] | None" = None,
        nearby_range: int = 300,
    ) -> bool:
        cls._ensure_skeleton()
        if cls.loaded_data is None:
            return False
        track_name = cls._to_project_name(track_name)

        existing_count = cls.frame_track_instance_count(frame_idx, track_name)
        if existing_count >= cls.get_max_instances_per_id():
            print("Cannot add new skeleton: maximum instances reached for this ID in this frame.")
            return False

        if not cls.is_multi_instance_enabled() and existing_count > 0:
            return False

        instance_id: Optional[int] = None
        if cls.is_multi_instance_enabled() or cls.INSTANCE_ID_COL in cls.loaded_data.columns:
            instance_id = cls._next_free_instance_id(frame_idx, track_name)

        new_row = {"track": track_name, "frame_idx": int(frame_idx), "instance.visibility": 2}
        if instance_id is not None:
            new_row[cls.INSTANCE_ID_COL] = instance_id

        init_coords: dict[str, tuple[float, float, int]] = {}
        src = cls.find_reference_instance_row(
            frame_idx,
            track_name,
            instance_id,
            nearby_range=nearby_range,
        )
        if src is not None:
            for kp in cls.kp_order:
                xcol, ycol, vcol = f"{kp}.x", f"{kp}.y", f"{kp}.visibility"
                init_coords[kp] = (src[xcol], src[ycol], int(src.get(vcol, 2)))

        if not init_coords:
            ax, ay = anchor_xy if anchor_xy is not None else (0.5, 0.5)
            xs = [node.x for node in cls.skeleton_model.nodes.values()]
            ys = [node.y for node in cls.skeleton_model.nodes.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            w0, h0 = max_x - min_x, max_y - min_y
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2

            target_size = 0.125
            scale = target_size / max(w0, h0) if max(w0, h0) else 1.0

            for kp, node in cls.skeleton_model.nodes.items():
                nx = ax + (node.x - cx) * scale
                ny = ay + (node.y - cy) * scale
                init_coords[kp] = (
                    max(0.0, min(nx, 1.0)),
                    max(0.0, min(ny, 1.0)),
                    2,
                )

        for kp in cls.kp_order:
            xcol, ycol, vcol = f"{kp}.x", f"{kp}.y", f"{kp}.visibility"
            nx, ny, vis = init_coords.get(kp, (0.5, 0.5, 1))
            new_row[xcol], new_row[ycol], new_row[vcol] = nx, ny, vis

        try:
            cls.loaded_data = pd.concat(
                [cls.loaded_data.reset_index(drop=True), pd.DataFrame([new_row])],
                ignore_index=True,
                sort=False,
            )
        except Exception as err:
            print(f"Failed to add new skeleton row: {err}")
            return False

        cls.loaded_data = cls._normalize_loaded_df(cls.loaded_data)
        cls._bump_label_version()
        return True

    @classmethod
    def swap_or_rename_instance(cls, frame_idx: int, old_track: str, new_track: str) -> Optional[str]:
        if cls.loaded_data is None or cls.loaded_data.empty:
            return None

        old_index = cls._resolve_visible_row_index(frame_idx, old_track)
        if old_index is None:
            return None

        old_base_track = cls.get_base_track_name(old_track)
        new_track = cls._to_project_name(new_track)
        if old_base_track == new_track:
            return old_track

        if cls.is_multi_instance_enabled():
            if cls.frame_track_instance_count(frame_idx, new_track) >= cls.get_max_instances_per_id():
                return None
            cls.loaded_data.loc[old_index, "track"] = new_track
            new_instance_id = None
            if cls.INSTANCE_ID_COL in cls.loaded_data.columns:
                new_instance_id = cls._next_free_instance_id(
                    frame_idx,
                    new_track,
                    exclude_index=old_index,
                )
                cls.loaded_data.loc[old_index, cls.INSTANCE_ID_COL] = new_instance_id
            cls.loaded_data = cls._normalize_loaded_df(cls.loaded_data)
            cls._bump_label_version()
            return cls.make_instance_key(new_track, new_instance_id)

        new_index = cls._resolve_visible_row_index(frame_idx, new_track)
        if new_index is None:
            cls.loaded_data.loc[old_index, "track"] = new_track
        else:
            cls.loaded_data.loc[old_index, "track"] = "__tmp_track__"
            cls.loaded_data.loc[new_index, "track"] = old_base_track
            cls.loaded_data.loc[old_index, "track"] = new_track

        cls.loaded_data = cls._normalize_loaded_df(cls.loaded_data)
        cls._bump_label_version()
        return new_track

    @classmethod
    def delete_instance(cls, frame_idx: int, track: str) -> bool:
        if cls.loaded_data is None or cls.loaded_data.empty:
            return False

        row_index = cls._resolve_visible_row_index(frame_idx, track)
        if row_index is None:
            print(f"DeleteInstance: nothing to delete ({track}@{frame_idx})")
            return False

        cls.loaded_data = cls.loaded_data.drop(index=row_index).reset_index(drop=True)
        if not cls.loaded_data.empty:
            cls.loaded_data = cls._normalize_loaded_df(cls.loaded_data)
        cls._bump_label_version()
        return True

    @classmethod
    def load_csv_data(cls, file_path: Union[str, Path]) -> bool:
        cls._ensure_skeleton()
        return cls._load_generic(file_path, read_func=pd.read_csv)

    @classmethod
    def load_txt_data(
        cls,
        path: Union[str, Path],
        sep: str = r"\s+",
        inference_mode: bool = False,
    ) -> bool:
        print("This may take some time.")
        cls._ensure_skeleton()
        path = Path(path)
        cls._inference_mode = inference_mode

        if path.is_dir():
            candidate_dirs = [path]
            if inference_mode:
                labels_dir = path / "labels"
                if labels_dir != path:
                    candidate_dirs.append(labels_dir)

            selected_dir: Path | None = None
            txt_files: list[Path] = []
            for candidate_dir in candidate_dirs:
                if not candidate_dir.is_dir():
                    continue
                candidate_txt_files = sorted(candidate_dir.glob("*.txt"))
                if candidate_txt_files:
                    selected_dir = candidate_dir
                    txt_files = candidate_txt_files
                    break

            if not txt_files:
                print("There is no txt file in the directory.")
                return False
            if selected_dir is not None and selected_dir != path:
                print(f"Using TXT files from: {selected_dir}")

            cls._init_txt_schema(txt_files[0], sep)
            if not cls._check_txt_skeleton_compat(txt_files[0]):
                return False

            cpu_n = max(multiprocessing.cpu_count() - 1, 1)
            chunks: list[pd.DataFrame] = []
            cls.records_tmp = []

            def _safe_parse(fp: Path):
                try:
                    f_idx = int(fp.stem.split("_")[-1])
                    return cls._txt_to_records(fp, sep, f_idx)
                except Exception as err:
                    print(f"{fp.name} ??skip ({err})")
                    return []

            with ThreadPoolExecutor(max_workers=cpu_n) as pool:
                for recs in tqdm(
                    pool.map(_safe_parse, txt_files),
                    total=len(txt_files),
                    desc="Loading TXT frames",
                ):
                    if not recs:
                        continue
                    cls.records_tmp.extend(recs)
                    if len(cls.records_tmp) >= cls._BATCH_ROWS:
                        chunks.append(pd.DataFrame.from_records(cls.records_tmp))
                        cls.records_tmp.clear()

            if cls.records_tmp:
                chunks.append(pd.DataFrame.from_records(cls.records_tmp))
                cls.records_tmp.clear()
            if not chunks:
                print("There is no readable txt.")
                return False

            df_total = pd.concat(chunks, ignore_index=True)
            return cls._load_generic(df_total, from_dataframe=True)

        print("Attempting to read incorrect txt directory")
        return False

    @classmethod
    def _txt_to_records(cls, fp: Path, sep: str, frame_idx: int) -> List[dict]:
        buf = Path(fp).read_bytes()
        arr = np.fromstring(buf, sep=" ", dtype=np.float32)

        if arr.size % cls._expected_cols:
            return []

        arr = arr.reshape(-1, cls._expected_cols)

        body_cols = arr.shape[1] - 5
        has_instance_id = body_cols % 3 == 1
        kp_n = (body_cols - 1) // 3 if has_instance_id else body_cols // 3
        if cls.kp_order and kp_n != len(cls.kp_order):
            raise ValueError(f"{fp.name}: {kp_n} kpts != {len(cls.kp_order)}")
        if not cls.kp_order:
            cls.kp_order = [f"kp{i + 1}" for i in range(kp_n)]

        records = []
        for row in arr:
            data_slice = row[5:-1] if has_instance_id else row[5:]
            instance_score = 0.0
            if cls._inference_mode and data_slice.size >= 3:
                try:
                    instance_score = float(np.max(data_slice[2::3]))
                except Exception:
                    instance_score = 0.0
            rec: dict = {
                "track": f"track_{int(row[0])}",
                "frame_idx": frame_idx,
                "instance.visibility": 2,
            }
            if cls._inference_mode:
                rec[cls.INSTANCE_SCORE_COL] = instance_score
            if has_instance_id:
                rec[cls.INSTANCE_ID_COL] = int(row[-1])

            off = 0
            for kp in cls.kp_order:
                x, y, vis = data_slice[off: off + 3]
                rec[f"{kp}.x"] = float(x)
                rec[f"{kp}.y"] = float(y)
                if cls._inference_mode:
                    rec[f"{kp}.score"] = float(vis)
                    rec[f"{kp}.visibility"] = 2
                else:
                    vis_int = int(round(float(vis)))
                    rec[f"{kp}.visibility"] = 1 if vis_int == 1 else 2
                off += 3
            records.append(rec)
        return records

    @classmethod
    def _load_generic(cls, src, read_func=None, *, from_dataframe: bool = False) -> bool:
        try:
            if from_dataframe:
                df = src.copy()
                cls.csv_path = None
            else:
                df = read_func(src)
                cls.csv_path = str(src)

            if not cls._check_skeleton_compat(df):
                return False

            unique_tracks = df["track"].astype(str).unique().tolist()
            if len(unique_tracks) > cls.max_animals:
                QMessageBox.critical(
                    None,
                    "Load Error",
                    f"The total number of tracks ({len(unique_tracks)}) exceeds the maximum allowed ({cls.max_animals}).",
                )
                return False
            if set(unique_tracks) != set(cls.animals_name):
                mapping = cls._match_tracks(unique_tracks, cls.animals_name)
                if mapping is None:
                    return False
                df["track"] = df["track"].map(mapping)
                cls.track_mapping = mapping

            df = cls._prune_loaded_df(df)

            for score_col in [c for c in df.columns if c.endswith(".score")]:
                vis_col = score_col.replace(".score", ".visibility")
                if vis_col not in df.columns:
                    df[vis_col] = 2
            df = df.drop(columns=[c for c in df.columns if c.endswith(".score")])

            file_kp_order: List[str] = []
            for col in df.columns:
                if isinstance(col, str) and col.endswith(".x"):
                    base = col[:-2]
                    if base not in file_kp_order:
                        file_kp_order.append(base)

            skeleton_kp_order = list(cls.kp_order or [])
            if (
                skeleton_kp_order
                and len(file_kp_order) == len(skeleton_kp_order)
                and set(file_kp_order) == set(skeleton_kp_order)
            ):
                kp_order = skeleton_kp_order
            else:
                kp_order = file_kp_order
            cls.kp_order = kp_order

            base_cols = ["track", "frame_idx", "instance.visibility"]
            if cls.INSTANCE_ID_COL in df.columns:
                base_cols.append(cls.INSTANCE_ID_COL)

            new_order = [col for col in base_cols if col in df.columns]
            for kp in kp_order:
                for suffix in (".x", ".y", ".visibility"):
                    col = f"{kp}{suffix}"
                    if col in df.columns:
                        new_order.append(col)
            new_order += [c for c in df.columns if c not in new_order]

            df = df[new_order]
            df = cls._normalize_loaded_df(df)

            first = cls._first_frame_row(df)
            if first is not None and cls._needs_normalize(first):
                cls.loaded_data = df
                cls._coords_normalized = False
                cls._normalize_df()
            else:
                cls.loaded_data = df
                cls._coords_normalized = True

            cls._bump_label_version()
            return True
        except Exception as err:
            print(f"Failed to load data: {err}")
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
        for track in tracks:
            row = QHBoxLayout()
            row.addWidget(QLabel(track))

            cb = QComboBox()
            cb.addItems(animal_names)
            if track in animal_names:
                cb.setCurrentIndex(animal_names.index(track))
            else:
                cb.setEditable(True)
                cb.setPlaceholderText("select name")
                cb.setEditable(False)
                cb.setCurrentIndex(-1)
            row.addWidget(cb, 1)

            layout.addLayout(row)
            self.comboboxes[track] = cb

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Okay")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self._validate_and_accept)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)

        note = QLabel(
            "The label file and the project config animal name do not match.\n"
            "Match the names using the dialog below.\n"
            "Or, press Cancel to cancel loading."
        )
        layout.addWidget(note)
        layout.addLayout(btn_row)

    def _validate_and_accept(self):
        mapping = {track: cb.currentText() for track, cb in self.comboboxes.items()}
        names = list(mapping.values())
        if len(names) != len(set(names)):
            QMessageBox.warning(self, "Warning", "Please select without duplication")
            return
        self.accept()

    def get_mapping(self) -> dict[str, str]:
        return {
            track: combo.currentText()
            for track, combo in self.comboboxes.items()
        }
