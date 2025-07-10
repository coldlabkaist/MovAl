import pandas as pd
from pathlib import Path
from typing import Union, Optional, List
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QDialog, QPushButton, QVBoxLayout
from utils.skeleton import SkeletonModel

class DataLoader:

    # --------------------------------------------------------------------- state
    parent: Optional[QDialog] = None

    data: Optional[pd.DataFrame] = None      # 로드된 DataFrame (또는 None)
    csv_path: Optional[str] = None
    skeleton_model: "SkeletonModel" = None
    kp_order: list = None
    _skeleton_loaded: bool = False  # Skeleton loaded flag

    img_width: Optional[int] = None   # 현재 로드된 영상 해상도
    img_height: Optional[int] = None
    _coords_normalized: bool = False

    # --------------------------------------------------- skeleton helpers
    @classmethod
    def load_skeleton_info(cls, skeleton_model: "SkeletonModel") -> None:
        """Register skeleton keypoint order. Call once after loading YAML skeleton."""
        if cls._skeleton_loaded:
            return
        cls.skeleton_model = skeleton_model
        cls.kp_order = list(skeleton_model.nodes)
        cls._skeleton_loaded = True

    @classmethod
    def _ensure_skeleton(cls) -> None:
        if not cls._skeleton_loaded:
            raise RuntimeError("Skeleton information has not been loaded. Call load_skeleton_info() first.")

    @classmethod
    def set_image_dims(cls, w: int, h: int) -> None:
        cls.img_width, cls.img_height = w, h
        if cls.data is None:
            return

        # ─ 픽셀 좌표 → 이번 해상도로 최초 정규화 ─
        if not cls._coords_normalized:
            cls._normalize_df()

    @classmethod
    def load_csv_data(cls, file_path: Union[str, Path]) -> bool:
        cls._ensure_skeleton()
        return cls._load_generic(file_path, read_func=pd.read_csv)

    @classmethod
    def load_txt_data(cls, path: Union[str, Path], sep: str = "\s+") -> bool:
        cls._ensure_skeleton()
        path = Path(path)

        # ───── 1. 폴더인 경우: 내부 *.txt 모두 결합 ─────
        if path.is_dir():
            txt_files = sorted(path.glob("*.txt"))      # test_1.txt, test_2.txt, …
            if not txt_files:
                print("❌ 폴더에 txt 파일이 없습니다.")
                return False

            # (a) 개별 파일 → DataFrame
            frames = []
            for fp in txt_files:
                try:
                    # frame 번호: 파일 이름 끝의 “_N” 숫자
                    f_idx = int(fp.stem.split("_")[-1])
                except ValueError:
                    print(f"⚠️  '{fp.name}' → frame 번호 해석 실패, 건너뜀")
                    continue

                df_part = cls._read_single_txt(fp, sep=sep, frame_idx=f_idx)
                frames.append(df_part)

            if not frames:
                print("❌ 읽어들일 수 있는 txt가 없습니다.")
                return False

            df_total = pd.concat(frames, ignore_index=True)
            return cls._load_generic(df_total, from_dataframe=True)   # 신규 분기

        # ───── 2. 단일 파일: 종전 로직 유지 ─────
        print("잘못된 파일 읽음")
        # TODO

    @classmethod
    def _read_single_txt(cls, fp: Path, sep: str, frame_idx: int) -> pd.DataFrame:
        """
        • TXT 한 장(test_12.txt 등) → DataFrame 1-frame 분량
        • TXT 포맷:  track  min_x min_y w h  kp.x kp.y kp.score …
        """
        raw = pd.read_table(fp, header=None, sep=sep, engine="python")
        if raw.empty:
            return pd.DataFrame()

        # ----- 열 이름 생성 -----
        kp_n   = (raw.shape[1] - 5) // 3           # 5개 이후 세개씩 (x,y,vis)
        if cls.kp_order and kp_n != len(cls.kp_order):
            raise ValueError(
                f"TXT file '{fp.name}' contains {kp_n} key-points, "
                f"but skeleton expects {len(cls.kp_order)}."
            )
        if not cls.kp_order:
            # 최초 호출: kp_order 재구성
            cls.kp_order = [f"kp{i+1}" for i in range(kp_n)]

        cols = (
            ["track.num", "bbox.x", "bbox.y", "bbox.w", "bbox.h"] +
            sum([[f"{kp}.x", f"{kp}.y", f"{kp}.visibility"] for kp in cls.kp_order], [])
        )
        raw.columns = cols[: raw.shape[1]]          # 안전하게 슬라이스

        # track_표기 복원
        raw["track"] = raw["track.num"].apply(lambda n: f"track_{int(n)}")
        raw["frame.idx"] = frame_idx
        raw.drop(columns=["track.num", "bbox.x", "bbox.y", "bbox.w", "bbox.h"], inplace=True)

        for kp in cls.kp_order:
            vis_col = f"{kp}.visibility"
            if vis_col in raw.columns:
                raw[vis_col] = 2

        return raw

    # ------------------------------------------------------------------ 내부 공통 루틴
    @classmethod
    def create_new_data(cls, n_tracks: int = 1) -> bool:
        """Prepare an empty DataFrame template in memory matching the current skeleton."""
        cls._ensure_skeleton()
        base_cols = ["track", "frame.idx"]
        cols = ["track", "frame.idx", "instance.visibility"]
        for kp in cls.kp_order:
            cols += [f"{kp}.x", f"{kp}.y", f"{kp}.visibility"]
        # 빈 DataFrame 생성
        df = pd.DataFrame(columns=cols)
        cls.data = df
        cls.csv_path = None
        cls._coords_normalized = False
        print("🆕 New label template prepared in memory")
        return True

    @classmethod
    def _load_generic(cls, src, read_func=None, *, from_dataframe: bool = False) -> bool:
        """read_func: pd.read_csv 또는 pd.read_table"""
        try:
            # 이 코드 검토할것. 현재 문제상황 : csv가 로드가 안됨!!!!! (TODO)
            # ── A. 데이터 획득 ─────────────────────────────────────────
            if from_dataframe:
                df = src                                # DataFrame 그대로
                origin = "<DataFrame>"
                cls.csv_path = None
            else:
                df = read_func(src)                     # 보통 pd.read_csv or read_table
                origin = str(src)                       # 로깅용
                cls.csv_path = origin

            # 0) skeleton 호환성 검사
            cls._check_skeleton_compat(df)

            # 1) .score → .visibility
            for col in list(df.columns):
                if col.endswith(".score"):
                    vis = col.replace(".score", ".visibility")
                    if vis not in df.columns:
                        df[vis] = 2
            df = df.drop(columns=[c for c in df.columns if c.endswith(".score")])

            # 2) 컬럼 순서 – CSV 입력 순서 그대로
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

            # 3) (frame.idx, track) 2-단계 인덱스 부여  ➜  조회 속도 ↑
            df = df[new_order] 
            df = (
                df.set_index(["frame.idx", "track"], drop=False)
                .sort_index()
            )
            # ---  좌표 정규화 여부 판정  ---
            first = cls._first_frame_row(df)
            if first is not None and cls._needs_normalize(first):
                cls.data = df            # 임시 보관 (아래 함수는 cls.data 사용)
                cls._coords_normalized = False
                cls._normalize_df()      # img_width 가 None 이면 나중에 재-호출됨
            else:
                cls.data = df
                cls._coords_normalized = True

            print(f"✅ Loaded: {origin}")
            return True

        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False
    
    @staticmethod
    def _first_frame_row(df):
        try:
            first_idx = df["frame.idx"].min() 
            return df[df["frame.idx"] == first_idx].iloc[0]
        except Exception:
            return None
            
    @classmethod
    def _needs_normalize(cls, row):
        # “.x” 컬럼 중 값이 1보다 큰 것이 하나라도 있으면 픽셀 좌표로 간주
        for col, val in row.items():
            if col.endswith(".x") and val > 1:
                y = row.get(col.replace(".x", ".y"), 0)
                if y > 1:                    # x·y 둘 다 픽셀 범위
                    return True
        return False

    # ------------------------------------------------------------------ skeleton check
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

    @classmethod
    def reset_data(cls):
        cls.data = None
        cls.csv_path = None
        cls._coords_normalized = False
        print("🗑️ DataLoader 상태 초기화 완료")

    @classmethod
    def get_keypoint_coordinates_by_frame(cls, frame_idx):
        if cls.is_empty(cls.data):
            return {}

        try:
            frame_df = cls.data.xs(frame_idx, level="frame.idx")   # track 이 인덱스
        except KeyError:
            return {t: {} for t in cls.data["track"].unique()}

        coords = {t: {} for t in cls.data["track"].unique()}
        for track, group in frame_df.groupby(level="track"):       # ← 여기 수정
            row = group.iloc[0]                                    # 1행만 사용
            for kp in cls.kp_order:
                xcol, ycol, scol = f"{kp}.x", f"{kp}.y", f"{kp}.visibility"
                if xcol in row and ycol in row and scol in row:
                    coords[track][kp] = (row[xcol], row[ycol], row[scol])

        return coords

    def is_empty(obj) -> bool:
        """DataFrame·시퀀스·None 모두 안전하게 검사."""
        # 1) None 자체
        if obj is None:
            return True
        
        # 2) pandas.DataFrame - 전용 속성 .empty 사용
        if isinstance(obj, pd.DataFrame):
            return obj.empty
        
        # 3) 길이를 가질 수 있는(Sized) 객체
        if isinstance(obj, Sized):
            return len(obj) == 0
        
        # 4) 그 밖의 객체 – 길이나 empty 개념이 없으므로 ‘비어 있지 않다’고 간주
        return False

    @classmethod
    def _normalize_df(cls):
        if cls.img_width is None or cls.img_height is None:
            print("⚠️ 해상도 정보 없음 → 정규화 건너뜀")
            return

        # kp_order 가 비어 있으면 즉석에서 추출
        if not cls.kp_order:
            cls.kp_order = [c[:-2] for c in cls.data.columns if c.endswith(".x")]

        for kp in cls.kp_order:
            cls.data[f"{kp}.x"] /= cls.img_width
            cls.data[f"{kp}.y"] /= cls.img_height
        cls._coords_normalized = True