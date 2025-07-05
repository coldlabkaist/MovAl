import pandas as pd
from pathlib import Path
from typing import Union, Optional
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QDialog, QPushButton, QVBoxLayout

class DataLoader:
    """Singleton‑style loader that keeps a global DataFrame (`data`) and the
    current runtime skeleton (`skeleton_data`).  The public method names remain
    identical to the original code so no other module breaks.
    """

    # --------------------------------------------------------------------- state
    data: Optional[pd.DataFrame] = None      # 로드된 DataFrame (또는 None)
    skeleton_data: dict = {}                 # track → [(kp1,kp2), …]
    csv_path: Optional[str] = None
    kp_order: list[str] = [] 

    img_width: Optional[int] = None   # 현재 로드된 영상 해상도
    img_height: Optional[int] = None
    _coords_normalized: bool = False
    norm_src_w: Optional[int] = None     # ← 정규화에 사용한 “이전” 해상도
    norm_src_h: Optional[int] = None
    # --------------------------------------------------------------------- CSV/TXT 로더

    @classmethod
    def set_image_dims(cls, w: int, h: int) -> None:
        cls.img_width, cls.img_height = w, h
        if cls.data is None:
            return

        # ─ 이미 0-1 좌표 → 새 해상도로 재-정규화 ─
        if cls._coords_normalized and cls.norm_src_w:
            if (w, h) != (cls.norm_src_w, cls.norm_src_h):
                fx, fy = cls.norm_src_w / w, cls.norm_src_h / h
                for kp in cls.kp_order:
                    cls.data[f"{kp}.x"] *= fx
                    cls.data[f"{kp}.y"] *= fy
                cls.norm_src_w, cls.norm_src_h = w, h
                print(f"🔄 좌표를 {w}×{h} 기준으로 재-정규화 완료")
            return

        # ─ 픽셀 좌표 → 이번 해상도로 최초 정규화 ─
        if not cls._coords_normalized:
            cls._normalize_df()

    @classmethod
    def load_csv_data(cls, file_path: Union[str, Path]) -> bool:
        return cls._load_generic(file_path, read_func=pd.read_csv)

    @classmethod
    def load_txt_data(cls, path: Union[str, Path], sep: str = "\s+") -> bool:
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
    def _load_generic(cls, src, read_func=None, *, from_dataframe: bool = False) -> bool:
        """read_func: pd.read_csv 또는 pd.read_table"""
        try:
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

    # ───────── 내부 유틸 ─────────
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
        cls.norm_src_w, cls.norm_src_h = cls.img_width, cls.img_height
        print(f"✅ (0-1) 정규화 완료 — 기준 {cls.norm_src_w}×{cls.norm_src_h}")


    # ------------------------------------------------------------------ skeleton check
    @classmethod
    def _check_skeleton_compat(cls, df: pd.DataFrame) -> None:
        """Reset skeleton if the loaded file is incompatible."""
        if not cls.skeleton_data:
            return  # skeleton 없음 → 자유롭게 유지

        file_kps = {c.split(".")[0] for c in df.columns if c.endswith(".x") or c.endswith(".y")}
        skel_kps = {kp for edges in cls.skeleton_data.values() for pair in edges for kp in pair}
        if not skel_kps.issubset(file_kps):
            try:
                from skeleton import SkeletonManager
                SkeletonManager.reset_skeleton_info()
                print("⚠️  Skeleton reset: incompatible with new file")
            except ImportError:
                cls.skeleton_data = {}


    # ------------------------------------------------------------------ keypoint helpers (원본 유지)
    @classmethod
    def get_keypoints(cls):
        if cls.data is None:
            return []
        keypoints = []
        tracks = cls.data["track"].unique()          # 입력 순서 유지
        for idx, track in enumerate(tracks, start=1):
            keypoints.append(f"<b>▶ Animal {idx}</b>")
            for kp in cls.kp_order:                  # ★엑셀 순서
                keypoints.append(f"  {kp}")
        return keypoints

    """@classmethod
    def get_keypoint_coordinates(cls):
        if cls.data is None:
            return {}
        coords = {}
        for track in sorted(cls.data["track"].unique()):
            row = cls.data[cls.data["track"] == track].iloc[0]
            track_coords = {}
            for col in [c for c in cls.data.columns if ".x" in c]:
                base = col.replace(".x", "")
                if f"{base}.y" in cls.data.columns:
                    track_coords[base] = (row[col], row[f"{base}.y"])
            coords[track] = track_coords
        return coords"""

    @classmethod
    def get_keypoint_coordinates_by_frame(cls, frame_idx):
        """{track: {kp:(x,y)…}}  track 이름 그대로, 누락 트랙은 빈 dict"""
        if cls.data is None:
            return {}

        try:
            frame_df = cls.data.xs(frame_idx, level="frame.idx")   # track 이 인덱스
        except KeyError:
            return {t: {} for t in cls.data["track"].unique()}

        coords = {t: {} for t in cls.data["track"].unique()}
        for track, group in frame_df.groupby(level="track"):       # ← 여기 수정
            row = group.iloc[0]                                    # 1행만 사용
            for kp in cls.kp_order:
                xcol, ycol = f"{kp}.x", f"{kp}.y"
                if xcol in row and ycol in row:
                    coords[track][kp] = (row[xcol], row[ycol])

        return coords

    @classmethod
    def update_point(cls, track, frame_idx, keypoint, norm_x, norm_y):
        if cls.data is None:
            print("DataLoader.update_point: No data loaded.")
            return
        mask = (cls.data["track"] == track) & (cls.data["frame.idx"] == frame_idx)
        if mask.sum() == 0:
            print(f"DataLoader.update_point: No row for track={track}, frame={frame_idx}")
            return
        col_x, col_y = f"{keypoint}.x", f"{keypoint}.y"
        if col_x not in cls.data.columns or col_y not in cls.data.columns:
            print(f"DataLoader.update_point: Columns {col_x} or {col_y} not found.")
            return
        cls.data.loc[mask, [col_x, col_y]] = [norm_x, norm_y]
        print(f"✅ Updated {col_x}, {col_y} to ({norm_x:.4f}, {norm_y:.4f})")

    @classmethod
    def update_visibility(cls, track, frame_idx, keypoint, visibility):
        if cls.data is None:
            print("DataLoader.update_visibility: No data loaded.")
            return
        mask = (cls.data["track"] == track) & (cls.data["frame.idx"] == frame_idx)
        if mask.sum() == 0:
            print(f"DataLoader.update_visibility: No row for track={track}, frame={frame_idx}")
            return
        v_col = f"{keypoint}.visibility"
        if v_col not in cls.data.columns:
            print(f"DataLoader.update_visibility: Column {v_col} not found.")
            return
        cls.data.loc[mask, v_col] = visibility
        print(f"✅ Updated {v_col} to {visibility} (track={track}, frame={frame_idx})")

    @classmethod
    def reset_data(cls):
        cls.data = None
        cls.csv_path = None
        cls._coords_normalized = False
        cls.norm_src_w = cls.norm_src_h = None
        print("🗑️ DataLoader 상태 초기화 완료")


# --------------------------------------------------------------------- simple dialog wrappers
class LoadDataDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Data Format")
        self.setFixedSize(300, 150)
        layout = QVBoxLayout(self)
        self.csv_button = QPushButton("Load CSV")
        self.txt_button = QPushButton("Load TXT")
        layout.addWidget(self.csv_button)
        layout.addWidget(self.txt_button)
        self.csv_button.clicked.connect(self.load_csv)
        self.txt_button.clicked.connect(self.load_txt)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV (*.csv)")
        if path and DataLoader.load_csv_data(path):
            self.accept()

    def load_txt(self):
        # 폴더 하나만 선택               ▼ 기존 getOpenFileName → getExistingDirectory
        dir_path = QFileDialog.getExistingDirectory(
            self, "Open TXT Folder", "", QFileDialog.Option.ShowDirsOnly
        )
        if dir_path and DataLoader.load_txt_data(dir_path):   # 폴더 경로 그대로 전달
            self.accept()