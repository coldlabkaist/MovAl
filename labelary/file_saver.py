# ───── ① 새 모듈 import ─────
import os                                   # ★ 추가
from pathlib import Path
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QCheckBox, QDialogButtonBox
from .data_loader import DataLoader
import pandas as pd
import re                                   # track 정렬용

class _SaveChoiceDialog(QDialog):
    """CSV / TXT 저장 여부를 고르는 간단한 팝업"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("저장 옵션 선택")
        self.setFixedSize(220, 120)

        self.chk_csv = QCheckBox("CSV 저장")
        self.chk_txt = QCheckBox("TXT 저장")

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(self.chk_csv)
        lay.addWidget(self.chk_txt)
        lay.addWidget(btns)

    # 헬퍼    
    def want_csv(self) -> bool: return self.chk_csv.isChecked()
    def want_txt(self) -> bool: return self.chk_txt.isChecked()

# ───── ② TXT 내보내기 헬퍼 ─────
def _export_txt_files(base_csv_path: str, df: pd.DataFrame) -> None:
    video_name = Path(base_csv_path).stem  
    video_dir  = Path(base_csv_path).parent
    base_dir   = video_dir / video_name  
    txt_dir    = base_dir / "txt"       # ⑤ 새 폴더
    txt_dir.mkdir(parents=True, exist_ok=True)

    max_f = int(df["frame.idx"].max()) - 1  
    pad   = max(2, len(str(max_f)))

    # frame 순서 고정
    for frame_idx in sorted(df["frame.idx"].unique()):
        frame_df = df[df["frame.idx"] == frame_idx]

        # track 정렬: track_0, track_1 … 의 숫자를 기준으로
        def _track_key(t):                   # "track_12" → 12
            m = re.search(r'(\d+)$', t)
            return int(m.group(1)) if m else 0
        frame_df = frame_df.sort_values("track", key=lambda s: s.map(_track_key))

        lines = []
        for _, row in frame_df.iterrows():
            track_num = _track_key(row["track"])

            # bounding-box 계산
            xs = [row[f"{kp}.x"] for kp in DataLoader.kp_order if f"{kp}.x" in row]
            ys = [row[f"{kp}.y"] for kp in DataLoader.kp_order if f"{kp}.y" in row]
            min_x, min_y = min(xs), min(ys)
            w, h         = max(xs) - min_x, max(ys) - min_y

            buf = [track_num, min_x, min_y, w, h]

            # keypoint 값 붙이기 (score가 없으면 visibility)
            for kp in DataLoader.kp_order:
                x, y = row.get(f"{kp}.x"), row.get(f"{kp}.y")
                s    = (row.get(f"{kp}.score") if f"{kp}.score" in row
                        else row.get(f"{kp}.visibility"))
                if x is not None and y is not None and s is not None:
                    buf.extend([x, y, s])

            # 공백-구분 문자열 생성 (float는 6자리까지)
            line = " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in buf)
            lines.append(line)

        txt_path = txt_dir / f"{Path(base_csv_path).stem}_{frame_idx:0{pad}d}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    print(f"✅ TXT files saved in: {txt_dir}")

def _sanitize_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    • 인덱스에 있는 'track' / 'frame.idx' 를
      ─ 이미 컬럼으로 존재하면 drop=True 로 제거
      ─ 없으면 drop=False 로 컬럼화
    • 최종적으로 RangeIndex 로 만들어 중복 문제를 원천 차단
    """
    idx_names = [n for n in df.index.names if n is not None]

    for lvl in idx_names:
        if lvl in df.columns:
            # 같은 이름의 컬럼이 있으면 인덱스 레벨만 제거
            df = df.reset_index(level=lvl, drop=True)
        else:
            # 컬럼이 없으면 컬럼으로 살려낸다
            df = df.reset_index(level=lvl, drop=False)

    # 여기까지 오면 인덱스에 이름 있는 레벨이 없을 수도 있고 있을 수도 있다.
    # RangeIndex 로 깔끔하게 통일해 중복 가능성 제거
    df.reset_index(drop=True, inplace=True)

    # 혹시 남아 있을 중복 컬럼(동일 라벨) 제거 - 첫 번째 것만 유지
    df = df.loc[:, ~df.columns.duplicated()]

    return df


# ───── ③ 기존 함수 수정 ─────
def save_modified_csv(parent):
    # ── (0) 데이터 유무 확인 ───────────────────────────
    if DataLoader.data is None:
        QMessageBox.warning(parent, "Warning", "먼저 CSV/TXT를 불러오세요.")
        return

    # ── (1) 형식 선택 팝업 ────────────────────────────
    dlg = _SaveChoiceDialog(parent)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    want_csv, want_txt = dlg.want_csv(), dlg.want_txt()
    if not (want_csv or want_txt):
        QMessageBox.information(parent, "알림", "저장 형식을 하나 이상 선택하십시오.")
        return

    # ── (2) 공통 전처리: 인덱스·스코어 정리 ───────────
    orig_df = _sanitize_index(DataLoader.data.copy())
    df_for_csv = orig_df.copy()
    for sc in [c for c in df_for_csv.columns if c.endswith(".score")]:
        df_for_csv[sc.replace(".score", ".visibility")] = 2
        df_for_csv.drop(columns=[sc], inplace=True)

    saved_paths = []

    # ── (3-A) CSV 저장 ───────────────────────────────
    if want_csv:
        csv_path, _ = QFileDialog.getSaveFileName(
            parent, "CSV 저장 위치 지정", "", "CSV Files (*.csv)"
        )
        if csv_path:
            try:
                df_for_csv.to_csv(csv_path, index=False)
                saved_paths.append(csv_path)
            except Exception as e:
                QMessageBox.critical(parent, "Error", f"CSV 저장 실패:\n{e}")

    # ── (3-B) TXT 저장 ───────────────────────────────
    if want_txt:
        # ▼ 파일 대신 폴더만 고를 수 있도록 getExistingDirectory 사용
        txt_dir = QFileDialog.getExistingDirectory(
            parent, "TXT 저장 폴더 선택", "", QFileDialog.Option.ShowDirsOnly
        )
        if txt_dir:
            try:
                _export_txt_files(txt_dir, orig_df)   # 폴더 경로 그대로 전달
                saved_paths.append(os.path.abspath(txt_dir))
            except Exception as e:
                QMessageBox.critical(parent, "Error", f"TXT 저장 실패:\n{e}")

    # ── (4) 저장 결과 안내 ────────────────────────────
    if saved_paths:
        msg = "아래 경로로 저장 완료:\n" + "\n".join(saved_paths)
        QMessageBox.information(parent, "Success", msg)
    else:
        QMessageBox.warning(parent, "Warning", "저장되지 않았습니다.")