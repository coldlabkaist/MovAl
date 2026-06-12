from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def interpolate_pose_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("DataFrame is required.")
    if df.empty:
        return df.copy()
    if "frame_idx" not in df.columns or "track" not in df.columns:
        raise ValueError("CSV must contain 'frame_idx' and 'track' columns.")

    work_df = df.copy()
    work_df["track"] = work_df["track"].astype(str)
    work_df["frame_idx"] = pd.to_numeric(work_df["frame_idx"], errors="coerce")
    work_df = work_df.dropna(subset=["frame_idx"]).copy()
    if work_df.empty:
        return work_df.reindex(columns=df.columns)
    work_df["frame_idx"] = work_df["frame_idx"].astype(int)

    group_cols = ["track"]
    if "instance.id" in work_df.columns:
        work_df["instance.id"] = pd.to_numeric(work_df["instance.id"], errors="coerce").astype("Int64")
        group_cols.append("instance.id")

    x_cols = [col for col in work_df.columns if col.endswith(".x")]
    y_cols = [col for col in work_df.columns if col.endswith(".y")]
    coord_bases = sorted({col[:-2] for col in x_cols} & {col[:-2] for col in y_cols})
    score_cols = [
        col for col in work_df.columns
        if col == "instance.score" or col.endswith(".score")
    ]

    work_df = work_df.sort_values(group_cols + ["frame_idx"], kind="stable")
    output_rows: list[dict] = []

    for _, group_df in work_df.groupby(group_cols, dropna=False, sort=False):
        group_rows = group_df.to_dict("records")
        if not group_rows:
            continue

        for index, current_row in enumerate(group_rows[:-1]):
            next_row = group_rows[index + 1]
            output_rows.append(dict(current_row))

            current_frame = int(current_row["frame_idx"])
            next_frame = int(next_row["frame_idx"])
            gap = next_frame - current_frame
            if gap <= 1:
                continue

            for frame_idx in range(current_frame + 1, next_frame):
                ratio = (frame_idx - current_frame) / gap
                interpolated_row = dict(current_row)
                interpolated_row["frame_idx"] = frame_idx

                for base in coord_bases:
                    x_col = f"{base}.x"
                    y_col = f"{base}.y"
                    start_x = pd.to_numeric(pd.Series([current_row.get(x_col)]), errors="coerce").iloc[0]
                    end_x = pd.to_numeric(pd.Series([next_row.get(x_col)]), errors="coerce").iloc[0]
                    start_y = pd.to_numeric(pd.Series([current_row.get(y_col)]), errors="coerce").iloc[0]
                    end_y = pd.to_numeric(pd.Series([next_row.get(y_col)]), errors="coerce").iloc[0]

                    interpolated_row[x_col] = (
                        float(start_x + (end_x - start_x) * ratio)
                        if pd.notna(start_x) and pd.notna(end_x)
                        else pd.NA
                    )
                    interpolated_row[y_col] = (
                        float(start_y + (end_y - start_y) * ratio)
                        if pd.notna(start_y) and pd.notna(end_y)
                        else pd.NA
                    )

                for score_col in score_cols:
                    interpolated_row[score_col] = 0.0

                output_rows.append(interpolated_row)

        output_rows.append(dict(group_rows[-1]))

    result_df = pd.DataFrame(output_rows, columns=work_df.columns)
    sort_cols = ["frame_idx", "track"]
    if "instance.id" in result_df.columns:
        result_df["instance.id"] = pd.to_numeric(result_df["instance.id"], errors="coerce").astype("Int64")
        sort_cols.append("instance.id")
    result_df["frame_idx"] = pd.to_numeric(result_df["frame_idx"], errors="coerce").fillna(0).astype(int)
    result_df["track"] = result_df["track"].astype(str)
    result_df = result_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return result_df.reindex(columns=df.columns)


def interpolate_pose_csv(
    csv_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    source_path = Path(csv_path)
    if output_path is None:
        output_file = source_path.with_name(f"{source_path.stem}.interpolated{source_path.suffix}")
    else:
        output_file = Path(output_path)

    df = pd.read_csv(source_path)
    interpolated_df = interpolate_pose_dataframe(df)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    interpolated_df.to_csv(output_file, index=False)
    return output_file
