from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QMessageBox,
)

from .csv_interpolation import interpolate_pose_csv


class CsvInterpolationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CSV Interpolation")
        self.setFixedSize(560, 420)

        self.csv_paths: list[str] = []

        layout = QVBoxLayout(self)

        instruction = QLabel(
            "Select one or more CSV files. Interpolated files will be saved next to the originals."
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        button_row = QHBoxLayout()
        self.load_btn = QPushButton("Load CSV Files")
        self.load_btn.clicked.connect(self.load_csv_files)
        self.clear_btn = QPushButton("Clear List")
        self.clear_btn.clicked.connect(self.clear_csv_files)
        button_row.addWidget(self.load_btn)
        button_row.addWidget(self.clear_btn)
        layout.addLayout(button_row)

        self.status_label = QLabel("No CSV files loaded.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.run_btn = QPushButton("Create Interpolated CSV")
        self.run_btn.clicked.connect(self.run_interpolation)
        layout.addWidget(self.run_btn)

    def load_csv_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV Files",
            "",
            "CSV Files (*.csv)",
        )
        if not paths:
            return

        existing = {str(Path(path)) for path in self.csv_paths}
        for path in paths:
            normalized = str(Path(path))
            if normalized in existing:
                continue
            existing.add(normalized)
            self.csv_paths.append(normalized)
            self.file_list.addItem(normalized)

        self._update_status()

    def clear_csv_files(self) -> None:
        self.csv_paths = []
        self.file_list.clear()
        self._update_status()

    def run_interpolation(self) -> None:
        if not self.csv_paths:
            QMessageBox.warning(self, "No CSV selected", "Load at least one CSV file first.")
            return

        created_paths: list[str] = []
        failures: list[str] = []

        for csv_path in self.csv_paths:
            try:
                output_path = interpolate_pose_csv(csv_path)
                created_paths.append(str(output_path))
            except Exception as err:
                failures.append(f"{Path(csv_path).name}: {err}")

        self._update_status(created_paths=created_paths, failures=failures)

        if created_paths and not failures:
            QMessageBox.information(
                self,
                "Interpolation complete",
                f"Created {len(created_paths)} interpolated CSV file(s).",
            )
            return

        if created_paths:
            QMessageBox.warning(
                self,
                "Interpolation completed with warnings",
                f"Created {len(created_paths)} interpolated CSV file(s), but some files failed.",
            )
            return

        QMessageBox.critical(
            self,
            "Interpolation failed",
            "No interpolated CSV files were created.",
        )

    def _update_status(
        self,
        *,
        created_paths: list[str] | None = None,
        failures: list[str] | None = None,
    ) -> None:
        if created_paths or failures:
            lines: list[str] = []
            if created_paths:
                lines.append(f"Created {len(created_paths)} interpolated CSV file(s).")
            if failures:
                lines.append(f"Failed: {len(failures)}")
                lines.extend(failures[:3])
            self.status_label.setText("\n".join(lines))
            return

        if not self.csv_paths:
            self.status_label.setText("No CSV files loaded.")
            return

        self.status_label.setText(f"{len(self.csv_paths)} CSV file(s) ready for interpolation.")
