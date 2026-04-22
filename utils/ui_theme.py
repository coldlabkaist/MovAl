from __future__ import annotations

from pathlib import Path
from typing import Mapping

THEME_COLORS: dict[str, str] = {
    "text_default": "#2f2f2f",
    "text_primary": "#1f1f1f",
    "text_muted": "#9b9b9b",
    "app_bg": "#f2f2f2",
    "surface": "#ffffff",
    "surface_soft": "#f0f0f0",
    "surface_tab": "#ececec",
    "surface_selected": "#CAD1F0",
    "list_item_hover": "#f4f8ff",
    "list_item_selected": "#f4f8ff",
    "video_item_highlight": "#dbe8ff",
    "control_side_bg": "#E0E1EB",
    "control_side_bg_hover": "#AFB3C7",
    "border": "#b8b8b8",
    "border_soft": "#b1b1b1",
    "border_tab": "#a1a1a1",
    "border_accent_soft": "#999999",
    "accent": "#4662DF",
    "accent_hover": "#1E3391",
}

def get_theme_colors(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    colors = dict(THEME_COLORS)
    if overrides:
        colors.update(dict(overrides))
    return colors


def build_app_stylesheet(theme_colors: Mapping[str, str] | None = None) -> str:
    assets_dir = Path(__file__).resolve().parent / "ui_assets"
    combo_down = (assets_dir / "chevron_down.svg").resolve().as_posix()
    spin_up = (assets_dir / "chevron_up.svg").resolve().as_posix()
    spin_down = (assets_dir / "chevron_down.svg").resolve().as_posix()
    check_white = (assets_dir / "check_white.svg").resolve().as_posix()
    c = get_theme_colors(theme_colors)

    base = """
    QWidget {
        color: @TEXT_DEFAULT@;
        font-size: 12px;
    }
    QMainWindow, QDialog, QMessageBox {
        background: @APP_BG@;
    }
    QGroupBox {
        background: @SURFACE@;
        border: 1px solid @BORDER_SOFT@;
        border-radius: 10px;
        margin-top: 8px;
        padding-top: 10px;
        font-weight: 600;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px 0 4px;
    }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTreeWidget, QPlainTextEdit, QTextEdit {
        background: @SURFACE@;
        color: @TEXT_PRIMARY@;
        border: 1px solid @BORDER@;
        border-radius: 8px;
        padding: 0px 6px;
    }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
        min-height: 20px;
        max-height: 20px;
    }
    QComboBox {
        padding-right: 20px;
    }
    QSpinBox, QDoubleSpinBox {
        padding-right: 20px;
    }
    QScrollArea, QAbstractScrollArea {
        background: @SURFACE@;
        border: 1px solid @BORDER@;
        border-radius: 8px;
    }
    QScrollArea > QWidget > QWidget {
        background: @SURFACE@;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 18px;
        border-left: 1px solid @BORDER_ACCENT_SOFT@;
        background: @CONTROL_SIDE_BG@;
        border-top-right-radius: 8px;
        border-bottom-right-radius: 8px;
    }
    QComboBox::drop-down:hover {
        background: @CONTROL_SIDE_BG_HOVER@;
    }
    QAbstractSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 18px;
        border-left: 1px solid @BORDER_ACCENT_SOFT@;
        border-bottom: 1px solid @BORDER_ACCENT_SOFT@;
        background: @CONTROL_SIDE_BG@;
        border-top-right-radius: 8px;
    }
    QAbstractSpinBox::up-button:hover {
        background: @CONTROL_SIDE_BG_HOVER@;
    }
    QAbstractSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 18px;
        border-left: 1px solid @BORDER_ACCENT_SOFT@;
        background: @CONTROL_SIDE_BG@;
        border-bottom-right-radius: 8px;
    }
    QAbstractSpinBox::down-button:hover {
        background: @CONTROL_SIDE_BG_HOVER@;
    }
    QCheckBox {
        spacing: 6px;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border: 1px solid @BORDER_ACCENT_SOFT@;
        border-radius: 4px;
        background: @SURFACE@;
    }
    QCheckBox::indicator:hover {
        border-color: @ACCENT@;
    }
    QCheckBox::indicator:disabled {
        background: @SURFACE_SOFT@;
        border-color: @BORDER@;
    }
    QSlider::groove:horizontal {
        border: 0;
        height: 6px;
        border-radius: 3px;
        background: @BORDER_TAB@;
    }
    QSlider::sub-page:horizontal {
        border: 0;
        background: @ACCENT@;
        border-radius: 3px;
    }
    QSlider::add-page:horizontal {
        border: 0;
        background: @BORDER_TAB@;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        width: 14px;
        margin: -5px 0;
        border: 1px solid @ACCENT@;
        border-radius: 7px;
        background: @SURFACE@;
    }
    QTabWidget::pane {
        border: 1px solid @BORDER_SOFT@;
        border-radius: 10px;
        background: @SURFACE@;
    }
    QTabBar::tab {
        background: @SURFACE_TAB@;
        color: @TEXT_DEFAULT@;
        border: 1px solid @BORDER_TAB@;
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 6px 12px;
        margin-right: 2px;
    }
    QTabBar::tab:selected {
        background: @SURFACE@;
        color: @TEXT_PRIMARY@;
    }
    QPushButton {
        background: @SURFACE@;
        color: @TEXT_PRIMARY@;
        border: 1px solid @BORDER@;
        border-radius: 8px;
        padding: 0px 8px;
        min-height: 20px;
        font-weight: 500;
    }
    QPushButton:hover {
        background: @SURFACE@;
    }
    QPushButton:pressed {
        background: @SURFACE_SOFT@;
    }
    QPushButton:disabled {
        background: @SURFACE_SOFT@;
        color: @TEXT_MUTED@;
        border-color: @BORDER_SOFT@;
    }
    QPushButton[primary="true"] {
        background: @ACCENT@;
        color: @SURFACE@;
        border: 1px solid @ACCENT@;
        font-weight: 600;
    }
    QPushButton[primary="true"]:hover {
        background: @ACCENT_HOVER@;
        border-color: @ACCENT_HOVER@;
    }
    QProgressBar {
        background: @SURFACE_SOFT@;
        color: @TEXT_DEFAULT@;
        border: 1px solid @BORDER_SOFT@;
        border-radius: 8px;
        text-align: center;
    }
    QProgressBar::chunk {
        background: @ACCENT@;
        border-radius: 7px;
    }
    QMessageBox QLabel {
        color: @TEXT_DEFAULT@;
        font-size: 12px;
    }
    QMessageBox QPushButton {
        min-width: 96px;
        min-height: 20px;
    }
    QMenu {
        background: @SURFACE@;
        border: 1px solid @BORDER@;
        padding: 4px;
    }
    QMenu::item {
        color: @TEXT_PRIMARY@;
        padding: 4px 24px 4px 12px;
        border-radius: 6px;
    }
    QMenu::item:selected {
        background: @SURFACE_SELECTED@;
        color: @TEXT_PRIMARY@;
    }
    QMenu::item:disabled {
        color: @TEXT_MUTED@;
        background: transparent;
    }
    QMenu::separator {
        height: 1px;
        background: @BORDER_SOFT@;
        margin: 4px 6px;
    }
    QLabel[stepLabel="true"] {
        font-weight: 700;
        color: @TEXT_PRIMARY@;
    }
    """

    icon_rules = """
    QComboBox::down-arrow {
        image: url("@ICON_COMBO_DOWN@");
        width: 10px;
        height: 10px;
    }
    QAbstractSpinBox::up-arrow {
        image: url("@ICON_SPIN_UP@");
        width: 9px;
        height: 9px;
    }
    QAbstractSpinBox::down-arrow {
        image: url("@ICON_SPIN_DOWN@");
        width: 9px;
        height: 9px;
    }
    QCheckBox::indicator:checked {
        background: @ACCENT@;
        border-color: @ACCENT@;
        image: url("@ICON_CHECK_WHITE@");
    }
    QScrollArea#inferenceTargetScroll {
        border: none;
        background: transparent;
    }
    QScrollArea#inferenceTargetScroll > QWidget > QWidget {
        background: transparent;
    }
    """

    tokens = {
        "TEXT_DEFAULT": c["text_default"],
        "TEXT_PRIMARY": c["text_primary"],
        "TEXT_MUTED": c["text_muted"],
        "APP_BG": c["app_bg"],
        "SURFACE": c["surface"],
        "SURFACE_SOFT": c["surface_soft"],
        "SURFACE_TAB": c["surface_tab"],
        "SURFACE_SELECTED": c["surface_selected"],
        "CONTROL_SIDE_BG": c["control_side_bg"],
        "CONTROL_SIDE_BG_HOVER": c["control_side_bg_hover"],
        "BORDER": c["border"],
        "BORDER_SOFT": c["border_soft"],
        "BORDER_TAB": c["border_tab"],
        "BORDER_ACCENT_SOFT": c["border_accent_soft"],
        "ACCENT": c["accent"],
        "ACCENT_HOVER": c["accent_hover"],
        "ICON_COMBO_DOWN": combo_down,
        "ICON_SPIN_UP": spin_up,
        "ICON_SPIN_DOWN": spin_down,
        "ICON_CHECK_WHITE": check_white,
    }

    rendered = base + icon_rules
    for key, value in tokens.items():
        rendered = rendered.replace(f"@{key}@", value)
    return rendered
