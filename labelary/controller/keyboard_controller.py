from PyQt6.QtCore import QObject, Qt, QEvent
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
)

class KeyboardController(QObject):
    def __init__(self, main_dialog, video_loader, mouse_controller, parent=None):
        super().__init__(parent)
        self.main_dialog = main_dialog
        self.video_loader = video_loader
        self.mouse_controller = mouse_controller

    def detach(self) -> None:
        self.main_dialog = None
        self.video_loader = None
        self.mouse_controller = None

    def eventFilter(self, obj, event):
        if self.main_dialog is None:
            return super().eventFilter(obj, event)
        if not getattr(self.main_dialog, "shortcuts_enabled", True):
            return super().eventFilter(obj, event)
        if event.type() == QEvent.Type.KeyPress:
            if not getattr(self.main_dialog, "shortcuts_enabled", True):
                return super().eventFilter(obj, event)
            if self._is_typing_target_active(obj):
                return super().eventFilter(obj, event)
            key = event.key()
            modifiers = event.modifiers()
            ctrl_pressed = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
            shift_pressed = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            if key == Qt.Key.Key_Z and ctrl_pressed and shift_pressed:
                self.mouse_controller.redo_current_frame_edit()
                return True
            elif key == Qt.Key.Key_Z and ctrl_pressed:
                self.mouse_controller.undo_current_frame_edit()
                return True
            if (
                key in (Qt.Key.Key_Right, Qt.Key.Key_D)
                and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            ):
                current = self.video_loader.current_frame
                if current + 1 < self.video_loader.total_frames:
                    self.video_loader.move_to_frame(current + 1)
                    self.main_dialog.auto_label_current_frame()
                return True
            elif (
                key in (Qt.Key.Key_Left, Qt.Key.Key_A)
                and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            ):
                current = self.video_loader.current_frame
                if current > 0:
                    self.video_loader.move_to_frame(current - 1)
                    self.main_dialog.auto_label_current_frame()
                return True
            elif (key == Qt.Key.Key_Right and
                  event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.video_loader.move_to_labeled_frame(+1)
                self.main_dialog.auto_label_current_frame()
                return True
            elif (key == Qt.Key.Key_Left and
                  event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.video_loader.move_to_labeled_frame(-1)
                self.main_dialog.auto_label_current_frame()
                return True

            elif key == Qt.Key.Key_Space and not event.isAutoRepeat():
                self.main_dialog.play_or_pause()
                return True
            elif key == Qt.Key.Key_Delete and not event.isAutoRepeat():
                self.mouse_controller._delete_selected_instance()
                return True

            elif (key == Qt.Key.Key_A and
                event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.mouse_controller._add_new_skeleton_label()
                return True
            elif (key == Qt.Key.Key_S and
                event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.main_dialog.open_quick_save_dialog()
                return True
            elif (key == Qt.Key.Key_D and
                  event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.mouse_controller._delete_selected_instance()
                return True
            elif (key == Qt.Key.Key_F and
                event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.mouse_controller._replace_selected_instance()
                return True
            elif (key == Qt.Key.Key_X and
                event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.mouse_controller._replace_selected_instance()
                return True
            elif (key == Qt.Key.Key_V and
                  event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.mouse_controller._toggle_selected_node_visibility()
                return True
            elif (key == Qt.Key.Key_T and
                  event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.main_dialog.toggle_automatic_labeling()
                return True
            elif (key == Qt.Key.Key_E and
                  event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.main_dialog.run_automatic_label_addition()
                return True
            elif (key == Qt.Key.Key_R and
                  event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.main_dialog.run_automatic_relabel()
                return True
            for idx, key_num in enumerate([1,2,3,4,5,6,7,8,9,0]):
                if (key == getattr(Qt.Key, f"Key_{key_num}") and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                    self.mouse_controller._change_instance_number_by_idx(idx)
                    return True
                    

        return super().eventFilter(obj, event)

    def _is_typing_target_active(self, obj) -> bool:
        focus_widget = QApplication.focusWidget()
        candidates = [focus_widget, obj]
        for widget in candidates:
            if widget is None:
                continue
            if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
                return True
            if isinstance(widget, QComboBox) and widget.isEditable():
                return True
        return False
