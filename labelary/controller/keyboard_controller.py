from PyQt6.QtCore import QObject, Qt, QEvent

class KeyboardController(QObject):
    def __init__(self, main_dialog, video_loader, mouse_controller, parent=None):
        super().__init__(parent)
        self.main_dialog = main_dialog
        self.video_loader = video_loader
        self.mouse_controller = mouse_controller

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Right and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                current = self.video_loader.current_frame
                self.video_loader.move_to_frame(current + 1)
                return True
            elif key == Qt.Key.Key_Left and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                current = self.video_loader.current_frame
                if current > 0:
                    self.video_loader.move_to_frame(current - 1)
                return True
            elif (key == Qt.Key.Key_Right and
                  event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.video_loader.move_to_labeled_frame(+1)
                return True
            elif (key == Qt.Key.Key_Left and
                  event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.video_loader.move_to_labeled_frame(-1)
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
                self.main_dialog.open_save_dialog()
                return True
            elif (key == Qt.Key.Key_D and
                event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.mouse_controller._delete_selected_instance()
                return True
            elif (key == Qt.Key.Key_V and
                event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.mouse_controller._toggle_selected_node_visibility()
                return True
            for idx, key_num in enumerate([1,2,3,4,5,6,7,8,9,0]):
                if (key == getattr(Qt.Key, f"Key_{key_num}") and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                    self.mouse_controller._change_instance_number_by_idx(idx)
                    return True
                    

        return super().eventFilter(obj, event)
