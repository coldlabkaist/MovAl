from PyQt6.QtCore import QObject, Qt, QEvent

class KeyboardController(QObject):
    def __init__(self, main_dialog, video_loader, parent=None):
        super().__init__(parent)
        self.main_dialog = main_dialog
        self.video_loader = video_loader

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Right:
                current = self.video_loader.current_frame
                self.video_loader.move_to_frame(current + 1)
                return True
            elif key == Qt.Key.Key_Left:
                current = self.video_loader.current_frame
                if current > 0:
                    self.video_loader.move_to_frame(current - 1)
                return True
            elif key == Qt.Key.Key_Space and not event.isAutoRepeat():
                self.main_dialog.play_or_pause()
                return True

            # -------------------------- TODO --------------------------
            elif (key == Qt.Key.Key_S and
                event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                print("save key pressed")
                pass
                return True

        return super().eventFilter(obj, event)
