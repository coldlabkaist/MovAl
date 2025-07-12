from PyQt6.QtCore import QObject, Qt, QEvent

class KeyboardController(QObject):
    def __init__(self, video_loader, parent=None):
        super().__init__(parent)
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
                self.video_loader.toggle_playback()
                return True

            # -------------------------- TODO --------------------------
            elif (key == Qt.Key.Key_S and
                event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                print("save key pressed")
                pass
                return True

        return super().eventFilter(obj, event)
