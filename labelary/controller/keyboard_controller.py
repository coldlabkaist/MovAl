from PyQt6.QtCore import QObject, Qt, QEvent

class KeyboardController(QObject):
    def __init__(self, video_player, parent=None):
        super().__init__(parent)
        self.video_player = video_player

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Right:
                self.video_player.next_frame()
                return True
            elif key == Qt.Key.Key_Left:
                current = self.video_player.current_frame
                if current > 0:
                    self.video_player.move_to_frame(current - 1)
                return True
            elif key == Qt.Key.Key_Space and not event.isAutoRepeat():
                self.video_player.toggle_playback()
                return True

            # -------------------------- TODO --------------------------
            elif (key == Qt.Key.Key_S and
                event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                print("save key pressed")
                pass
                return True

        return super().eventFilter(obj, event)
