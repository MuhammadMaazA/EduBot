import sys
import os
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QWidget

from activity_window import ActivityWindow
from camera_manager import list_cameras

# path to config_window.ui
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(CURRENT_DIR, "config_window.ui")


class ConfigWindow(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_PATH, self)
        self._activity_window = None

        # Populate camera dropdown: [(-1, "No camera"), (0, "Camera 0"), ...]
        self._cameras = list_cameras()
        for _idx, label in self._cameras:
            self.camera_selector.addItem(label)

        self.connect_signals()

    def connect_signals(self):
        self.save_settings_button.clicked.connect(self.save_settings)
        self.start_button.clicked.connect(self.start_activity)

    def save_settings(self):
        name = self.robot_name.text()
        volume = self.robot_volume.value()
        camera_index = self._selected_camera_index()
        print(f"Saved! Name: {name}, Volume: {volume}, Camera index: {camera_index}")

    def _selected_camera_index(self) -> int:
        """Return the actual OpenCV camera index for the current combo selection."""
        combo_pos = self.camera_selector.currentIndex()
        if 0 <= combo_pos < len(self._cameras):
            return self._cameras[combo_pos][0]
        return -1

    def start_activity(self):
        self.hide()
        self._open_activity(1)

    def _open_activity(self, number: int):
        """Open the given activity number, closing any existing activity window."""
        if self._activity_window is not None:
            self._activity_window.close()
            self._activity_window = None

        self._activity_window = ActivityWindow(
            activity_number=number,
            on_navigate=self._open_activity,
            camera_index=self._selected_camera_index(),
        )
        self._activity_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConfigWindow()
    window.show()
    sys.exit(app.exec())
