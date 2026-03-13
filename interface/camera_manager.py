"""
camera_manager.py
~~~~~~~~~~~~~~~~~
Shared camera utilities used by both the config window and the activity window.

  list_cameras()   – returns [(index, label), ...] for every working camera.
                     Always includes a "No camera" sentinel at index -1.

  CameraThread     – QThread that captures frames from a given camera index
                     and emits each frame as a QImage via the `frame_ready`
                     signal.  Call start() / stop() to control it.
"""

from __future__ import annotations

import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


# ── camera enumeration ────────────────────────────────────────────────────────

def list_cameras() -> list[tuple[int, str]]:
    """
    Probe camera indices 0-9 and return the ones that open successfully.

    Returns a list of (index, label) tuples, always starting with the
    (-1, "No camera") sentinel so combo-box index 0 always means "none".
    """
    results: list[tuple[int, str]] = [(-1, "No camera")]
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if cap is not None and cap.isOpened():
            ret, _ = cap.read()
            if ret:
                results.append((idx, f"Camera {idx}"))
            cap.release()
    return results


# ── camera capture thread ─────────────────────────────────────────────────────

class CameraThread(QThread):
    """
    Captures frames from the chosen camera index in a background thread
    and emits each decoded frame as a QImage.

    Usage
    -----
        thread = CameraThread(camera_index=0)
        thread.frame_ready.connect(my_label.update_frame)
        thread.start()
        ...
        thread.stop()
    """

    frame_ready = pyqtSignal(QImage)

    def __init__(self, camera_index: int, parent=None):
        super().__init__(parent)
        self._camera_index = camera_index
        self._running = False

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_ANY)
        if not cap.isOpened():
            self._running = False
            return

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR → RGB then wrap in QImage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            # Emit a *copy* so the buffer stays valid after the loop continues
            self.frame_ready.emit(img.copy())

            # ~30 fps
            self.msleep(33)

        cap.release()
        self._running = False

    def stop(self):
        """Request the thread to stop and wait for it to finish."""
        self._running = False
        self.wait(2000)
