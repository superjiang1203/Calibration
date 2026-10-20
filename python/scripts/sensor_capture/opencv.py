"""opencv Python SDK frame adapter."""
from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np


from .base import FrameSource

class OpenCVSource(FrameSource):
    backend_name = "opencv-video-capture"

    def __init__(
        self, sensor_index: int, width: int, height: int, acquisition_fps: float
    ) -> None:
        self.sensor_index = int(sensor_index)
        self.width = int(width)
        self.height = int(height)
        self.acquisition_fps = float(acquisition_fps)
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self.sensor_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open OpenCV camera index {self.sensor_index}")
        if self.width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.acquisition_fps > 0:
            capture.set(cv2.CAP_PROP_FPS, self.acquisition_fps)
        self.capture = capture

    def read(self, timeout_ms: int) -> np.ndarray | None:
        del timeout_ms
        if self.capture is None:
            raise RuntimeError("OpenCV camera is not open")
        ok, frame = self.capture.read()
        return frame if ok else None

    def metadata(self) -> dict[str, Any]:
        result = {
            "backend": self.backend_name,
            "sensor_index_0_based": self.sensor_index,
        }
        if self.capture is not None:
            result["features"] = {
                "width": self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                "fps": self.capture.get(cv2.CAP_PROP_FPS),
            }
        return result

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
