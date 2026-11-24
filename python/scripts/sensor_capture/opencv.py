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


from .frame_source import FrameSource

class OpenCVSource(FrameSource):
    backend_name = "opencv-video-capture"

    def __init__(
        self, sensor_index: int, width: int, height: int, acquisition_fps: float,
        gain_sdk: float | None = None, exposure_sdk: float | None = None,
    ) -> None:
        self.sensor_index = int(sensor_index)
        self.width = int(width)
        self.height = int(height)
        self.acquisition_fps = float(acquisition_fps)
        if gain_sdk is not None and (isinstance(gain_sdk, bool) or not isinstance(gain_sdk, (int, float))
                or not math.isfinite(gain_sdk) or gain_sdk < 0):
            raise ValueError('OpenCV gain_sdk must be a finite nonnegative number or None')
        self.gain_sdk = gain_sdk
        if exposure_sdk is not None and (isinstance(exposure_sdk, bool) or not isinstance(exposure_sdk, (int, float))
                or not math.isfinite(exposure_sdk)):
            raise ValueError('OpenCV exposure must be a finite number or None')
        self.exposure_sdk = exposure_sdk
        self.gain_info = {}
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
        if self.gain_sdk is not None or self.exposure_sdk is not None:
            try:
                for name, prop, requested in (('exposure', cv2.CAP_PROP_EXPOSURE, self.exposure_sdk),
                                               ('gain', cv2.CAP_PROP_GAIN, self.gain_sdk)):
                    if requested is None:
                        continue
                    if not capture.set(prop, float(requested)):
                        raise RuntimeError(f'OpenCV camera/backend does not support setting {name}')
                    actual = capture.get(prop)
                    if not math.isfinite(actual) or not math.isclose(actual, requested, rel_tol=1e-6, abs_tol=1e-6):
                        raise RuntimeError(f'OpenCV {name} readback {actual} differs from requested {requested}')
                    self.gain_info.update({f'{name}_sdk': actual, f'requested_{name}_sdk': requested,
                                           f'{name}_property': 'CAP_PROP_' + name.upper(),
                                           f'{name}_units': 'backend_native'})
                    print(f'OpenCV: {name}={actual:g} (backend units)')
            except Exception:
                capture.release()
                raise
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
                **self.gain_info,
            }
        return result

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
