"""Shared frame source interface and preview support."""
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


class FrameSource:
    backend_name = "unknown"

    def open(self) -> None:
        raise NotImplementedError

    def read(self, timeout_ms: int) -> np.ndarray | None:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {"backend": self.backend_name}

    def close(self) -> None:
        raise NotImplementedError


def require_gui_support():
    gui = next((line.split(':', 1)[1].strip() for line in
                cv2.getBuildInformation().splitlines() if line.strip().startswith('GUI:')), 'UNKNOWN')
    if gui in ('NONE', 'NO', ''):
        raise RuntimeError(f'OpenCV has no GUI support (GUI={gui}). Current Python: {sys.executable}. '
                           'Run conda activate cylmarker, or use '
                           r'C:\Users\lijy8\.conda\envs\cylmarker\python.exe directly.')
