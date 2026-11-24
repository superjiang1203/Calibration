"""Common interface for single-stream camera adapters."""

from __future__ import annotations

from typing import Any

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
