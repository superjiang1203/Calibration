"""Shared per-image chessboard detection and unsmoothed preview rendering."""
from __future__ import annotations
import cv2
import numpy as np
from .core import detect_checkerboard, refine_checkerboard_corners
_REALTIME_DETECTION_MAX_DIMENSION = 1600


def preview_image(gray8: np.ndarray, max_width: int, max_height: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, max_width / gray8.shape[1], max_height / gray8.shape[0])
    if scale == 1.0:
        preview = gray8.copy()
    else:
        preview = cv2.resize(
            gray8,
            (max(1, int(round(gray8.shape[1] * scale))), max(1, int(round(gray8.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return preview, scale


def detect_checkerboard_realtime(
    gray8: np.ndarray,
    pattern_size: tuple[int, int],
    *,
    max_dimension: int = _REALTIME_DETECTION_MAX_DIMENSION,
) -> tuple[bool, np.ndarray | None, str, tuple[int, int], float]:
    """Bound live detection latency while retaining full-resolution corners.

    Searching an entire 20 MP frame is unnecessarily expensive, especially
    when glare temporarily removes the checkerboard and the classic detector
    falls back to an exhaustive search. Locate the board on a bounded image,
    then map and refine the result against the original pixels.
    """

    if max_dimension <= 0:
        raise ValueError("max_dimension must be positive")
    detection_image, scale = preview_image(
        gray8, int(max_dimension), int(max_dimension)
    )
    found, corners, detector = detect_checkerboard(
        detection_image,
        pattern_size,
        detector="auto",
        exhaustive=False,
        fast_check=True,
    )

    # Direct illumination often creates a strong brightness gradient. CLAHE
    # is a bounded second attempt that restores local black/white contrast
    # without allowing a failed full-resolution search to monopolize the
    # detection worker.
    if not found:
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            detection_image
        )
        found, corners, enhanced_detector = detect_checkerboard(
            enhanced,
            pattern_size,
            detector="auto",
            exhaustive=False,
            fast_check=True,
        )
        detector = (
            f"clahe-{enhanced_detector}"
            if found
            else f"{detector}/clahe-{enhanced_detector}"
        )

    detection_size = (detection_image.shape[1], detection_image.shape[0])
    detector = f"{detector}@{detection_size[0]}x{detection_size[1]}"
    if not found or corners is None:
        return False, None, detector, detection_size, scale

    axis_scale = np.array([detection_size[0] / gray8.shape[1], detection_size[1] / gray8.shape[0]], dtype=np.float32)
    full_corners = np.ascontiguousarray(corners / axis_scale, dtype=np.float32)
    if scale < 1.0:
        try:
            full_corners = refine_checkerboard_corners(gray8, full_corners, pattern_size)
            full_corners = np.ascontiguousarray(full_corners, dtype=np.float32)
            detector += "+fullres-subpix"
        except cv2.error:
            # The scaled subpixel result is still valid if an extreme
            # near-border pose leaves no full-resolution refinement window.
            detector += "+scaled-subpix"
    return True, full_corners, detector, detection_size, scale



def draw_subpixel_corners(image, corners):
    """Draw small markers with fractional pixel coordinates, without a grid."""
    shift = 8
    unit = 1 << shift
    for x, y in np.asarray(corners).reshape(-1, 2):
        cx, cy = int(round(float(x) * unit)), int(round(float(y) * unit))
        radius = 3 * unit
        for color, thickness in ((0, 2), (255, 1)):
            cv2.line(image, (cx-radius, cy), (cx+radius, cy), color, thickness, cv2.LINE_AA, shift)
            cv2.line(image, (cx, cy-radius), (cx, cy+radius), color, thickness, cv2.LINE_AA, shift)
