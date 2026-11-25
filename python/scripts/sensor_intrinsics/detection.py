"""Shared per-image chessboard detection and unsmoothed preview rendering."""
from __future__ import annotations
import cv2
import numpy as np
_REALTIME_DETECTION_MAX_DIMENSION = 1600


def prepare_gray8(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit grayscale image without changing its geometry.

    OpenCV's checkerboard detectors require 8-bit input.  Mono10/12/16 input
    is linearly mapped using robust endpoints; clipping only the outer 0.1%
    avoids a handful of hot pixels destroying contrast.  The original image is
    never modified and should still be retained for production use.
    """

    if image is None or image.size == 0:
        raise ValueError("image is empty")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"unsupported image shape: {image.shape}")

    if gray.dtype == np.uint8:
        return np.ascontiguousarray(gray)

    finite = np.asarray(gray, dtype=np.float64)
    valid = finite[np.isfinite(finite)]
    if valid.size == 0:
        raise ValueError("image has no finite pixels")
    low, high = np.percentile(valid, (0.1, 99.9))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(gray.shape, dtype=np.uint8)
    scaled = (finite - low) * (255.0 / (high - low))
    scaled[~np.isfinite(scaled)] = 0.0
    return np.ascontiguousarray(np.clip(scaled, 0, 255).astype(np.uint8))


def image_quality(gray8: np.ndarray) -> dict[str, float]:
    """Compute diagnostic (not universal pass/fail) image-quality metrics."""

    if gray8.dtype != np.uint8 or gray8.ndim != 2:
        raise ValueError("image_quality expects a uint8 grayscale image")
    return {
        "sharpness": float(cv2.Laplacian(gray8, cv2.CV_64F).var()),
        "dark_fraction": float(np.mean(gray8 <= 2)),
        "bright_fraction": float(np.mean(gray8 >= 253)),
        "mean_gray": float(np.mean(gray8)),
    }


def create_object_points(
    pattern_size: tuple[int, int], square_size: float
) -> np.ndarray:
    """Create planar points in row-major detector order.

    ``pattern_size`` is ``(inner_columns, inner_rows)``.  ``square_size`` is
    normally expressed in millimetres, which makes the resulting translations
    use millimetres too.
    """

    columns, rows = (int(pattern_size[0]), int(pattern_size[1]))
    if columns < 2 or rows < 2:
        raise ValueError("checkerboard needs at least 2x2 inner corners")
    if not np.isfinite(square_size) or square_size <= 0:
        raise ValueError("square_size must be a positive finite number")
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= float(square_size)
    return points


def refine_checkerboard_corners(gray8, corners, pattern_size):
    """Refine within a local window sized for the actual square pitch."""
    grid = np.asarray(corners, np.float32).reshape(pattern_size[1], pattern_size[0], 2)
    horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    vertical = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    pitch = min(float(np.median(horizontal)), float(np.median(vertical)))
    if not np.isfinite(pitch) or pitch < 4:
        raise ValueError('Checkerboard square pitch is too small for subpixel refinement')
    radius = max(2, min(11, int(np.floor(pitch * 0.5))))
    return cv2.cornerSubPix(gray8, np.ascontiguousarray(corners, dtype=np.float32).copy(),
        (radius, radius), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))


def detect_checkerboard(
    gray8: np.ndarray,
    pattern_size: tuple[int, int],
    *,
    detector: str = "auto",
    exhaustive: bool = True,
    fast_check: bool = False,
) -> tuple[bool, np.ndarray | None, str]:
    """Detect a complete checkerboard and return subpixel corner locations.

    Auto uses classic detection plus subpixel refinement first, with SB as
    fallback. On the tested Orbbec images this avoids SB's repeat variation
    and reduces sensitivity to saturated white squares.
    """

    if gray8.dtype != np.uint8 or gray8.ndim != 2:
        raise ValueError("detect_checkerboard expects a uint8 grayscale image")
    if detector not in {"auto", "sb", "classic"}:
        raise ValueError(f"unsupported detector: {detector}")

    size = (int(pattern_size[0]), int(pattern_size[1]))
    can_use_sb = hasattr(cv2, "findChessboardCornersSB")
    if detector == "auto":
        found, corners, used = detect_checkerboard(gray8, size, detector="classic",
                                                  exhaustive=exhaustive, fast_check=True)
        if found or not can_use_sb:
            return found, corners, used
        found, corners, used = detect_checkerboard(gray8, size, detector="sb",
                                                  exhaustive=exhaustive, fast_check=fast_check)
        return found, corners, used if found else "classic+sb"
    if detector == "sb" and can_use_sb:
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
        if exhaustive:
            flags |= cv2.CALIB_CB_EXHAUSTIVE
        found, corners = cv2.findChessboardCornersSB(gray8, size, flags=flags)
        if found:
            return True, np.ascontiguousarray(corners, dtype=np.float32), "sb"
        if detector == "sb":
            return False, None, "sb"

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast_check:
        # Live capture must reject a temporarily obscured or saturated board
        # promptly. Without FAST_CHECK, the classic fallback can spend tens
        # of seconds searching a high-resolution cluttered frame.
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray8, size, flags=flags)
    if not found:
        used = "classic" if detector == "classic" or not can_use_sb else "sb+classic"
        return False, None, used
    try:
        refined = refine_checkerboard_corners(gray8, corners, size)
    except ValueError:
        return False, None, "classic-small-pitch"
    return True, np.ascontiguousarray(refined, dtype=np.float32), "classic"


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
        for color, thickness in (((0, 0, 0), 2), ((255, 255, 255), 1)):
            cv2.line(image, (cx-radius, cy), (cx+radius, cy), color, thickness, cv2.LINE_AA, shift)
            cv2.line(image, (cx, cy-radius), (cx, cy+radius), color, thickness, cv2.LINE_AA, shift)
