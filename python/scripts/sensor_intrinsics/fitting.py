"""Numerical sensor intrinsic fitting and held-out validation.

No camera SDK dependency; each image is measured independently, and held-out
images validate a fitted model without refitting its intrinsic parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from .detection import create_object_points


INTRINSIC_STD_NAMES = (
    "fx",
    "fy",
    "cx",
    "cy",
    "k1",
    "k2",
    "p1",
    "p2",
    "k3",
    "k4",
    "k5",
    "k6",
    "s1",
    "s2",
    "s3",
    "s4",
    "tau_x",
    "tau_y",
)


@dataclass
class Observation:
    """One successfully detected checkerboard view."""

    path: str
    corners: np.ndarray
    image_size: tuple[int, int]
    sharpness: float = math.nan
    dark_fraction: float = math.nan
    bright_fraction: float = math.nan
    detector: str = "unknown"


@dataclass
class CalibrationFit:
    """Numerical result for one fit over a fixed set of observations."""

    rms: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    std_intrinsics: np.ndarray
    std_extrinsics: np.ndarray
    per_view_errors: np.ndarray
    point_errors: list[np.ndarray]
    residual_vectors: list[np.ndarray]


def _manual_residuals(
    object_points: np.ndarray,
    observations: Sequence[Observation],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    per_view: list[float] = []
    point_errors: list[np.ndarray] = []
    residual_vectors: list[np.ndarray] = []
    for observation, rvec, tvec in zip(observations, rvecs, tvecs):
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        observed_xy = observation.corners.reshape(-1, 2).astype(np.float64)
        projected_xy = projected.reshape(-1, 2).astype(np.float64)
        residual = observed_xy - projected_xy
        errors = np.linalg.norm(residual, axis=1)
        per_view.append(float(np.sqrt(np.mean(np.square(errors)))))
        point_errors.append(errors)
        residual_vectors.append(residual)
    return np.asarray(per_view), point_errors, residual_vectors


def calibrate_once(
    observations: Sequence[Observation],
    pattern_size: tuple[int, int],
    square_size: float,
    *,
    model: str = "standard",
) -> CalibrationFit:
    """Fit standard 5-coefficient or rational Brown-Conrady intrinsics."""

    if not observations:
        raise ValueError("at least one observation is required")
    if model not in {"standard", "rational"}:
        raise ValueError("model must be 'standard' or 'rational'")
    image_size = observations[0].image_size
    if any(observation.image_size != image_size for observation in observations):
        raise ValueError("all observations must have the same resolution")

    object_template = create_object_points(pattern_size, square_size)
    object_points = [object_template.copy() for _ in observations]
    image_points = [observation.corners for observation in observations]
    flags = cv2.CALIB_RATIONAL_MODEL if model == "rational" else 0
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-9,
    )

    if hasattr(cv2, "calibrateCameraExtended"):
        (
            rms,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            std_intrinsics,
            std_extrinsics,
            _opencv_per_view,
        ) = cv2.calibrateCameraExtended(
            object_points,
            image_points,
            image_size,
            None,
            None,
            flags=flags,
            criteria=criteria,
        )
    else:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
            flags=flags,
            criteria=criteria,
        )
        std_intrinsics = np.zeros((len(INTRINSIC_STD_NAMES), 1), dtype=np.float64)
        std_extrinsics = np.zeros((len(observations) * 6, 1), dtype=np.float64)

    arrays = (camera_matrix, dist_coeffs, std_intrinsics, std_extrinsics)
    if not np.isfinite(float(rms)) or any(not np.all(np.isfinite(a)) for a in arrays):
        raise RuntimeError("OpenCV returned a non-finite calibration result")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise RuntimeError("OpenCV returned a non-positive focal length")

    per_view, point_errors, residual_vectors = _manual_residuals(
        object_template,
        observations,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
    )
    return CalibrationFit(
        rms=float(rms),
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        dist_coeffs=np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
        rvecs=[np.asarray(value, dtype=np.float64).reshape(3, 1) for value in rvecs],
        tvecs=[np.asarray(value, dtype=np.float64).reshape(3, 1) for value in tvecs],
        std_intrinsics=np.asarray(std_intrinsics, dtype=np.float64).reshape(-1),
        std_extrinsics=np.asarray(std_extrinsics, dtype=np.float64).reshape(-1),
        per_view_errors=per_view,
        point_errors=point_errors,
        residual_vectors=residual_vectors,
    )


def coverage_statistics(
    observations: Sequence[Observation], image_size: tuple[int, int]
) -> dict[str, Any]:
    width, height = image_size
    points = np.concatenate(
        [observation.corners.reshape(-1, 2) for observation in observations], axis=0
    ).astype(np.float64)
    normalized = points / np.asarray([width, height], dtype=np.float64)
    centroids = np.asarray(
        [observation.corners.reshape(-1, 2).mean(axis=0) for observation in observations],
        dtype=np.float64,
    ) / np.asarray([width, height], dtype=np.float64)
    hull = cv2.convexHull(points.astype(np.float32))
    hull_fraction = float(cv2.contourArea(hull) / float(width * height))

    board_area_fractions: list[float] = []
    for observation in observations:
        board_hull = cv2.convexHull(observation.corners.astype(np.float32))
        board_area_fractions.append(
            float(cv2.contourArea(board_hull) / float(width * height))
        )
    return {
        "all_corner_bounds_normalized": {
            "x_min": float(normalized[:, 0].min()),
            "x_max": float(normalized[:, 0].max()),
            "y_min": float(normalized[:, 1].min()),
            "y_max": float(normalized[:, 1].max()),
        },
        "corner_convex_hull_image_fraction": hull_fraction,
        "board_centroid_bounds_normalized": {
            "x_min": float(centroids[:, 0].min()),
            "x_max": float(centroids[:, 0].max()),
            "y_min": float(centroids[:, 1].min()),
            "y_max": float(centroids[:, 1].max()),
        },
        "board_area_fraction": {
            "min": float(np.min(board_area_fractions)),
            "median": float(np.median(board_area_fractions)),
            "max": float(np.max(board_area_fractions)),
        },
    }


def pose_statistics(fit: CalibrationFit) -> tuple[dict[str, Any], list[float]]:
    tilts: list[float] = []
    depths: list[float] = []
    for rvec, tvec in zip(fit.rvecs, fit.tvecs):
        rotation, _ = cv2.Rodrigues(rvec)
        cosine = float(np.clip(abs(rotation[2, 2]), 0.0, 1.0))
        tilts.append(float(np.degrees(np.arccos(cosine))))
        depths.append(float(tvec.reshape(-1)[2]))
    depth_array = np.asarray(depths)
    positive_depths = depth_array[depth_array > 0]
    depth_ratio = (
        float(positive_depths.max() / positive_depths.min())
        if positive_depths.size and positive_depths.min() > 0
        else math.nan
    )
    stats = {
        "tilt_deg": {
            "min": float(np.min(tilts)),
            "median": float(np.median(tilts)),
            "max": float(np.max(tilts)),
        },
        "board_depth": {
            "min": float(np.min(depths)),
            "median": float(np.median(depths)),
            "max": float(np.max(depths)),
            "max_to_min_ratio": depth_ratio,
        },
    }
    return stats, tilts


def quality_warnings(
    fit: CalibrationFit,
    coverage: dict[str, Any],
    poses: dict[str, Any],
    image_size: tuple[int, int],
) -> list[str]:
    """Return engineering review prompts, not universal acceptance criteria."""

    warnings: list[str] = []
    width, height = image_size
    bounds = coverage["all_corner_bounds_normalized"]
    missing_edges = []
    if bounds["x_min"] > 0.12:
        missing_edges.append("left")
    if bounds["x_max"] < 0.88:
        missing_edges.append("right")
    if bounds["y_min"] > 0.12:
        missing_edges.append("top")
    if bounds["y_max"] < 0.88:
        missing_edges.append("bottom")
    if missing_edges:
        warnings.append(
            "角点没有充分覆盖图像边缘：" + ", ".join(missing_edges)
        )
    if coverage["corner_convex_hull_image_fraction"] < 0.55:
        warnings.append("角点凸包覆盖不足 55%，边缘畸变约束可能偏弱")
    centroid_bounds = coverage["board_centroid_bounds_normalized"]
    if centroid_bounds["x_max"] - centroid_bounds["x_min"] < 0.35:
        warnings.append("棋盘中心的水平位置变化不足")
    if centroid_bounds["y_max"] - centroid_bounds["y_min"] < 0.35:
        warnings.append("棋盘中心的垂直位置变化不足")
    if poses["tilt_deg"]["max"] < 20.0:
        warnings.append("最大棋盘倾角不足 20°，焦距与平面姿态可能耦合")
    if poses["board_depth"]["max_to_min_ratio"] < 1.35:
        warnings.append("棋盘远近尺度变化不足")
    if fit.rms > 1.0:
        warnings.append("全局重投影 RMS 超过 1 px，应检查模糊、板平面度和模型")
    elif fit.rms > 0.5:
        warnings.append("全局重投影 RMS 超过 0.5 px，建议检查逐图残差")

    fx, fy = fit.camera_matrix[0, 0], fit.camera_matrix[1, 1]
    cx, cy = fit.camera_matrix[0, 2], fit.camera_matrix[1, 2]
    if not 0.8 <= fx / fy <= 1.2:
        warnings.append("fx/fy 偏离 1 较多，请核对像素比例和棋盘列/行顺序")
    if not (-0.1 * width <= cx <= 1.1 * width) or not (
        -0.1 * height <= cy <= 1.1 * height
    ):
        warnings.append("主点明显位于图像范围之外，结果很可能不稳定")
    if fit.std_intrinsics.size >= 2:
        if fit.std_intrinsics[0] / fx > 0.02 or fit.std_intrinsics[1] / fy > 0.02:
            warnings.append("焦距标准差超过估计值的 2%，建议增加姿态覆盖")
    return warnings


def verify_observations(
    observations: Sequence[Observation],
    pattern_size: tuple[int, int],
    square_size_mm: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    """Estimate only each board pose and measure fixed-intrinsic holdout error."""

    object_points = create_object_points(pattern_size, square_size_mm)
    entries: list[dict[str, Any]] = []
    all_errors: list[np.ndarray] = []
    all_residuals: list[np.ndarray] = []
    for observation in observations:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            observation.corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            entries.append(
                {"path": observation.path, "status": "pose_failed"}
            )
            continue
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        residual = observation.corners.reshape(-1, 2).astype(np.float64) - projected.reshape(
            -1, 2
        )
        errors = np.linalg.norm(residual, axis=1)
        all_errors.append(errors)
        all_residuals.append(residual)
        entries.append(
            {
                "path": observation.path,
                "status": "ok",
                "view_rms_px": float(np.sqrt(np.mean(np.square(errors)))),
                "point_error_mean_px": float(np.mean(errors)),
                "point_error_p95_px": float(np.percentile(errors, 95)),
                "rvec": rvec.reshape(-1).tolist(),
                "tvec_mm": tvec.reshape(-1).tolist(),
            }
        )
    return entries, all_errors, all_residuals
