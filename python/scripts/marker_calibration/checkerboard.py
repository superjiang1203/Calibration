#!/usr/bin/env python3
"""Metric calibration of rigid marker objects photographed with a known checkerboard.

Never changes source images/metadata. The board fixes scale, not a marker edge.
Each capture group may have its own object-to-board rigid transformations.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def gray8(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = np.clip(
            image.astype(np.float64) * 255 / max(float(image.max()), 1), 0, 255
        ).astype(np.uint8)
    return image


def refine(gray, points, window=5):
    points = np.asarray(points, np.float32).reshape(-1, 1, 2)
    return cv2.cornerSubPix(
        gray,
        points.copy(),
        (window, window),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
    ).reshape(-1, 2)


def black_border_corners(smooth, corners, *, return_quality=False):
    """Fit the four black-to-white edges; do not snap to the white label rim.

    Edge profiles have an outward signed gradient. An unsigned corner detector
    can instead lock on the opposite-polarity white-label-to-background edge.
    """
    p = np.asarray(corners, np.float64).reshape(4, 2)
    center = p.mean(axis=0)
    lines = []
    support = []

    def finish(points, valid, reason):
        quality = {
            "valid": bool(valid),
            "reason": reason,
            "edge_support": support,
            "maximum_corner_shift_px": float(
                np.linalg.norm(np.asarray(points) - p, axis=1).max()
            ),
        }
        return (np.asarray(points), quality) if return_quality else np.asarray(points)

    offsets = np.arange(-6.0, 6.01, 0.5)
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        tangent = b - a
        length = np.linalg.norm(tangent)
        tangent /= length
        outward = np.array([tangent[1], -tangent[0]])
        if outward @ ((a + b) / 2 - center) < 0:
            outward = -outward
        n = max(16, min(70, int(length * 0.65)))
        samples = a[None] + np.linspace(0.18, 0.82, n)[:, None] * (b - a)[None]
        positions = samples[:, None, :] + offsets[None, :, None] * outward
        profile = cv2.remap(
            smooth,
            positions[:, :, 0].astype(np.float32),
            positions[:, :, 1].astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float64)
        gradient = np.gradient(profile, 0.5, axis=1)
        # A weak localization preference resolves multiple positive edges.
        scored = gradient * np.exp(-0.5 * (offsets[None] / 4.0) ** 2)
        best = np.argmax(scored[:, 1:-1], axis=1) + 1
        peak = gradient[np.arange(n), best]
        rows = np.arange(n)
        left = gradient[rows, best - 1]
        right = gradient[rows, best + 1]
        denominator = left - 2 * peak + right
        shift = np.zeros(n)
        good = np.abs(denominator) > 1e-6
        shift[good] = np.clip(
            0.5 * (left[good] - right[good]) / denominator[good], -1, 1
        )
        edge = samples + (offsets[best] + shift * 0.5)[:, None] * outward
        valid = (peak > 5) & (best > 1) & (best < len(offsets) - 2)
        support.append(int(valid.sum()))
        if valid.sum() < 8:
            return finish(p, False, "insufficient_signed_edge_support")
        points = edge[valid]
        weights = peak[valid]
        for iteration in range(3):
            origin = np.average(points, axis=0, weights=weights)
            _, _, vt = np.linalg.svd(
                (points - origin) * np.sqrt(weights[:, None]), full_matrices=False
            )
            normal = vt[-1]
            deviation = np.abs((points - origin) @ normal)
            keep = deviation < max(
                0.3,
                float(
                    np.median(deviation)
                    + 3 * 1.4826 * np.median(abs(deviation - np.median(deviation)))
                ),
            )
            if keep.all() or keep.sum() < 8:
                break
            points = points[keep]
            weights = weights[keep]
        if abs(normal @ outward) < 0.9:
            return finish(p, False, "edge_direction_inconsistent")
        lines.append((normal, float(normal @ origin)))
    result = []
    for i in range(4):
        first, second = lines[i - 1], lines[i]
        a = np.stack([first[0], second[0]])
        if abs(np.linalg.det(a)) < 0.05:
            return finish(p, False, "near_parallel_edge_intersection")
        result.append(np.linalg.solve(a, np.array([first[1], second[1]])))
    result = np.asarray(result)
    # One badly thresholded corner can be far from the intersection of two
    # otherwise well supported black/white edges. Scale this safety bound with
    # marker size rather than silently reverting to a known bad corner.
    limit = max(
        12.0, 0.25 * float(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1).max())
    )
    if np.max(np.linalg.norm(result - p, axis=1)) > limit or not cv2.isContourConvex(
        result.astype(np.float32)
    ):
        return finish(p, False, "invalid_or_excessive_edge_intersection")
    return finish(result, True, "signed_edge_intersections")


def redetect_markers(record, root):
    gray = gray8(cv2.imread(str(root / record["image"]), cv2.IMREAD_UNCHANGED))
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.65)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    params.minMarkerPerimeterRate = 0.008
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 43
    params.adaptiveThreshWinSizeStep = 10
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100), params
    )
    corners, ids, _ = detector.detectMarkers(gray)
    counts = Counter([] if ids is None else ids.ravel().tolist())
    markers = {}
    for mid, p in zip([] if ids is None else ids.ravel(), corners):
        if counts[int(mid)] == 1:
            markers[int(mid)] = black_border_corners(smooth, p).tolist()
    return dict(
        record,
        markers=markers,
        marker_corner_method="signed_black_to_white_border_line_fit",
    )


def refine_all_markers(args):
    cv2.setNumThreads(1)
    output = Path(args.output)
    root = Path(args.images)
    data = json.loads((output / "detections.json").read_text())
    if not (output / "detections.first_pass.json").exists():
        write_json(output / "detections.first_pass.json", data)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(redetect_markers, r, root) for r in data["frames"]]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 20 == 0:
                print("REFINED", len(records), flush=True)
    data["frames"] = sorted(records, key=lambda r: r["index"])
    write_json(output / "detections.json", data)
    print(
        "REFINED_COUNTS",
        dict(
            sorted(Counter(int(mid) for r in records for mid in r["markers"]).items())
        ),
        flush=True,
    )


def detect_one(path, pattern):
    metadata = json.loads(path.with_suffix(".json").read_text())
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Unreadable image: {path}")
    gray = gray8(raw)
    scale = min(1.0, 1800 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    found, board = cv2.findChessboardCornersSB(
        small, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    stored = np.asarray(metadata.get("board_corners", []), np.float64).reshape(-1, 2)
    source = "independent_sb_then_original_resolution_subpix"
    if found:
        axes = np.array(
            [small.shape[1] / gray.shape[1], small.shape[0] / gray.shape[0]]
        )
        board = (board.reshape(-1, 2) + 0.5) / axes - 0.5
        if stored.shape == board.shape and np.mean(
            (board[::-1] - stored) ** 2
        ) < np.mean((board - stored) ** 2):
            board = board[::-1].copy()
        board = refine(gray, board, 7)
    elif len(stored) == pattern[0] * pattern[1]:
        source = "capture_metadata_then_original_resolution_subpix"
        board = refine(gray, stored, 7)
    else:
        source, board = "not_found", np.empty((0, 2))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 4
    params.cornerRefinementMaxIterations = 40
    params.cornerRefinementMinAccuracy = 0.001
    params.minMarkerPerimeterRate = 0.008
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 43
    params.adaptiveThreshWinSizeStep = 10
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100), params
    )
    points, ids, rejected = detector.detectMarkers(gray)
    counts = Counter([] if ids is None else ids.flatten().tolist())
    markers = {
        int(mid): corners.reshape(4, 2).tolist()
        for mid, corners in zip([] if ids is None else ids.flatten(), points)
        if counts[int(mid)] == 1
    }
    return {
        "image": path.name,
        "index": int(path.stem.split("_")[-1]),
        "group": int(metadata.get("group", 0)),
        "image_size": [gray.shape[1], gray.shape[0]],
        "board": board.tolist(),
        "board_source": source,
        "board_capture_delta_rms_px": None
        if stored.shape != board.shape
        else float(np.sqrt(np.mean(np.sum((stored - board) ** 2, axis=1)))),
        "markers": markers,
        "duplicate_ids": [mid for mid, count in counts.items() if count > 1],
        "capture_marker_ids": metadata.get("marker_ids", {}),
        "focus": metadata.get("focus", {}),
    }


def detect_all(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.images)
    session = json.loads((root / "session.json").read_text())
    pattern = tuple(session["board"]["pattern_size"])
    if float(session["board"]["square_size_mm"]) != 35.0:
        raise ValueError("Expected the supplied 35 mm checkerboard")
    cv2.setNumThreads(1)
    paths = sorted(root.glob("frame_*.png"))
    records, started = [], time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(detect_one, p, pattern): p for p in paths}
        for future in as_completed(futures):
            result = future.result()
            records.append(result)
            if len(records) % 10 == 0 or len(records) == len(paths):
                print(
                    f"Detected {len(records)}/{len(paths)}; {time.monotonic() - started:.1f}s",
                    flush=True,
                )
                write_json(
                    out / "detections.partial.json",
                    {"frames": sorted(records, key=lambda r: r["index"])},
                )
    records.sort(key=lambda r: r["index"])
    write_json(
        out / "detections.json",
        {"source": str(root), "board": session["board"], "frames": records},
    )
    print(
        "ID_COUNTS",
        dict(
            sorted(Counter(int(mid) for r in records for mid in r["markers"]).items())
        ),
        flush=True,
    )
    cells = []
    for group in sorted(set(r["group"] for r in records)):
        candidates = [r for r in records if r["group"] == group]
        for r in (candidates[0], candidates[len(candidates) // 2]):
            image = gray8(cv2.imread(str(root / r["image"]), cv2.IMREAD_UNCHANGED))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for mid, p in r["markers"].items():
                p = np.asarray(p)
                cv2.polylines(
                    image, [np.rint(p).astype(np.int32)], True, (0, 255, 0), 3
                )
                cv2.putText(
                    image,
                    str(mid),
                    tuple(np.rint(p.mean(axis=0)).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )
            factor = 700 / image.shape[1]
            image = cv2.resize(image, None, fx=factor, fy=factor)
            cv2.putText(
                image,
                f"group {group} / {r['image']}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cells.append(image)
    montage = np.vstack([np.hstack(cells[i : i + 2]) for i in range(0, len(cells), 2)])
    cv2.imwrite(str(out / "capture_montage.jpg"), montage)


def pose(vector):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.asarray(vector[:3]))[0]
    result[:3, 3] = vector[3:6]
    return result


def vector(matrix):
    return np.r_[cv2.Rodrigues(matrix[:3, :3])[0].ravel(), matrix[:3, 3]]


def transform(matrix, points):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def square(length):
    return (
        np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float64)
        * length
        / 2
    )


def project(points, matrix, k, d):
    return cv2.projectPoints(
        np.asarray(points, np.float64), vector(matrix)[:3], matrix[:3, 3], k, d
    )[0].reshape(-1, 2)


def stats(values):
    values = np.asarray(values, np.float64).ravel()
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def object_board(pattern, spacing):
    result = np.zeros((pattern[0] * pattern[1], 3), np.float64)
    result[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * spacing
    return result


def detect_full_board(image, pattern):
    """Return every configured inner corner in original image coordinates."""
    from ..sensor_intrinsics.detection import detect_checkerboard_realtime, prepare_gray8

    found, corners, *_ = detect_checkerboard_realtime(prepare_gray8(image), pattern)
    if not found or corners is None:
        return None
    pixels = corners.reshape(-1, 2)
    if (len(pixels) != pattern[0] * pattern[1] or not np.isfinite(pixels).all()
            or np.any(pixels < 0) or np.any(pixels >= np.asarray(image.shape[1::-1]))):
        return None
    return pixels


def pnp(points, measured, k, d):
    ok, r, t = cv2.solvePnP(
        np.asarray(points, np.float64),
        np.asarray(measured, np.float64),
        k,
        d,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("PnP failed")
    return pose(np.r_[r.ravel(), t.ravel()])


def checked_board_pnp(points, measured, k, d, options):
    """Estimate from a RANSAC consensus, then gate the entire frame on RMS."""
    limit = options["max_board_pnp_rmse_px"]
    min_inliers = int(np.ceil(len(points) * (1 - options["max_outlier_fraction"])))
    quality = {"corner_count": len(points), "min_inliers": min_inliers,
               "inliers": 0, "rmse_px": None, "limit_px": limit}
    points = np.ascontiguousarray(points, dtype=np.float64)
    measured = np.ascontiguousarray(measured, dtype=np.float64)
    try:
        ok, _, _, indices = cv2.solvePnPRansac(
            points, measured, k, d, iterationsCount=200, reprojectionError=limit,
            confidence=0.999, flags=cv2.SOLVEPNP_AP3P,
        )
        if not ok or indices is None:
            quality["reason"] = "棋盘 PnP RANSAC 未找到有效位姿。"
            return None, None, quality
        indices = indices.ravel()
        quality["inliers"] = len(indices)
        if len(indices) < min_inliers:
            quality["reason"] = f"棋盘 PnP RANSAC 内点不足（{len(indices)}/{len(points)}，至少需要 {min_inliers}）。"
            return None, None, quality
        # The pose returned by RANSAC can be unstable for a planar board.
        matrix = pnp(points[indices], measured[indices], k, d)
        squared = np.sum((project(points, matrix, k, d) - measured)**2, axis=1)
    except (ValueError, cv2.error):
        quality["reason"] = "棋盘 PnP 无法求解有效位姿。"
        return None, None, quality
    valid = squared <= limit**2
    rmse = float(np.sqrt(np.mean(squared)))
    quality["inliers"] = int(valid.sum())
    if not np.isfinite(rmse):
        quality["reason"] = "棋盘 PnP 重投影误差无效。"
        return None, None, quality
    quality["rmse_px"] = rmse
    if rmse > limit:
        quality["reason"] = f"棋盘 PnP 全角点 RMS {rmse:.3f} px 超过上限 {limit:g} px。"
    elif np.any(transform(matrix, points)[:, 2] <= 0):
        quality["reason"] = "棋盘 PnP 存在位于相机后方的角点。"
    elif quality["inliers"] < min_inliers:
        quality["reason"] = f'棋盘 PnP 精化后内点不足（{quality["inliers"]}/{len(points)}，至少需要 {min_inliers}）。'
    else:
        return matrix, valid, quality
    return None, None, quality


def board_halfturn(board):
    """Map reversed corner indexing back to the original checkerboard basis."""
    rotation = np.diag([-1.0, -1.0, 1.0, 1.0])
    rotation[:3, 3] = board[0] + board[-1]
    return rotation


def fit_intrinsics(frames, board, output):
    train = [r for r in frames if r["group"] == 0 and r["split"] == "train"]
    original = list(train)
    rejected = []
    for iteration in range(3):
        result = cv2.calibrateCameraExtended(
            [board.astype(np.float32)] * len(train),
            [np.asarray(r["board"], np.float32).reshape(-1, 1, 2) for r in train],
            tuple(train[0]["image_size"]),
            None,
            None,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-10),
        )
        rms, k, d, rv, tv, std, _, errors = result
        errors = errors.ravel()
        limit = max(
            0.5,
            float(
                np.median(errors)
                + 4 * 1.4826 * np.median(abs(errors - np.median(errors)))
            ),
        )
        bad = errors > limit
        if not np.any(bad) or iteration == 2:
            break
        rejected.extend(r["image"] for r, b in zip(train, bad) if b)
        train = [r for r, b in zip(train, bad) if not b]
        if len(train) < 20:
            raise ValueError("Too few reliable intrinsic images")
    held = []
    for r in frames:
        if r["group"] == 0 and r["split"] == "test":
            measured = np.asarray(r["board"])
            # Solve from half the corners and evaluate the complementary half.
            select = np.arange(len(board)) % 2 == 0
            c = pnp(board[select], measured[select], k, d)
            held.extend(
                np.linalg.norm(
                    project(board[~select], c, k, d) - measured[~select], axis=1
                )
            )
    fs = cv2.FileStorage(str(output / "camera_intrinsics.yml"), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", frames[0]["image_size"][0])
    fs.write("image_height", frames[0]["image_size"][1])
    fs.write("camera_matrix", k)
    fs.write("distortion_coefficients", d)
    fs.write("square_size_mm", 35.0)
    fs.write("calibration_rms_px", rms)
    fs.release()
    report = {
        "model": "Brown-Conrady 5 coefficients",
        "camera_matrix": k.tolist(),
        "distortion": d.ravel().tolist(),
        "training_image_count": len(train),
        "candidate_training_count": len(original),
        "rejected_training_images": rejected,
        "training_rms_px": float(rms),
        "training_per_image_rms_px": errors.tolist(),
        "opencv_parameter_std": std.ravel().tolist(),
        "heldout_half_board_corner_error_px": stats(held),
        "training_images": [r["image"] for r in train],
    }
    write_json(output / "intrinsics_report.json", report)
    print("INTRINSICS", json.dumps(report), flush=True)
    return k, d, report


def orient_boards(frames, board, k, d, *, poses_estimated=False):
    if not poses_estimated:
        for r in frames:
            r["board"] = np.asarray(r["board"], np.float64)
            r["camera"] = pnp(board, r["board"], k, d)
    halfturn = board_halfturn(board)
    flipped = []
    for group in sorted(set(r["group"] for r in frames)):
        candidates = [r for r in frames if r["group"] == group]
        rotations = np.stack([r["camera"][:3, :3] for r in candidates])
        # Rectangular boards have a 180-degree indexing ambiguity. Pick a
        # consistent branch inside each rigid capture group, using rotation only.
        angles = []
        for rotation in rotations:
            direct = Rotation.from_matrix(rotation.T @ rotations).magnitude()
            other = Rotation.from_matrix(
                rotation.T @ (rotations @ halfturn[:3, :3])
            ).magnitude()
            angles.append(np.minimum(direct, other).sum())
        anchor = rotations[int(np.argmin(angles))]
        for r in candidates:
            direct = Rotation.from_matrix(anchor.T @ r["camera"][:3, :3]).magnitude()
            other = Rotation.from_matrix(
                anchor.T @ r["camera"][:3, :3] @ halfturn[:3, :3]
            ).magnitude()
            if other + 1e-6 < direct:
                r["board"] = r["board"][::-1].copy()
                if poses_estimated:
                    r["camera"] = r["camera"] @ halfturn
                    r["board_inliers"] = r["board_inliers"][::-1].copy()
                else:
                    r["camera"] = pnp(board, r["board"], k, d)
                flipped.append(r["image"])
    return flipped


def triangulate(records, mid, k, d):
    matrices = []
    rays = []
    centers = []
    for r in records:
        p = np.asarray(r["markers"][mid])
        xy = cv2.undistortPoints(p.reshape(-1, 1, 2), k, d).reshape(-1, 2)
        camera = r["camera"]
        ray = np.c_[xy, np.ones(4)] @ camera[:3, :3]
        ray /= np.linalg.norm(ray, axis=1)[:, None]
        rays.append(ray)
        centers.append(-camera[:3, :3].T @ camera[:3, 3])
    rays = np.stack(rays)
    centers = np.stack(centers)
    points = []
    conditions = []
    angles = []
    for corner in range(4):
        directions = rays[:, corner]
        a = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
        summed = a.sum(axis=0)
        point = np.linalg.solve(summed, np.einsum("nij,nj->i", a, centers))
        points.append(point)
        conditions.append(float(np.linalg.cond(summed)))
        angles.append(
            float(
                np.rad2deg(np.arccos(np.clip((directions @ directions.T).min(), -1, 1)))
            )
        )
    return np.asarray(points), max(conditions), min(angles)


def fit_square_3d(points):
    q = square(1.0)
    centered = points - points.mean(axis=0)
    u, s, vt = np.linalg.svd(q.T @ centered)
    sign = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ np.diag([1, 1, sign]) @ u.T
    length = float((s * np.array([1, 1, sign])).sum() / np.sum(q * q))
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = points.mean(axis=0)
    return result, length


def init_geometry(frames, tools, k, d, output):
    groups = sorted(set(r["group"] for r in frames))
    measurements = {}
    report = []
    for group in groups:
        records = [r for r in frames if r["group"] == group and r["split"] == "train"]
        for mid in sorted(set(mid for ids in tools.values() for mid in ids)):
            visible = [r for r in records if mid in r["markers"]]
            if len(visible) < 3:
                continue
            points, condition, angle = triangulate(visible, mid, k, d)
            initial, length = fit_square_3d(points)
            if not 2 < length < 60:
                report.append({"group": group, "id": mid, "failed_length": length})
                continue

            def residual(x):
                obj = transform(pose(x[:6]), square(np.exp(x[6])))
                return np.concatenate(
                    [
                        (project(obj, r["camera"], k, d) - r["markers"][mid]).ravel()
                        for r in visible
                    ]
                )

            result = least_squares(
                residual,
                np.r_[vector(initial), np.log(length)],
                loss="soft_l1",
                f_scale=1.0,
                bounds=(
                    np.r_[np.full(6, -np.inf), np.log(2.0)],
                    np.r_[np.full(6, np.inf), np.log(60.0)],
                ),
                max_nfev=150,
                x_scale="jac",
            )
            error = float(np.sqrt(np.mean(residual(result.x) ** 2) * 2))
            measurements[group, mid] = (
                pose(result.x[:6]),
                float(np.exp(result.x[6])),
                len(visible),
            )
            report.append(
                {
                    "group": group,
                    "id": mid,
                    "length_mm": float(np.exp(result.x[6])),
                    "rmse_px": error,
                    "triangulation_condition": condition,
                    "triangulation_angle_deg": angle,
                    "images": len(visible),
                }
            )
    geometry = {}
    group_poses = {}
    lengths = {}
    references = {}
    for tool, ids in tools.items():
        counts = {
            mid: sum(
                count for (g, m), (_, _, count) in measurements.items() if m == mid
            )
            for mid in ids
        }
        ref = max(ids, key=lambda mid: counts[mid])
        references[tool] = ref
        geometry[ref] = np.eye(4)
        remaining = set(ids) - {ref}
        while remaining or any((g, tool) not in group_poses for g in groups):
            candidates = []
            for (g, mid), (matrix, length, count) in measurements.items():
                if mid in ids and mid in geometry and (g, tool) not in group_poses:
                    candidates.append((count, "group", g, mid, matrix))
                if mid in remaining and (g, tool) in group_poses:
                    candidates.append((count, "marker", g, mid, matrix))
            if not candidates:
                raise ValueError(
                    f"Disconnected group/marker graph for {tool}: {remaining}"
                )
            _, kind, g, mid, matrix = max(candidates, key=lambda x: x[0])
            if kind == "group":
                group_poses[g, tool] = matrix @ np.linalg.inv(geometry[mid])
            else:
                geometry[mid] = np.linalg.inv(group_poses[g, tool]) @ matrix
                remaining.remove(mid)
        for mid in ids:
            lengths[mid] = float(
                np.median(
                    [
                        length
                        for (g, m), (_, length, _) in measurements.items()
                        if m == mid
                    ]
                )
            )
    write_json(
        output / "triangulation_initialization.json",
        {"observations": report, "references": references, "lengths": lengths},
    )
    return geometry, lengths, group_poses, references


class JointProblem:
    """Analytic sparse BA: shared square geometry, group placements, camera poses."""

    def __init__(
        self,
        frames,
        tools,
        references,
        geometry,
        lengths,
        placements,
        board,
        k,
        d,
        vary_cameras,
    ):
        self.frames = frames
        self.tools = tools
        self.references = references
        self.board = board
        self.k = k
        self.d = d
        self.owner = {mid: tool for tool, ids in tools.items() for mid in ids}
        self.geometry0 = geometry
        self.lengths0 = lengths
        self.placements0 = placements
        self.marker_slices = {}
        self.length_slices = {}
        self.group_slices = {}
        self.camera_slices = {}
        x = []
        for mid in sorted(geometry):
            if mid not in references.values():
                self.marker_slices[mid] = slice(len(x), len(x) + 6)
                x.extend(vector(geometry[mid]))
            self.length_slices[mid] = slice(len(x), len(x) + 1)
            x.append(np.log(lengths[mid]))
        for key in sorted(placements):
            self.group_slices[key] = slice(len(x), len(x) + 6)
            x.extend(vector(placements[key]))
        if vary_cameras:
            for i, r in enumerate(frames):
                self.camera_slices[i] = slice(len(x), len(x) + 6)
                x.extend(vector(r["camera"]))
        self.x = np.asarray(x)
        self.lower = np.full(len(x), -np.inf)
        self.upper = np.full(len(x), np.inf)
        for section in self.length_slices.values():
            self.lower[section] = np.log(2.0)
            self.upper[section] = np.log(60.0)
        self.cached_x = None
        self.cached = None

    def unpack(self, x):
        geometry = {
            mid: pose(x[self.marker_slices[mid]])
            if mid in self.marker_slices
            else np.eye(4)
            for mid in self.geometry0
        }
        lengths = {mid: float(np.exp(x[s][0])) for mid, s in self.length_slices.items()}
        placements = {key: pose(x[s]) for key, s in self.group_slices.items()}
        cameras = [
            pose(x[self.camera_slices[i]]) if i in self.camera_slices else r["camera"]
            for i, r in enumerate(self.frames)
        ]
        return geometry, lengths, placements, cameras

    def evaluate(self, x):
        if self.cached_x is not None and np.array_equal(x, self.cached_x):
            return self.cached
        geometry, lengths, placements, cameras = self.unpack(x)
        marker_dr = {
            mid: cv2.Rodrigues(x[s][:3])[1].reshape(3, 3, 3)
            for mid, s in self.marker_slices.items()
        }
        group_dr = {
            key: cv2.Rodrigues(x[s][:3])[1].reshape(3, 3, 3)
            for key, s in self.group_slices.items()
        }
        rows = []
        cols = []
        data = []
        residuals = []
        offset = 0

        def add(start, section, values):
            values = np.asarray(values).reshape(-1, section.stop - section.start)
            rows.append(
                np.repeat(np.arange(start, start + len(values)), values.shape[1])
            )
            cols.append(np.tile(np.arange(section.start, section.stop), len(values)))
            data.append(values.ravel())

        for i, (r, camera) in enumerate(zip(self.frames, cameras)):
            ids = sorted(mid for mid in r["markers"] if mid in geometry)
            local = {mid: square(lengths[mid]) for mid in ids}
            world = {mid: transform(geometry[mid], local[mid]) for mid in ids}
            placed = {
                mid: transform(placements[r["group"], self.owner[mid]], world[mid])
                for mid in ids
            }
            points = np.concatenate([self.board, *[placed[mid] for mid in ids]])
            measured = np.concatenate([r["board"], *[r["markers"][mid] for mid in ids]])
            pixels, jac = cv2.projectPoints(
                points, vector(camera)[:3], camera[:3, 3], self.k, self.d
            )
            residuals.append((pixels.reshape(-1, 2) - measured).ravel())
            if i in self.camera_slices:
                add(offset, self.camera_slices[i], jac[:, :6])
            jp = jac[:, 3:6].reshape(-1, 2, 3)
            begin = len(self.board)
            for mid in ids:
                key = (r["group"], self.owner[mid])
                placement = placements[key]
                jpc = jp[begin : begin + 4] @ camera[:3, :3]
                drg = np.einsum("arc,nc->nra", group_dr[key], world[mid])
                add(
                    offset + begin * 2,
                    self.group_slices[key],
                    np.concatenate([jpc @ drg, jpc], axis=2),
                )
                jpm = jpc @ placement[:3, :3]
                if mid in self.marker_slices:
                    drm = np.einsum("arc,nc->nra", marker_dr[mid], local[mid])
                    add(
                        offset + begin * 2,
                        self.marker_slices[mid],
                        np.concatenate([jpm @ drm, jpm], axis=2),
                    )
                dl = local[mid] @ geometry[mid][:3, :3].T
                add(
                    offset + begin * 2,
                    self.length_slices[mid],
                    np.einsum("nij,nj->ni", jpm, dl).reshape(-1, 1),
                )
                begin += 4
            offset += len(points) * 2
        result = np.concatenate(residuals)
        jacobian = coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(offset, len(x)),
        ).tocsr()
        self.cached_x = x.copy()
        self.cached = (result, jacobian)
        return self.cached

    def solve(self, max_nfev=150):
        initial = self.evaluate(self.x)[0]
        print(
            f"BA variables={len(self.x)} residuals={len(initial)} initial_RMS={np.sqrt(np.mean(initial**2) * 2):.4f}",
            flush=True,
        )
        result = least_squares(
            lambda x: self.evaluate(x)[0],
            self.x,
            jac=lambda x: self.evaluate(x)[1],
            bounds=(self.lower, self.upper),
            method="trf",
            loss="soft_l1",
            f_scale=0.7,
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-8,
            xtol=1e-9,
            gtol=1e-5,
            tr_options={"atol": 1e-8, "btol": 1e-8},
            verbose=1,
        )
        self.x = result.x
        final = self.evaluate(self.x)[0]
        report = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "optimality": float(result.optimality),
            "cost": float(result.cost),
            "initial_rms_px": float(np.sqrt(np.mean(initial**2) * 2)),
            "final_rms_px": float(np.sqrt(np.mean(final**2) * 2)),
        }
        print("BA_RESULT", json.dumps(report), flush=True)
        return self.unpack(self.x), report


def prepare_frames(cache):
    frames = cache["frames"]
    for r in frames:
        r["markers"] = {
            int(mid): np.asarray(p, np.float64) for mid, p in r["markers"].items()
        }
    for group in sorted(set(r["group"] for r in frames)):
        records = [r for r in frames if r["group"] == group]
        cutoff = int(len(records) * 0.8)
        for i, r in enumerate(records):
            r["split"] = "train" if i < cutoff else "test"
    return frames


def fit_all(args):
    cv2.setNumThreads(1)
    output = Path(args.output)
    cache = json.loads((output / "detections.json").read_text())
    frames = prepare_frames(cache)
    board = object_board(
        tuple(cache["board"]["pattern_size"]), float(cache["board"]["square_size_mm"])
    )
    k, d, intrinsics = fit_intrinsics(frames, board, output)
    flipped = orient_boards(frames, board, k, d)
    # IDs are established by repeated decoded observations, not by assumed edge lengths.
    counts = Counter(mid for r in frames for mid in r["markers"])
    memberships = {}
    for r in frames:
        for tool, ids in r["capture_marker_ids"].items():
            for mid in ids:
                memberships.setdefault(mid, Counter())[tool] += 1
    tools = {
        tool: sorted(
            mid
            for mid, c in memberships.items()
            if c.most_common(1)[0][0] == tool and counts[mid] >= 8
        )
        for tool in "ABC"
    }
    print("TOOLS", tools, "FLIPPED_BOARDS", flipped, flush=True)
    geometry, lengths, placements, references = init_geometry(
        frames, tools, k, d, output
    )
    train = [r for r in frames if r["split"] == "train"]
    problem = JointProblem(
        train, tools, references, geometry, lengths, placements, board, k, d, False
    )
    (geometry, lengths, placements, cameras), fixed_report = problem.solve()
    problem = JointProblem(
        train, tools, references, geometry, lengths, placements, board, k, d, True
    )
    (geometry, lengths, placements, cameras), joint_report = problem.solve(200)
    for r, c in zip(train, cameras):
        r["camera"] = c
    result = {
        "tools": tools,
        "references": references,
        "lengths_mm": lengths,
        "marker_poses": {mid: m.tolist() for mid, m in geometry.items()},
        "group_poses": {
            f"{g}:{tool}": m.tolist() for (g, tool), m in placements.items()
        },
        "training_cameras": {r["image"]: c.tolist() for r, c in zip(train, cameras)},
        "board_order_flipped_images": flipped,
        "camera_matrix": k.tolist(),
        "distortion": d.ravel().tolist(),
        "stages": {
            "fixed_board_camera_poses": fixed_report,
            "joint_board_marker_ba": joint_report,
        },
        "split": {r["image"]: r["split"] for r in frames},
        "counts": dict(counts),
    }
    write_json(output / "fit.json", result)


def load_result(output):
    cache = json.loads((output / "detections.json").read_text())
    fit = json.loads((output / "fit.json").read_text())
    if fit.get("geometry_model", "square") != "square":
        raise ValueError(
            "Non-square geometry must use its own validator; do not regenerate corners with square(length)."
        )
    frames = prepare_frames(cache)
    k = np.asarray(fit["camera_matrix"])
    d = np.asarray(fit["distortion"])
    board = object_board(
        tuple(cache["board"]["pattern_size"]), float(cache["board"]["square_size_mm"])
    )
    orient_boards(frames, board, k, d)
    geometry = {
        int(mid): np.asarray(matrix) for mid, matrix in fit["marker_poses"].items()
    }
    lengths = {int(mid): length for mid, length in fit["lengths_mm"].items()}
    placements = {
        (int(key.split(":")[0]), key.split(":")[1]): np.asarray(matrix)
        for key, matrix in fit["group_poses"].items()
    }
    return cache, fit, frames, k, d, board, geometry, lengths, placements


def triangulate_generic(cameras, pixels, k, d):
    rays = []
    centers = []
    for camera, pixel in zip(cameras, pixels):
        xy = cv2.undistortPoints(
            np.asarray(pixel, np.float64).reshape(1, 1, 2), k, d
        ).ravel()
        ray = camera[:3, :3].T @ np.r_[xy, 1.0]
        ray /= np.linalg.norm(ray)
        rays.append(ray)
        centers.append(-camera[:3, :3].T @ camera[:3, 3])
    rays = np.asarray(rays)
    centers = np.asarray(centers)
    a = np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]
    return np.linalg.solve(a.sum(axis=0), np.einsum("nij,nj->i", a, centers))


def metric_crosscheck(frames, tools, geometry, lengths, placements, k, d, board):
    records = []
    for group in sorted(set(r["group"] for r in frames)):
        test = [r for r in frames if r["group"] == group and r["split"] == "test"]
        for tool, ids in tools.items():
            for mid in ids:
                visible = [r for r in test if mid in r["markers"]]
                if len(visible) < 5:
                    continue
                points, condition, angle = triangulate(visible, mid, k, d)
                if angle < 6:
                    continue
                initial = placements[group, tool] @ geometry[mid]

                def residual(x):
                    obj = transform(pose(x[:6]), square(np.exp(x[6])))
                    return np.concatenate(
                        [
                            (
                                project(obj, r["camera"], k, d) - r["markers"][mid]
                            ).ravel()
                            for r in visible
                        ]
                    )

                solved = least_squares(
                    residual,
                    np.r_[vector(initial), np.log(lengths[mid])],
                    loss="soft_l1",
                    f_scale=0.7,
                    x_scale="jac",
                    max_nfev=150,
                )
                estimated = pose(solved.x[:6])
                edge = float(np.exp(solved.x[6]))
                predicted = transform(initial, square(lengths[mid]))
                recovered = transform(estimated, square(edge))
                records.append(
                    {
                        "group": group,
                        "tool": tool,
                        "id": mid,
                        "image_count": len(visible),
                        "ray_angle_deg": angle,
                        "triangulation_condition": condition,
                        "test_edge_mm": edge,
                        "training_edge_mm": lengths[mid],
                        "absolute_edge_difference_mm": abs(edge - lengths[mid]),
                        "center_difference_mm": float(
                            np.linalg.norm(estimated[:3, 3] - initial[:3, 3])
                        ),
                        "corner_difference_mm": np.linalg.norm(
                            recovered - predicted, axis=1
                        ).tolist(),
                        "independent_test_fit_rmse_px": float(
                            np.sqrt(np.mean(residual(solved.x) ** 2) * 2)
                        ),
                    }
                )
    # Board scale check: reconstruct checkerboard corners that were NOT used
    # for pose estimation. Distances of two grid steps are 70 mm.
    test = [r for r in frames if r["split"] == "test"]
    selection = np.arange(len(board)) % 2 == 0
    cameras = [pnp(board[selection], r["board"][selection], k, d) for r in test]
    recovered = {
        idx: triangulate_generic(cameras, [r["board"][idx] for r in test], k, d)
        for idx in np.flatnonzero(~selection)
    }
    corner_errors = [
        np.linalg.norm(value - board[idx]) for idx, value in recovered.items()
    ]
    grid_errors = []
    grid_distances = []
    cols = 11
    for idx, point in recovered.items():
        for delta in (2, 2 * cols):
            other = idx + delta
            if other not in recovered:
                continue
            if delta == 2 and idx // cols != other // cols:
                continue
            distance = float(np.linalg.norm(point - recovered[other]))
            grid_errors.append(abs(distance - 70.0))
            grid_distances.append(distance)
    return {
        "method": "Independently fit squares using only held-out image measurements and board-only camera poses; no similarity alignment.",
        "minimum_images": 5,
        "minimum_ray_angle_deg": 6.0,
        "observations": records,
        "per_tool": {
            tool: {
                "edge_difference_mm": stats(
                    [
                        r["absolute_edge_difference_mm"]
                        for r in records
                        if r["tool"] == tool
                    ]
                ),
                "center_difference_mm": stats(
                    [r["center_difference_mm"] for r in records if r["tool"] == tool]
                ),
                "corner_difference_mm": stats(
                    [
                        v
                        for r in records
                        if r["tool"] == tool
                        for v in r["corner_difference_mm"]
                    ]
                ),
            }
            for tool in tools
        },
        "unused_board_corner_reconstruction_error_mm": stats(corner_errors),
        "unused_board_two_step_70mm_distance_error_mm": stats(grid_errors),
        "unused_board_two_step_distances_mm": grid_distances,
    }


def validate_all(args):
    output = Path(args.output)
    cache, fit, frames, k, d, board, geometry, lengths, placements = load_result(output)
    owner = {mid: tool for tool, ids in fit["tools"].items() for mid in ids}
    samples = []
    board_test = []
    depths = []
    for r in frames:
        if r["split"] == "train":
            camera = np.asarray(fit["training_cameras"][r["image"]])
        else:
            camera = r["camera"]
            selected = np.arange(len(board)) % 2 == 0
            pose_half = pnp(board[selected], r["board"][selected], k, d)
            board_test.extend(
                np.linalg.norm(
                    project(board[~selected], pose_half, k, d) - r["board"][~selected],
                    axis=1,
                )
            )
        for mid, measured in r["markers"].items():
            if mid not in geometry:
                continue
            points = transform(
                placements[r["group"], owner[mid]],
                transform(geometry[mid], square(lengths[mid])),
            )
            predicted = project(points, camera, k, d)
            delta = np.linalg.norm(predicted - measured, axis=1)
            depths.extend(transform(camera, points)[:, 2])
            samples.append(
                {
                    "image": r["image"],
                    "group": r["group"],
                    "split": r["split"],
                    "tool": owner[mid],
                    "id": mid,
                    "corner_errors_px": delta.tolist(),
                    "rmse_px": float(np.sqrt(np.mean(delta**2))),
                    "measured": measured.tolist(),
                    "predicted": predicted.tolist(),
                }
            )
    summary = {
        split: {
            tool: stats(
                [
                    error
                    for r in samples
                    if r["split"] == split and r["tool"] == tool
                    for error in r["corner_errors_px"]
                ]
            )
            for tool in fit["tools"]
        }
        for split in ("train", "test")
    }
    per_id = {
        mid: stats(
            [
                error
                for r in samples
                if r["split"] == "test" and r["id"] == mid
                for error in r["corner_errors_px"]
            ]
        )
        for mid in geometry
    }
    metric = metric_crosscheck(
        frames, fit["tools"], geometry, lengths, placements, k, d, board
    )
    counts = Counter(int(mid) for r in frames for mid in r["markers"])
    report = {
        "source": cache["source"],
        "image_count": len(frames),
        "training_images": sum(r["split"] == "train" for r in frames),
        "heldout_images": sum(r["split"] == "test" for r in frames),
        "split_policy": "Within each of the seven capture groups, the final contiguous 20% of images are held out.",
        "scale_source": "11x8 checkerboard with fixed 35 mm grid; all marker edge lengths estimated freely",
        "validation_policy": "Held-out marker predictions use only checkerboard camera PnP, shared training geometry, and training group placements. No held-out marker pose refit or error-based test exclusion.",
        "marker_counts": {tool: len(ids) for tool, ids in fit["tools"].items()},
        "excluded_rare_ids": {
            mid: count for mid, count in counts.items() if mid not in geometry
        },
        "summary_px": summary,
        "per_id_test_px": per_id,
        "heldout_complementary_board_corners_px": stats(board_test),
        "metric_crosscheck": metric,
        "minimum_depth_mm": float(min(depths)),
        "largest_heldout_observation_errors": sorted(
            [r for r in samples if r["split"] == "test"],
            key=lambda r: r["rmse_px"],
            reverse=True,
        )[:20],
        "observations_flagged_over_3px_rmse": {
            split: {
                "count": sum(r["split"] == split and r["rmse_px"] > 3 for r in samples),
                "total": sum(r["split"] == split for r in samples),
                "records": [
                    {"image": r["image"], "id": r["id"], "rmse_px": r["rmse_px"]}
                    for r in samples
                    if r["split"] == split and r["rmse_px"] > 3
                ],
            }
            for split in ("train", "test")
        },
        "completeness_passed": all(len(ids) == 9 for ids in fit["tools"].values()),
        "scope": "Marker geometry only; instrument tip/ball center is not calibrated.",
        "limitations": (
            [
                f"Tool {tool} has only {len(ids)} reliably decoded markers; missing geometry is not invented."
                for tool, ids in fit["tools"].items()
                if len(ids) != 9
            ]
            + [
                "Millimeter checks measure consistency against the supplied board and split data, not independent metrology."
            ]
        ),
        "stages": fit["stages"],
    }
    write_json(output / "validation_report.json", report)
    write_json(output / "reprojection_observations.json", {"observations": samples})
    for tool, ids in fit["tools"].items():
        reference = {"B": 81, "C": 84}.get(tool, fit["references"][tool])
        rebase = np.linalg.inv(geometry[reference])
        positions = {
            mid: transform(
                rebase, transform(geometry[mid], square(lengths[mid]))
            ).tolist()
            for mid in ids
        }
        document = {
            "schema_version": 1,
            "instrument": {
                "id": f"board-calibrated-tool-{tool.lower()}",
                "display_name": f"器械 {tool}：35 mm 棋盘尺度标定",
                "enabled": False,
                "model": "marked_object",
                "marker_dictionary": "DICT_4X4_100",
                "kind": "navigation_stick",
            },
            "marker_positions": positions,
            "marker_lengths": {mid: lengths[mid] for mid in ids},
            "marker_length": lengths[reference],
            "reference_marker_id": reference,
            "units": "mm",
            "scale_source": "checkerboard_35_mm",
            "reference_marker_length_fixed": False,
            "optimization_method": "shared_square_geometry_group_placements_board_marker_joint_ba",
            "marker_count": len(ids),
            "expected_marker_count": 9,
            "geometry_complete": len(ids) == 9,
            "tip_calibrated": False,
            "coordinate_frame": "Reference marker center; decoded +X right, +Y up, +Z outward",
            "heldout_corner_rmse_px": summary["test"][tool]["rms"],
        }
        (output / f"tool_{tool}_geometry.yaml").write_text(
            yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    print("VALIDATION", json.dumps(summary), flush=True)
    print("METRIC", json.dumps(metric["per_tool"]), flush=True)
    return report


def render_validation(args):
    output = Path(args.output)
    root = Path(args.images)
    data = json.loads((output / "reprojection_observations.json").read_text())[
        "observations"
    ]
    report = json.loads((output / "validation_report.json").read_text())
    selected = []
    for group in range(7):
        names = sorted(
            set(
                r["image"] for r in data if r["group"] == group and r["split"] == "test"
            )
        )
        selected.append(names[len(names) // 2])
    selected.extend(
        r["image"] for r in report["largest_heldout_observation_errors"][:5]
    )
    selected = list(dict.fromkeys(selected))
    (output / "overlays").mkdir(exist_ok=True)
    gallery = []
    for name in selected:
        image = cv2.imread(str(root / name))
        records = [r for r in data if r["image"] == name]
        panels = []
        for tool in "ABC":
            obs = [r for r in records if r["tool"] == tool]
            if not obs:
                continue
            points = np.concatenate([r["measured"] for r in obs])
            minimum = np.maximum(0, np.floor(points.min(axis=0) - 25).astype(int))
            maximum = np.minimum(
                [image.shape[1], image.shape[0]],
                np.ceil(points.max(axis=0) + 25).astype(int),
            )
            x, y = minimum
            xx, yy = maximum
            crop = image[y:yy, x:xx].copy()
            factor = min(3.0, 600 / max(crop.shape[:2]))
            crop = cv2.resize(crop, None, fx=factor, fy=factor)
            for r in obs:
                measured = (np.array(r["measured"]) - minimum) * factor
                predicted = (np.array(r["predicted"]) - minimum) * factor
                for a, b in zip(measured, predicted):
                    a = tuple(np.rint(a).astype(int))
                    b = tuple(np.rint(b).astype(int))
                    cv2.circle(crop, a, 3, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.drawMarker(
                        crop, b, (0, 0, 255), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA
                    )
                    cv2.line(crop, a, b, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    crop,
                    str(r["id"]),
                    tuple(np.rint(measured.mean(axis=0)).astype(int)),
                    0,
                    0.5,
                    (255, 180, 0),
                    1,
                    cv2.LINE_AA,
                )
            canvas = np.full((340, 640, 3), 30, np.uint8)
            if crop.shape[0] > 285:
                crop = cv2.resize(
                    crop, None, fx=285 / crop.shape[0], fy=285 / crop.shape[0]
                )
            canvas[45 : 45 + crop.shape[0], 10 : 10 + crop.shape[1]] = crop
            rms = np.sqrt(np.mean([e * e for r in obs for e in r["corner_errors_px"]]))
            cv2.putText(
                canvas,
                f"{name} / {tool} / RMS {rms:.3f} px",
                (10, 25),
                0,
                0.55,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            panels.append(canvas)
        panel = np.hstack(panels)
        destination = output / "overlays" / name.replace(".png", ".jpg")
        cv2.imwrite(str(destination), panel)
        gallery.append(panel)
    cv2.imwrite(str(output / "validation_montage.jpg"), np.vstack(gallery))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["detect", "refine-markers", "fit", "validate", "render"]
    )
    parser.add_argument("--images")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not args.output:
        parser.error("--output is required")
    if args.stage in ("detect", "refine-markers", "render") and not args.images:
        parser.error("--images is required")
    if args.stage == "detect":
        detect_all(args)
    elif args.stage == "refine-markers":
        refine_all_markers(args)
    elif args.stage == "fit":
        fit_all(args)
    elif args.stage == "validate":
        validate_all(args)
    elif args.stage == "render":
        render_validation(args)


if __name__ == "__main__":
    main()
