"""Calibrate each imager independently before fitting pairwise extrinsics.

All observations are measurements in individual images. Validation images are
reserved before fitting either intrinsics or extrinsics. Detection failures are
reported separately from numerical residuals; neither implies wrong corners.
"""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import re
import uuid

import cv2
import numpy as np

from ..sensor_intrinsics.core import (
    INTRINSIC_STD_NAMES, Observation, CalibrationFit, _manual_residuals, calibrate_once, coverage_statistics, utc_now_iso,
    commit_staged_files, create_object_points, image_quality, prepare_gray8,
    pose_statistics, quality_warnings, verify_observations,
)
from ..sensor_intrinsics.detection import detect_checkerboard_realtime
from .report import save_quality_report
from ..io import sensor_image_directories


def _safe_label(label):
    label = str(label)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
        raise ValueError(f"Sensor labels must contain only letters, digits, _ or -: {label!r}")
    return label


def _image_path(session, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Image paths must be nonempty strings")
    path = (session / relative).resolve()
    try:
        path.relative_to(session)
    except ValueError as exc:
        raise ValueError(f"Image path escapes the capture session: {relative}") from exc
    return path, path.relative_to(session).as_posix()


def _load_manifest(session):
    source = session / "capture_metadata.json"
    if not source.is_file():
        raise FileNotFoundError(f"No capture_metadata.json in {session}")
    raw = json.loads(source.read_text(encoding="utf-8-sig"))
    if raw.get("schema_version") != 2 or not isinstance(raw.get("sensors"), list):
        raise ValueError("Capture metadata must use schema_version 2 and a sensor list")
    sensors, images, board = raw["sensors"], raw["images"], raw["board"]
    pairs = raw["pairs"]
    labels = [_safe_label(sensor["label"]) for sensor in sensors]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Capture metadata must have unique sensor labels")
    if not isinstance(images, dict) or set(images) != set(labels):
        raise ValueError("Image lists must match the declared sensors")
    if any(not isinstance(paths, list) for paths in images.values()):
        raise ValueError("Each sensor must declare a list of image paths")
    for pair in pairs:
        if "images" not in pair:
            raise ValueError("Every pair must declare its sensor-to-image mapping")
        if len(pair["images"]) < 2 or any(label not in images for label in pair["images"]):
            raise ValueError("Every pair must refer to two or more declared sensors")
        for label, path in pair["images"].items():
            if path not in images[label]:
                raise ValueError(f"Pair image is absent from the sensor image list: {label}/{path}")
    expected_directories = sensor_image_directories(sensors)
    declared_directories = set()
    for sensor in sensors:
        directory = sensor.get("image_directory")
        if directory != expected_directories[sensor["label"]]:
            raise ValueError(f"Incorrect vendor/channel image directory for {sensor['label']}: {directory}")
        folder, _ = _image_path(session, directory)
        if folder == session or not folder.is_dir():
            raise ValueError(f"Missing or invalid image directory for {sensor['label']}: {directory}")
        key = str(folder).casefold()
        if key in declared_directories:
            raise ValueError("Each sensor must have its own image directory")
        declared_directories.add(key)
        for relative in images[sensor["label"]]:
            path, _ = _image_path(session, relative)
            if path.parent != folder:
                raise ValueError(f"Image for {sensor['label']} is outside its declared directory: {relative}")
    pattern = (int(board["pattern_cols"]), int(board["pattern_rows"]))
    objects = create_object_points(pattern, float(board["square_size_mm"]))
    return raw, source, sensors, images, pairs, pattern, objects, float(board["square_size_mm"])


def _options(options):
    result = {"model": "standard", "min_views": 15, "validation_stride": 5,
              "detect_max_dim": 1600, "max_rms_px": 1.0, "max_validation_rmse_px": 2.0,
              "max_pair_delta_ms": 5.0, "reject_outliers": True,
              "max_view_rmse_px": 1.0, "max_outlier_fraction": 0.2}
    result.update(options or {})
    for key in ("min_views", "validation_stride", "detect_max_dim"):
        result[key] = int(result[key])
    if result["min_views"] < 3 or result["validation_stride"] < 2 or result["detect_max_dim"] < 1:
        raise ValueError("min_views >=3, validation_stride >=2, detect_max_dim >=1 are required")
    for key in ("max_rms_px", "max_validation_rmse_px", "max_pair_delta_ms", "max_view_rmse_px"):
        result[key] = float(result[key])
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if result["model"] not in {"standard", "rational"}:
        raise ValueError("model must be standard or rational")
    if not isinstance(result["reject_outliers"], bool):
        raise ValueError("reject_outliers must be true or false")
    if "use_known_intrinsics" in result and result["use_known_intrinsics"] is not None and type(result["use_known_intrinsics"]) is not bool:
        raise ValueError("use_known_intrinsics must be true or false")
    result["max_outlier_fraction"] = float(result["max_outlier_fraction"])
    if not 0 < result["max_outlier_fraction"] <= 0.5:
        raise ValueError("max_outlier_fraction must be > 0 and <= 0.5")
    return result


def _fixed_camera(sensor, opts, image_size):
    """Select known geometry, or request checkerboard fitting explicitly.

    An omitted switch retains the previous SDK/measured mixed workflow.
    Explicit false estimates every imager from checkerboards, even with SDK K/D.
    Explicit true requires known K/D for every imager and never fits a fallback.
    """
    if opts.get("use_known_intrinsics") is False:
        return None
    path = opts.get("fixed_intrinsics", {}).get(sensor["label"])
    if path:
        from ..validation.stereo import _camera
        camera = _camera(Path(path))
        source = str(path)
    elif sensor.get("sensor_type") in {"vcamera", "orbbec"}:
        data = sensor.get("capture_metadata", {}).get("sdk_intrinsics")
        if not data:
            raise ValueError(f"{sensor['sensor_type']} factory intrinsics missing for {sensor['label']}: recapture with SDK K/D or export SDK K/D to OpenCV YAML and set calibration.fixed_intrinsics for this label; measured intrinsics are not a fallback")
        matrix = np.asarray(data["camera_matrix"], np.float64)
        distortion = np.asarray(data["distortion_coefficients"], np.float64).reshape(-1)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or min(matrix[0, 0], matrix[1, 1]) <= 0
                or not np.allclose(matrix[2], [0, 0, 1])
                or distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all()):
            raise ValueError("Invalid captured SDK K/D")
        camera = dict(camera_matrix=matrix, distortion_coefficients=distortion,
                      image_size=tuple(data.get("image_size") or sensor.get("capture_metadata", {}).get("image_size") or ()))
        source = ("capture_metadata.GetFactoryCalibInfo(Texture)" if sensor["sensor_type"] == "vcamera"
                  else "capture_metadata.OrbbecStreamProfile.get_intrinsic/get_distortion")
    else:
        if opts.get("use_known_intrinsics") is True:
            raise ValueError(f"Known intrinsics required for {sensor['label']}: set calibration.fixed_intrinsics.{sensor['label']} to a measured OpenCV YAML file; checkerboard fitting is disabled")
        return None
    if tuple(camera["image_size"]) != tuple(image_size):
        raise ValueError(f"Factory model resolution {camera['image_size']} differs from images {image_size}; provide the SDK model for the actual mode")
    return camera, source


def _fixed_fit(training, pattern, square, camera):
    objects = create_object_points(pattern, square)
    rvecs, tvecs = [], []
    for view in training:
        ok, rvec, tvec = cv2.solvePnP(objects, view.corners, camera["camera_matrix"],
                                    camera["distortion_coefficients"], flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError(f"Fixed-intrinsic board pose failed: {view.path}")
        rvecs.append(rvec)
        tvecs.append(tvec)
    errors, points, residuals = _manual_residuals(objects, training, camera["camera_matrix"],
                                               camera["distortion_coefficients"], rvecs, tvecs)
    return CalibrationFit(float(np.sqrt(np.mean(errors ** 2))), camera["camera_matrix"].copy(),
                          camera["distortion_coefficients"].copy(), rvecs, tvecs,
                          np.empty(0), np.empty(0), errors, points, residuals)


def _fit_intrinsics_with_rejection(training, pattern, square, opts, sensor_result, session, label, fixed=None):
    """Refit after removing the worst training view; never inspect holdout errors."""
    retained = list(training)
    budget = int(len(retained) * opts["max_outlier_fraction"])
    rejected = sensor_result["rejected_training_views"]
    while True:
        fit = (_fixed_fit(retained, pattern, square, fixed) if fixed is not None
               else calibrate_once(retained, pattern, square, model=opts["model"]))
        sensor_result.setdefault("initial_training_rms_px", fit.rms)
        errors = np.asarray(fit.per_view_errors, dtype=float)
        if errors.size != len(retained) or not np.isfinite(errors).all():
            raise ValueError("Invalid per-view residuals; automatic rejection cannot continue")
        for view, error in zip(retained, errors):
            if error > opts["max_view_rmse_px"]:
                sensor_result.setdefault("image_warnings", []).append({
                    "image": Path(view.path).relative_to(session).as_posix(),
                    "stage": "内参拟合", "iteration": len(rejected) + 1,
                    "message": f"逐图训练 RMSE {error:.6f} px 超过 {opts['max_view_rmse_px']:.6f} px 阈值"})
        worst = int(np.argmax(errors))
        if not opts["reject_outliers"] or errors[worst] <= opts["max_view_rmse_px"]:
            return retained, fit
        if len(retained) - 1 < opts["min_views"]:
            raise ValueError("High-residual training views remain, but rejection would violate min_views")
        if len(rejected) >= budget:
            raise ValueError("High-residual training views remain after reaching max_outlier_fraction; inspect the data")
        view = retained.pop(worst)
        entry = {"image": Path(view.path).relative_to(session).as_posix(),
                 "rmse_px_before_removal": float(errors[worst]), "threshold_px": opts["max_view_rmse_px"],
                 "iteration": len(rejected) + 1, "reason": "intrinsic_training_rmse_exceeds_threshold"}
        rejected.append(entry)
        sensor_result["training_image_count"] = len(retained)
        print(f"{label}: excluded training image {entry['image']}: {errors[worst]:.6f} px; refitting intrinsics")


def _rmse(observed, projected):
    difference = np.asarray(observed).reshape(-1, 2) - np.asarray(projected).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def _summary(entries, key):
    values = np.asarray([entry[key] for entry in entries if key in entry], np.float64)
    if not values.size:
        return {"count": 0, "mean_rmse_px": None, "max_rmse_px": None}
    return {"count": int(values.size), "mean_rmse_px": float(np.mean(values)),
            "max_rmse_px": float(np.max(values))}


def _save_bundle(outputs):
    """Publish all minimal YAML results together after every quality check passes."""
    staged = []
    try:
        for path, fields in outputs:
            if path.exists():
                raise FileExistsError(f"Calibration output already exists: {path}")
            temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.yml")
            staged.append((temporary, path))
            storage = cv2.FileStorage(str(temporary), cv2.FILE_STORAGE_WRITE)
            if not storage.isOpened():
                raise OSError(f"Cannot create calibration file: {temporary}")
            try:
                for key, value in fields.items():
                    storage.write(key, value)
            finally:
                storage.release()
        commit_staged_files(staged, replace_existing=False)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def _detect_image(path, pattern, maximum):
    try:
        image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None, {"status": "unreadable_image"}
        gray = prepare_gray8(image)
    except (OSError, ValueError, cv2.error) as error:
        return None, {"status": "unreadable_image", "detail": str(error)}
    size = (gray.shape[1], gray.shape[0])
    quality = image_quality(gray)
    try:
        found, corners, detector, _, _ = detect_checkerboard_realtime(gray, pattern, max_dimension=maximum)
    except (ValueError, cv2.error) as error:
        return None, {"status": "detection_error", "detail": str(error), "image_size": list(size), **quality}
    audit = {"status": "detected" if found and corners is not None else "complete_checkerboard_not_found",
             "detector": detector, "image_size": list(size),
             "corner_count": int(len(corners)) if corners is not None else 0, **quality}
    if not found or corners is None:
        return None, audit
    if not np.isfinite(corners).all() or len(corners) != pattern[0] * pattern[1]:
        audit["status"] = "invalid_corner_coordinates"
        return None, audit
    return Observation(str(path), corners, size, quality["sharpness"], quality["dark_fraction"],
                       quality["bright_fraction"], detector), audit


def _board_pose(objects, corners, camera):
    ok, rotation, translation = cv2.solvePnP(objects, corners, camera["camera_matrix"], camera["distortion_coefficients"])
    if not ok or not np.isfinite(translation).all() or float(translation[2, 0]) <= 0:
        raise ValueError("Board pose could not be estimated in front of sensor")
    return cv2.Rodrigues(rotation)[0], translation.reshape(3, 1)


def _predict_right(objects, left, left_camera, right_camera, transform):
    rotation, translation = _board_pose(objects, left, left_camera)
    right_rotation = transform[:3, :3] @ rotation
    right_translation = transform[:3, :3] @ translation + transform[:3, 3:]
    if not np.all((right_rotation @ objects.T + right_translation)[2] > 0):
        raise ValueError("Projected board lies behind right sensor")
    return cv2.projectPoints(objects, cv2.Rodrigues(right_rotation)[0], right_translation,
                             right_camera["camera_matrix"], right_camera["distortion_coefficients"])[0]


def _order_candidates(corners, pattern):
    # A square grid permits quarter turns as well as a half turn. A rectangular
    # grid cannot exchange row and column counts, so its two rotations suffice.
    grid = corners.reshape(pattern[1], pattern[0], 1, 2)
    turns = (0, 1, 2, 3) if pattern[0] == pattern[1] else (0, 2)
    return [(turn * 90, np.ascontiguousarray(np.rot90(grid, turn).reshape(-1, 1, 2))) for turn in turns]


def _select_order(objects, view, cameras, transform, pattern):
    predicted = _predict_right(objects, view["left"].corners, cameras[0], cameras[1], transform)
    candidates = [(angle, corners, _rmse(corners, predicted)) for angle, corners in _order_candidates(view["right"].corners, pattern)]
    candidates.sort(key=lambda value: value[2])
    best, other = candidates[:2]
    gap = other[2] - best[2]
    return {**view, "right_corners": best[1], "right_order_rotation_deg": best[0],
            "ordering_best_rmse_px": best[2], "ordering_alternative_rmse_px": other[2],
            "ordering_ambiguous": bool(gap < 0.5 or other[2] <= best[2] * 1.1)}


def _fit_stereo(objects, views, cameras):
    flags = cv2.CALIB_FIX_INTRINSIC
    # Retain all SDK coefficients, including rational and thin-prism terms.
    count = max(np.asarray(camera["distortion_coefficients"]).size for camera in cameras)
    if count >= 8:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if count >= 12:
        flags |= cv2.CALIB_THIN_PRISM_MODEL
    if count >= 14:
        flags |= cv2.CALIB_TILTED_MODEL
    rms, _, _, _, _, rotation, translation, _, _ = cv2.stereoCalibrate(
        [objects] * len(views), [view["left"].corners for view in views],
        [view["right_corners"] for view in views],
        cameras[0]["camera_matrix"].copy(), cameras[0]["distortion_coefficients"].copy(),
        cameras[1]["camera_matrix"].copy(), cameras[1]["distortion_coefficients"].copy(),
        cameras[0]["image_size"], flags=flags,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-8))
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3:] = rotation, translation
    if not np.isfinite(rms) or not np.isfinite(transform).all():
        raise ValueError("Stereo calibration returned non-finite parameters")
    return float(rms), transform


def _initial_transform(objects, views, cameras, pattern, prior):
    if prior is not None:
        prior = np.asarray(prior, np.float64)
        if (prior.shape != (4, 4) or not np.isfinite(prior).all() or not np.allclose(prior[3], [0, 0, 0, 1])
                or not np.allclose(prior[:3, :3].T @ prior[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(prior[:3, :3]), 1.0, atol=1e-4)):
            raise ValueError("SDK ordering prior must be a finite 4x4 transform")
        return prior, {"method": "sdk_geometry_used_only_to_resolve_corner_order", "global_ambiguity": False}
    seeds = []
    for view in views:
        lr, lt = _board_pose(objects, view["left"].corners, cameras[0])
        for _, corners in _order_candidates(view["right"].corners, pattern):
            rr, rt = _board_pose(objects, corners, cameras[1])
            transform = np.eye(4)
            transform[:3, :3] = rr @ lr.T
            transform[:3, 3:] = rt - transform[:3, :3] @ lt
            seeds.append(transform)
    hypotheses = {}
    for seed in seeds:
        try:
            selected = [_select_order(objects, view, cameras, seed, pattern) for view in views]
        except (ValueError, cv2.error):
            continue
        key = tuple(view["right_order_rotation_deg"] for view in selected)
        score = float(np.median([view["ordering_best_rmse_px"] for view in selected]))
        if key not in hypotheses or score < hypotheses[key][0]:
            hypotheses[key] = (score, seed)
    ranked = sorted(hypotheses.values(), key=lambda value: value[0])
    if not ranked:
        raise ValueError("No consistent corner-order geometry could be established")
    ambiguous = len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.5
    return ranked[0][1], {"method": "training_view_board_pose_consensus",
                          "best_consensus_median_rmse_px": ranked[0][0],
                          "alternative_consensus_median_rmse_px": ranked[1][0] if len(ranked) > 1 else None,
                          "global_ambiguity": bool(ambiguous)}


def calibrate_session(session: Path, output_dir: Path, options: dict, mode: str) -> dict:
    """Fit image geometry, publish passing YAML and always retain quality reports.

    Every sensor uses its own training images. Held-out images never participate
    in fitting. If any numerical quality check fails, no result YAML is written.
    Image failures and correspondence decisions are retained in quality reports.
    """
    session, output_dir = Path(session).resolve(), Path(output_dir).resolve()
    opts = _options(options)
    if mode not in {"sensor_intrinsics", "sensor_extrinsics"}:
        raise ValueError("Unsupported calibration mode")
    if mode == "sensor_intrinsics" and opts.get("use_known_intrinsics") is True:
        raise ValueError("use_known_intrinsics=true requires mode sensor_extrinsics")
    intrinsics_mode = ("estimate_intrinsics" if mode == "sensor_intrinsics" or opts.get("use_known_intrinsics") is False
                       else "known_intrinsics" if opts.get("use_known_intrinsics") is True else "legacy_mixed")
    print(f"Intrinsic calibration strategy: {intrinsics_mode}")
    raw, manifest, sensors, images, captures, pattern, objects, square = _load_manifest(session)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "created_at_utc": utc_now_iso(),
              "session": str(session), "manifest": str(manifest), "mode": mode,
              "intrinsics_mode": intrinsics_mode,
              "opencv_version": cv2.__version__, "options": opts,
              "board": {"pattern_cols": pattern[0], "pattern_rows": pattern[1], "square_size_mm": square},
              "metric_definitions": {
                  "training_rms_px": "sqrt(mean squared 2D corner distance), fitted intrinsic training views",
                  "stereo_rms_px": "OpenCV stereo RMS over both cameras, fixed intrinsics",
                  "validation": "held-out images; mean/max of per-view RMSE, not a pooled point RMSE",
                  "pair_validation": "first-camera board pose projected through fixed R/T into second camera"},
              "repeatability": {"status": "not_measured", "value": None,
                                "reason": "Requires independent repeated acquisitions and calibrations; rerunning identical images is not such a measurement."},
              "sensors": {}, "pairs": [], "output_files": [], "passed": True}
    outputs = []
    heldout, paired_paths = set(), set()
    for index, capture in enumerate(captures):
        for label, relative in capture["images"].items():
            key = (label, _image_path(session, relative)[1])
            paired_paths.add(key)
            if index % opts["validation_stride"] == 0:
                heldout.add(key)
    observations, all_observations, cameras = {}, {}, {}
    for sensor in sensors:
        label = sensor["label"]
        accepted, seen, detection_audits = {}, set(), []
        expected = sensor.get("image_size", (sensor.get("sdk_intrinsics") or {}).get("image_size"))
        if isinstance(expected, dict):
            expected = [expected["width"], expected["height"]]
        expected = tuple(expected) if expected else None
        for index, relative in enumerate(images[label]):
            path, relative = _image_path(session, relative)
            key = (label, relative)
            if key in seen:
                detection_audits.append({"image": relative, "status": "duplicate_manifest_image"})
                print(f"{label}: {relative}: duplicate_manifest_image")
                continue
            seen.add(key)
            if key not in paired_paths and index % opts["validation_stride"] == 0:
                heldout.add(key)
            observation, audit = _detect_image(path, pattern, opts["detect_max_dim"])
            if observation is not None:
                expected = expected or observation.image_size
                if observation.image_size != expected:
                    audit.update({"status": "resolution_mismatch", "detail": f"expected {expected}, received {observation.image_size}"})
                    observation = None
            if observation is not None:
                accepted[relative] = observation
            detection_audits.append({"image": relative,
                                     "split": "validation" if key in heldout else "training", **audit})
            detail = audit.get("detail", audit.get("detector", ""))
            print(f"{label}: {relative}: {audit['status']}" + (f" ({detail})" if detail else ""))
        training = [observation for relative, observation in accepted.items() if (label, relative) not in heldout]
        validation = [observation for relative, observation in accepted.items() if (label, relative) in heldout]
        sensor_result = {"source_image_count": len(seen), "valid_image_count": len(accepted),
                         "training_image_count": len(training), "initial_training_image_count": len(training),
                         "rejected_training_views": [], "validation_image_count": len(validation),
                         "detections": detection_audits, "image_warnings": [], "warnings": [], "passed": False}
        result["sensors"][label] = sensor_result
        observations[label] = accepted
        all_observations[label] = dict(accepted)
        if len(training) < opts["min_views"] or not validation:
            reason = f"Need >= {opts['min_views']} training images plus held-out images; got {len(training)} + {len(validation)}"
            sensor_result["reason"] = reason
            print(f"{label}: FAIL: {reason}")
            result["passed"] = False
            continue
        try:
            fixed = _fixed_camera(sensor, opts, training[0].image_size) if mode == "sensor_extrinsics" else None
            sensor_result["intrinsics_source"] = fixed[1] if fixed else "checkerboard_fit"
            sensor_result["intrinsics_fixed"] = fixed is not None
            print(f"{label}: intrinsics {'fixed' if fixed else 'estimated'}; source: {sensor_result['intrinsics_source']}")
            training, fit = _fit_intrinsics_with_rejection(training, pattern, square, opts, sensor_result, session, label,
                                                          fixed[0] if fixed else None)
            for entry in sensor_result["rejected_training_views"]:
                accepted.pop(entry["image"], None)
            if (not np.isfinite(fit.rms) or not np.isfinite(fit.camera_matrix).all()
                    or not np.isfinite(fit.dist_coeffs).all()
                    or min(fit.camera_matrix[0, 0], fit.camera_matrix[1, 1]) <= 0):
                raise ValueError("Intrinsic calibration returned invalid or non-finite parameters")
            coverage = coverage_statistics(training, training[0].image_size)
            poses, _ = pose_statistics(fit)
            warnings = quality_warnings(fit, coverage, poses, training[0].image_size)
            entries, _, _ = verify_observations(validation, pattern, square, fit.camera_matrix, fit.dist_coeffs)
            excluded_observations = [all_observations[label][entry["image"]]
                                     for entry in sensor_result["rejected_training_views"]]
            excluded_entries, _, _ = verify_observations(
                excluded_observations, pattern, square, fit.camera_matrix, fit.dist_coeffs)
            sensor_result["excluded_validation"] = _summary(excluded_entries, "view_rms_px")
            sensor_result["excluded_validation_views"] = [
                {**entry, "path": Path(entry["path"]).relative_to(session).as_posix()}
                for entry in excluded_entries]
            summary = _summary(entries, "view_rms_px")
            passed = bool(fit.rms <= opts["max_rms_px"] and summary["count"] == len(validation)
                          and summary["max_rmse_px"] is not None
                          and summary["max_rmse_px"] <= opts["max_validation_rmse_px"])
            sensor_result.update({"passed": passed, "training_rms_px": fit.rms,
                                  "validation": summary, "warnings": warnings,
                                  "image_size": training[0].image_size, "coverage": coverage, "poses": poses,
                                  "intrinsic_parameter_std": dict(zip(INTRINSIC_STD_NAMES, fit.std_intrinsics.reshape(-1))),
                                  "training_views": [{"image": Path(view.path).relative_to(session).as_posix(),
                                                      "rmse_px": float(error)}
                                                     for view, error in zip(training, fit.per_view_errors)],
                                  "validation_views": [{**entry, "path": Path(entry["path"]).relative_to(session).as_posix()}
                                                       for entry in entries]})
            for warning in warnings:
                print(f"{label}: WARNING: {warning}")
            for observation, error in zip(training, fit.per_view_errors):
                if error > opts["max_rms_px"]:
                    print(f"{label}: training residual {Path(observation.path).name}: {float(error):.6f} px (image retained)")
            for entry in entries:
                if entry.get("status") != "ok":
                    print(f"{label}: validation {Path(entry['path']).name}: {entry.get('status')}")
                elif entry["view_rms_px"] > opts["max_validation_rmse_px"]:
                    print(f"{label}: validation residual {Path(entry['path']).name}: {entry['view_rms_px']:.6f} px")
            if not passed:
                sensor_result["reason"] = "Intrinsic training RMS, held-out pose estimation or held-out residual exceeds the configured quality limits"
                print(f"{label}: FAIL: {sensor_result['reason']}")
            camera = {"camera_matrix": fit.camera_matrix, "distortion_coefficients": fit.dist_coeffs,
                      "image_size": training[0].image_size}
            cameras[label] = camera
            outputs.append((output_dir / f"sensor_{label}_intrinsics.yml", {
                "image_width": camera["image_size"][0], "image_height": camera["image_size"][1],
                "camera_matrix": np.asarray(fit.camera_matrix, np.float64).reshape(3, 3),
                "distortion_coefficients": np.asarray(fit.dist_coeffs, np.float64).reshape(1, -1)}))
            result["passed"] = result["passed"] and passed
        except (ValueError, RuntimeError, cv2.error) as error:
            sensor_result["reason"] = str(error)
            print(f"{label}: FAIL: {error}")
            result["passed"] = False

    if mode != "sensor_intrinsics":
        groups, seen_pairs = {}, set()
        for index, capture in enumerate(captures):
            ordered_labels = [sensor["label"] for sensor in sensors if sensor["label"] in capture["images"]]
            for left, right in combinations(ordered_labels, 2):
                key = (left, right)
                group = groups.setdefault(key, {"views": [], "rejected_pairs": []})
                lp, rp = (_image_path(session, capture["images"][label])[1] for label in key)
                entry = {"capture_index": index, "left_image": lp, "right_image": rp}
                def reject(reason):
                    group["rejected_pairs"].append({**entry, "reason": reason})
                    print(f"{left} -> {right}: {lp} / {rp}: skipped pair: {reason}")
                if lp == rp:
                    reject("same_image_for_both_sensors")
                    continue
                duplicate_key = (left, lp, right, rp)
                if duplicate_key in seen_pairs:
                    reject("duplicate_pair")
                    continue
                seen_pairs.add(duplicate_key)
                if left not in cameras or right not in cameras:
                    reject("sensor_intrinsics_unavailable")
                    continue
                missing = [label for label, image in ((left, lp), (right, rp)) if image not in observations[label]]
                if missing:
                    excluded = [label for label, image in ((left, lp), (right, rp))
                                if any(view["image"] == image for view in result["sensors"][label]["rejected_training_views"])]
                    reject(("intrinsic_training_outlier: " + ", ".join(excluded)) if excluded else
                           "missing_sensor_observation: " + ", ".join(missing))
                    continue
                delta = capture.get("capture_delta_ms")
                static_capture = raw.get("capture_mode") == "static_board_sequential_reads"
                if delta is None and static_capture:
                    pass  # Independent device clocks cannot support a fake timestamp delta.
                elif delta is None or not np.isfinite(delta) or delta < 0:
                    reject("invalid_or_missing_timestamp_difference")
                    continue
                if delta is not None and delta > opts["max_pair_delta_ms"] and not static_capture:
                    reject(f"timestamp_difference_exceeds_limit: {delta:.6f} ms")
                    continue
                split = "validation" if (left, lp) in heldout or (right, rp) in heldout else "training"
                group["views"].append({**entry, "left": observations[left][lp], "right": observations[right][rp],
                                       "split": split, "timestamp_basis": capture.get("timestamp_basis", "unknown")})
        if not groups:
            result["passed"] = False
            print("FAIL: no_sensor_pairs_in_manifest")
        for (left, right), group in groups.items():
            training = [view for view in group["views"] if view["split"] == "training"]
            validation = [view for view in group["views"] if view["split"] == "validation"]
            pair_result = {"left_sensor": left, "right_sensor": right, "passed": False,
                           "training_pair_count": len(training), "validation_pair_count": len(validation),
                           "rejected_pairs": group["rejected_pairs"], "image_warnings": [], "warnings": []}
            result["pairs"].append(pair_result)
            if len(training) < opts["min_views"] or not validation:
                reason = f"Need >= {opts['min_views']} training pairs plus held-out pairs; got {len(training)} + {len(validation)}"
                pair_result["reason"] = reason
                print(f"{left} -> {right}: FAIL: {reason}")
                result["passed"] = False
                continue
            try:
                pair_cameras = (cameras[left], cameras[right])
                sdk_left = raw.get("sdk_reference_source_label")
                sdk_right = raw.get("sdk_reference_target_label")
                prior = raw.get("sdk_transform_ir_left_to_ir_right") if left == sdk_left and right == sdk_right else None
                if left == sdk_right and right == sdk_left and raw.get("sdk_transform_ir_left_to_ir_right") is not None:
                    prior = np.linalg.inv(np.asarray(raw["sdk_transform_ir_left_to_ir_right"], np.float64))
                if prior is None:
                    for candidate in raw.get("sdk_transforms", []):
                        if candidate.get("left_label") == left and candidate.get("right_label") == right:
                            prior = candidate["transform"]
                initial, ordering = _initial_transform(objects, training, pair_cameras, pattern, prior)
                pair_result["ordering"] = ordering
                if ordering["global_ambiguity"]:
                    raise ValueError("Different corner-order hypotheses have indistinguishable training geometry; collect more diverse poses or use an identified board")
                print(f"{left} -> {right}: corner ordering: {ordering['method']}")
                ordered = [_select_order(objects, view, pair_cameras, initial, pattern) for view in training]
                for view in ordered:
                    if view["ordering_ambiguous"]:
                        pair_result["rejected_pairs"].append({"left_image": view["left_image"],
                                                             "right_image": view["right_image"], "reason": "ambiguous_corner_order"})
                        print(f"{left} -> {right}: {view['left_image']} / {view['right_image']}: skipped pair: ambiguous_corner_order")
                ordered = [view for view in ordered if not view["ordering_ambiguous"]]
                if len(ordered) < opts["min_views"]:
                    raise ValueError("Too few unambiguous training pairs")
                for _ in range(3):
                    rms, transform = _fit_stereo(objects, ordered, pair_cameras)
                    revised = [_select_order(objects, view, pair_cameras, transform, pattern) for view in ordered]
                    if any(view["ordering_ambiguous"] for view in revised):
                        raise ValueError("Fitted extrinsics leave corner order ambiguous")
                    if all(a["right_order_rotation_deg"] == b["right_order_rotation_deg"] for a, b in zip(ordered, revised)):
                        ordered = revised
                        break
                    ordered = revised
                else:
                    raise ValueError("Corner ordering did not converge")
                for view in ordered:
                    if view["right_order_rotation_deg"]:
                        pair_result["image_warnings"].append({"left_image": view["left_image"],
                            "right_image": view["right_image"], "stage": "外参拟合",
                            "message": f"第二相机角点编号旋转 {view['right_order_rotation_deg']} 度后参与拟合"})
                        print(f"{left} -> {right}: {view['right_image']}: corner order rotated {view['right_order_rotation_deg']} degrees")
                errors = []
                for original in validation:
                    view = _select_order(objects, original, pair_cameras, transform, pattern)
                    errors.append({"capture_index": view["capture_index"], "left_image": view["left_image"],
                                   "right_image": view["right_image"],
                                   "right_order_rotation_deg": view["right_order_rotation_deg"],
                                   "right_reprojection_rmse_px": view["ordering_best_rmse_px"],
                                   "ordering_ambiguous": view["ordering_ambiguous"]})
                    if view["ordering_ambiguous"]:
                        print(f"{left} -> {right}: validation {view['right_image']}: ambiguous_corner_order")
                    elif view["ordering_best_rmse_px"] > opts["max_validation_rmse_px"]:
                        print(f"{left} -> {right}: validation {view['right_image']}: residual {view['ordering_best_rmse_px']:.6f} px")
                summary = _summary(errors, "right_reprojection_rmse_px")
                excluded_errors = []
                for rejected in pair_result["rejected_pairs"]:
                    if not rejected["reason"].startswith("intrinsic_training_outlier:"):
                        continue
                    original = {**rejected,
                                "left": all_observations[left][rejected["left_image"]],
                                "right": all_observations[right][rejected["right_image"]]}
                    try:
                        view = _select_order(objects, original, pair_cameras, transform, pattern)
                        excluded_errors.append({"left_image": rejected["left_image"],
                                                "right_image": rejected["right_image"],
                                                "right_reprojection_rmse_px": view["ordering_best_rmse_px"],
                                                "ordering_ambiguous": view["ordering_ambiguous"], "status": "ok"})
                    except (ValueError, cv2.error) as error:
                        excluded_errors.append({"left_image": rejected["left_image"],
                                                "right_image": rejected["right_image"],
                                                "status": "pose_failed", "reason": str(error)})
                pair_result["excluded_validation"] = _summary(excluded_errors, "right_reprojection_rmse_px")
                pair_result["excluded_validation_views"] = excluded_errors
                passed = bool(rms <= opts["max_rms_px"] and summary["max_rmse_px"] <= opts["max_validation_rmse_px"]
                              and not any(entry["ordering_ambiguous"] for entry in errors))
                baseline = float(np.linalg.norm(transform[:3, 3]))
                pair_result.update({"passed": passed, "baseline_mm": baseline, "stereo_rms_px": rms,
                                    "training_pair_count_used": len(ordered), "validation": summary,
                                    "validation_views": errors,
                                    "training_views": [{"left_image": view["left_image"], "right_image": view["right_image"],
                                                        "right_order_rotation_deg": view["right_order_rotation_deg"]}
                                                       for view in ordered]})
                if not passed:
                    pair_result["reason"] = "Stereo training RMS, held-out residual or held-out corner ordering failed the configured quality checks"
                    print(f"{left} -> {right}: FAIL: {pair_result['reason']}")
                if any(view["timestamp_basis"] not in {"device", "sensor_exposure"} for view in group["views"]):
                    warning = "Host/unknown timestamps do not guarantee simultaneous sensor exposure; the board and rig must be stationary when each pair is saved."
                    pair_result["warnings"].append(warning)
                    for view in group["views"]:
                        if view["timestamp_basis"] not in {"device", "sensor_exposure"}:
                            pair_result["image_warnings"].append({"left_image": view["left_image"],
                                "right_image": view["right_image"],
                                "stage": "外参拟合" if view["split"] == "training" else "外参留出验证",
                                "message": "曝光同步无法确认；该配对要求棋盘和相机保持静止"})
                    print(f"{left} -> {right}: WARNING: {warning}")
                outputs.append((output_dir / f"pair_{left}_to_{right}_extrinsics.yml", {
                    "R": np.asarray(transform[:3, :3], np.float64).reshape(3, 3),
                    "T": np.asarray(transform[:3, 3:], np.float64).reshape(3, 1), "baseline_mm": baseline}))
                result["passed"] = result["passed"] and passed
            except (ValueError, RuntimeError, cv2.error) as error:
                pair_result["reason"] = str(error)
                print(f"{left} -> {right}: FAIL: {error}")
                result["passed"] = False
    if result["passed"]:
        _save_bundle(outputs)
        result["output_files"] = [str(path) for path, _ in outputs]
    else:
        print("Calibration failed quality checks; no result YAML files were written.")
    save_quality_report(output_dir, result)
    return result
