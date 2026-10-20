"""Read-only validation of saved intrinsics and extrinsics on another capture."""

from pathlib import Path

import cv2
import numpy as np

from ..sensor_intrinsics.core import verify_observations
from ..stereo_calibration.core import _detect_image, _image_path, _load_manifest, _options, _select_order, _summary


def _read_yaml(path, keys):
    if not path.is_file():
        raise FileNotFoundError(f"Missing calibration for this sensor label/direction: {path}")
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"Cannot read OpenCV calibration YAML: {path}")
    try:
        values = {}
        for key in keys:
            node = storage.getNode(key)
            if node.empty():
                raise ValueError(f"Missing {key} in {path}")
            values[key] = node.mat() if node.isMap() else node.real()
        return values
    finally:
        storage.release()


def _camera(path):
    camera = _read_yaml(path, ("image_width", "image_height", "camera_matrix", "distortion_coefficients"))
    dimensions = [camera["image_width"], camera["image_height"]]
    if any(not np.isfinite(value) or value <= 0 or value != int(value) for value in dimensions):
        raise ValueError(f"Invalid image dimensions in {path}")
    matrix = np.asarray(camera["camera_matrix"], np.float64)
    distortion = np.asarray(camera["distortion_coefficients"], np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-9)
            or distortion.ndim != 2 or 1 not in distortion.shape or distortion.size not in (4, 5, 8, 12, 14)
            or not np.isfinite(distortion).all()):
        raise ValueError(f"Invalid K or distortion coefficients in {path}")
    return {"image_size": tuple(int(value) for value in dimensions), "camera_matrix": matrix,
            "distortion_coefficients": distortion.reshape(-1)}


def _transform(path):
    values = _read_yaml(path, ("R", "T", "baseline_mm"))
    rotation = np.asarray(values["R"], np.float64)
    translation = np.asarray(values["T"], np.float64)
    baseline = values["baseline_mm"]
    if (rotation.shape != (3, 3) or translation.shape != (3, 1)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
            or not np.isfinite(baseline) or baseline < 0
            or not np.isclose(baseline, np.linalg.norm(translation), rtol=1e-6, atol=1e-6)):
        raise ValueError(f"Invalid R/T or baseline differs from norm(T) in {path}")
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3:] = rotation, translation
    return transform, float(baseline)


def verify_session(session, calibration_dir, options, mode):
    """Evaluate all usable images with fixed saved K/D/R/T; write no files."""
    session, calibration_dir = Path(session).resolve(), Path(calibration_dir).resolve()
    opts = _options(options)
    if mode not in {"sensor_intrinsics", "sensor_extrinsics"}:
        raise ValueError("Unsupported verification mode")
    raw, _, sensors, images, captures, pattern, objects, square = _load_manifest(session)
    labels = [sensor["label"] for sensor in sensors]
    if mode == "sensor_extrinsics" and len(labels) != 2:
        raise ValueError("Extrinsic verification requires exactly two sensors")
    cameras = {label: _camera(calibration_dir / f"sensor_{label}_intrinsics.yml") for label in labels}
    if mode == "sensor_extrinsics":
        transform, baseline = _transform(calibration_dir / f"pair_{labels[0]}_to_{labels[1]}_extrinsics.yml")
    result = {"passed": True, "sensors": {}, "pairs": [], "output_files": []}
    observations = {}
    limit = opts["max_validation_rmse_px"]
    for label in labels:
        accepted, seen = {}, set()
        for relative in images[label]:
            path, relative = _image_path(session, relative)
            if relative in seen:
                print(f"{label}: {relative}: skipped verification image: duplicate_manifest_image")
                continue
            seen.add(relative)
            observation, audit = _detect_image(path, pattern, opts["detect_max_dim"])
            actual_size = observation.image_size if observation is not None else tuple(audit.get("image_size", ()))
            if actual_size and actual_size != cameras[label]["image_size"]:
                raise ValueError(f"{label}: {relative}: resolution {actual_size} differs from saved intrinsics {cameras[label]['image_size']}")
            if observation is None:
                print(f"{label}: {relative}: skipped verification image: {audit['status']} ({audit.get('detail', audit.get('detector', ''))})")
            else:
                accepted[relative] = observation
        observations[label] = accepted
        camera = cameras[label]
        entries, _, _ = verify_observations(list(accepted.values()), pattern, square,
                                           camera["camera_matrix"], camera["distortion_coefficients"])
        summary = _summary(entries, "view_rms_px")
        passed = bool(len(accepted) >= 3 and summary["count"] == len(accepted)
                      and summary["max_rmse_px"] is not None and summary["max_rmse_px"] <= limit)
        result["sensors"][label] = {"valid_image_count": len(accepted), "validation": summary, "passed": passed}
        if not passed:
            result["sensors"][label]["reason"] = "Need >=3 valid images, successful board poses and all fixed-intrinsic residuals within the configured limit"
        for entry in entries:
            print(f"{label}: verify {Path(entry['path']).name}: {entry.get('view_rms_px', entry.get('status'))} px")
        result["passed"] = result["passed"] and passed
    if mode == "sensor_extrinsics":
        left, right = labels
        errors, seen = [], set()
        static_capture = raw.get("capture_mode") == "static_board_sequential_reads"
        synchronized_capture = raw.get("capture_mode") in {"synchronized_ir", "synchronized_raw_ir"}
        if static_capture:
            print("WARNING: Separate-device exposures are not synchronized. Keep the board and both sensors stationary when each pair is captured.")
        for capture in captures:
            if not all(label in capture["images"] for label in labels):
                print(f"{left} -> {right}: skipped verification pair: missing_sensor_image")
                continue
            lp, rp = (_image_path(session, capture["images"][label])[1] for label in labels)
            reason = None
            if (lp, rp) in seen:
                reason = "duplicate_pair"
            elif lp == rp:
                reason = "same_image_for_both_sensors"
            elif lp not in observations[left] or rp not in observations[right]:
                reason = "missing_sensor_observation"
            seen.add((lp, rp))
            delta = capture.get("capture_delta_ms")
            if reason is None and not static_capture and not synchronized_capture:
                reason = "unknown_capture_timing_mode"
            if reason is None and synchronized_capture and capture.get("timestamp_basis") not in {"device", "sensor_exposure"}:
                reason = "unsupported_synchronized_timestamp_basis"
            if reason is None and not (delta is None and static_capture):
                if delta is None or not np.isfinite(delta) or delta < 0:
                    reason = "invalid_or_missing_timestamp_difference"
                elif delta > opts["max_pair_delta_ms"] and not static_capture:
                    reason = "timestamp_difference_exceeds_limit"
            if reason:
                print(f"{left} -> {right}: {lp} / {rp}: skipped verification pair: {reason}")
                continue
            view = {"left": observations[left][lp], "right": observations[right][rp]}
            try:
                ordered = _select_order(objects, view, (cameras[left], cameras[right]), transform, pattern)
            except (ValueError, cv2.error) as error:
                print(f"{left} -> {right}: verify {lp} / {rp}: projection_failed: {error}")
                errors.append({"projection_failed": True})
                continue
            error = ordered["ordering_best_rmse_px"]
            errors.append({"right_reprojection_rmse_px": error, "ordering_ambiguous": ordered["ordering_ambiguous"]})
            print(f"{left} -> {right}: verify {lp} / {rp}: {error:.6f} px; order rotation={ordered['right_order_rotation_deg']} degrees; ambiguous={ordered['ordering_ambiguous']}")
        summary = _summary(errors, "right_reprojection_rmse_px")
        passed = bool(len(errors) >= 3 and summary["count"] == len(errors) and summary["max_rmse_px"] is not None
                      and summary["max_rmse_px"] <= limit and not any(entry.get("ordering_ambiguous", False) for entry in errors))
        pair = {"left_sensor": left, "right_sensor": right, "baseline_mm": baseline, "validation": summary, "passed": passed}
        if not passed:
            pair["reason"] = "Need >=3 usable pairs, unambiguous corner ordering and all fixed-extrinsic cross-projection residuals within the configured limit"
        result["pairs"].append(pair)
        result["passed"] = result["passed"] and passed
    return result
