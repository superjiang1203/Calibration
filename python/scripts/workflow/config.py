"""Validate mode-independent sensor profiles and calibration settings."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "calibration.json"
MODES = ("sensor_intrinsics", "sensor_extrinsics")
_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_PROFILE_KEYS = {
    "serial_number", "ip_address", "schema_version", "sensor_type", "sensor_index", "stream", "width", "height",
    "acquisition_fps", "exposure_auto", "exposure_us", "exposure_sdk",
    "gain_auto", "gain_db", "gain_sdk", "gamma_enabled", "offset_x", "offset_y",
    "pixel_format", "laser_mode", "white_balance_mode", "white_balance_kelvin",
    "white_balance_warmup_frames", "galaxy_backend", "gxipy_path", "notes",
    "_comment", "_comments",
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unexpected = set(value) - allowed
    if unexpected:
        raise ValueError(f"Unknown {name} key(s): {', '.join(sorted(unexpected))}")
    if "_comment" in value and not isinstance(value["_comment"], str):
        raise ValueError(f"{name} _comment must be a string")
    if "_comments" in value:
        comments = _object(value["_comments"], f"{name} _comments")
        if any(not isinstance(text, str) for text in comments.values()):
            raise ValueError(f"{name} comments must be strings")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, minimum: float = 0, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum or (positive and value == minimum)):
        comparison = ">" if positive else ">="
        raise ValueError(f"{name} must be a finite number {comparison} {minimum}")
    return float(value)


def _read_json(path: Path) -> dict[str, Any]:
    return _object(json.loads(path.read_text(encoding="utf-8-sig")), str(path))


def _resolve_path(value: Any, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _sensor(entry: Any, config_dir: Path) -> dict[str, Any]:
    entry = _object(entry, "sensor")
    _keys(entry, {"label", "sensor_config", "_comment"}, "sensor")
    label = entry.get("label")
    if not isinstance(label, str) or not _LABEL.fullmatch(label):
        raise ValueError("Sensor label must start with an ASCII letter and use letters, digits, _ or - (max 64 characters)")
    profile_path = _resolve_path(entry.get("sensor_config"), config_dir, "sensor_config")
    profile = _read_json(profile_path)
    _keys(profile, _PROFILE_KEYS, f"sensor profile {label}")
    if type(profile.get("schema_version")) is not int or profile["schema_version"] != 1:
        raise ValueError(f"Sensor profile {label} schema_version must be 1")
    sensor_type = profile.get("sensor_type")
    if sensor_type not in ("orbbec", "galaxy", "opencv", "vcamera"):
        raise ValueError(f"Sensor {label} sensor_type must be orbbec, galaxy, opencv or vcamera")
    for selector in ("serial_number", "ip_address"):
        if selector in profile and (sensor_type != "vcamera" or not isinstance(profile[selector], str)):
            raise ValueError(f"{label}.{selector} must be a string and is only available for Vcamera")
    if sensor_type == "vcamera":
        allowed = {"schema_version", "sensor_type", "sensor_index", "stream", "width", "height",
                   "serial_number", "ip_address", "acquisition_fps", "notes", "_comment", "_comments"}
        _keys(profile, allowed, f"Vcamera profile {label}")
        if profile.get("stream") != "color":
            raise ValueError(f"{label}: Vcamera currently supports raw color texture only")
        if profile.get("acquisition_fps", 0) != 0:
            raise ValueError(f"{label}: Vcamera acquisition_fps must be 0 (device continuous rate)")
    stream = profile.get("stream")
    if stream not in ("color", "gray", "ir_left", "ir_right"):
        raise ValueError(f"Sensor {label} stream must be color, gray, ir_left or ir_right")
    if sensor_type != "orbbec" and stream.startswith("ir_"):
        raise ValueError(f"Sensor {label}: named left/right IR streams require Orbbec SDK")
    if sensor_type == "orbbec" and stream == "gray":
        raise ValueError(f"Sensor {label}: select Orbbec color, ir_left or ir_right explicitly")
    if sensor_type == "galaxy" and stream != "gray":
        raise ValueError(f"Sensor {label}: current Galaxy backend supports a Mono8 gray sensor")
    _integer(profile.get("sensor_index"), f"{label}.sensor_index", 1 if sensor_type == "galaxy" else 0)
    for key in ("width", "height"):
        _integer(profile.get(key), f"{label}.{key}", 1 if sensor_type == "orbbec" else 0)
    fps = _number(profile.get("acquisition_fps", 0), f"{label}.acquisition_fps")
    if sensor_type == "orbbec" and (fps <= 0 or not fps.is_integer()):
        raise ValueError(f"{label}.acquisition_fps must be a positive integer for Orbbec")
    profile["acquisition_fps"] = fps
    for key in ("exposure_auto", "gain_auto", "gamma_enabled"):
        if key in profile and profile[key] is not False:
            raise ValueError(f"{label}.{key} must be false during calibration")
    if sensor_type == "orbbec":
        if profile.get("exposure_us") is not None:
            raise ValueError(f"{label}: Orbbec requires native exposure_sdk, not exposure_us")
        if profile.get("exposure_sdk") is not None:
            _integer(profile["exposure_sdk"], f"{label}.exposure_sdk", 1)
        for key in ("gain_db", "offset_x", "offset_y", "galaxy_backend", "gxipy_path"):
            if key in profile:
                raise ValueError(f"{label}.{key} is only available for Galaxy sensors")
        if stream.startswith("ir_"):
            _integer(profile.get("exposure_sdk"), f"{label}.exposure_sdk", 1)
            for key in ("white_balance_mode", "white_balance_kelvin", "white_balance_warmup_frames"):
                if key in profile:
                    raise ValueError(f"{label}: white balance applies only to the color stream")
            _integer(profile.get("gain_sdk", 16), f"{label}.gain_sdk")
            profile.setdefault("gain_sdk", 16)
            _integer(profile.get("laser_mode", 0), f"{label}.laser_mode")
            if profile.get("laser_mode", 0) not in (0, 1, 2):
                raise ValueError(f"{label}.laser_mode must be 0, 1 or 2")
            profile.setdefault("laser_mode", 0)
            if profile.get("pixel_format", "Y8") not in ("Y8", "Y16"):
                raise ValueError(f"{label}.pixel_format must be Y8 or Y16 for IR")
            profile.setdefault("pixel_format", "Y8")
        else:
            if "pixel_format" in profile:
                raise ValueError(f"{label}: color pixel format is negotiated by the SDK adapter; pixel_format is configurable for IR and Galaxy")
            if any(key in profile for key in ("gain_sdk", "laser_mode")):
                raise ValueError(f"{label}: gain_sdk and laser_mode belong to the IR streams")
            wb_mode = profile.get("white_balance_mode", "auto_lock")
            if wb_mode not in ("auto_lock", "auto", "manual"):
                raise ValueError(f"{label}.white_balance_mode must be auto_lock, auto or manual")
            temperature = profile.get("white_balance_kelvin")
            if wb_mode == "manual":
                _integer(temperature, f"{label}.white_balance_kelvin", 1)
            elif temperature is not None:
                raise ValueError(f"{label}.white_balance_kelvin requires manual white balance")
            warmup = _integer(profile.get("white_balance_warmup_frames", 30), f"{label}.white_balance_warmup_frames", 5)
            if warmup > 120:
                raise ValueError(f"{label}.white_balance_warmup_frames must be <= 120")
            profile.setdefault("white_balance_mode", wb_mode)
            profile.setdefault("white_balance_kelvin", temperature)
            profile.setdefault("white_balance_warmup_frames", warmup)
    elif sensor_type == "galaxy":
        if any(key in profile for key in ("exposure_sdk", "gain_sdk", "laser_mode", "white_balance_mode", "white_balance_kelvin", "white_balance_warmup_frames")):
            raise ValueError(f"{label}: Orbbec controls cannot be used by Galaxy")
        if profile.get("exposure_us") is not None:
            _number(profile["exposure_us"], f"{label}.exposure_us", positive=True)
        _number(profile.get("gain_db", 0), f"{label}.gain_db")
        for key in ("offset_x", "offset_y"):
            if profile.get(key) is not None:
                _integer(profile[key], f"{label}.{key}")
        if profile.get("pixel_format", "Mono8") != "Mono8":
            raise ValueError(f"{label}.pixel_format must be Mono8")
        if profile.get("galaxy_backend", "gxipy") not in ("gxipy", "ctypes", "auto"):
            raise ValueError(f"{label}.galaxy_backend must be gxipy, ctypes or auto")
        if profile.get("gxipy_path") is not None:
            profile["gxipy_path"] = str(_resolve_path(profile["gxipy_path"], profile_path.parent, f"{label}.gxipy_path"))
    else:
        unsupported = {"gain_sdk", "laser_mode", "gain_db", "offset_x", "offset_y", "pixel_format", "galaxy_backend", "gxipy_path", "white_balance_mode", "white_balance_kelvin", "white_balance_warmup_frames"}
        if set(profile) & unsupported or profile.get("exposure_us") is not None or profile.get("exposure_sdk") is not None:
            raise ValueError(f"{label}: OpenCV exposure controls are backend-specific; use the vendor SDK for controlled exposure")
    profile.update(label=label, sensor_config=str(profile_path))
    return profile


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load JSON; profile references resolve relative to the workflow file."""
    config_path = DEFAULT_CONFIG if path is None else Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    raw = _read_json(config_path)
    _keys(raw, {"schema_version", "mode", "board", "sensors", "capture", "calibration", "_comment", "_comments"}, "workflow")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 2:
        raise ValueError("Workflow schema_version must be 2")
    mode = raw.get("mode")
    if mode not in MODES:
        raise ValueError("mode must be sensor_intrinsics or sensor_extrinsics; stereo is two-sensor extrinsic calibration")
    board = _object(raw.get("board"), "board")
    _keys(board, {"pattern_cols", "pattern_rows", "square_size_mm", "_comment"}, "board")
    for key in ("pattern_cols", "pattern_rows"):
        _integer(board.get(key), f"board.{key}", 2)
    _number(board.get("square_size_mm"), "board.square_size_mm", positive=True)
    entries = raw.get("sensors")
    if not isinstance(entries, list) or len(entries) not in (1, 2):
        raise ValueError("sensors must contain one or two sensor entries")
    if mode == "sensor_extrinsics" and len(entries) != 2:
        raise ValueError("sensor_extrinsics requires exactly two sensors")
    sensors = [_sensor(entry, config_path.parent) for entry in entries]
    if len({sensor["label"].casefold() for sensor in sensors}) != len(sensors):
        raise ValueError("Sensor labels must be unique (case-insensitive)")
    identities = [(s["sensor_type"], (s.get("serial_number") or s.get("ip_address") or s["sensor_index"]) if s["sensor_type"] == "vcamera" else s["sensor_index"], s["stream"] if s["sensor_type"] == "orbbec" else None) for s in sensors]
    if len(set(identities)) != len(identities):
        raise ValueError("The same sensor stream cannot be listed twice")
    if len(sensors) == 2 and all(s["sensor_type"] == "orbbec" for s in sensors) and sensors[0]["sensor_index"] == sensors[1]["sensor_index"]:
        if {s["stream"] for s in sensors} != {"ir_left", "ir_right"}:
            raise ValueError("One Orbbec device currently supports a paired ir_left/ir_right session; color/IR simultaneous capture is not implemented")
        for key in ("width", "height", "acquisition_fps", "exposure_sdk", "gain_sdk", "pixel_format", "laser_mode"):
            if sensors[0].get(key) != sensors[1].get(key):
                raise ValueError(f"Shared-device IR pair requires matching {key}")
    capture_defaults = dict(target_count=40, preview_max_width=1400, preview_max_height=900,
                            timeout_ms=1500, min_interval_seconds=1.0, max_pair_delta_ms=5.0)
    capture = _object(raw.get("capture", {}), "capture")
    _keys(capture, set(capture_defaults) | {"_comment", "_comments"}, "capture")
    capture_defaults.update(capture)
    for key in ("target_count", "preview_max_width", "preview_max_height", "timeout_ms"):
        _integer(capture_defaults[key], f"capture.{key}", 0 if key == "target_count" else 1)
    _number(capture_defaults["min_interval_seconds"], "capture.min_interval_seconds")
    _number(capture_defaults["max_pair_delta_ms"], "capture.max_pair_delta_ms", positive=True)
    calibration_defaults = dict(model="standard", min_views=15, validation_stride=5, detect_max_dim=1600,
                                max_rms_px=1.0, max_validation_rmse_px=2.0, reject_outliers=True,
                                max_view_rmse_px=1.0, max_outlier_fraction=0.2, fixed_intrinsics={},
                                use_known_intrinsics=None)
    calibration = _object(raw.get("calibration", {}), "calibration")
    _keys(calibration, set(calibration_defaults) | {"_comment", "_comments"}, "calibration")
    calibration_defaults.update(calibration)
    if "use_known_intrinsics" in calibration and type(calibration["use_known_intrinsics"]) is not bool:
        raise ValueError("calibration.use_known_intrinsics must be a boolean")
    known = calibration_defaults["use_known_intrinsics"]
    if mode == "sensor_intrinsics" and known is True:
        raise ValueError("calibration.use_known_intrinsics=true requires mode sensor_extrinsics")
    fixed = _object(calibration_defaults["fixed_intrinsics"], "calibration.fixed_intrinsics")
    if set(fixed) - {s["label"] for s in sensors}:
        raise ValueError("fixed_intrinsics contains an unknown sensor label")
    calibration_defaults["fixed_intrinsics"] = {
        label: str(_resolve_path(value, config_path.parent, f"fixed_intrinsics.{label}"))
        for label, value in fixed.items()}
    if mode == "sensor_extrinsics" and known is not False:
        for sensor in sensors:
            label = sensor["label"]
            fixed_path = calibration_defaults["fixed_intrinsics"].get(label)
            if fixed_path:
                # Fail before opening devices if the selected model is missing,
                # malformed, or belongs to another configured image resolution.
                from ..validation.stereo import _camera
                camera = _camera(Path(fixed_path))
                size = (sensor["width"], sensor["height"])
                if all(size) and camera["image_size"] != size:
                    raise ValueError(f"Known model resolution {camera['image_size']} differs from {label} profile {size}")
            elif known is True and sensor["sensor_type"] not in {"orbbec", "vcamera"}:
                raise ValueError(f"Known intrinsics required for {label}: set calibration.fixed_intrinsics.{label} to a measured OpenCV YAML file")
    if calibration_defaults["model"] not in ("standard", "rational"):
        raise ValueError("calibration.model must be standard or rational")
    _integer(calibration_defaults["min_views"], "calibration.min_views", 3)
    _integer(calibration_defaults["validation_stride"], "calibration.validation_stride", 2)
    _integer(calibration_defaults["detect_max_dim"], "calibration.detect_max_dim", 1)
    for key in ("max_rms_px", "max_validation_rmse_px", "max_view_rmse_px"):
        _number(calibration_defaults[key], f"calibration.{key}", positive=True)
    if type(calibration_defaults["reject_outliers"]) is not bool:
        raise ValueError("calibration.reject_outliers must be a boolean")
    fraction = _number(calibration_defaults["max_outlier_fraction"], "calibration.max_outlier_fraction", positive=True)
    if fraction > 0.5:
        raise ValueError("calibration.max_outlier_fraction must be <= 0.5")
    return dict(schema_version=2, mode=mode, board=board, sensors=sensors,
                capture=capture_defaults, calibration=calibration_defaults,
                _config_path=str(config_path))
