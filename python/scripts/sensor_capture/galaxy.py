"""galaxy Python SDK frame adapter."""
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


from .base import FrameSource

def _align_int(value: int, value_range: dict[str, Any]) -> int:
    minimum = int(value_range.get("min", 0))
    maximum = int(value_range["max"])
    increment = max(1, int(value_range.get("inc", 1)))
    clamped = min(maximum, max(minimum, int(value)))
    return minimum + ((clamped - minimum) // increment) * increment


def _install_numpy_compat_long_shim() -> None:
    """Support Galaxy SDK gxipy releases predating NumPy 2.x.

    Galaxy SDK 2.6's Python sample imports ``numpy.compat.long`` even though
    NumPy 2.x removed that module.  The SDK only needs Python's integer type.
    This in-memory compatibility module does not modify NumPy or vendor files.
    """

    try:
        compat = importlib.import_module("numpy.compat")
    except ModuleNotFoundError:
        compat = types.ModuleType("numpy.compat")
        sys.modules["numpy.compat"] = compat
    if not hasattr(compat, "long"):
        compat.long = int  # type: ignore[attr-defined]


def _find_gxipy_samples(explicit_path: str | None) -> Path | None:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    development = os.environ.get("GALAXY_SDK_DEVELOPMENT")
    if development:
        candidates.append(Path(development) / "Samples" / "Python")
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.append(
            Path(program_files)
            / "Daheng Imaging"
            / "GalaxySDK"
            / "Development"
            / "Samples"
            / "Python"
        )
    for candidate in candidates:
        if (candidate / "gxipy" / "__init__.py").is_file():
            return candidate.resolve()
    return None


def import_gxipy(explicit_path: str | None = None):
    _install_numpy_compat_long_shim()
    if importlib.util.find_spec("gxipy") is None:
        samples = _find_gxipy_samples(explicit_path)
        if samples is None:
            raise ImportError(
                "gxipy was not found. Install Galaxy SDK or pass "
                "--gxipy-path <GalaxySDK/Development/Samples/Python>."
            )
        sys.path.insert(0, str(samples))
    return importlib.import_module("gxipy")


class GxipyGalaxySource(FrameSource):
    """Official Galaxy SDK Python backend, primarily for native Windows."""

    backend_name = "galaxy-gxipy"

    def __init__(
        self,
        *,
        sensor_index: int,
        width: int,
        height: int,
        offset_x: int | None,
        offset_y: int | None,
        exposure_us: float | None,
        gain_db: float,
        acquisition_fps: float,
        gxipy_path: str | None,
    ) -> None:
        self.sensor_index = int(sensor_index)
        self.requested_width = int(width)
        self.requested_height = int(height)
        self.requested_offset_x = offset_x
        self.requested_offset_y = offset_y
        self.exposure_us = None if exposure_us is None else float(exposure_us)
        self.gain_db = float(gain_db)
        self.acquisition_fps = float(acquisition_fps)
        self.gxipy_path = gxipy_path
        self.gx = None
        self.manager = None
        self.camera = None
        self.feature_control = None
        self.device_info: dict[str, Any] = {}
        self._streaming = False
        self._configuration_verified = False
        self._configuration_warnings: list[str] = []
        self._expected_roi: dict[str, int] = {}

    def _is_implemented(self, name: str) -> bool:
        try:
            return bool(self.feature_control.is_implemented(name))
        except Exception:
            return False

    def _is_writable(self, name: str) -> bool:
        try:
            return bool(self.feature_control.is_writable(name))
        except Exception:
            return False

    def _enum(self, name: str):
        return self.feature_control.get_enum_feature(name)

    def _integer(self, name: str):
        return self.feature_control.get_int_feature(name)

    def _floating(self, name: str):
        return self.feature_control.get_float_feature(name)

    def _boolean(self, name: str):
        return self.feature_control.get_bool_feature(name)

    def _set_enum_if_writable(self, name: str, value: str) -> None:
        if self._is_writable(name):
            self._enum(name).set(value)

    def _set_float_if_writable(self, name: str, value: float) -> None:
        if self._is_writable(name):
            feature = self._floating(name)
            value_range = feature.get_range()
            requested = float(value)
            clipped = min(float(value_range["max"]), max(float(value_range["min"]), requested))
            if clipped != requested:
                self._configuration_warnings.append(
                    f"{name} requested {requested} was clipped to {clipped}"
                )
            feature.set(float(clipped))

    def _verify_configuration(self) -> None:
        def require_enum(name: str, expected: str, *, optional: bool = False) -> None:
            if not self._is_implemented(name):
                if optional:
                    return
                raise RuntimeError(f"Galaxy camera does not expose required node {name}")
            actual = self._read_feature("enum", name)
            if actual is None or str(actual).casefold() != expected.casefold():
                raise RuntimeError(
                    f"Galaxy camera {name} is {actual!r}; calibration requires {expected!r}"
                )

        def require_float(
            name: str, expected: float | None, *, positive: bool = False
        ) -> None:
            if not self._is_implemented(name):
                raise RuntimeError(f"Galaxy camera does not expose required node {name}")
            actual = self._read_feature("float", name)
            try:
                numeric = float(actual)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Galaxy camera {name} cannot be read") from exc
            if not math.isfinite(numeric) or (positive and numeric <= 0):
                raise RuntimeError(f"Galaxy camera {name} has invalid value {actual!r}")
            if expected is not None and not math.isclose(
                numeric, float(expected), rel_tol=1e-6, abs_tol=1e-6
            ):
                raise RuntimeError(
                    f"Galaxy camera {name} is {numeric}, requested {float(expected)}; "
                    "the requested calibration setting was not applied"
                )

        def require_int(name: str, expected: int) -> None:
            if not self._is_implemented(name):
                raise RuntimeError(f"Galaxy camera does not expose required node {name}")
            actual = self._read_feature("int", name)
            if actual is None or int(actual) != int(expected):
                raise RuntimeError(
                    f"Galaxy camera {name} is {actual!r}, expected {int(expected)}; "
                    "the requested ROI was not applied"
                )

        require_enum("AcquisitionMode", "Continuous")
        require_enum("TriggerMode", "Off")
        require_enum("PixelFormat", "Mono8")
        require_enum("ExposureAuto", "Off", optional=True)
        require_enum("GainAuto", "Off", optional=True)
        if self._is_implemented("GammaEnable"):
            gamma_enabled = self._read_feature("bool", "GammaEnable")
            if gamma_enabled is None or bool(gamma_enabled):
                raise RuntimeError(
                    f"Galaxy camera GammaEnable is {gamma_enabled!r}; "
                    "calibration requires False"
                )
        require_float("ExposureTime", self.exposure_us, positive=True)
        require_float("Gain", self.gain_db)
        expected_fps_mode = "On" if self.acquisition_fps > 0 else "Off"
        require_enum(
            "AcquisitionFrameRateMode", expected_fps_mode, optional=True
        )
        if self.acquisition_fps > 0:
            require_float("AcquisitionFrameRate", self.acquisition_fps, positive=True)
        for name, expected in self._expected_roi.items():
            require_int(name, expected)

    def _set_roi(self) -> None:
        self._expected_roi = {}
        # Offsets must be reset before expanding Width/Height to their maxima.
        for name in ("OffsetX", "OffsetY"):
            if self._is_writable(name):
                feature = self._integer(name)
                feature.set(_align_int(0, feature.get_range()))
            elif self._is_implemented(name):
                actual = self._read_feature("int", name)
                if actual is None or int(actual) != 0:
                    raise RuntimeError(
                        f"Galaxy camera {name} cannot be reset to zero; "
                        "the requested ROI extent cannot be applied safely"
                    )

        width_feature = self._integer("Width")
        height_feature = self._integer("Height")
        width_range = width_feature.get_range()
        height_range = height_feature.get_range()
        requested_width = self.requested_width or int(width_range["max"])
        requested_height = self.requested_height or int(height_range["max"])
        width = _align_int(requested_width, width_range)
        height = _align_int(requested_height, height_range)
        if width != requested_width or height != requested_height:
            self._configuration_warnings.append(
                f"ROI {requested_width}x{requested_height} aligned to {width}x{height}"
            )
        width_feature.set(width)
        height_feature.set(height)
        self._expected_roi.update({"Width": width, "Height": height})

        for name, requested, extent in (
            ("OffsetX", self.requested_offset_x, width),
            ("OffsetY", self.requested_offset_y, height),
        ):
            if not self._is_implemented(name):
                raise RuntimeError(f"Galaxy camera does not expose required node {name}")
            feature = self._integer(name)
            value_range = feature.get_range()
            if requested is None:
                sensor_extent_name = "WidthMax" if name == "OffsetX" else "HeightMax"
                if self._is_implemented(sensor_extent_name):
                    sensor_extent = int(self._integer(sensor_extent_name).get())
                else:
                    sensor_extent = int(value_range["max"]) + extent
                requested = max(0, (sensor_extent - extent) // 2)
            aligned = _align_int(int(requested), value_range)
            if aligned != int(requested):
                self._configuration_warnings.append(
                    f"{name} requested {requested} aligned to {aligned}"
                )
            if self._is_writable(name):
                feature.set(aligned)
            self._expected_roi[name] = aligned

    def open(self) -> None:
        self._configuration_verified = False
        self._configuration_warnings = []
        if self.sensor_index < 1:
            raise ValueError("Galaxy camera index is 1-based and must be >= 1")
        self.gx = import_gxipy(self.gxipy_path)
        self.manager = self.gx.DeviceManager()
        if hasattr(self.manager, "update_all_device_list"):
            device_count, device_infos = self.manager.update_all_device_list()
        else:
            device_count, device_infos = self.manager.update_device_list()
        # GigE discovery also returns other vendors (e.g. Percipio Vcamera).
        # User indices refer to Daheng devices, not the mixed SDK device list.
        galaxy_devices = [info for info in device_infos
                          if 'daheng' in str(info.get('vendor_name', '')).casefold()]
        if not galaxy_devices:
            raise RuntimeError("Galaxy SDK found no Daheng camera; discovered vendors: "
                               + ', '.join(str(info.get('vendor_name', 'unknown')) for info in device_infos))
        if self.sensor_index > len(galaxy_devices):
            raise RuntimeError(
                f"Galaxy camera index {self.sensor_index} exceeds {len(galaxy_devices)} Daheng device(s)"
            )
        self.device_info = dict(galaxy_devices[self.sensor_index - 1])
        self.camera = self.manager.open_device_by_index(self.device_info['index'])
        try:
            self.feature_control = self.camera.get_remote_device_feature_control()
            self._set_enum_if_writable("AcquisitionMode", "Continuous")
            self._set_enum_if_writable("TriggerMode", "Off")
            self._set_enum_if_writable("PixelFormat", "Mono8")
            self._set_enum_if_writable("ExposureAuto", "Off")
            self._set_enum_if_writable("GainAuto", "Off")
            if self._is_writable("GammaEnable"):
                self._boolean("GammaEnable").set(False)
            self._set_roi()
            if self.exposure_us is not None:
                self._set_float_if_writable("ExposureTime", self.exposure_us)
            self._set_float_if_writable("Gain", self.gain_db)
            if self.acquisition_fps > 0:
                self._set_enum_if_writable("AcquisitionFrameRateMode", "On")
                self._set_float_if_writable(
                    "AcquisitionFrameRate", self.acquisition_fps
                )
            else:
                self._set_enum_if_writable("AcquisitionFrameRateMode", "Off")
            self._verify_configuration()
            self._configuration_verified = True

            data_stream = self.camera.data_stream[0]
            try:
                data_stream.set_acquisition_buffer_number(3)
            except Exception as exc:
                self._configuration_warnings.append(
                    f"could not set acquisition buffer count: {exc}"
                )
            try:
                stream_control = data_stream.get_feature_control()
                if stream_control.is_writable("StreamBufferHandlingMode"):
                    stream_control.get_enum_feature("StreamBufferHandlingMode").set(
                        "NewestOnly"
                    )
            except Exception as exc:
                self._configuration_warnings.append(
                    f"could not select newest-frame buffering: {exc}"
                )
            self.camera.stream_on()
            self._streaming = True
        except BaseException:
            self.close()
            raise

    def read(self, timeout_ms: int) -> np.ndarray | None:
        if self.camera is None or not self._streaming:
            raise RuntimeError("Galaxy camera is not open")
        raw = self.camera.data_stream[0].get_image(timeout=int(timeout_ms))
        if raw is None:
            return None
        if hasattr(raw, "get_status"):
            status = raw.get_status()
            if status != 0:  # GxFrameStatusList.SUCCESS
                return None
        array = raw.get_numpy_array()
        if array is None:
            return None
        return np.asarray(array).copy()

    def _read_feature(self, kind: str, name: str) -> Any:
        if not self._is_implemented(name):
            return None
        try:
            feature = getattr(self.feature_control, f"get_{kind}_feature")(name)
            value = feature.get()
            if isinstance(value, tuple):
                return value[-1]
            return value
        except Exception:
            return None

    def _read_float_range(self, name: str) -> dict[str, Any] | None:
        if not self._is_implemented(name):
            return None
        try:
            value_range = self._floating(name).get_range()
            return {
                str(key): value
                for key, value in value_range.items()
                if key in {"min", "max", "inc", "unit"}
            }
        except Exception:
            return None

    def _read_int_range(self, name: str) -> dict[str, Any] | None:
        if not self._is_implemented(name):
            return None
        try:
            value_range = self._integer(name).get_range()
            return {
                str(key): value
                for key, value in value_range.items()
                if key in {"min", "max", "inc", "unit"}
            }
        except Exception:
            return None

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "backend": self.backend_name,
            "sensor_index_1_based": self.sensor_index,
            "device_info": self.device_info,
            "configuration_verified": self._configuration_verified,
            "configuration_warnings": self._configuration_warnings,
        }
        if self.feature_control is None:
            return metadata
        metadata["features"] = {
            "device_model": self._read_feature("string", "DeviceModelName"),
            "device_serial": self._read_feature("string", "DeviceSerialNumber"),
            "width": self._read_feature("int", "Width"),
            "width_range": self._read_int_range("Width"),
            "height": self._read_feature("int", "Height"),
            "height_range": self._read_int_range("Height"),
            "offset_x": self._read_feature("int", "OffsetX"),
            "offset_x_range": self._read_int_range("OffsetX"),
            "offset_y": self._read_feature("int", "OffsetY"),
            "offset_y_range": self._read_int_range("OffsetY"),
            "sensor_width": self._read_feature("int", "WidthMax"),
            "sensor_height": self._read_feature("int", "HeightMax"),
            "binning_horizontal": self._read_feature("int", "BinningHorizontal"),
            "binning_vertical": self._read_feature("int", "BinningVertical"),
            "decimation_horizontal": self._read_feature("int", "DecimationHorizontal"),
            "decimation_vertical": self._read_feature("int", "DecimationVertical"),
            "pixel_format": self._read_feature("enum", "PixelFormat"),
            "exposure_auto": self._read_feature("enum", "ExposureAuto"),
            "exposure_us": self._read_feature("float", "ExposureTime"),
            "exposure_us_range": self._read_float_range("ExposureTime"),
            "gain_auto": self._read_feature("enum", "GainAuto"),
            "gain_db": self._read_feature("float", "Gain"),
            "gain_db_range": self._read_float_range("Gain"),
            "gamma_enabled": self._read_feature("bool", "GammaEnable"),
            "acquisition_fps_mode": self._read_feature(
                "enum", "AcquisitionFrameRateMode"
            ),
            "acquisition_fps": self._read_feature(
                "float", "AcquisitionFrameRate"
            ),
            "acquisition_fps_range": self._read_float_range(
                "AcquisitionFrameRate"
            ),
            "reverse_x": self._read_feature("bool", "ReverseX"),
            "reverse_y": self._read_feature("bool", "ReverseY"),
        }
        return metadata

    def close(self) -> None:
        if self.camera is None:
            return
        if self._streaming:
            try:
                self.camera.stream_off()
            except Exception:
                pass
            self._streaming = False
        try:
            self.camera.close_device()
        except Exception:
            pass
        self.camera = None
        self.feature_control = None


class CtypesGalaxySource(FrameSource):
    """Workspace's libgxiapi.so ctypes backend for Linux/WSL deployments."""

    backend_name = "galaxy-ctypes"

    def __init__(
        self,
        *,
        sensor_index: int,
        width: int,
        height: int,
        offset_x: int | None,
        offset_y: int | None,
        exposure_us: float | None,
        gain_db: float,
        acquisition_fps: float,
    ) -> None:
        self.sensor_index = int(sensor_index)
        self.width = int(width)
        self.height = int(height)
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.exposure_us = None if exposure_us is None else float(exposure_us)
        self.gain_db = float(gain_db)
        self.acquisition_fps = float(acquisition_fps)
        self.camera = None
        self.devices: list[Any] = []
        self._streaming = False
        self._configuration_verified = False
        self._configuration_warnings: list[str] = []
        self._expected_roi: dict[str, int] = {}

    def _verify_configuration(self) -> None:
        def available(name: str) -> bool:
            try:
                return bool(self.camera.feature_is_available(name))
            except Exception:
                return False

        def require_enum(name: str, expected: str, *, optional: bool = False) -> None:
            if not available(name):
                if optional:
                    return
                raise RuntimeError(f"Galaxy camera does not expose required node {name}")
            actual = self._safe_get("enum", name)
            if actual is None or str(actual).casefold() != expected.casefold():
                raise RuntimeError(
                    f"Galaxy camera {name} is {actual!r}; calibration requires {expected!r}"
                )

        def require_float(
            name: str, expected: float | None, *, positive: bool = False
        ) -> None:
            if not available(name):
                raise RuntimeError(f"Galaxy camera does not expose required node {name}")
            actual = self._safe_get("float", name)
            try:
                numeric = float(actual)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Galaxy camera {name} cannot be read") from exc
            if not math.isfinite(numeric) or (positive and numeric <= 0):
                raise RuntimeError(f"Galaxy camera {name} has invalid value {actual!r}")
            if expected is not None and not math.isclose(
                numeric, float(expected), rel_tol=1e-6, abs_tol=1e-6
            ):
                raise RuntimeError(
                    f"Galaxy camera {name} is {numeric}, requested {float(expected)}; "
                    "the requested calibration setting was not applied"
                )

        def require_int(name: str, expected: int) -> None:
            if not available(name):
                raise RuntimeError(f"Galaxy camera does not expose required node {name}")
            actual = self._safe_get("int", name)
            if actual is None or int(actual) != int(expected):
                raise RuntimeError(
                    f"Galaxy camera {name} is {actual!r}, expected {int(expected)}; "
                    "the requested ROI was not applied"
                )

        require_enum("AcquisitionMode", "Continuous")
        require_enum("TriggerMode", "Off")
        require_enum("PixelFormat", "Mono8")
        require_enum("ExposureAuto", "Off", optional=True)
        require_enum("GainAuto", "Off", optional=True)
        if available("GammaEnable"):
            gamma_enabled = self._safe_get("bool", "GammaEnable")
            if gamma_enabled is None or bool(gamma_enabled):
                raise RuntimeError(
                    f"Galaxy camera GammaEnable is {gamma_enabled!r}; "
                    "calibration requires False"
                )
        require_float("ExposureTime", self.exposure_us, positive=True)
        require_float("Gain", self.gain_db)
        expected_fps_mode = "On" if self.acquisition_fps > 0 else "Off"
        require_enum(
            "AcquisitionFrameRateMode", expected_fps_mode, optional=True
        )
        if self.acquisition_fps > 0:
            require_float("AcquisitionFrameRate", self.acquisition_fps, positive=True)
        for name, expected in self._expected_roi.items():
            require_int(name, expected)

    def open(self) -> None:
        self._configuration_verified = False
        self._configuration_warnings = []
        # Sensor adapters live two package levels below python/common.
        common_path = Path(__file__).resolve().parents[2] / "common"
        if str(common_path) not in sys.path:
            sys.path.insert(0, str(common_path))
        from GalaxyCamera import GalaxyCamera

        camera = GalaxyCamera()
        self.camera = camera
        try:
            camera.init()
            self.devices = camera.list_devices()
            if not self.devices:
                raise RuntimeError("Galaxy SDK found no camera")
            camera.open_by_index(self.sensor_index)
            camera.set_enum("AcquisitionMode", "Continuous")
            camera.set_enum("TriggerMode", "Off")
            if camera.feature_is_writable("PixelFormat"):
                camera.set_enum("PixelFormat", "Mono8")
            if camera.feature_is_writable("ExposureAuto"):
                camera.set_enum("ExposureAuto", "Off")
            if camera.feature_is_writable("GainAuto"):
                camera.set_enum("GainAuto", "Off")
            if camera.feature_is_writable("GammaEnable"):
                camera.set_bool("GammaEnable", False)
            camera.set_int_aligned("OffsetX", 0)
            camera.set_int_aligned("OffsetY", 0)
            requested_width = self.width or int(camera.get_int_range("Width")["max"])
            requested_height = self.height or int(camera.get_int_range("Height")["max"])
            initial_roi = camera.set_roi(
                requested_width, requested_height, offset_x=0, offset_y=0
            )
            actual_width = int(initial_roi["width"])
            actual_height = int(initial_roi["height"])
            sensor_width = initial_roi.get("sensor_width")
            sensor_height = initial_roi.get("sensor_height")
            if sensor_width is None:
                sensor_width = int(camera.get_int_range("OffsetX")["max"]) + actual_width
            if sensor_height is None:
                sensor_height = int(camera.get_int_range("OffsetY")["max"]) + actual_height
            requested_offset_x = (
                max(0, (int(sensor_width) - actual_width) // 2)
                if self.offset_x is None
                else self.offset_x
            )
            requested_offset_y = (
                max(0, (int(sensor_height) - actual_height) // 2)
                if self.offset_y is None
                else self.offset_y
            )
            final_roi = camera.set_roi(
                actual_width,
                actual_height,
                offset_x=requested_offset_x,
                offset_y=requested_offset_y,
            )
            self._expected_roi = {
                "Width": int(final_roi["width"]),
                "Height": int(final_roi["height"]),
                "OffsetX": int(final_roi["offset_x"]),
                "OffsetY": int(final_roi["offset_y"]),
            }
            if self.width and int(final_roi["width"]) != self.width:
                self._configuration_warnings.append(
                    f"Width requested {self.width} aligned to {final_roi['width']}"
                )
            if self.height and int(final_roi["height"]) != self.height:
                self._configuration_warnings.append(
                    f"Height requested {self.height} aligned to {final_roi['height']}"
                )
            if int(final_roi["offset_x"]) != int(requested_offset_x):
                self._configuration_warnings.append(
                    f"OffsetX requested {requested_offset_x} aligned to {final_roi['offset_x']}"
                )
            if int(final_roi["offset_y"]) != int(requested_offset_y):
                self._configuration_warnings.append(
                    f"OffsetY requested {requested_offset_y} aligned to {final_roi['offset_y']}"
                )
            if self.exposure_us is not None:
                camera.set_exposure_time_us(self.exposure_us)
            camera.set_gain(self.gain_db)
            if self.acquisition_fps > 0:
                camera.set_acquisition_fps(self.acquisition_fps)
            elif camera.feature_is_writable("AcquisitionFrameRateMode"):
                camera.set_enum("AcquisitionFrameRateMode", "Off")
            self._verify_configuration()
            self._configuration_verified = True
            camera.set_buffer_number(3)
            try:
                camera.set_stream_buffer_mode("newest_only")
            except Exception:
                pass
            camera.stream_on()
            self._streaming = True
        except BaseException:
            self.close()
            raise

    def read(self, timeout_ms: int) -> np.ndarray | None:
        if self.camera is None or not self._streaming:
            raise RuntimeError("Galaxy camera is not open")
        get_frame = getattr(self.camera, "get_latest_frame", self.camera.get_frame)
        frame = get_frame(timeout_ms=int(timeout_ms))
        if frame is None:
            return None
        array = frame.to_numpy()
        return None if array is None else np.asarray(array).copy()

    def _safe_get(self, kind: str, name: str) -> Any:
        try:
            if not self.camera.feature_is_available(name):
                return None
            value = getattr(self.camera, f"get_{kind}")(name)
            if isinstance(value, dict) and "symbolic" in value:
                return value["symbolic"]
            return value
        except Exception:
            return None

    def _safe_get_float_range(self, name: str) -> dict[str, Any] | None:
        try:
            if not self.camera.feature_is_available(name):
                return None
            return self.camera.get_float_range(name)
        except Exception:
            return None

    def _safe_get_int_range(self, name: str) -> dict[str, Any] | None:
        try:
            if not self.camera.feature_is_available(name):
                return None
            return self.camera.get_int_range(name)
        except Exception:
            return None

    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "backend": self.backend_name,
            "sensor_index_1_based": self.sensor_index,
            "devices": self.devices,
            "configuration_verified": self._configuration_verified,
            "configuration_warnings": self._configuration_warnings,
        }
        if self.camera is not None:
            result["features"] = {
                "device_model": self._safe_get("string", "DeviceModelName"),
                "device_serial": self._safe_get("string", "DeviceSerialNumber"),
                "width": self._safe_get("int", "Width"),
                "width_range": self._safe_get_int_range("Width"),
                "height": self._safe_get("int", "Height"),
                "height_range": self._safe_get_int_range("Height"),
                "offset_x": self._safe_get("int", "OffsetX"),
                "offset_x_range": self._safe_get_int_range("OffsetX"),
                "offset_y": self._safe_get("int", "OffsetY"),
                "offset_y_range": self._safe_get_int_range("OffsetY"),
                "sensor_width": self._safe_get("int", "WidthMax"),
                "sensor_height": self._safe_get("int", "HeightMax"),
                "binning_horizontal": self._safe_get("int", "BinningHorizontal"),
                "binning_vertical": self._safe_get("int", "BinningVertical"),
                "decimation_horizontal": self._safe_get(
                    "int", "DecimationHorizontal"
                ),
                "decimation_vertical": self._safe_get("int", "DecimationVertical"),
                "pixel_format": self._safe_get("enum", "PixelFormat"),
                "exposure_auto": self._safe_get("enum", "ExposureAuto"),
                "exposure_us": self._safe_get("float", "ExposureTime"),
                "exposure_us_range": self._safe_get_float_range("ExposureTime"),
                "gain_auto": self._safe_get("enum", "GainAuto"),
                "gain_db": self._safe_get("float", "Gain"),
                "gain_db_range": self._safe_get_float_range("Gain"),
                "gamma_enabled": self._safe_get("bool", "GammaEnable"),
                "acquisition_fps_mode": self._safe_get(
                    "enum", "AcquisitionFrameRateMode"
                ),
                "acquisition_fps": self._safe_get(
                    "float", "AcquisitionFrameRate"
                ),
                "acquisition_fps_range": self._safe_get_float_range(
                    "AcquisitionFrameRate"
                ),
                "reverse_x": self._safe_get("bool", "ReverseX"),
                "reverse_y": self._safe_get("bool", "ReverseY"),
            }
        return result

    def close(self) -> None:
        if self.camera is None:
            return
        if self._streaming:
            try:
                self.camera.stream_off()
            except Exception:
                pass
            self._streaming = False
        try:
            self.camera.close()
        except Exception:
            pass
        try:
            self.camera.uninit()
        except Exception:
            pass
        self.camera = None
