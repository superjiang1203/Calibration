"""orbbec Python SDK frame adapter."""
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

def disable_orbbec_logging(sdk):
    """Keep SDK file logs out of the calibration workspace; exceptions still propagate."""
    sdk.Context.set_logger_level(sdk.OBLogLevel.NONE)


def _orbbec_camera_class():
    common_root = Path(__file__).resolve().parents[2] / "common"
    if str(common_root) not in sys.path:
        sys.path.insert(0, str(common_root))
    try:
        from OrbbecCamera import OrbbecCamera
    except ImportError as error:
        raise RuntimeError(
            f"Orbbec SDK import failed: {error}. Current Python: {sys.executable}. "
            "Install with this Python: python -m python.scripts.install_sdks orbbec"
        ) from error
    return OrbbecCamera


class OrbbecSensorSource(FrameSource):
    backend_name = 'orbbec-color'

    def __init__(self, args):
        self.args = args
        self.camera = None
        self.features = {}
        self.serial = None

    def open(self):
        try:
            self._open_sdk()
        except Exception as error:
            self.close()
            if type(error).__module__.startswith('pyorbbecsdk'):
                raise RuntimeError(f'Orbbec SDK open failed: {error}') from error
            raise

    def _open_sdk(self):
        camera_class = _orbbec_camera_class()
        import pyorbbecsdk as sdk
        disable_orbbec_logging(sdk)
        context = sdk.Context()
        devices = context.query_devices()
        if not 0 <= self.args.sensor_index < devices.get_count():
            raise RuntimeError('Orbbec sensor_index is out of range or no device is connected')
        device = devices.get_device_by_index(self.args.sensor_index)
        camera = camera_class(target_width=self.args.width, target_height=self.args.height,
            target_fps=int(self.args.acquisition_fps))
        self.camera = camera
        camera.pipeline = sdk.Pipeline(device)
        camera.config = sdk.Config()
        profiles = camera.pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
        camera.color_profile = None
        for name in ('RGB', 'BGR', 'MJPG', 'YUYV', 'UYVY', 'NV12', 'NV21', 'I420'):
            try:
                camera.color_profile = profiles.get_video_stream_profile(self.args.width,
                    self.args.height, getattr(sdk.OBFormat, name), int(self.args.acquisition_fps))
                break
            except sdk.OBError:
                continue
        if camera.color_profile is None:
            raise ValueError('Requested Orbbec resolution/fps has no supported color profile')
        profile = camera.color_profile
        if (profile.get_width(), profile.get_height(), profile.get_fps()) != (self.args.width, self.args.height, int(self.args.acquisition_fps)):
            raise ValueError('Requested Orbbec resolution/fps is unsupported; fallback profiles are rejected')
        camera.config.enable_stream(profile)
        # SDK exposure units depend on the device: exposure_sdk is native, not microseconds.
        auto = sdk.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL
        exposure = sdk.OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT
        device.set_bool_property(auto, False)
        if self.args.exposure_sdk is not None:
            device.set_int_property(exposure, self.args.exposure_sdk)
        actual = device.get_int_property(exposure)
        if device.get_bool_property(auto):
            raise RuntimeError('Orbbec auto exposure was not disabled')
        if self.args.exposure_sdk is not None and actual != self.args.exposure_sdk:
            raise RuntimeError(f'Orbbec exposure readback differs: requested {self.args.exposure_sdk}, actual {actual}')
        camera.pipeline.start(camera.config)
        camera._is_open = True
        wb = self._configure_white_balance(device, sdk)
        self.serial = device.get_device_info().get_serial_number()
        self.features = dict(width=profile.get_width(), height=profile.get_height(),
            device_serial=self.serial,
            fps=profile.get_fps(), exposure_auto=False, exposure_sdk=actual,
            **wb,
            pixel_format=str(profile.get_format()))

    def _configure_white_balance(self, device, sdk):
        auto = sdk.OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL
        temperature = sdk.OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT
        mode = self.args.white_balance_mode
        if mode == 'manual':
            device.set_bool_property(auto, False)
            device.set_int_property(temperature, self.args.white_balance_kelvin)
        else:
            device.set_bool_property(auto, True)
            if not device.get_bool_property(auto):
                raise RuntimeError('Orbbec automatic white balance was not enabled')
            if mode == 'auto_lock':
                history = []
                frames = 0
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and frames < 150:
                    if self.camera.read(timeout_ms=500) is None:
                        continue
                    frames += 1
                    history.append(device.get_int_property(temperature))
                    if frames >= self.args.white_balance_warmup_frames and max(history[-5:]) - min(history[-5:]) <= 100:
                        break
                else:
                    raise RuntimeError('Orbbec white balance did not stabilize; check illumination and retry')
                settled = history[-1]
                device.set_bool_property(auto, False)
                device.set_int_property(temperature, settled)
        actual_auto = device.get_bool_property(auto)
        actual_kelvin = device.get_int_property(temperature)
        if actual_auto != (mode == 'auto'):
            raise RuntimeError('Orbbec white balance mode readback differs from requested mode')
        expected = self.args.white_balance_kelvin if mode == 'manual' else (settled if mode == 'auto_lock' else None)
        if expected is not None and actual_kelvin != expected:
            raise RuntimeError(f'Orbbec white balance readback differs: requested {expected}, actual {actual_kelvin}')
        return dict(white_balance_mode=mode, white_balance_auto=actual_auto,
                    white_balance_kelvin=actual_kelvin)

    def read(self, timeout_ms):
        try:
            return self.camera.read(timeout_ms)
        except Exception as error:
            if type(error).__module__.startswith('pyorbbecsdk'):
                raise RuntimeError(f'Orbbec SDK read failed: {error}') from error
            raise

    def metadata(self):
        profile = self.camera.color_profile
        intrinsic, distortion = profile.get_intrinsic(), profile.get_distortion()
        return dict(backend=self.backend_name, sensor_type='orbbec',
            sensor_index=self.args.sensor_index, serial_number=self.serial, features=self.features,
            sdk_intrinsics=dict(
                camera_matrix=[[float(intrinsic.fx), 0, float(intrinsic.cx)],
                               [0, float(intrinsic.fy), float(intrinsic.cy)], [0, 0, 1]],
                distortion_coefficients=[float(getattr(distortion, key))
                    for key in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')],
                image_size=[profile.get_width(), profile.get_height()],
                sdk_distortion_model=str(getattr(distortion, 'model', 'UNAVAILABLE'))))

    def close(self):
        if self.camera is not None:
            self.camera.close()
            self.camera = None
