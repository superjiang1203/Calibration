"""orbbec Python SDK frame adapter."""
from __future__ import annotations

import importlib
import importlib.metadata
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


from .frame_source import FrameSource

def disable_orbbec_logging(sdk):
    """Keep SDK file logs out of the calibration workspace; exceptions still propagate."""
    sdk.Context.set_logger_level(sdk.OBLogLevel.NONE)


def configure_orbbec_gain(device, sdk, stream, requested, original_settings=None):
    """Set color/IR gain in native integer units; None preserves the device value."""
    if requested is None:
        return {}
    if type(requested) is not int or requested < 0:
        raise ValueError('Orbbec gain_sdk must be a nonnegative integer or None')
    name = 'OB_PROP_COLOR_GAIN_INT' if stream == 'color' else 'OB_PROP_IR_GAIN_INT'
    prop = getattr(sdk.OBPropertyID, name)
    if not all(device.is_property_supported(prop, permission) for permission in (
            sdk.OBPermissionType.PERMISSION_READ, sdk.OBPermissionType.PERMISSION_WRITE)):
        raise RuntimeError(f'Orbbec {name} does not support reading and setting gain')
    limits = device.get_int_property_range(prop)
    low, high, step = int(limits.min), int(limits.max), int(limits.step)
    if low > high or step < 0:
        raise RuntimeError(f'Orbbec {name}: invalid gain range')
    if not low <= requested <= high:
        raise ValueError(f'Orbbec {name}: gain_sdk={requested} outside supported range [{low}, {high}]')
    if step > 0 and (requested - low) % step:
        raise ValueError(f'Orbbec {name}: gain_sdk must follow step {step} from {low}')
    if original_settings is not None:
        original_settings.append((prop, 'int', device.get_int_property(prop)))
    device.set_int_property(prop, requested)
    actual = device.get_int_property(prop)
    if actual != requested:
        raise RuntimeError(f'Orbbec {name}: gain readback {actual} differs from requested {requested}')
    print(f'Orbbec: gain={actual} (device units)')
    return dict(gain_sdk=actual, requested_gain_sdk=requested, gain_range_sdk=[low, high],
                gain_step_sdk=step, gain_property=name, gain_units='sdk_native')


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
        gain = configure_orbbec_gain(device, sdk, 'color', getattr(self.args, 'gain_sdk', None))
        camera.pipeline.start(camera.config)
        camera._is_open = True
        wb = self._configure_white_balance(device, sdk)
        self.serial = device.get_device_info().get_serial_number()
        self.features = dict(width=profile.get_width(), height=profile.get_height(),
            device_serial=self.serial,
            fps=profile.get_fps(), exposure_auto=False, exposure_sdk=actual,
            **wb, **gain,
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


class OrbbecIRSource:
    """One SDK pipeline for one or both IR streams of the same device."""

    def __init__(self, sensors):
        self.sensors = sensors
        self.sdk = self.device = self.pipeline = None
        self.started = False
        self.original_settings = []
        self.profiles = {}
        self.info = {}
        self.gain_info = {}

    def open(self):
        try:
            self._open()
        except Exception as error:
            if type(error).__module__.startswith('pyorbbecsdk'):
                raise RuntimeError(f'Orbbec IR SDK open failed: {error}') from error
            raise

    def _open(self):
        try:
            import pyorbbecsdk as sdk
        except ImportError as error:
            raise RuntimeError(
                f'Orbbec SDK import failed: {error}. '
                'Install with this Python: python -m python.scripts.install_sdks orbbec'
            ) from error
        self.sdk = sdk
        disable_orbbec_logging(sdk)
        sensor = self.sensors[0]
        self.context = sdk.Context()
        devices = self.context.query_devices()
        index = sensor['sensor_index']
        if not 0 <= index < devices.get_count():
            raise RuntimeError('No Orbbec device at configured sensor_index')
        self.device = device = devices.get_device_by_index(index)
        self.pipeline = sdk.Pipeline(device)
        config = sdk.Config()
        for item in self.sensors:
            kind = sdk.OBSensorType.LEFT_IR_SENSOR if item['stream'] == 'ir_left' else sdk.OBSensorType.RIGHT_IR_SENSOR
            profile = self.pipeline.get_stream_profile_list(kind).get_video_stream_profile(
                item['width'], item['height'], getattr(sdk.OBFormat, item['pixel_format']), int(item['acquisition_fps']))
            config.enable_stream(profile)
            self.profiles[item['label']] = profile
        if len(self.sensors) == 2:
            config.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            self.pipeline.enable_frame_sync()
        properties = sdk.OBPropertyID
        settings = [(properties.OB_PROP_IR_AUTO_EXPOSURE_BOOL, 'bool', False),
                    (properties.OB_PROP_IR_EXPOSURE_INT, 'int', sensor['exposure_sdk'])]
        laser_mode = sensor.get('laser_mode', 0)
        laser = getattr(properties, 'OB_PROP_LASER_CONTROL_INT', None)
        if laser is not None and device.is_property_supported(laser, sdk.OBPermissionType.PERMISSION_WRITE):
            settings.append((laser, 'int', laser_mode))
        elif laser_mode in (0, 1):
            settings.append((properties.OB_PROP_LASER_BOOL, 'bool', bool(laser_mode)))
        else:
            raise ValueError('Device does not support automatic laser mode')
        for prop, kind, value in settings:
            previous = getattr(device, 'get_' + kind + '_property')(prop)
            self.original_settings.append((prop, kind, previous))
            getattr(device, 'set_' + kind + '_property')(prop, value)
            actual = getattr(device, 'get_' + kind + '_property')(prop)
            if actual != value:
                raise RuntimeError(f'IR setting readback mismatch: {prop}, requested={value}, actual={actual}')
        self.gain_info = configure_orbbec_gain(device, sdk, 'ir', sensor.get('gain_sdk', 16),
                                               self.original_settings)
        self.pipeline.start(config)
        self.started = True
        for _ in range(10):
            self.pipeline.wait_for_frames(1000)
        version = 'unknown'
        for name in ('pyorbbecsdk2', 'pyorbbecsdk'):
            try:
                version = importlib.metadata.version(name)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        self.info = {'device_serial': device.get_device_info().get_serial_number(),
                     'sdk_python_version': version,
                     'image_geometry': 'native SDK stream; no application D2C, undistortion or rectification; firmware processing unverified'}

    def read(self, timeout_ms):
        try:
            return self._read(timeout_ms)
        except Exception as error:
            if type(error).__module__.startswith('pyorbbecsdk'):
                raise RuntimeError(f'Orbbec IR SDK read failed: {error}') from error
            raise

    def _read(self, timeout_ms):
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return {}, {}
        images, timestamps = {}, {}
        for sensor in self.sensors:
            frame = frames.get_left_ir_frame() if sensor['stream'] == 'ir_left' else frames.get_right_ir_frame()
            if frame is None:
                continue
            fmt = frame.get_format()
            if fmt not in (self.sdk.OBFormat.Y8, self.sdk.OBFormat.Y16):
                raise ValueError('IR stream must return uncompressed Y8/Y16')
            dtype = np.uint8 if fmt == self.sdk.OBFormat.Y8 else np.uint16
            data = np.frombuffer(frame.get_data(), dtype=dtype)
            width, height = frame.get_width(), frame.get_height()
            if data.size != width * height:
                raise ValueError('Unsupported padded/compressed IR image')
            label = sensor['label']
            images[label] = data.reshape(height, width).copy()
            timestamps[label] = int(frame.get_timestamp_us())
            self.profiles[label] = frame.get_stream_profile().as_video_stream_profile()
        return images, timestamps

    def metadata(self, label):
        profile = self.profiles[label]
        intrinsic, distortion = profile.get_intrinsic(), profile.get_distortion()
        return {**self.info, **self.gain_info, 'backend': 'orbbec-ir',
                'image_size': [profile.get_width(), profile.get_height()],
                'sdk_intrinsics': {'image_size': [profile.get_width(), profile.get_height()],
                                   'camera_matrix': [[intrinsic.fx, 0, intrinsic.cx], [0, intrinsic.fy, intrinsic.cy], [0, 0, 1]],
                                   'distortion_coefficients': [float(getattr(distortion, key)) for key in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')],
                                   'sdk_distortion_model': str(getattr(distortion, 'model', 'UNAVAILABLE'))}}

    def sdk_pair_metadata(self):
        if len(self.sensors) != 2:
            return {}
        by_stream = {sensor['stream']: sensor['label'] for sensor in self.sensors}
        lp, rp = self.profiles[by_stream['ir_left']], self.profiles[by_stream['ir_right']]
        extrinsic = lp.get_extrinsic_to(rp)
        translation = np.asarray(extrinsic.transform if hasattr(extrinsic, 'transform') else extrinsic.trans, np.float64).reshape(3)
        rotation = np.asarray(extrinsic.rot, np.float64).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, translation
        baseline = float(np.linalg.norm(translation))
        if not np.isfinite(transform).all() or not 1 < baseline < 500:
            raise ValueError('Invalid SDK reference IR extrinsics')
        return {'sdk_baseline_mm': baseline, 'sdk_transform_ir_left_to_ir_right': transform.tolist(),
                'sdk_reference_source_label': by_stream['ir_left'], 'sdk_reference_target_label': by_stream['ir_right']}

    def close(self):
        try:
            if self.started:
                self.pipeline.stop()
                self.started = False
        finally:
            if self.device is not None:
                failures = []
                for prop, kind, value in reversed(self.original_settings):
                    try:
                        getattr(self.device, 'set_' + kind + '_property')(prop, value)
                    except Exception as error:
                        failures.append(str(error))
                self.original_settings.clear()
                if failures:
                    raise RuntimeError('Could not restore original IR settings: ' + '; '.join(failures))
