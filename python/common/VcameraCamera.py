"""图漾 Vcamera 官方 Python SDK 的独立彩色/灰度采集封装。

与 OrbbecCamera 类似，支持 open/read/close、is_open 和上下文管理。
read 返回所选通道的 BGR uint8 或单通道 uint8/uint16 ndarray，超时返回 None。
不执行深度对齐、去畸变或软件缩放，适用于内外参标定。

用法（工程根目录）：
    from python.common.VcameraCamera import VcameraCamera
    with VcameraCamera(target_width=1280, target_height=960) as camera:
        image = camera.read(timeout_ms=1500)

stream: color/gray 选择 Texture，ir_left/ir_right 选择 Left/Right。
gray 要求 Texture 本身输出灰度，不会将彩色图转换成灰度或自动切换镜头。
宽高为 0 时选择最大可用模式；多设备建议通过 serial_number 选择。
exposure_us 单位为微秒；指定时手动曝光，未指定时默认启动自动曝光、稳定后锁定。
gain_sdk 使用设备原生单位，None 不指定初值；auto_lock 模式将曝光和实际增益一起锁定。
"""
import importlib
import math
import queue

import numpy as np

from types import SimpleNamespace


class VcameraCamera:
    backend_name = 'vcamera-sdk'
    stream_sensors = dict(color='Texture', gray='Texture', ir_left='Left', ir_right='Right')

    def __init__(self, target_width=0, target_height=0, sensor_index=0,
                 serial_number='', ip_address='', stream='color', exposure_us=None,
                 gain_sdk=None, gain_feature=None, exposure_mode=None, exposure_warmup_frames=30):
        if sensor_index < 0 or target_width < 0 or target_height < 0:
            raise ValueError('Device index and image dimensions must be nonnegative')
        if stream not in self.stream_sensors:
            raise ValueError('Vcamera stream must be color, gray, ir_left or ir_right')
        if exposure_us is not None and (isinstance(exposure_us, bool)
                or not isinstance(exposure_us, (int, float))
                or not math.isfinite(exposure_us) or exposure_us <= 0):
            raise ValueError('Vcamera exposure_us must be a finite positive number or None')
        if gain_sdk is not None and (isinstance(gain_sdk, bool)
                or not isinstance(gain_sdk, (int, float))
                or not math.isfinite(gain_sdk) or gain_sdk < 0):
            raise ValueError('Vcamera gain_sdk must be a finite nonnegative number or None')
        if gain_feature not in (None, 'DigitalGain', 'AnalogGain', 'Gain'):
            raise ValueError('Vcamera gain_feature must be DigitalGain, AnalogGain or Gain')
        if exposure_mode is None:
            exposure_mode = 'manual' if exposure_us is not None else 'auto_lock'
        if exposure_mode not in ('manual', 'auto_lock', 'preserve'):
            raise ValueError('Vcamera exposure_mode must be manual, auto_lock or preserve')
        if (exposure_mode == 'manual') != (exposure_us is not None):
            raise ValueError('Vcamera manual mode requires exposure_us; auto_lock/preserve require None')
        if type(exposure_warmup_frames) is not int or not 5 <= exposure_warmup_frames <= 120:
            raise ValueError('Vcamera exposure_warmup_frames must be an integer from 5 to 120')
        self.args = SimpleNamespace(width=target_width, height=target_height,
                                    sensor_index=sensor_index, serial_number=serial_number,
                                    ip_address=ip_address, stream=stream, exposure_us=exposure_us,
                                    gain_sdk=gain_sdk, gain_feature=gain_feature,
                                    exposure_mode=exposure_mode, exposure_warmup_frames=exposure_warmup_frames)
        self.camera = None
        self.connected = False
        self.started = False
        self.frames = queue.Queue(maxsize=1)
        self.info = {}

    @staticmethod
    def check(status, operation):
        if not status:
            raise RuntimeError(f'Vcamera {operation}: {status.message()}')

    @staticmethod
    def discover_devices(sdk):
        return [device for device in sdk.CameraUtils.DiscoverCameras()
                if device.model or device.network_info.ip]

    def open(self):
        if self.is_open:
            return
        try:
            self.sdk = importlib.import_module('vcamera')
        except ImportError as exc:
            raise RuntimeError('Vcamera Python SDK is unavailable. Run with this Python: '
                               'python -m python.scripts.install_sdks vcamera') from exc
        sdk = self.sdk
        self.check(sdk.CameraUtils.Init(True), 'initialize')
        devices = self.discover_devices(sdk)
        matches = [d for d in devices
                   if (not self.args.serial_number or d.serial_number == self.args.serial_number)
                   and (not self.args.ip_address or d.network_info.ip == self.args.ip_address)]
        if self.args.serial_number or self.args.ip_address:
            if len(matches) != 1:
                raise RuntimeError('Vcamera selector must match exactly one discovered camera')
            device = matches[0]
        else:
            if self.args.sensor_index >= len(devices):
                raise RuntimeError(f'Vcamera index {self.args.sensor_index}: discovered {len(devices)} camera(s)')
            device = devices[self.args.sensor_index]
        self.info = dict(serial_number=device.serial_number, ip_address=device.network_info.ip,
                         model=device.model)
        self.camera = sdk.CameraFactory.GetCameraByCameraInfo(device)
        try:
            self.check(self.camera.Connect(), 'connect')
            self.connected = True
            self.check(self.camera.StopCapture(), 'stop previous capture')
            self.sensor_name = self.stream_sensors[self.args.stream]
            self.sensor = sensor = getattr(sdk.SensorType, self.sensor_name)
            present, status = self.camera.HasSensor(sensor)
            self.check(status, f'query {self.sensor_name} sensor')
            if not present:
                raise RuntimeError(f'Vcamera {device.model} has no {self.sensor_name} sensor; '
                                   'check the stream in the selected camera profile')
            modes, status = self.camera.GetImageModes(sensor)
            self.check(status, f'query {self.sensor_name} modes')
            modes = [m for m in modes if (not self.args.width or m.width == self.args.width)
                     and (not self.args.height or m.height == self.args.height)]
            if not modes:
                raise RuntimeError(f'Vcamera has no {self.sensor_name} mode matching requested resolution')
            mode = max(modes, key=lambda m: m.width * m.height)
            current, status = self.camera.GetCurrentImageMode(sensor)
            self.check(status, f'query current {self.sensor_name} mode')
            # Fixed-resolution models can reject writes even to the current mode.
            if (current.width, current.height, current.pixel_format) != (mode.width, mode.height, mode.pixel_format):
                self.check(self.camera.SetImageMode(sensor, mode), f'set {self.sensor_name} mode')
            self.check(self.camera.SetUndistortionEnabled(sensor, False), f'disable {self.sensor_name} undistortion')
            calib, status = self.camera.GetFactoryCalibInfo(sensor)
            self.check(status, f'read factory {self.sensor_name} calibration')
            if (calib.intrinsic.width, calib.intrinsic.height) != (mode.width, mode.height):
                raise RuntimeError('Vcamera factory intrinsics resolution differs from selected mode; '
                                   f'use native resolution {calib.intrinsic.width}x{calib.intrinsic.height}')
            geometry = 'raw_texture' if self.sensor_name == 'Texture' else 'raw_' + self.sensor_name.lower()
            self.info['sdk_intrinsics'] = dict(
                image_size=[int(calib.intrinsic.width), int(calib.intrinsic.height)],
                camera_matrix=np.asarray(calib.intrinsic.matrix).reshape(3, 3).tolist(),
                distortion_coefficients=np.asarray(calib.distortion.coefficients).reshape(-1).tolist(),
                source=f'GetFactoryCalibInfo({self.sensor_name})', image_geometry=geometry)
            self.info.update(width=mode.width, height=mode.height, image_geometry=geometry,
                             sdk_sensor=self.sensor_name,
                             raw_pixel_format=str(mode.pixel_format))
            feature, status = self.camera.GetFeature('AcquisitionMode')
            self.check(status, 'get AcquisitionMode')
            self.check(feature.SetValue(2), 'set continuous acquisition')
            # Mapping only applies to cameras that expose both Depth and Texture.
            available = {}
            for name in ('Depth', 'Texture', 'Left', 'Right'):
                exists, status = self.camera.HasSensor(getattr(sdk.SensorType, name))
                self.check(status, f'query {name} sensor')
                available[name] = exists
            if available['Depth'] and available['Texture']:
                self.check(self.camera.SetMapDepthToTextureEnabled(False), 'disable depth to texture mapping')
                self.check(self.camera.SetMapTextureToDepthEnabled(False), 'disable texture to depth mapping')
            for name, exists in available.items():
                if exists:
                    self.check(self.camera.SetSensorEnabled(getattr(sdk.SensorType, name), name == self.sensor_name),
                               f'configure {name} sensor')
            self._configure_exposure()
            self._configure_gain()
            self.info['exposure_mode'] = self.args.exposure_mode
            if self.args.exposure_mode == 'auto_lock':
                self._set_auto_control(f'{self.sensor_name}/ExposureAuto', True, required=True)
            self.camera.RegisterFrameSetCallback(self._callback)
            self.check(self.camera.StartCapture(), 'start capture')
            self.started = True
            if self.args.exposure_mode == 'auto_lock':
                self._warmup_and_lock_exposure()
        except Exception:
            self.close()
            raise

    def _configure_exposure(self):
        requested = self.args.exposure_us
        if requested is None:
            return
        self._set_exposure(requested)

    def _set_exposure(self, requested):
        # ExposureTime is the SDK's microsecond API. IntExposureTime has native
        # device units and must never be used as a microsecond fallback.
        name = f'{self.sensor_name}/ExposureTime'
        feature, status = self.camera.GetFeature(name)
        self.check(status, f'get {name} (microseconds)')
        if feature.GetType() != self.sdk.FeatureType.Float64:
            raise RuntimeError(f'Vcamera {name}: expected a floating-point microsecond feature')
        limits, status = feature.GetFloatRange()
        self.check(status, f'query {name} range')
        low, high = float(limits.minValue), float(limits.maxValue)
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high and high > 0):
            raise RuntimeError(f'Vcamera {name}: invalid exposure range {low}..{high} us')
        if not low <= requested <= high:
            raise ValueError(f'Vcamera {name}: exposure_us={requested:g} outside supported '
                             f'range [{low:g}, {high:g}] us')
        auto_supported = self._disable_auto_control(f'{self.sensor_name}/ExposureAuto')
        self.check(feature.SetValue(float(requested)), f'set {name}')
        value, status = feature.GetValue()
        self.check(status, f'read back {name}')
        actual = float(value)
        if not math.isfinite(actual) or actual <= 0 or not low <= actual <= high:
            raise RuntimeError(f'Vcamera {name}: invalid exposure readback {actual} us')
        # Sensors can quantize the requested duration to integral line times.
        self.info.update(exposure_us=actual, requested_exposure_us=float(requested),
                         exposure_range_us=[low, high], exposure_feature=name,
                         exposure_auto_control_available=auto_supported)
        if auto_supported:
            self.info['exposure_auto'] = False
        print(f'Vcamera: exposure={actual:g} us')

    def _disable_auto_control(self, auto_name):
        return self._set_auto_control(auto_name, False)

    def _set_auto_control(self, auto_name, enabled, required=False):
        # This SDK reports FeatureInvalid for absent optional controls. Enumerate
        # features so absence is distinct from a failure reading an existing one.
        features, status = self.camera.GetAllFeatures()
        self.check(status, 'list automatic controls')
        auto = next((item for item in features if item.GetName() == auto_name), None)
        auto_supported = auto is not None
        if required and not auto_supported:
            raise RuntimeError(f'Vcamera {auto_name}: unavailable; configure a manual exposure instead')
        if auto_supported:
            if auto.GetType() == self.sdk.FeatureType.Bool:
                requested = bool(enabled)
            elif auto.GetType() == self.sdk.FeatureType.Enumeration:
                items, status = auto.GetEnumItems()
                self.check(status, f'query {auto_name} values')
                names = ('continuous', 'on', 'enabled', 'auto') if enabled else ('off', 'disabled')
                matches = [item for name in names for item in items if item.name.lower() == name]
                if len(matches) != 1:
                    # Continuous takes precedence when both On and Continuous are exposed.
                    matches = [item for item in items if enabled and item.name.lower() == 'continuous']
                if len(matches) != 1:
                    raise RuntimeError(f'Vcamera {auto_name}: no unambiguous {"On" if enabled else "Off"} value')
                requested = matches[0].value
            else:
                raise RuntimeError(f'Vcamera {auto_name}: unsupported automatic control type')
            self.check(auto.SetValue(requested), f'set {auto_name}')
            value, status = auto.GetValue()
            self.check(status, f'read back {auto_name}')
            actual = bool(value) if auto.GetType() == self.sdk.FeatureType.Bool else int(value)
            if actual != requested:
                raise RuntimeError(f'Vcamera {auto_name}: automatic control readback mismatch')
        return auto_supported

    def _photometric_controls(self):
        features, status = self.camera.GetAllFeatures()
        self.check(status, 'list exposure/gain controls')
        wanted = {f'{self.sensor_name}/{name}' for name in ('ExposureTime', 'DigitalGain', 'AnalogGain', 'Gain')}
        return {feature.GetName(): feature for feature in features if feature.GetName() in wanted}

    def _control_values(self, controls):
        values = {}
        for name, feature in controls.items():
            value, status = feature.GetValue()
            self.check(status, f'read {name}')
            integer = feature.GetType() == self.sdk.FeatureType.Int64
            native = int(value) if integer else float(value)
            number = float(native)
            if not math.isfinite(number) or number < 0 or (name.endswith('/ExposureTime') and number <= 0):
                raise RuntimeError(f'Vcamera {name}: invalid control value {number}')
            values[name] = native
        return values

    def _warmup_and_lock_exposure(self):
        controls = self._photometric_controls()
        exposure_name = f'{self.sensor_name}/ExposureTime'
        if exposure_name not in controls:
            raise RuntimeError(f'Vcamera {exposure_name}: cannot read automatic exposure in microseconds')
        previous, stable = None, 0
        print('Vcamera: automatic exposure warmup; keep the calibration scene in view.', flush=True)
        for count in range(1, 121):
            if self.read(1500) is None:
                raise RuntimeError('Vcamera: no frame during automatic exposure warmup')
            values = self._control_values(controls)
            settled = previous is not None and all(
                math.isclose(values[name], previous[name], rel_tol=0.01, abs_tol=1e-6) for name in values)
            stable = stable + 1 if settled else 0
            previous = values
            if count >= self.args.exposure_warmup_frames and stable >= 6:
                break
        else:
            raise RuntimeError('Vcamera: exposure/gain did not stabilize in 120 frames; improve illumination or use manual exposure')
        self._set_auto_control(f'{self.sensor_name}/ExposureAuto', False, required=True)
        gain_auto = self._disable_auto_control(f'{self.sensor_name}/GainAuto')
        locked = self._control_values(controls)
        self._set_exposure(locked[exposure_name])
        for name, feature in controls.items():
            if name != exposure_name:
                self.check(feature.SetValue(locked[name]), f'lock {name}')
        # Discard automatic-exposure frames already queued in the device/SDK and
        # verify the fixed control values before returning any calibration frame.
        verified = self._control_values(controls)
        for _ in range(5):
            if self.read(1500) is None:
                raise RuntimeError('Vcamera: no frame after locking exposure')
            actual = self._control_values(controls)
            if any(not math.isclose(actual[name], verified[name], rel_tol=1e-6, abs_tol=1e-6) for name in actual):
                raise RuntimeError('Vcamera: exposure/gain changed after disabling automatic controls')
        self.info.update(exposure_auto=False, exposure_lock_verified=True,
                         exposure_warmup_frame_count=count, locked_controls=verified,
                         gain_auto_control_available=gain_auto)
        if gain_auto:
            self.info['gain_auto'] = False
        print(f'Vcamera: exposure/gain locked after {count} frames: {verified}', flush=True)

    def _configure_gain(self):
        requested = self.args.gain_sdk
        if requested is None:
            return
        selected = self.args.gain_feature
        if selected is None:
            features, status = self.camera.GetAllFeatures()
            self.check(status, 'query gain controls')
            names = {item.GetName() for item in features}
            selected = next((item for item in ('DigitalGain', 'Gain', 'AnalogGain')
                             if f'{self.sensor_name}/{item}' in names), None)
            if selected is None:
                raise RuntimeError(f'Vcamera {self.sensor_name}: adjustable gain is unavailable')
        name = f'{self.sensor_name}/{selected}'
        feature, status = self.camera.GetFeature(name)
        self.check(status, f'get {name}; select a supported gain_feature for this model')
        kind = feature.GetType()
        integer = kind == self.sdk.FeatureType.Int64
        if integer:
            if not float(requested).is_integer():
                raise ValueError(f'Vcamera {name}: gain_sdk must be an integer')
            limits, status = feature.GetIntRange()
        elif kind == self.sdk.FeatureType.Float64:
            limits, status = feature.GetFloatRange()
        else:
            raise RuntimeError(f'Vcamera {name}: unsupported gain feature type')
        self.check(status, f'query {name} range')
        low, high, step = float(limits.minValue), float(limits.maxValue), float(limits.step)
        if not all(math.isfinite(v) for v in (low, high, step)) or low > high or step < 0:
            raise RuntimeError(f'Vcamera {name}: invalid gain range')
        if not low <= requested <= high:
            raise ValueError(f'Vcamera {name}: gain_sdk={requested:g} outside supported range [{low:g}, {high:g}]')
        if step > 0 and not math.isclose((requested - low) / step, round((requested - low) / step), abs_tol=1e-6, rel_tol=0):
            raise ValueError(f'Vcamera {name}: gain_sdk must follow step {step:g} from {low:g}')
        auto_supported = self._disable_auto_control(f'{self.sensor_name}/GainAuto')
        self.check(feature.SetValue(int(requested) if integer else float(requested)), f'set {name}')
        value, status = feature.GetValue()
        self.check(status, f'read back {name}')
        actual = int(value) if integer else float(value)
        if (not math.isfinite(actual) or not low <= actual <= high
                or not math.isclose(actual, requested, rel_tol=1e-6, abs_tol=1e-6)):
            raise RuntimeError(f'Vcamera {name}: gain readback {actual} differs from requested {requested}')
        actual = int(actual) if integer else actual
        self.info.update(gain_sdk=actual, requested_gain_sdk=requested,
                         gain_range_sdk=[low, high], gain_step_sdk=step, gain_feature=name,
                         gain_units='sdk_native', gain_auto_control_available=auto_supported)
        if auto_supported:
            self.info['gain_auto'] = False
        print(f'Vcamera: gain={actual:g} (device units)')

    def _callback(self, frameset):
        try:
            image = frameset.GetImage(self.sensor)
            if not image.IsValid():
                return
            fmt = image.pixel_format()
            if self.args.stream == 'color':
                formats = {self.sdk.PixelFormat.Bgr24: ('BGR24', np.uint8, 3)}
            else:
                formats = {self.sdk.PixelFormat.Gray8: ('Gray8', np.uint8, 1),
                           self.sdk.PixelFormat.Gray16: ('Gray16', np.uint16, 1)}
            if fmt not in formats:
                raise RuntimeError(f'Vcamera {self.sensor_name} returned {fmt} for stream={self.args.stream}; '
                                   'select the stream matching the physical sensor')
            pixel_format, dtype, channels = formats[fmt]
            # Copy inside callback: vendor buffers expire when the callback returns.
            frame = np.asarray(image.to_numpy()).copy()
            shape = (self.info['height'], self.info['width']) + ((3,) if channels == 3 else ())
            if frame.shape != shape or frame.dtype != dtype:
                raise RuntimeError(f'Unexpected Vcamera {self.sensor_name} image: {frame.shape}, {frame.dtype}')
            self.info['pixel_format'] = pixel_format
            if hasattr(image, 'laser_val'):
                self.info['last_frame_laser_value'] = int(image.laser_val())
        except Exception as exc:
            frame = exc
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put_nowait(frame)

    @property
    def is_open(self):
        return self.connected and self.started

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def read(self, timeout_ms=100):
        if not self.is_open:
            raise RuntimeError('VcameraCamera must be opened before read()')
        if timeout_ms < 0:
            raise ValueError('timeout_ms must be nonnegative')
        try:
            frame = self.frames.get(timeout=timeout_ms / 1000)
        except queue.Empty:
            return None
        if isinstance(frame, Exception):
            raise frame
        return frame

    def metadata(self):
        return dict(backend=self.backend_name, sensor_type='vcamera',
                    sensor_index=self.args.sensor_index, stream=self.args.stream,
                    raw_texture=self.stream_sensors[self.args.stream] == 'Texture', **self.info)

    def close(self):
        camera, self.camera = self.camera, None
        try:
            if camera is not None and self.connected:
                try:
                    if self.started:
                        self.check(camera.StopCapture(), 'stop capture')
                finally:
                    self.check(camera.Disconnect(), 'disconnect')
        finally:
            self.started = self.connected = False
            while True:
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    break
