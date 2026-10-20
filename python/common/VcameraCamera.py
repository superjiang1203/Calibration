"""图漾 Vcamera 官方 Python SDK 的独立彩色采集封装。

与 OrbbecCamera 类似，支持 open/read/close、is_open 和上下文管理。
read 返回原始 Texture 通道的 BGR uint8 ndarray，超时返回 None。
不执行深度对齐、去畸变或软件缩放，适用于彩色内外参标定。

用法（工程根目录）：
    from python.common.VcameraCamera import VcameraCamera
    with VcameraCamera(target_width=1280, target_height=960) as camera:
        image = camera.read(timeout_ms=1500)

宽高为 0 时选择最大可用模式；多设备建议通过 serial_number 选择。
"""
import importlib
import queue

import numpy as np

from types import SimpleNamespace


class VcameraCamera:
    backend_name = 'vcamera-sdk'

    def __init__(self, target_width=0, target_height=0, sensor_index=0,
                 serial_number='', ip_address=''):
        if sensor_index < 0 or target_width < 0 or target_height < 0:
            raise ValueError('Device index and image dimensions must be nonnegative')
        self.args = SimpleNamespace(width=target_width, height=target_height,
                                    sensor_index=sensor_index, serial_number=serial_number,
                                    ip_address=ip_address)
        self.camera = None
        self.connected = False
        self.started = False
        self.frames = queue.Queue(maxsize=1)
        self.info = {}

    @staticmethod
    def check(status, operation):
        if not status:
            raise RuntimeError(f'Vcamera {operation}: {status.message()}')

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
        devices = sdk.CameraUtils.DiscoverCameras()
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
            sensor = sdk.SensorType.Texture
            modes, status = self.camera.GetImageModes(sensor)
            self.check(status, 'query texture modes')
            modes = [m for m in modes if (not self.args.width or m.width == self.args.width)
                     and (not self.args.height or m.height == self.args.height)]
            if not modes:
                raise RuntimeError('Vcamera has no texture mode matching requested resolution')
            mode = max(modes, key=lambda m: m.width * m.height)
            self.check(self.camera.SetImageMode(sensor, mode), 'set texture mode')
            self.check(self.camera.SetUndistortionEnabled(sensor, False), 'disable texture undistortion')
            calib, status = self.camera.GetFactoryCalibInfo(sensor)
            self.check(status, 'read factory texture calibration')
            self.info['sdk_intrinsics'] = dict(
                image_size=[int(calib.intrinsic.width), int(calib.intrinsic.height)],
                camera_matrix=np.asarray(calib.intrinsic.matrix).reshape(3, 3).tolist(),
                distortion_coefficients=np.asarray(calib.distortion.coefficients).reshape(-1).tolist(),
                source='GetFactoryCalibInfo(Texture)', image_geometry='raw_texture')
            self.info.update(width=mode.width, height=mode.height, pixel_format='BGR24',
                             raw_pixel_format=str(mode.pixel_format))
            feature, status = self.camera.GetFeature('AcquisitionMode')
            self.check(status, 'get AcquisitionMode')
            self.check(feature.SetValue(2), 'set continuous acquisition')
            self.check(self.camera.SetMapDepthToTextureEnabled(False), 'disable depth to texture mapping')
            self.check(self.camera.SetMapTextureToDepthEnabled(False), 'disable texture to depth mapping')
            self.check(self.camera.SetSensorEnabled(sdk.SensorType.Depth, False), 'disable depth')
            self.check(self.camera.SetSensorEnabled(sensor, True), 'enable texture')
            self.camera.RegisterFrameSetCallback(self._callback)
            self.check(self.camera.StartCapture(), 'start capture')
            self.started = True
        except Exception:
            self.close()
            raise

    def _callback(self, frameset):
        try:
            image = frameset.GetImage(self.sdk.SensorType.Texture)
            if not image.IsValid():
                return
            if image.pixel_format() != self.sdk.PixelFormat.Bgr24:
                raise RuntimeError('Vcamera returned a non-BGR24 texture frame')
            # Copy inside callback: vendor buffers expire when the callback returns.
            frame = np.asarray(image.to_numpy()).copy()
            if frame.shape != (self.info['height'], self.info['width'], 3):
                raise RuntimeError(f'Unexpected Vcamera texture shape: {frame.shape}')
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
                    sensor_index=self.args.sensor_index, stream='color',
                    raw_texture=True, **self.info)

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
