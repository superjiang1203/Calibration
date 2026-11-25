"""Calibration adapter for the reusable Vcamera Python camera wrapper."""
import argparse
import importlib
import json
import sys
from pathlib import Path
from ...common.VcameraCamera import VcameraCamera
from .camera_options import sensor_options
from .frame_source import FrameSource


class VcameraSource(FrameSource):
    backend_name = 'vcamera-sdk'

    def __init__(self, args):
        self.camera = VcameraCamera(
            target_width=args.width, target_height=args.height,
            sensor_index=args.sensor_index, serial_number=args.serial_number,
            ip_address=args.ip_address, stream=getattr(args, 'stream', 'color'),
            exposure_us=getattr(args, 'exposure_us', None),
            gain_sdk=getattr(args, 'gain_sdk', None),
            gain_feature=getattr(args, 'gain_feature', None),
            exposure_mode=getattr(args, 'exposure_mode', None),
            exposure_warmup_frames=getattr(args, 'exposure_warmup_frames', 30))

    def open(self):
        self.camera.open()

    def read(self, timeout_ms):
        return self.camera.read(timeout_ms)

    def metadata(self):
        return self.camera.metadata()

    def close(self):
        self.camera.close()


def inspect_device(sdk, device):
    """Read modes and controls without changing acquisition settings."""
    camera = sdk.CameraFactory.GetCameraByCameraInfo(device)
    VcameraCamera.check(camera.Connect(), 'connect for inspection')
    try:
        sensors = {}
        for name in dict.fromkeys(VcameraCamera.stream_sensors.values()):
            sensor = getattr(sdk.SensorType, name)
            present, status = camera.HasSensor(sensor)
            VcameraCamera.check(status, f'query {name} sensor')
            if not present:
                continue
            modes, status = camera.GetImageModes(sensor)
            VcameraCamera.check(status, f'query {name} modes')
            sensors[name] = {
                'streams': [s for s, n in VcameraCamera.stream_sensors.items() if n == name],
                'modes': [dict(width=m.width, height=m.height,
                               raw_pixel_format=str(m.pixel_format)) for m in modes],
            }
        features, status = camera.GetAllFeatures()
        VcameraCamera.check(status, 'list features')
        controls = {}
        for feature in features:
            name = feature.GetName()
            if name != 'AcquisitionMode' and 'laser' not in name.lower() and not name.endswith((
                    '/ExposureTime', '/IntExposureTime', '/ExposureAuto',
                    '/DigitalGain', '/AnalogGain', '/Gain', '/GainAuto', '/HDREnable')):
                continue
            kind = feature.GetType()
            entry = {'type': str(kind)}
            value, status = feature.GetValue()
            if status:
                if kind == sdk.FeatureType.Bool:
                    entry['value'] = bool(value)
                elif kind in (sdk.FeatureType.Int64, sdk.FeatureType.Enumeration):
                    entry['value'] = int(value)
                elif kind == sdk.FeatureType.Float64:
                    entry['value'] = float(value)
                else:
                    entry['value'] = str(value)
            else:
                entry['read_error'] = status.message()
            if kind in (sdk.FeatureType.Int64, sdk.FeatureType.Float64):
                limits, status = (feature.GetIntRange() if kind == sdk.FeatureType.Int64
                                  else feature.GetFloatRange())
                if status:
                    entry.update(min=limits.minValue, max=limits.maxValue, step=limits.step)
                else:
                    entry['range_error'] = status.message()
            controls[name] = entry
        depth, status = camera.HasSensor(sdk.SensorType.Depth)
        VcameraCamera.check(status, 'query Depth sensor')
        return dict(sensors=sensors, has_depth=depth, controls=controls)
    finally:
        VcameraCamera.check(camera.Disconnect(), 'disconnect after inspection')


def capture_frame(options, destination, timeout_ms):
    import cv2

    with VcameraCamera(
            target_width=options.width, target_height=options.height,
            sensor_index=options.sensor_index, serial_number=options.serial_number,
            ip_address=options.ip_address, stream=options.stream,
            exposure_us=options.exposure_us, gain_sdk=options.gain_sdk,
            gain_feature=options.gain_feature, exposure_mode=options.exposure_mode,
            exposure_warmup_frames=options.exposure_warmup_frames) as camera:
        frame = camera.read(timeout_ms)
        if frame is None:
            raise RuntimeError(f'No Vcamera frame within {timeout_ms} ms')
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # imencode/tofile also supports Windows paths containing Chinese text.
        ok, encoded = cv2.imencode(destination.suffix, frame)
        if not ok:
            raise RuntimeError(f'Cannot encode Vcamera frame: {destination}')
        encoded.tofile(destination)
        return dict(path=str(destination), shape=list(frame.shape), dtype=str(frame.dtype),
                    metadata=camera.metadata())


def inspect_main(argv=None):
    from ..workflow.config import PROJECT_ROOT, validate_sensor_profile

    parser = argparse.ArgumentParser(
        description="Inspect Vcamera devices, modes and controls, and optionally save one frame.",
        prog="calibrate.py inspect-vcamera",
    )
    default_profile = PROJECT_ROOT / "configs/cameras/vcamera_ir_left.json"

    parser.add_argument('--config', type=Path, default=default_profile,
                        help='Camera profile JSON (default: vcamera_ir_left.json)')
    parser.add_argument('--list-devices', action='store_true',
                        help='Only discover devices; do not connect or read a profile')
    parser.add_argument('--save-frame', type=Path,
                        help='Apply profile settings and save a single PNG frame')
    parser.add_argument('--factory-intrinsics', type=Path, help='Export factory camera intrinsics to OpenCV YAML')
    parser.add_argument('--timeout-ms', type=int, default=5000)
    parser.add_argument('--output', type=Path, help='Also save the JSON report')
    args = parser.parse_args(argv)
    if args.timeout_ms <= 0:
        parser.error('--timeout-ms must be positive')
    if args.list_devices and (args.save_frame or args.factory_intrinsics):
        parser.error('--save-frame cannot be combined with --list-devices')
    if args.save_frame and args.save_frame.suffix.lower() != '.png':
        parser.error('--save-frame requires .png to preserve grayscale values')
    try:
        options = None
        if not args.list_devices:
            profile_path = args.config.resolve()
            profile = validate_sensor_profile(
                json.loads(profile_path.read_text(encoding='utf-8-sig')), profile_path)
            if profile['sensor_type'] != 'vcamera':
                raise ValueError('--config must select a vcamera profile')
            options = sensor_options(profile)
        try:
            sdk = importlib.import_module('vcamera')
        except ImportError as exc:
            raise RuntimeError('Vcamera Python SDK is unavailable. Install with: '
                               'python -m python.scripts.install_sdks vcamera') from exc
        VcameraCamera.check(sdk.CameraUtils.Init(True), 'initialize')
        devices = VcameraCamera.discover_devices(sdk)
        report = {'devices': [dict(sensor_index=i, model=d.model,
                                   serial_number=d.serial_number, ip_address=d.network_info.ip)
                              for i, d in enumerate(devices)]}
        if options is not None:
            matches = [i for i, d in enumerate(devices)
                       if (not options.serial_number or d.serial_number == options.serial_number)
                       and (not options.ip_address or d.network_info.ip == options.ip_address)]
            if options.serial_number or options.ip_address:
                if len(matches) != 1:
                    raise RuntimeError('Vcamera selector must match exactly one discovered camera')
                index = matches[0]
            else:
                index = options.sensor_index
            if index >= len(devices):
                raise RuntimeError(f'Vcamera index {index}: discovered {len(devices)} camera(s)')
            report['selected_device'] = report['devices'][index]
            if args.factory_intrinsics:
                export_factory_intrinsics(options, args.factory_intrinsics)
                print(f'Saved OpenCV YAML: {args.factory_intrinsics.resolve()}')
                return 0
            report['capabilities'] = inspect_device(sdk, devices[index])
            # Pin capture to the inspected device even if discovery order changes.
            options.serial_number = devices[index].serial_number
            report['profile'] = dict(path=str(profile_path), stream=options.stream,
                                     width=options.width, height=options.height,
                                     exposure_us=options.exposure_us, gain_sdk=options.gain_sdk,
                                     exposure_mode=options.exposure_mode,
                                     exposure_warmup_frames=options.exposure_warmup_frames)
            if args.save_frame:
                report['capture'] = capture_frame(options, args.save_frame, args.timeout_ms)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + '\n', encoding='utf-8')
        print(rendered)
        return 0
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f'Vcamera inspection failed: {exc}', file=sys.stderr)
        return 1


def export_factory_intrinsics(options, path):
    adapter = VcameraSource(options)
    try:
        adapter.open()
        intrinsics = adapter.metadata().get("sdk_intrinsics")
        if not intrinsics:
            raise ValueError("Vcamera SDK 未提供当前分辨率的出厂内参")
    finally:
        adapter.close()
    write_factory_intrinsics(intrinsics, path)


def write_factory_intrinsics(intrinsics, path):
    import cv2
    import numpy as np

    path = path.resolve()
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("出厂内参输出需要 .yaml 或 .yml 文件")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    storage = cv2.FileStorage(".yml", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY | cv2.FILE_STORAGE_FORMAT_YAML)
    if not storage.isOpened():
        raise OSError(f"Cannot write {path}")
    width, height = intrinsics["image_size"]
    storage.write("image_width", width)
    storage.write("image_height", height)
    storage.write("camera_matrix", np.asarray(intrinsics["camera_matrix"], np.float64))
    storage.write("distortion_coefficients", np.asarray(intrinsics["distortion_coefficients"], np.float64).reshape(1, -1))
    with path.open("x", encoding="utf-8") as stream:
        stream.write(storage.releaseAndGetString())
