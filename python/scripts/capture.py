"""One capture workflow for RGB, monochrome and Orbbec left/right IR sensors."""
from __future__ import annotations

from pathlib import Path
import importlib.metadata
import time

import cv2
import numpy as np

from .io import sensor_image_directories, write_image, write_json
from .sensor_capture.config import sensor_options
from .sensor_capture.sources import create_frame_source, require_gui_support, disable_orbbec_logging
from .sensor_intrinsics.core import create_object_points, prepare_gray8, utc_now_iso
from .sensor_intrinsics.detection import detect_checkerboard_realtime, draw_subpixel_corners, preview_image


class OrbbecIRSource:
    """One SDK pipeline for one or both IR streams of the same device."""

    def __init__(self, sensors):
        self.sensors = sensors
        self.sdk = self.device = self.pipeline = None
        self.started = False
        self.original_settings = []
        self.profiles = {}
        self.info = {}

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
            raise RuntimeError('Use the cylmarker environment with pyorbbecsdk2 installed') from error
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
                    (properties.OB_PROP_IR_EXPOSURE_INT, 'int', sensor['exposure_sdk']),
                    (properties.OB_PROP_IR_GAIN_INT, 'int', sensor.get('gain_sdk', 16))]
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
        return {**self.info, 'backend': 'orbbec-ir',
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


def open_sources(sensors):
    """Open adapters, grouping IR streams that share one Orbbec device."""
    adapters = []
    ir_groups = {}
    for sensor in sensors:
        if sensor['sensor_type'] == 'orbbec' and sensor['stream'].startswith('ir_'):
            ir_groups.setdefault(sensor['sensor_index'], []).append(sensor)
        else:
            adapters.append((create_frame_source(sensor_options(sensor)), [sensor]))
    adapters.extend((OrbbecIRSource(group), group) for group in ir_groups.values())
    try:
        for adapter, _ in adapters:
            adapter.open()
    except Exception:
        for adapter, _ in reversed(adapters):
            adapter.close()
        raise
    return adapters


def read_sources(adapters, timeout_ms):
    images, timestamps = {}, {}
    for adapter, sensors in adapters:
        if isinstance(adapter, OrbbecIRSource):
            received, device_times = adapter.read(timeout_ms)
            images.update(received)
            timestamps.update(device_times)
        else:
            image = adapter.read(timeout_ms)
            if image is not None:
                label = sensors[0]['label']
                images[label] = image
                timestamps[label] = time.monotonic_ns() // 1000
    synchronized = len(adapters) == 1 and isinstance(adapters[0][0], OrbbecIRSource) and len(adapters[0][1]) == 2
    return images, timestamps, synchronized


def capture_session(config, session_path):
    require_gui_support()
    session = Path(session_path).resolve()
    if session.exists() and any(session.iterdir()):
        raise FileExistsError(f'Capture directory is not empty: {session}')
    sensors, board, settings = config['sensors'], config['board'], config['capture']
    pattern = (board['pattern_cols'], board['pattern_rows'])
    create_object_points(pattern, board['square_size_mm'])
    image_directories = sensor_image_directories(sensors)
    adapters = open_sources(sensors)
    metadata = {'schema_version': 2, 'calibration_type': config['mode'], 'created_at_utc': utc_now_iso(),
                'board': board, 'sensors': [dict(sensor, image_directory=image_directories[sensor['label']])
                                          for sensor in sensors],
                'images': {sensor['label']: [] for sensor in sensors}, 'pairs': []}
    saved_batches = 0
    last_saved = -float('inf')
    labels = [sensor['label'] for sensor in sensors]
    windows = []
    try:
        session.mkdir(parents=True, exist_ok=True)
        for label, directory in image_directories.items():
            (session / directory).mkdir(exist_ok=True)
            print(f'{label}: original images -> {session / directory}')
        print(f'Capture session: {session}')
        print('Space: save original images, including any detection failures for review. Q/Esc: finish.')
        print('Stop the board and sensors before saving. For separate devices, exposure synchronization is not guaranteed.')
        while True:
            images, timestamps, synchronized = read_sources(adapters, settings['timeout_ms'])
            metadata['capture_mode'] = 'synchronized_ir' if synchronized else 'static_board_sequential_reads'
            detected = {}
            for label in labels:
                image = images.get(label)
                name = f'Sensor calibration: {label}'
                if name not in windows:
                    windows.append(name)
                if image is None:
                    detected[label] = False
                    preview = np.zeros((200, 480), np.uint8)
                    cv2.putText(preview, 'No frame received', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, 255, 2)
                else:
                    gray = prepare_gray8(image)
                    found, corners, route, _, _ = detect_checkerboard_realtime(gray, pattern, max_dimension=config['calibration']['detect_max_dim'])
                    detected[label] = found
                    preview, _ = preview_image(gray, settings['preview_max_width'], settings['preview_max_height'])
                    if found:
                        axis_scale = np.array([preview.shape[1] / gray.shape[1], preview.shape[0] / gray.shape[0]], np.float32)
                        draw_subpixel_corners(preview, corners * axis_scale)
                    text = f"{label}: {'found' if found else 'not found'} / {len(metadata['images'][label])} saved"
                    cv2.putText(preview, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, 180, 2)
                cv2.imshow(name, preview)
            key = cv2.waitKey(1) & 0xff
            closed = any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in windows)
            if closed or key in (27, ord('q'), ord('Q')):
                break
            if key != ord(' '):
                continue
            if not images:
                print('Nothing saved: no sensor frames received')
                continue
            if time.monotonic() - last_saved < settings.get('min_interval_seconds', 1.0):
                print('Nothing saved: move the board to a different pose before saving again')
                continue
            names = {}
            for label in labels:
                if label not in images:
                    continue
                name = f"{image_directories[label]}/{len(metadata['images'][label]):04d}.png"
                write_image(session / name, images[label])
                metadata['images'][label].append(name)
                names[label] = name
            capture = {'images': names, 'detection_found': detected,
                       'timestamp_basis': 'sensor_exposure' if synchronized else 'host_receive_or_device_mixed',
                       'timestamps_us': {label: timestamps[label] for label in names}}
            if len(names) == 2:
                # Compare device timestamps only within a synchronized IR pipeline.
                capture['capture_delta_ms'] = abs(timestamps[labels[1]] - timestamps[labels[0]]) / 1000 if synchronized else None
                metadata['pairs'].append(capture.copy())
            for entry in metadata['sensors']:
                label = entry['label']
                if label in images:
                    entry['image_size'] = [images[label].shape[1], images[label].shape[0]]
            for adapter, group in adapters:
                for sensor in group:
                    entry = next(item for item in metadata['sensors'] if item['label'] == sensor['label'])
                    entry['capture_metadata'] = adapter.metadata(sensor['label']) if isinstance(adapter, OrbbecIRSource) else adapter.metadata()
                if synchronized:
                    metadata.update(adapter.sdk_pair_metadata())
            write_json(session / 'capture_metadata.json', metadata)
            saved_batches += 1
            last_saved = time.monotonic()
            missing = [label for label, found in detected.items() if not found]
            print(f'Saved capture {saved_batches}; detection not found: {missing or "none"}; paired captures: {len(metadata["pairs"])}')
            count = settings.get('target_count', 40)
            if count and saved_batches >= count:
                break
        if not (session / 'capture_metadata.json').exists():
            write_json(session / 'capture_metadata.json', metadata)
    finally:
        failures = []
        for adapter, _ in reversed(adapters):
            try:
                adapter.close()
            except Exception as error:
                failures.append(str(error))
        cv2.destroyAllWindows()
        if failures:
            raise RuntimeError('Sensor cleanup failed: ' + '; '.join(failures))
    return session
