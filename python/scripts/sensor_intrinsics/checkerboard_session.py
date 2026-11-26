"""Capture a checkerboard session for sensor intrinsic or extrinsic calibration."""

from pathlib import Path
import time
from collections import deque

import cv2
import numpy as np

from ..session_storage import sensor_image_directories, utc_now_iso, write_image, write_json
from ..sensor_capture.orbbec import OrbbecIRSource
from ..sensor_capture.sources import read_sources
from ..sensor_capture.preview import CaptureView
from .detection import create_object_points, detect_checkerboard_realtime, draw_subpixel_corners, prepare_gray8, preview_image


def capture_session(config, session_path):
    with CaptureView() as view:
        return _capture_session(config, session_path, view)


def _capture_session(config, session_path, view):
    session = Path(session_path).resolve()
    if session.exists() and any(session.iterdir()):
        raise FileExistsError(f'Capture directory is not empty: {session}')
    sensors, board, settings = config['sensors'], config['board'], config['capture']
    pattern = (board['pattern_cols'], board['pattern_rows']) if board else None
    if pattern:
        create_object_points(pattern, board['square_size_mm'])
    image_directories = sensor_image_directories(sensors)
    adapters = []
    metadata = {'schema_version': 2, 'calibration_type': config['mode'], 'created_at_utc': utc_now_iso(),
                'board': board, 'sensors': [dict(sensor, image_directory=image_directories[sensor['label']])
                                          for sensor in sensors],
                'images': {sensor['label']: [] for sensor in sensors}, 'pairs': []}
    saved_batches = 0
    last_saved = -float('inf')
    labels = [sensor['label'] for sensor in sensors]
    view.title = {
        'sensor_intrinsics': '普通内参 · 相机采集',
        'sensor_extrinsics': '双目外参 · 相机采集',
        'camera_preview': '实时相机预览',
    }[config['mode']]
    shared_ir = len(sensors) == 2 and all(s['sensor_type'] == 'orbbec' and s['stream'].startswith('ir_') for s in sensors) and sensors[0]['sensor_index'] == sensors[1]['sensor_index']
    groups = {' / '.join(labels): sensors} if shared_ir else {s['label']: [s] for s in sensors}
    view.device_states = dict.fromkeys(groups, 'disconnected')
    view.required_frames = len(labels)
    view.camera_slots = [f"Sensor calibration: {label}" for label in labels]
    view.pair_ready = False
    failed = []
    tracking = config.get('live_tracking')
    tracking_error = None
    arrivals = deque()
    preview_only = config['mode'] == 'camera_preview'
    if view.web and preview_only and len(labels) == 1:
        view.video_directory = session

    def on_read_error(adapter, group, error):
        failed.append((adapter, group))
        view.log(f'{", ".join(sensor["label"] for sensor in group)} 取帧失败：{error}')

    try:
        if not preview_only:
            session.mkdir(parents=True, exist_ok=True)
            for label, directory in image_directories.items():
                (session / directory).mkdir(exist_ok=True)
                print(f'{label}: original images -> {session / directory}')
            print(f'Capture session: {session}')
        print('Space: save original images, including any detection failures for review. Q/Esc: finish.')
        if pattern:
            print('Stop the board and sensors before saving. For separate devices, exposure synchronization is not guaranteed.')
        while True:
            if not view.video_requested.is_set() and view.video_writer is not None:
                view.stop_video()
            if view.finish.is_set():
                break
            if not view.prepare_sources(adapters, groups):
                arrivals.clear()
                view.preview_timing = None
                time.sleep(0.05)
                continue
            images, timestamps, synchronized = read_sources(
                adapters, settings['timeout_ms'], on_error=on_read_error if view.web else None)
            received = time.monotonic()
            if not images:
                arrivals.clear()
                view.preview_timing = None
            for adapter, group in failed:
                adapters.remove((adapter, group))
                key = next(key for key, configured in groups.items() if group[0]["label"] == configured[0]["label"])
                view.device_states[key] = 'failed'
                try:
                    adapter.close()
                except Exception as error:
                    view.log(f'{key} 关闭失败：{error}')
            if failed:
                view.stop_video()
                view.refresh_controls(adapters)
                failed.clear()
            view.pair_ready = all(label in images for label in labels)
            if view.pair_ready and synchronized:
                delta_ms = abs(timestamps[labels[1]] - timestamps[labels[0]]) / 1000
                view.pair_ready = delta_ms <= settings['max_pair_delta_ms']
            metadata['capture_mode'] = ('single_camera' if not pattern else
                                        'synchronized_ir' if synchronized else 'static_board_sequential_reads')
            detected = {}
            for label in labels:
                image = images.get(label)
                name = f'Sensor calibration: {label}'
                if image is None:
                    detected[label] = False
                    if view.web:
                        view.images.pop(name, None)
                        with view.lock:
                            view.frames = {title: frame for title, frame in view.frames.items() if title != name}
                        continue
                    preview = np.zeros((200, 480), np.uint8)
                    cv2.putText(preview, 'No frame received', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, 255, 2)
                elif not pattern:
                    arrivals.append(received)
                    while arrivals and arrivals[0] < received - 2:
                        arrivals.popleft()
                    acquisition_fps = ((len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
                                       if len(arrivals) > 1 and arrivals[-1] > arrivals[0] else None)
                    display = image if image.dtype == np.uint8 else prepare_gray8(image)
                    preview, _ = preview_image(display, settings['preview_max_width'], settings['preview_max_height'])
                    tracking_ms = None
                    if tracking:
                        tracking_started = time.monotonic()
                        try:
                            preview = tracking.overlay(image, preview, adapters[0][0])
                            tracking_error = None
                            tracking_ms = (time.monotonic() - tracking_started) * 1000
                        except (ValueError, KeyError, RuntimeError, cv2.error) as error:
                            preview = tracking.with_hud(preview, ["Marker tracking unavailable; see camera log"])
                            if str(error) != tracking_error:
                                tracking_error = str(error)
                                view.log(f'Marker 追踪无法叠加：{error}')
                    view.preview_timing = {
                        'acquisition_fps': acquisition_fps, 'tracking_ms': tracking_ms,
                        'processing_ms': (time.monotonic() - received) * 1000,
                    }
                else:
                    gray = prepare_gray8(image)
                    found, corners, route, _, _ = detect_checkerboard_realtime(gray, pattern, max_dimension=config['calibration']['detect_max_dim'])
                    detected[label] = found
                    display = image if image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8 else gray
                    preview, _ = preview_image(display, settings['preview_max_width'], settings['preview_max_height'])
                    if found:
                        axis_scale = np.array([preview.shape[1] / gray.shape[1], preview.shape[0] / gray.shape[0]], np.float32)
                        draw_subpixel_corners(preview, corners * axis_scale)
                    text = f"{label}: {'found' if found else 'not found'} / {len(metadata['images'][label])} saved"
                    cv2.putText(preview, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (180, 180, 180), 2)
                view.show(name, preview)
            key = view.key()
            if key in (27, ord('q'), ord('Q')):
                break
            if key != ord(' '):
                continue
            if not view.pair_ready:
                view.log('未保存：当前没有收到全部所需相机的有效帧。')
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
            if preview_only and not saved_batches:
                print(f'Capture session: {session}', flush=True)
            saved_batches += 1
            last_saved = time.monotonic()
            missing = [label for label, found in detected.items() if not found]
            print(f'Saved capture {saved_batches}'
                  + (f'; detection not found: {missing or "none"}; paired captures: {len(metadata["pairs"])}' if pattern else ''))
            count = settings.get('target_count', 40)
            if count and saved_batches >= count:
                break
        if not preview_only and not (session / 'capture_metadata.json').exists():
            write_json(session / 'capture_metadata.json', metadata)
    finally:
        view.stop_video()
        failures = []
        for adapter, _ in reversed(adapters):
            try:
                adapter.close()
            except Exception as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError('Sensor cleanup failed: ' + '; '.join(failures))
    return session
