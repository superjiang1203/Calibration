"""Capture a checkerboard session for sensor intrinsic or extrinsic calibration."""

from pathlib import Path
import time

import cv2
import numpy as np

from ..session_storage import sensor_image_directories, utc_now_iso, write_image, write_json
from ..sensor_capture.orbbec import OrbbecIRSource
from ..sensor_capture.sources import open_sources, read_sources
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
    pattern = (board['pattern_cols'], board['pattern_rows'])
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
    try:
        session.mkdir(parents=True, exist_ok=True)
        for label, directory in image_directories.items():
            (session / directory).mkdir(exist_ok=True)
            print(f'{label}: original images -> {session / directory}')
        print(f'Capture session: {session}')
        print('Space: save original images, including any detection failures for review. Q/Esc: finish.')
        print('Stop the board and sensors before saving. For separate devices, exposure synchronization is not guaranteed.')
        while True:
            if view.finish.is_set():
                break
            if not view.prepare(adapters, lambda: open_sources(sensors)):
                time.sleep(0.05)
                continue
            images, timestamps, synchronized = read_sources(adapters, settings['timeout_ms'])
            metadata['capture_mode'] = 'synchronized_ir' if synchronized else 'static_board_sequential_reads'
            detected = {}
            for label in labels:
                image = images.get(label)
                name = f'Sensor calibration: {label}'
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
                view.show(name, preview)
            key = view.key()
            if key in (27, ord('q'), ord('Q')):
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
        if failures:
            raise RuntimeError('Sensor cleanup failed: ' + '; '.join(failures))
    return session
