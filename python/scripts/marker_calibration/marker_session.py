"""Capture raw ArUco views with the existing camera adapters and timestamp layout."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter

import cv2
import numpy as np

from ..sensor_capture.sources import open_sources, read_sources
from ..session_storage import sensor_image_directories, utc_now_iso, write_image, write_json
from ..sensor_capture.preview import CaptureView
from ..sensor_intrinsics.detection import prepare_gray8
from .marker_io import create_detector, detect, detect_gray, detection_gray, load_camera, refine_marker_corners


class StableLabels:
    """Display-only label hysteresis and a short, explicitly marked lost state."""

    def __init__(self, deadband_px=2.0, hold_seconds=0.35):
        self.deadband_px = deadband_px
        self.hold_seconds = hold_seconds
        self.labels = {}

    def update(self, corners, now=None):
        now = time.monotonic() if now is None else now
        for mid in list(self.labels):
            if now - self.labels[mid]['seen'] > self.hold_seconds:
                del self.labels[mid]
        for mid, points in corners.items():
            center = np.asarray(points).mean(axis=0)
            if mid not in self.labels:
                self.labels[mid] = {'position': center.copy(), 'seen': now}
            else:
                record = self.labels[mid]
                delta = center - record['position']
                distance = np.linalg.norm(delta)
                if distance > 40:
                    record['position'] = center.copy()
                elif distance > self.deadband_px:
                    record['position'] += 0.4 * delta
                record['seen'] = now
        return {mid: (tuple(np.rint(record['position']).astype(int)), mid in corners)
                for mid, record in self.labels.items()}


class CaptureGroups:
    """Track qualified markers, overlap and geometric baseline in each placement."""

    def __init__(self, marker, options, metadata, camera):
        from .checkerboard import object_board

        self.ids = set(marker['ids'])
        self.reference = marker['reference_marker_id']
        self.options = options
        self.metadata = metadata
        if camera is not None:
            self.set_camera(camera)
        spec = marker['checkerboard']
        self.pattern = (spec['pattern_cols'], spec['pattern_rows'])
        self.board = object_board(self.pattern, spec['square_size_mm'])
        self.center_cache_key = None
        self.triangulated_corners = {}

    @property
    def index(self):
        return self.metadata['capture_group']

    def set_camera(self, camera):
        self.k, self.d = camera['matrix'], camera['distortion']

    def records(self, group):
        return [record for record in self.metadata['saved_views'] if record['group'] == group]

    def coverage(self, group):
        records = self.records(group)
        coverage = {}
        for mid in sorted(self.ids):
            train = [r['rays'][str(mid)] for r in records if r['split'] == 'train' and str(mid) in r['rays']]
            test = sum(mid in r['ids'] for r in records if r['split'] == 'test')
            angle = 0.0
            if len(train) >= 2:
                rays = np.asarray(train)
                angle = float(np.rad2deg(np.arccos(np.clip((rays @ rays.T).min(), -1, 1))))
            if train or test:
                coverage[mid] = (len(train), test, angle)
        return coverage

    def qualified(self, group):
        return {mid for mid, (train, test, angle) in self.coverage(group).items()
                if train >= self.options['min_marker_views'] and test
                and angle >= self.options['min_view_angle_deg']}

    def status(self):
        qualified = self.qualified(self.index)
        previous = set().union(*(self.qualified(i) for i in range(self.index)))
        records = self.records(self.index)
        missing = sorted(self.ids - previous - qualified)
        train_total = sum(r['split'] == 'train' for r in self.metadata['saved_views'])
        bridge = qualified & previous
        minimum = self.options['min_group_views']
        ready = len(records) >= minimum and len(qualified) >= 2 and (
            bool(bridge) if self.index else self.reference in qualified
        )
        ready_groups = self.index + int(ready)
        return {
            'group': self.index + 1, 'images': len(records), 'minimum_images': minimum, 'ready': ready,
            'ready_groups': ready_groups, 'minimum_groups': self.options['min_groups'],
            'complete': (ready or self.index > 0 and not records) and not missing
                        and ready_groups >= self.options['min_groups']
                        and train_total >= self.options['min_views'],
            'missing': missing, 'qualified': sorted(qualified), 'bridge': sorted(bridge),
            'previous': sorted(previous),
        }

    def board_pose(self, image, records):
        from .checkerboard import board_halfturn, checked_board_pnp, detect_full_board

        pixels = detect_full_board(image, self.pattern)
        if pixels is None:
            cols, rows = self.pattern
            raise ValueError(f'棋盘格未完整识别，需要 {cols}×{rows} 共 {cols * rows} 个内角点。'
                             '请让棋盘格完整入镜并避免遮挡或反光。')
        pose, _, quality = checked_board_pnp(self.board, pixels, self.k, self.d, self.options)
        if pose is None:
            raise ValueError(quality["reason"])
        if records:
            previous = np.asarray(records[-1]['board_rotation'])
            flipped = pose @ board_halfturn(self.board)
            if np.trace(previous.T @ flipped[:3, :3]) > np.trace(previous.T @ pose[:3, :3]):
                pose = flipped
        return pose

    def measure(self, image, detector):
        records = self.records(self.index)
        pose = self.board_pose(image, records)
        corners, _ = detect(image, detector)
        visible = {mid: corners[mid] for mid in sorted(self.ids & corners.keys())}
        if len(visible) < 2:
            raise ValueError('Show at least two target markers')
        status = self.status()
        if not status['ready'] and self.index and not set(status['previous']).intersection(visible):
            raise ValueError(f'Keep an overlap marker visible from {status["previous"]}')
        if not self.index and self.reference not in visible:
            raise ValueError(f'Show top ID {self.reference} together with a side marker in group 1')
        centers = np.array([points.mean(axis=0) for points in visible.values()])
        xy = cv2.undistortPoints(centers.reshape(-1, 1, 2), self.k, self.d).reshape(-1, 2)
        rays = np.c_[xy, np.ones(len(xy))] @ pose[:3, :3]
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        measured = {str(mid): ray.tolist() for mid, ray in zip(visible, rays)}
        position = -pose[:3, :3].T @ pose[:3, 3]
        corner_pixels = np.concatenate(list(visible.values()))
        normalized = cv2.undistortPoints(corner_pixels.reshape(-1, 1, 2), self.k, self.d).reshape(-1, 2)
        world_rays = (np.c_[normalized, np.ones(len(normalized))] @ pose[:3, :3]).reshape(-1, 4, 3)
        corner_rays = dict(zip(measured, world_rays))
        epipolar_errors = []
        for record in records:
            overlap = measured.keys() & record['rays'].keys()
            baseline = position - np.asarray(record['camera_position'])
            if np.linalg.norm(baseline) < 1.0:
                raise ValueError('Move the camera to a different position, including sideways or up/down')
            if overlap and all(np.dot(measured[mid], record['rays'][mid]) > np.cos(np.deg2rad(1))
                               for mid in overlap):
                raise ValueError('View is too close to a saved view. Move the camera sideways or change height')
            for mid in overlap:
                normals = np.cross(baseline, record['corner_rays'][mid])
                norms = np.linalg.norm(normals, axis=1)
                valid = norms > 0.05 * np.linalg.norm(baseline)
                distances = np.abs(np.sum(normals[valid] * corner_rays[mid][valid], axis=1)) / norms[valid]
                epipolar_errors.extend(distances * min(self.k[0, 0], self.k[1, 1]))
        if epipolar_errors and np.median(epipolar_errors) > self.options['max_validation_rmse_px']:
            raise ValueError('Target position appears to have changed. Restore it or start a new group when ready')
        return {
            'group': self.index, 'ids': list(visible), 'ids_source': 'full_resolution',
            'split': 'test' if (len(records) + 1) % self.options['validation_stride'] == 0 else 'train',
            'rays': measured, 'board_rotation': pose[:3, :3].tolist(),
            'camera_position': position.tolist(),
            'corner_rays': {mid: rays.tolist() for mid, rays in corner_rays.items()},
        }

    def triangulate_markers(self, records):
        key = self.index, len(records)
        if key == self.center_cache_key:
            return self.triangulated_corners
        coverage = self.coverage(self.index)
        markers = {}
        for mid, (count, _, angle) in coverage.items():
            if count < 3 or angle < self.options['min_view_angle_deg']:
                continue
            observations = [record for record in records if record['split'] == 'train'
                            and str(mid) in record.get('corner_rays', {})]
            if len(observations) < 3:
                continue
            rays = np.asarray([record['corner_rays'][str(mid)] for record in observations])
            cameras = np.asarray([record['camera_position'] for record in observations])
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
            perpendicular = np.eye(3) - rays[..., :, None] * rays[..., None, :]
            lhs = perpendicular.sum(axis=0)
            rhs = np.einsum('ncij,nj->ci', perpendicular, cameras)
            try:
                corners = np.linalg.solve(lhs, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                continue
            if np.all(np.sum((corners[None, ...] - cameras[:, None, :]) * rays, axis=-1) > 0):
                markers[mid] = corners
        self.center_cache_key = key
        self.triangulated_corners = markers
        return markers

    def live_distances(self, image, preview, corners):
        from .checkerboard import project

        ids = sorted(self.ids & corners.keys())
        distances = dict.fromkeys(ids)
        records = self.records(self.index)
        markers = self.triangulate_markers(records)
        ready = [mid for mid in ids if mid in markers]
        if ready:
            try:
                pose = self.board_pose(image, records)
            except (ValueError, cv2.error):
                return {'status': 'board_unavailable', 'mm': distances}
            position = -pose[:3, :3].T @ pose[:3, 3]
            scale = np.asarray(image.shape[1::-1]) / np.asarray(preview.shape[1::-1])
            for mid in ready:
                points = markers[mid]
                if np.any((points @ pose[:3, :3].T + pose[:3, 3])[:, 2] <= 0):
                    continue
                prediction = project(points, pose, self.k, self.d)
                observed = (corners[mid] + 0.5) * scale - 0.5
                rmse = np.sqrt(np.mean(np.sum((prediction - observed)**2, axis=1)))
                if rmse <= 2 * self.options['max_validation_rmse_px']:
                    distances[mid] = float(np.linalg.norm(points.mean(axis=0) - position))
        return {'status': 'ok' if any(v is not None for v in distances.values()) else 'unavailable',
                'mm': distances}


def live_preview(image, detector, max_dimension=1200, label_state=None):
    """Bound live work to preview pixels; calibration redetects the saved originals."""
    original = prepare_gray8(image)
    gray = original
    scale = min(1.0, max_dimension / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, (max(1, round(gray.shape[1] * scale)),
                                 max(1, round(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    gray = detection_gray(gray)
    corners, duplicates = detect_gray(gray, detector)
    axis_scale = np.array([gray.shape[1] / original.shape[1], gray.shape[0] / original.shape[0]])
    full_corners = {mid: (points + 0.5) / axis_scale - 0.5 for mid, points in corners.items()}
    full_corners = refine_marker_corners(original, full_corners)
    corners = {mid: (points + 0.5) * axis_scale - 0.5 for mid, points in full_corners.items()}
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for mid, points in corners.items():
        polygon = np.rint(points * 256).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(preview, [polygon], True, (0, 255, 0), 1, cv2.LINE_AA, 8)
        for point in polygon[:, 0]:
            cv2.circle(preview, tuple(int(v) for v in point), 2 * 256, (0, 0, 255), 1, cv2.LINE_AA, 8)
    label_state = StableLabels() if label_state is None else label_state
    for mid, (position, current) in label_state.update(corners).items():
        text = str(mid) if current else f'{mid}?'
        color = (0, 255, 0) if current else (160, 160, 160)
        cv2.putText(preview, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return preview, corners, duplicates


def capture_session(config, session, camera, resume=False):
    with CaptureView() as view:
        return _capture_session(config, session, camera, resume, view)


def _capture_session(config, session, camera, resume, view):
    view.title = config.get('capture_title', 'Marker 几何标定 · 相机采集')
    sensor = dict(config['sensor'], label='marker_camera')
    directories = sensor_image_directories([sensor])
    directory = directories['marker_camera']
    detector = create_detector(config['marker']['dictionary'])
    label_state = StableLabels()
    adapters, paths, seen = [], [], Counter()
    use_factory = config["camera_parameters"] is None
    session_camera_path = session / 'camera_params.yml'
    sensor_record = {key: str(value) if hasattr(value, '__fspath__') else value for key, value in sensor.items()}
    metadata = {'schema_version': 2, 'calibration_type': 'marker_geometry',
                'created_at_utc': utc_now_iso(),
                'sensors': [dict(sensor_record, image_directory=directory)],
                'images': {'marker_camera': paths}, 'marker': config['marker'],
                'camera_parameters_source': 'Vcamera SDK' if use_factory else str(config['camera_parameters']),
                'camera_parameters': 'camera_params.yml', 'saved_views': []}
    session.mkdir(parents=True, exist_ok=True)
    if resume:
        metadata = json.loads((session / 'capture_metadata.json').read_text(encoding='utf-8-sig'))
        if (metadata.get('calibration_type') != 'marker_geometry' or
                metadata.get('marker', {}).get('dictionary') != config['marker']['dictionary'] or
                metadata['sensors'][0].get('image_directory') != directory):
            raise ValueError('Resume session must match the camera image directory and marker dictionary')
        paths = metadata['images']['marker_camera']
        if any(not (session / path).is_file() for path in paths):
            raise ValueError('Resume session has missing original images')
        seen.update(mid for view in metadata['saved_views'] for mid in view['ids'])
    else:
        if any(session.iterdir()):
            raise ValueError(f'Capture directory is not empty: {session}; use resume explicitly')
        if not use_factory:
            shutil.copyfile(config['camera_parameters'], session_camera_path)
    groups = None
    if config['marker'].get('checkerboard'):
        if resume and ('capture_group' not in metadata or metadata['marker'] != config['marker']):
            raise ValueError('Resume requires an existing grouped capture with the same marker and checkerboard settings')
        metadata.setdefault('capture_group', 0)
        groups = CaptureGroups(config['marker'], config['calibration'], metadata, camera)
        if resume:
            short = [i + 1 for i in range(groups.index)
                     if len(groups.records(i)) < groups.options['min_group_views']]
            if short:
                raise ValueError(f'已结束的组 {short} 未达到每组 {groups.options["min_group_views"]} 张有效图，请新建会话采集。')
        view.group_status = groups.status()
        metadata['capture_complete'] = view.group_status['complete']
        view.log(f'Group {groups.index + 1}: keep board and target still, move camera. '
                 f'At least {groups.options["min_groups"]} groups and '
                 f'{groups.options["min_group_views"]} accepted images per group. '
                 f'Each qualified ID needs {groups.options["min_marker_views"]} training shots, '
                 f'one held-out shot and {groups.options["min_view_angle_deg"]:g} deg of viewing spread.')
        if resume:
            view.log('Resume the same physical placement in the current group. Use Next group before moving the target.')
    write_json(session / 'capture_metadata.json', metadata)
    title = 'Instrument calibration | SPACE save | Q / ESC finish'
    def open_camera():
        nonlocal camera
        opened = open_sources([sensor])
        try:
            current = opened[0][0].metadata()
            previous = metadata['sensors'][0].get('capture_metadata', {})
            old_serial = previous.get('serial_number') or previous.get('device_serial') or previous.get('device_info', {}).get('sn')
            new_serial = current.get('serial_number') or current.get('device_serial') or current.get('device_info', {}).get('sn')
            if old_serial and new_serial != old_serial:
                raise ValueError('Cannot capture into the same session with a different camera serial number')
            factory = current.get('sdk_intrinsics') or {}
            if use_factory:
                from ..sensor_capture.vcamera import write_factory_intrinsics

                if not factory:
                    raise ValueError('Vcamera SDK 未提供当前采集模式的出厂内参。')
                if camera is None:
                    write_factory_intrinsics(factory, session_camera_path)
                    camera = load_camera(session_camera_path)
                    metadata['camera_parameters_source'] = factory.get('source', 'Vcamera SDK')
                    if groups:
                        groups.set_camera(camera)
                    view.log(f'已读取 Vcamera SDK 出厂内参：{camera["image_size"]}。')
                else:
                    matrix = np.asarray(factory['camera_matrix'])
                    distortion = np.asarray(factory['distortion_coefficients']).reshape(-1)
                    if (matrix.shape != camera['matrix'].shape or distortion.shape != camera['distortion'].shape
                            or not np.allclose(camera['matrix'], matrix) or not np.allclose(camera['distortion'], distortion)):
                        raise ValueError('Vcamera 当前出厂内参与会话内参不同，请新建采集会话。')
            size = current.get('image_size') or factory.get('image_size')
            if size is not None and list(size) != list(camera['image_size']):
                raise ValueError(f'相机图像尺寸 {size} 与所选内参的尺寸 {camera["image_size"]} 不一致。')
            metadata['sensors'][0]['capture_metadata'] = current
            write_json(session / 'capture_metadata.json', metadata)
            return opened
        except Exception:
            for adapter, _ in reversed(opened):
                adapter.close()
            raise

    try:
        view.open_window(title, 1100, 1000)
        started = time.monotonic()
        last_frame = started
        next_distance_update = started
        while True:
            if view.finish.is_set():
                break
            if not view.prepare(adapters, open_camera):
                view.marker_distances = None
                next_distance_update = 0
                last_frame = time.monotonic()
                time.sleep(0.05)
                continue
            images, _, _ = read_sources(adapters, 1000)
            if 'marker_camera' not in images:
                if time.monotonic() - last_frame > 15:
                    raise RuntimeError('Camera returned no frames for 15 seconds')
                if view.key() in (27, ord('q'), ord('Q')):
                    break
                continue
            last_frame = time.monotonic()
            image = images['marker_camera']
            if [image.shape[1], image.shape[0]] != camera['image_size']:
                raise ValueError('Live image dimensions do not match camera calibration')
            preview, corners, duplicates = live_preview(image, detector, label_state=label_state)
            lines = [f'SPACE: save   Q/ESC: finish   Saved: {len(paths)}',
                     f'Preview IDs: {sorted(corners)}', f'Observed IDs: {dict(sorted(seen.items()))}']
            if groups:
                status = view.group_status
                if last_frame >= next_distance_update:
                    view.marker_distances = groups.live_distances(image, preview, corners)
                    next_distance_update = last_frame + 0.5
                lines += [
                    f'Group {status["group"]}: {status["images"]} saved   Ready groups: '
                    f'{status["ready_groups"]}/{status["minimum_groups"]}',
                    f'Missing overall: {status["missing"]}   Bridge IDs: {status["bridge"]}',
                    'N: next group, then move target' if status['ready'] else
                    (f'Show top {groups.reference} + side' if not groups.index else
                     f'Keep overlap with earlier groups: {status["previous"]}'),
                ]
                if not view.web and view.marker_distances['mm']:
                    ranges = [f'{mid} ~{mm:.0f}' if mm is not None else f'{mid} --'
                              for mid, mm in view.marker_distances['mm'].items()]
                    lines.append('Marker range (mm): ' + ' / '.join(ranges))
                if not view.web and view.marker_save_feedback and not view.marker_save_feedback['saved']:
                    lines.append('Last shot rejected. See terminal for reason.')
                for mid, (train, test, angle) in groups.coverage(groups.index).items():
                    lines.append(f'{mid}: train {train}, holdout {test}, spread {angle:.1f} deg')
            if duplicates:
                lines.append(f'Duplicate IDs rejected: {duplicates}')
            for i, line in enumerate(lines):
                cv2.putText(preview, line, (15, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            view.show(title, preview)
            key = view.key()
            if key in (27, ord('q'), ord('Q')):
                break
            if groups and key in (ord('n'), ord('N')):
                if not view.group_status['ready']:
                    view.log('Current group is incomplete. Continue moving the camera and check the ready IDs.')
                    continue
                metadata['capture_group'] += 1
                view.group_status = groups.status()
                metadata['capture_complete'] = view.group_status['complete']
                view.marker_distances = None
                view.marker_save_feedback = None
                next_distance_update = 0
                write_json(session / 'capture_metadata.json', metadata)
                view.log(f'Group {groups.index + 1} opened. Reposition the target now, then keep it still. '
                         f'Keep an overlap ID from {view.group_status["previous"]}.')
                continue
            if key in (32, ord('c'), ord('C')):
                allowed = config['marker']['ids']
                visible = corners if allowed is None else {mid: p for mid, p in corners.items() if mid in allowed}
                if groups:
                    try:
                        record = groups.measure(image, detector)
                    except (ValueError, cv2.error) as error:
                        message = f'未保存当前帧。{error}'
                        view.marker_save_feedback = {'saved': False, 'message': message}
                        view.log(message)
                        continue
                else:
                    record = {'ids': sorted(visible), 'ids_source': 'preview'}
                if not groups and len(visible) < 2:
                    print('Warning: fewer than two target IDs; saving raw image for offline detection. Show multiple faces for co-visibility.', flush=True)
                relative = f'{directory}/{len(paths):04d}.png'
                if (session / relative).exists():
                    raise ValueError(f'Refusing to overwrite original capture: {relative}')
                write_image(session / relative, image)
                paths.append(relative)
                seen.update(record['ids'])
                current = adapters[0][0].metadata()
                metadata['sensors'][0]['capture_metadata'] = current
                metadata['saved_views'].append(dict(record, image=relative, elapsed_s=time.monotonic() - started))
                if groups:
                    view.group_status = groups.status()
                    metadata['capture_complete'] = view.group_status['complete']
                write_json(session / 'capture_metadata.json', metadata)
                if groups:
                    view.marker_save_feedback = {'saved': True, 'message': f'已保存 {relative}'}
                view.log(f'Saved {relative}: IDs={record["ids"]}; counts={dict(sorted(seen.items()))}')
                if groups:
                    status = view.group_status
                    if status['complete']:
                        view.log('Coverage complete. Finish capture to calibrate.')
                    elif status['ready']:
                        view.log(f'Group ready. Capture more views or start the next group. Missing IDs: {status["missing"]}')
    finally:
        for adapter, _ in reversed(adapters):
            adapter.close()
    if groups and not groups.status()['complete']:
        view.log(f'Partial session saved. Missing qualified IDs: {groups.status()["missing"]}.')
    return len(paths)
