"""Configured image detection, fixed camera parameters and server-compatible YAML."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..session_storage import write_image, write_json
from ..sensor_intrinsics.detection import prepare_gray8
from ..workflow.config import PROJECT_ROOT, validate_sensor_profile

DEFAULT_CONFIG = PROJECT_ROOT / 'configs' / 'markers' / 'double_cube.json'
DEFAULT_OPTIONS = {
    'min_views': 15, 'min_marker_views': 5, 'min_pair_views': 3,
    'validation_stride': 5, 'max_pnp_rmse_px': 3.0,
    'vote_translation_ratio': 0.5, 'vote_rotation_deg': 25.0,
    'loss_scale_px': 1.0, 'max_observation_rmse_px': 2.0,
    'max_validation_rmse_px': 3.0, 'max_outlier_fraction': 0.2,
    'min_length_ratio': 0.7, 'max_length_ratio': 1.3,
    'min_view_angle_deg': 10.0, 'max_nfev': 500,
}


def load_config(path):
    path = Path(path)
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    allowed = {'schema_version', 'mode', 'sensor_config', 'camera_parameters', 'marker', 'calibration', 'output'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('Unknown marker workflow keys')
    if config.get('schema_version') != 2 or config.get('mode') != 'marker_geometry':
        raise ValueError('Expected schema_version 2 and mode marker_geometry')
    for key in ('sensor_config', 'camera_parameters'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f'{key} must be a path string')
        value = Path(config[key])
        config[key] = (value if value.is_absolute() else path.parent / value).resolve()
    profile_path = config['sensor_config']
    config['sensor'] = validate_sensor_profile(json.loads(profile_path.read_text(encoding='utf-8-sig')), profile_path, 'marker_camera')
    marker = config['marker']
    if set(marker) - {'dictionary', 'expected_count', 'ids', 'reference_marker_id', 'reference_marker_length_mm', 'length_source', 'geometry_model', 'checkerboard'}:
        raise ValueError('Unknown marker settings')
    model = marker.setdefault('geometry_model', 'square')
    if model not in ('square', 'rectangle'):
        raise ValueError('marker.geometry_model must be square or rectangle')
    board = marker.get('checkerboard')
    if board is not None:
        if model != 'rectangle' or not isinstance(board, dict) or set(board) != {'pattern_cols', 'pattern_rows', 'square_size_mm'}:
            raise ValueError('Rectangle checkerboard requires pattern_cols, pattern_rows and square_size_mm')
        if any(type(board[key]) is not int or board[key] < 3 for key in ('pattern_cols', 'pattern_rows')):
            raise ValueError('Checkerboard pattern must have at least 3 x 3 inner corners')
        spacing = board['square_size_mm']
        if type(spacing) not in (int, float) or not np.isfinite(spacing) or spacing <= 0:
            raise ValueError('Checkerboard square_size_mm must be positive and finite')
        if marker.get('reference_marker_length_mm') is not None:
            raise ValueError('Checkerboard scale requires reference_marker_length_mm to be null')
        marker['length_source'] = f'checkerboard_{spacing:g}_mm'
    count = marker.get('expected_count')
    if type(count) is not int or count < 1:
        raise ValueError('marker.expected_count must be a positive integer')
    length = marker.get('reference_marker_length_mm')
    if not (model == 'rectangle' and length is None):
        if isinstance(length, bool) or not isinstance(length, (int, float)) or not np.isfinite(length) or length <= 0:
            raise ValueError('reference_marker_length_mm must be a positive known length')
    dictionary = getattr(cv2.aruco, marker['dictionary'], None)
    if dictionary is None:
        raise ValueError('Unsupported ArUco dictionary')
    dictionary_size = len(cv2.aruco.getPredefinedDictionary(dictionary).bytesList)
    ids = marker.get('ids')
    if board is not None and ids is None:
        raise ValueError('Guided checkerboard capture requires an explicit marker.ids list')
    if count > dictionary_size:
        raise ValueError('marker.expected_count exceeds dictionary size')
    if ids is not None and (not isinstance(ids, list) or len(ids) != count or
                            any(type(mid) is not int or not 0 <= mid < dictionary_size for mid in ids) or
                            len(set(ids)) != count):
        raise ValueError(f'marker.ids must be null (discover) or {count} unique dictionary IDs')
    reference = marker.get('reference_marker_id')
    if reference is not None and (type(reference) is not int or not 0 <= reference < dictionary_size or
                                  (ids is not None and reference not in ids)):
        raise ValueError('Invalid reference_marker_id')
    if reference is None and (model == 'rectangle' or count != 9):
        raise ValueError('Set marker.reference_marker_id explicitly for this target')
    options = config.get('calibration', {})
    if not isinstance(options, dict) or set(options) - set(DEFAULT_OPTIONS):
        raise ValueError('Unknown marker calibration options')
    options = dict(DEFAULT_OPTIONS, **options)
    for key, value in options.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('min_views', 'min_marker_views', 'min_pair_views', 'validation_stride', 'max_nfev'):
        if type(options[key]) is not int:
            raise ValueError(f'{key} must be an integer')
    if options['validation_stride'] < 2 or options['min_views'] < 3 or options['min_marker_views'] < 3 or options['min_pair_views'] < 2:
        raise ValueError('Insufficient coverage or validation settings')
    if not 0 < options['min_length_ratio'] < 1 < options['max_length_ratio'] or options['max_outlier_fraction'] >= 0.5:
        raise ValueError('Invalid length bounds or outlier fraction')
    output = config['output']
    if set(output) - {'filename', 'instrument', 'tip_offset'}:
        raise ValueError('Unknown output settings')
    name = output['filename']
    if not isinstance(name, str) or Path(name).name != name or Path(name).suffix not in ('.yaml', '.yml'):
        raise ValueError('output.filename must be a YAML filename')
    instrument = output['instrument']
    if not isinstance(instrument, dict) or instrument.get('model') != 'marked_object':
        raise ValueError('Geometry-only calibration requires instrument.model marked_object')
    if instrument.get('marker_dictionary') != marker['dictionary']:
        raise ValueError('Output and detection dictionaries must match')
    tip = np.asarray(output['tip_offset'], np.float64)
    if tip.shape != (3,) or not np.isfinite(tip).all():
        raise ValueError('tip_offset must contain three finite values')
    config['calibration'], config['_config_path'] = options, path
    return config


def load_camera(path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f'Cannot open camera parameters: {path}')
    try:
        def matrix(names):
            for name in names:
                node = storage.getNode(name)
                if not node.empty():
                    return np.asarray(node.mat(), np.float64)
            raise ValueError(f'Camera parameters missing one of {names}')
        k = matrix(('camera_matrix', 'intrinsics'))
        d = matrix(('distortion_coefficients', 'dist_coeffs', 'distortion')).reshape(-1)
        sizes = [(storage.getNode(a), storage.getNode(b)) for a, b in
                 [('image_width', 'image_height'), ('width', 'height')]]
        size = next(([int(a.real()), int(b.real())] for a, b in sizes if not a.empty() and not b.empty()), None)
    finally:
        storage.release()
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError('Invalid camera matrix')
    if d.size not in (4, 5, 8, 12, 14) or not np.isfinite(d).all():
        raise ValueError('Invalid distortion coefficients')
    if size is None or min(size) <= 0:
        raise ValueError('Camera YAML must specify calibrated image_width/image_height; image scaling is not inferred')
    return {'matrix': k, 'distortion': d, 'image_size': size}


def create_detector(dictionary):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    params.cornerRefinementWinSize = 2
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 33
    params.adaptiveThreshWinSizeStep = 10
    params.adaptiveThreshConstant = 7
    params.useAruco3Detection = False
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 2.0
    params.minSideLengthCanonicalImg = 16
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params)


def detection_gray(image):
    """Adjust thresholding intensity for dim captures without changing pixel geometry."""
    gray = prepare_gray8(image)
    high = float(np.percentile(gray, 99))
    if 0 < high < 160:
        gray = cv2.convertScaleAbs(gray, alpha=min(4.0, 220.0 / high))
    return gray


def detect(image, detector):
    corners, duplicates = detect_gray(detection_gray(image), detector)
    return refine_marker_corners(prepare_gray8(image), corners), duplicates


def refine_marker_corners(gray, corners):
    """Refine decoded outer corners in original-pixel ROIs, without temporal filtering."""
    refined = {}
    height, width = gray.shape
    for mid, points in corners.items():
        points = np.asarray(points, np.float32).reshape(4, 2)
        shortest = float(np.min(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)))
        window = max(2, min(5, int(shortest / 8)))
        padding = window * 2 + 3
        x0, y0 = np.maximum(0, np.floor(points.min(axis=0)).astype(int) - padding)
        x1, y1 = np.minimum([width, height], np.ceil(points.max(axis=0)).astype(int) + padding + 1)
        patch = gray[y0:y1, x0:x1]
        local = points - [x0, y0]
        if (min(patch.shape) <= window * 2 + 4 or np.any(local < window + 1) or
                np.any(local >= np.array([patch.shape[1], patch.shape[0]]) - window - 1)):
            refined[mid] = points.astype(np.float64)
            continue
        # Small spatial denoising only; it never averages different camera frames.
        patch = cv2.GaussianBlur(patch, (3, 3), 0.6)
        candidate = cv2.cornerSubPix(patch, np.ascontiguousarray(local, np.float32).reshape(4, 1, 2),
                                      (window, window), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        candidate = candidate.reshape(4, 2).astype(np.float64) + [x0, y0]
        valid = (np.isfinite(candidate).all() and
                 np.max(np.linalg.norm(candidate - points, axis=1)) <= window * 1.5 and
                 cv2.isContourConvex(candidate.astype(np.float32)))
        refined[mid] = candidate if valid else points.astype(np.float64)
    return refined


def detect_gray(gray, detector):
    """Detect on an already prepared image, sharing normalization with rendering."""
    try:
        corners, ids, _ = detector.detectMarkers(gray)
    except cv2.error as error:
        # OpenCV contour refinement can fail on degenerate contours in noisy frames.
        if 'nContours' not in str(error):
            raise
        parameters = detector.getDetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        retry = cv2.aruco.ArucoDetector(detector.getDictionary(), parameters)
        corners, ids, _ = retry.detectMarkers(gray)
    if ids is None:
        return {}, []
    counts = Counter(int(mid) for mid in ids.ravel())
    duplicates = sorted(mid for mid, count in counts.items() if count > 1)
    return {int(mid): points.reshape(4, 2).astype(np.float64) for mid, points in zip(ids.ravel(), corners)
            if int(mid) not in duplicates}, duplicates


def detect_session(session, config, camera, output):
    session = Path(session)
    metadata_file = session / 'capture_metadata.json'
    group_data = {}
    if metadata_file.is_file():
        metadata = json.loads(metadata_file.read_text(encoding='utf-8-sig'))
        images = metadata.get('images', {})
        if 'capture_group' in metadata:
            group_data = {(session / record['image']).resolve():
                          {'group': record['group'], 'split': record['split']}
                          for record in metadata['saved_views']}
        if not isinstance(images, dict) or len(images) != 1:
            raise ValueError('Select one raw sensor image directory, or a marker capture session')
        paths = [(session / name).resolve() for name in next(iter(images.values()))]
        if any(not path.is_relative_to(session.resolve()) for path in paths):
            raise ValueError('Image path escapes session')
    else:
        paths = sorted(path for path in session.iterdir() if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    if not paths:
        raise ValueError(f'No images in {session}')
    detector, views, records = create_detector(config['marker']['dictionary']), [], []
    preview_dir = output / 'detections'
    for path in paths:
        if group_data and path not in group_data:
            raise ValueError(f'Grouped image is missing its capture metadata: {path}')
        placement = group_data.get(path, {})
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            records.append({'image': str(path), 'error': 'unreadable'})
            continue
        size = [image.shape[1], image.shape[0]]
        if size != camera['image_size']:
            raise ValueError(f'Image size {size} != calibrated size {camera["image_size"]}: {path}')
        corners, duplicates = detect(image, detector)
        records.append({'image': str(path), **placement, 'ids': sorted(corners), 'duplicate_ids_rejected': duplicates,
                        'corners': {mid: points.tolist() for mid, points in corners.items()}})
        if corners:
            views.append({'image': str(path), **placement, 'corners': corners})
        preview = cv2.cvtColor(detection_gray(image), cv2.COLOR_GRAY2BGR)
        if corners:
            cv2.aruco.drawDetectedMarkers(preview, [p.astype(np.float32).reshape(1, 4, 2) for p in corners.values()],
                                          np.array(list(corners), np.int32).reshape(-1, 1))
        factor = min(1, 1600 / max(preview.shape[:2]))
        preview = cv2.resize(preview, None, fx=factor, fy=factor)
        write_image(preview_dir / (path.stem + '.png'), preview)
        print(f'{path.name}: IDs={sorted(corners)}', flush=True)
    counts = Counter(mid for view in views for mid in view['corners'])
    ids = config['marker']['ids']
    discovered = sorted(mid for mid, count in counts.items() if count >= config['calibration']['min_marker_views'])
    detection_report = {'dictionary': config['marker']['dictionary'], 'id_image_counts': dict(sorted(counts.items())),
                        'candidate_ids': discovered, 'images': records}
    write_json(output / 'marker_detections.json', detection_report)
    if ids is None:
        ids = discovered
    count = config['marker']['expected_count']
    if len(ids) != count:
        raise ValueError(f'Expected {count} reliably observed IDs; found {ids}. Inspect marker_detections.json and capture missing faces or configure marker.ids.')
    missing = set(ids) - set(discovered)
    if missing:
        raise ValueError(f'Markers lack observations: {sorted(missing)}')
    selected = []
    for view in views:
        corners = {mid: points for mid, points in view['corners'].items() if mid in ids}
        if corners:
            selected.append(dict(view, corners=corners))
    return selected, sorted(ids), detection_report


class FlowSequence(list):
    pass


class GeometryDumper(yaml.SafeDumper):
    pass


GeometryDumper.add_representer(FlowSequence, lambda dumper, values: dumper.represent_sequence('tag:yaml.org,2002:seq', values, flow_style=True))


def save_geometry(path, geometry, lengths, reference, config, *,
                  optimization_method='pnp_voting_independent_square_ba',
                  reference_marker_length_fixed=True):
    document = {'schema_version': 1, 'instrument': dict(config['output']['instrument']),
                'marker_positions': {mid: [FlowSequence(float(v) for v in point) for point in geometry[mid]] for mid in sorted(geometry)},
                'tip_offset': FlowSequence(config['output']['tip_offset']),
                'marker_length': float(lengths[reference]),
                'marker_lengths': {mid: float(lengths[mid]) for mid in sorted(lengths)},
                'reference_marker_id': reference,
                'optimization_method': optimization_method,
                'scale_source': config['marker']['length_source'],
                'reference_marker_length_fixed': bool(reference_marker_length_fixed),
                'tip_offset_source': 'configured_not_calibrated'}
    if config['marker'].get('geometry_model') == 'rectangle' and optimization_method == 'pnp_voting_independent_rectangle_ba':
        document['reference_marker_width_mm'] = float(config['marker']['reference_marker_length_mm'])
    temporary = path.with_name(f'.{path.stem}.{uuid.uuid4().hex}{path.suffix}')
    try:
        temporary.write_text(yaml.dump(document, Dumper=GeometryDumper, allow_unicode=True, sort_keys=False), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
def result_path(directory, filename):
    """Read previously generated reports with the old dataset prefix."""
    path = directory / filename
    if path.exists():
        return path
    legacy = directory / filename.replace("marker_", "C_", 1)
    return legacy if legacy.exists() else path
