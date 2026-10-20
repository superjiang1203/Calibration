"""Instrument geometry, pose estimation and RGB projection for extrinsic checks."""
from pathlib import Path
import json

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def matrix_file(path, *names):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        if not fs.isOpened():
            raise ValueError(f'Cannot read calibration file: {path}')
        result = [fs.getNode(name).mat() for name in names]
        if any(value is None or not np.isfinite(value).all() for value in result):
            raise ValueError(f'Missing/invalid matrices {names} in {path}')
        return result
    finally:
        fs.release()


def intrinsics(path):
    k, d = matrix_file(path, 'camera_matrix', 'distortion_coefficients')
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        size = tuple(int(fs.getNode(name).real()) for name in ('image_width', 'image_height'))
    finally:
        fs.release()
    if k.shape != (3, 3) or min(size) <= 0:
        raise ValueError(f'Invalid intrinsics: {path}')
    return k, d, size


def project(point, k, d):
    point = np.asarray(point, np.float64).reshape(3)
    if not np.isfinite(point).all() or point[2] <= 0:
        return None
    pixels, _ = cv2.projectPoints(point.reshape(1, 3), np.zeros(3), np.zeros(3), k, d)
    return pixels.reshape(2)


def transform_tip(tip_galaxy, rotation, translation, delta, k_rgb, d_rgb):
    """Galaxy optical XYZ (mm) -> RGB optical XYZ (mm) -> raw RGB pixel."""
    original = rotation @ np.asarray(tip_galaxy).reshape(3) + translation.reshape(3)
    adjusted = original + np.asarray(delta).reshape(3)
    return original, adjusted, project(original, k_rgb, d_rgb), project(adjusted, k_rgb, d_rgb)


class InstrumentTracker:
    def __init__(self, config, geometry=None):
        self.max_side = int(config['detect_max_dim'])
        self.max_rms = float(config['max_pose_rms_px'])
        self.min_markers = int(config.get('min_visible_markers', 1))
        self.auto_single_id = False
        if geometry:
            import yaml
            raw = yaml.safe_load(Path(geometry).read_text(encoding='utf-8-sig'))
            self.dictionary = raw['instrument']['marker_dictionary']
            self.positions = {int(key): np.asarray(value, np.float64).reshape(4, 3)
                              for key, value in raw['marker_positions'].items()}
            self.tip = np.asarray(raw['tip_offset'], np.float64).reshape(3)
            self.fix = np.eye(3)
        else:
            marker = config['single_marker']
            self.auto_single_id = marker['id'] is None and marker.get('allow_unique_id', False)
            if marker['id'] is None and not self.auto_single_id:
                raise ValueError('Select --geometry <instrument YAML>, or configure single_marker.id first')
            self.dictionary = marker['dictionary']
            half = float(marker['size_mm']) / 2
            if half <= 0:
                raise ValueError('Marker size must be positive')
            # Same tag axes as OpenCvTag36h11MarkerModel; generic IPPE supports this order.
            self.positions = {0 if self.auto_single_id else int(marker['id']): np.array(
                [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])}
            self.tip = np.asarray(marker['tip_offset_mm'], np.float64).reshape(3)
            self.fix = np.diag([1., -1., -1.]) if marker['apply_rx180'] else np.eye(3)
        if not self.positions or not np.isfinite(self.tip).all() or any(
                not np.isfinite(value).all() for value in self.positions.values()):
            raise ValueError('Invalid instrument geometry')
        if self.max_side <= 0 or self.max_rms <= 0:
            raise ValueError('Detection size and pose RMS limit must be positive')
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 45
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = .01
        params.maxMarkerPerimeterRate = 2.
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

    def solve(self, objects, pixels, k, d):
        objects = np.asarray(objects, np.float64).reshape(-1, 3)
        pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
        planar = np.linalg.matrix_rank(objects - objects.mean(axis=0), tol=1e-6) <= 2
        result = cv2.solvePnPGeneric(objects, pixels, k, d,
                                    flags=cv2.SOLVEPNP_IPPE if planar else cv2.SOLVEPNP_SQPNP)
        candidates = []
        for rv, tv in zip(result[1], result[2]):
            rotation = cv2.Rodrigues(rv)[0]
            if np.any((objects @ rotation.T + tv.reshape(3))[:, 2] <= 0):
                continue
            reproj, _ = cv2.projectPoints(objects, rv, tv, k, d)
            rms = float(np.sqrt(np.mean(np.sum((reproj.reshape(-1, 2) - pixels)**2, axis=1))))
            candidates.append((rms, rotation, tv.reshape(3)))
        if not candidates:
            return None
        rms, rotation, center = min(candidates, key=lambda item: item[0])
        if rms > self.max_rms:
            return None
        return dict(rms=rms, rotation=rotation, center=center,
                    tip=center + rotation @ self.fix @ self.tip,
                    objects=objects, pixels=pixels,
                    ambiguity_rms_gap=(sorted(v[0] for v in candidates)[1] - sorted(v[0] for v in candidates)[0]
                                       if len(candidates) > 1 else None))

    def detect(self, image, k, d):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        scale = min(1., self.max_side / max(gray.shape))
        small = cv2.resize(gray, (round(gray.shape[1]*scale), round(gray.shape[0]*scale)))
        corners, ids, _ = self.detector.detectMarkers(small)
        if ids is None:
            return None
        if self.auto_single_id:
            if len(ids) != 1:
                return None
            shape = next(iter(self.positions.values()))
            self.positions = {int(ids.ravel()[0]): shape}
        objects, pixels, used = [], [], []
        axis_scale = np.array([small.shape[1]/gray.shape[1], small.shape[0]/gray.shape[0]])
        for marker, points in zip(ids.ravel(), corners):
            if int(marker) not in self.positions:
                continue
            full = np.float32(points.reshape(4, 2)/axis_scale).reshape(4, 1, 2)
            cv2.cornerSubPix(gray, full, (3, 3), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .001))
            objects.extend(self.positions[int(marker)])
            pixels.extend(full.reshape(4, 2))
            used.append(int(marker))
        if len(used) < self.min_markers:
            return None
        pose = self.solve(objects, pixels, k, d)
        if pose:
            pose['ids'] = used
        return pose


def export_adjustment(folder, rotation, translation, delta, metadata):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / 'pair_galaxy_gray_to_vcamera_color_extrinsics.yml'
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    try:
        fs.write('R', rotation)
        fs.write('T', (translation + delta).reshape(3, 1))
        fs.write('baseline_mm', float(np.linalg.norm(translation + delta)))
    finally:
        fs.release()
    (folder / 'adjustment.json').write_text(json.dumps(dict(
        **metadata, original_T_mm=translation.tolist(), delta_T_rgb_mm=delta.tolist(),
        adjusted_T_mm=(translation+delta).tolist(), rotation_unchanged=True),
        ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return path
