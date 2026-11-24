"""Initialize and optimize square and rectangular marker geometry from calibrated images."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def rectangle(width, height):
    """OpenCV decoded corner order: top left, top right, bottom right, bottom left."""
    return np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float64) * np.array([width, height, 0]) / 2


def transform(pose, points):
    return points @ pose[:3, :3].T + pose[:3, 3]


def matrix_pose(rvec, tvec):
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.asarray(rvec).reshape(3))[0]
    pose[:3, 3] = np.asarray(tvec).reshape(3)
    return pose


def vector_pose(pose):
    return np.r_[cv2.Rodrigues(pose[:3, :3])[0].reshape(3), pose[:3, 3]]


def project(points, pose, camera):
    return cv2.projectPoints(points, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3],
                             camera['matrix'], camera['distortion'])[0].reshape(-1, 2)


def pnp_marker(corners, dimensions, camera):
    width, height = dimensions
    points = rectangle(width, height)
    method = cv2.SOLVEPNP_IPPE_SQUARE if width == height else cv2.SOLVEPNP_IPPE
    solved, rotations, translations, _ = cv2.solvePnPGeneric(
        points, np.asarray(corners, np.float64), camera['matrix'], camera['distortion'],
        flags=method)
    candidates = []
    if solved:
        for rotation, translation in zip(rotations, translations):
            pose = matrix_pose(rotation, translation)
            if np.min(transform(pose, points)[:, 2]) <= 0:
                continue
            error = np.sqrt(np.mean(np.sum((project(points, pose, camera) - corners) ** 2, axis=1)))
            candidates.append((float(error), pose))
    return min(candidates, key=lambda item: item[0]) if candidates else None


def vote(poses, weights, length, translation_limit, rotation_limit):
    """Medoid consensus in SE(3); average rotations on SO(3), never Rodrigues vectors."""
    rotations = np.stack([pose[:3, :3] for pose in poses])
    translations = np.stack([pose[:3, 3] for pose in poses])
    # Pairwise geodesic angle from trace(R_i^T R_j).
    trace = np.einsum('iab,jab->ij', rotations, rotations)
    angles = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    distances = np.linalg.norm(translations[:, None] - translations[None, :], axis=2)
    agree = (distances <= length * translation_limit) & (angles <= np.deg2rad(rotation_limit))
    weights = np.asarray(weights, np.float64)
    scores = agree @ weights
    index = max(range(len(poses)), key=lambda i: (scores[i], -float(np.median(distances[i]))))
    mask = agree[index]
    selected_weights = weights[mask]
    fused = np.eye(4)
    fused[:3, :3] = Rotation.from_matrix(rotations[mask]).mean(weights=selected_weights).as_matrix()
    fused[:3, 3] = np.average(translations[mask], axis=0, weights=selected_weights)
    return fused, np.flatnonzero(mask)


def connected(views, marker_ids, reference):
    found = {reference}
    while True:
        previous = len(found)
        for view in views:
            ids = set(view['corners']) & set(marker_ids)
            if ids & found:
                found.update(ids)
        if len(found) == previous:
            return found


def infer_top(marker_poses):
    """A top normal is perpendicular to all eight side normals; reject ambiguity."""
    ids = sorted(marker_poses)
    normals = np.stack([marker_poses[mid][:3, 2] for mid in ids])
    candidates = []
    for i, mid in enumerate(ids):
        side_normals = np.delete(normals, i, axis=0)
        _, _, vh = np.linalg.svd(side_normals)
        axis = vh[-1]
        if abs(axis @ normals[i]) > np.cos(np.deg2rad(20)) and np.max(np.abs(side_normals @ axis)) < np.sin(np.deg2rad(20)):
            candidates.append(mid)
    if len(candidates) != 1:
        raise ValueError(f'Cannot uniquely identify the top marker ({candidates}); set reference_marker_id from an inspected image')
    return candidates[0]


def initialize(views, marker_ids, length, camera, options, reference=None):
    """Vote relative poses on co-visible edges, then grow a maximum-support tree."""
    pairs = defaultdict(list)
    rejected = []
    for view in views:
        poses = {}
        for mid, corners in view['corners'].items():
            if mid not in marker_ids:
                continue
            solved = pnp_marker(corners, (length, length), camera)
            if solved is None or solved[0] > options['max_pnp_rmse_px']:
                rejected.append({'image': view['image'], 'marker_id': mid,
                                 'rmse_px': None if solved is None else solved[0]})
            else:
                poses[mid] = solved
        for a, b in combinations(sorted(poses), 2):
            ea, pa = poses[a]
            eb, pb = poses[b]
            pairs[a, b].append((np.linalg.inv(pa) @ pb, 1 / (1 + ea * ea + eb * eb)))
    edges, graph = [], defaultdict(dict)
    for (a, b), samples in pairs.items():
        pose, inliers = vote([s[0] for s in samples], [s[1] for s in samples], length,
                             options['vote_translation_ratio'], options['vote_rotation_deg'])
        edges.append({'ids': [a, b], 'samples': len(samples), 'inliers': len(inliers)})
        if len(inliers) >= options['min_pair_views']:
            graph[a][b] = (pose, len(inliers))
            graph[b][a] = (np.linalg.inv(pose), len(inliers))
    root = marker_ids[0] if reference is None else reference
    marker_poses = {root: np.eye(4)}
    while len(marker_poses) < len(marker_ids):
        candidates = [(weight, a, b, pose) for a in marker_poses
                      for b, (pose, weight) in graph[a].items() if b not in marker_poses]
        if not candidates:
            raise ValueError(f'PnP co-visibility graph is disconnected; missing IDs: {sorted(set(marker_ids) - set(marker_poses))}. Capture bridging views.')
        _, a, b, relative = max(candidates, key=lambda item: item[0])
        marker_poses[b] = marker_poses[a] @ relative
    reference = infer_top(marker_poses) if reference is None else reference
    rebase = np.linalg.inv(marker_poses[reference])
    marker_poses = {mid: rebase @ pose for mid, pose in marker_poses.items()}
    marker_poses[reference] = np.eye(4)
    return marker_poses, reference, {'edges': edges, 'rejected_pnp': rejected}


def estimate_view(view, geometry, camera, options, *, rectangular=False):
    """Each visible marker votes for the same instrument-to-camera transform."""
    poses, weights = [], []
    for mid, measured in view['corners'].items():
        if mid not in geometry:
            continue
        points = geometry[mid]
        width = np.linalg.norm(points[1] - points[0])
        height = np.linalg.norm(points[0] - points[3])
        solved = pnp_marker(measured, (width, height if rectangular else width), camera)
        if solved is None or solved[0] > options['max_pnp_rmse_px']:
            continue
        # Derive the orthonormal local basis from the fitted rectangle.
        x_axis = (points[1] - points[0]) / width
        y_axis = (points[0] - points[3]) / height
        marker_pose = np.eye(4)
        marker_pose[:3, :3] = np.column_stack([x_axis, y_axis, np.cross(x_axis, y_axis)])
        marker_pose[:3, 3] = points.mean(axis=0)
        poses.append(solved[1] @ np.linalg.inv(marker_pose))
        weights.append(1 / (1 + solved[0] ** 2))
    if not poses:
        return None
    length = float(np.mean([np.linalg.norm(p[1] - p[0]) for p in geometry.values()]))
    fused, _ = vote(poses, weights, length, options['vote_translation_ratio'], options['vote_rotation_deg'])
    # Score all hypotheses against all markers to avoid a low-error planar flip.
    def score(pose):
        errors = [float(np.sqrt(np.mean(np.sum((project(geometry[mid], pose, camera) - corners) ** 2, axis=1))))
                  for mid, corners in view['corners'].items() if mid in geometry]
        return sum(error <= options['max_pnp_rmse_px'] for error in errors), -float(np.median(errors))
    pose = max([fused, *poses], key=score)
    selected = [mid for mid, corners in view['corners'].items() if mid in geometry and
                np.sqrt(np.mean(np.sum((project(geometry[mid], pose, camera) - corners) ** 2, axis=1))) <= options['max_pnp_rmse_px']]
    if not selected:
        return None
    objects = np.concatenate([geometry[mid] for mid in selected])
    measured = np.concatenate([view['corners'][mid] for mid in selected])
    rvec, tvec = vector_pose(pose)[:3].copy(), pose[:3, 3].copy()
    ok, rvec, tvec = cv2.solvePnP(objects, measured, camera['matrix'], camera['distortion'],
                                  rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    result = matrix_pose(rvec, tvec)
    if np.min(transform(result, objects)[:, 2]) <= 0:
        return None
    return result


def check_coverage(views, marker_ids, reference, options, *, connectivity_views=None):
    counts = Counter(mid for view in views for mid in view['corners'] if mid in marker_ids)
    insufficient = {mid: counts[mid] for mid in marker_ids if counts[mid] < options['min_marker_views']}
    if len(views) < options['min_views'] or insufficient:
        raise ValueError(f'Insufficient training coverage: {len(views)} views; under-observed IDs/counts: {insufficient}')
    missing = set(marker_ids) - connected(
        views if connectivity_views is None else connectivity_views, marker_ids, reference)
    if missing:
        raise ValueError(f'Training co-visibility graph is disconnected: {sorted(missing)}')


def optimize(views, marker_poses, view_poses, dimensions, reference, camera, options,
             vary_lengths, rectangular):
    """Fix the reference pose and width. In rectangle mode its height remains free."""
    variable_ids = sorted(set(marker_poses) - {reference})
    pose_slices = {mid: slice(i * 6, (i + 1) * 6) for i, mid in enumerate(variable_ids)}
    cursor = len(variable_ids) * 6
    width_ids = variable_ids if vary_lengths else []
    height_ids = sorted(marker_poses) if vary_lengths and rectangular else []
    width_indices = {mid: cursor + i for i, mid in enumerate(width_ids)}
    cursor += len(width_indices)
    height_indices = {mid: cursor + i for i, mid in enumerate(height_ids)}
    cursor += len(height_indices)
    view_slices = [slice(cursor + i * 6, cursor + (i + 1) * 6) for i in range(len(views))]
    initial = np.zeros(cursor + 6 * len(views))
    lower, upper = np.full_like(initial, -np.inf), np.full_like(initial, np.inf)
    for mid, section in pose_slices.items():
        initial[section] = vector_pose(marker_poses[mid])
    anchor = dimensions[reference][0]
    for axis, indices in enumerate((width_indices, height_indices)):
        for mid, index in indices.items():
            initial[index] = np.log(dimensions[mid][axis])
            lower[index] = np.log(anchor * options['min_length_ratio'])
            upper[index] = np.log(anchor * options['max_length_ratio'])
    for section, pose in zip(view_slices, view_poses):
        initial[section] = vector_pose(pose)
    observation_count = sum(len(view['corners']) for view in views)
    sparsity = lil_matrix((observation_count * 8, initial.size), dtype=np.int8)
    row = 0
    for i, view in enumerate(views):
        for mid in sorted(view['corners']):
            for columns in (pose_slices, width_indices, height_indices):
                if mid in columns:
                    sparsity[row:row + 8, columns[mid]] = 1
            sparsity[row:row + 8, view_slices[i]] = 1
            row += 8

    def unpack(values):
        poses = {reference: np.eye(4)}
        for mid, section in pose_slices.items():
            vector = values[section]
            poses[mid] = matrix_pose(vector[:3], vector[3:6])
        sizes = {}
        for mid, (width, height) in dimensions.items():
            if mid in width_indices:
                width = float(np.exp(values[width_indices[mid]]))
            if mid in height_indices:
                height = float(np.exp(values[height_indices[mid]]))
            sizes[mid] = (width, height if rectangular else width)
        geometry = {mid: transform(pose, rectangle(*sizes[mid])) for mid, pose in poses.items()}
        frames = [matrix_pose(values[section][:3], values[section][3:6]) for section in view_slices]
        return poses, sizes, geometry, frames

    def residual(values):
        _, _, geometry, frames = unpack(values)
        residuals = []
        for view, pose in zip(views, frames):
            ids = sorted(view['corners'])
            points = np.concatenate([geometry[mid] for mid in ids])
            measured = np.concatenate([view['corners'][mid] for mid in ids])
            residuals.append((project(points, pose, camera) - measured).ravel())
        return np.concatenate(residuals)

    before = residual(initial)
    result = least_squares(residual, initial, jac_sparsity=sparsity.tocsr(), method='trf',
                           bounds=(lower, upper), loss='soft_l1', f_scale=options['loss_scale_px'],
                           x_scale='jac', max_nfev=options['max_nfev'], ftol=1e-8,
                           xtol=1e-8, gtol=1e-4)
    poses, sizes, geometry, frames = unpack(result.x)
    after = residual(result.x)
    report = {'success': bool(result.success), 'status': int(result.status), 'message': result.message,
              'nfev': int(result.nfev), 'initial_rmse_px': float(np.sqrt(np.mean(before ** 2) * 2)),
              'final_rmse_px': float(np.sqrt(np.mean(after ** 2) * 2)),
              'optimality': float(result.optimality)}
    return poses, sizes, geometry, frames, report


def residual_records(views, geometry, poses, camera):
    records = []
    for view, pose in zip(views, poses):
        for mid, measured in view['corners'].items():
            points = geometry[mid]
            delta = project(points, pose, camera) - measured
            records.append({'image': view['image'], 'marker_id': mid,
                            'rmse_px': float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
                            'positive_depth': bool(np.min(transform(pose, points)[:, 2]) > 0)})
    return records


def calibrate(views, marker_ids, reference, length, camera, options, *, model='square'):
    rectangular = model == 'rectangle'
    # Reserve complete images before geometric initialization, voting, or fitting.
    validation = [view for i, view in enumerate(views) if (i + 1) % options['validation_stride'] == 0]
    training = [view for i, view in enumerate(views) if (i + 1) % options['validation_stride'] != 0]
    if not validation:
        raise ValueError('No held-out validation images; capture more views')
    marker_poses, reference, initialization = initialize(training, marker_ids, length, camera, options, reference)
    geometry = {mid: transform(pose, rectangle(length, length)) for mid, pose in marker_poses.items()}
    poses, usable, failed_initialization = [], [], []
    for view in training:
        pose = estimate_view(view, geometry, camera, options, rectangular=rectangular)
        if pose is None:
            failed_initialization.append(view['image'])
        else:
            poses.append(pose)
            usable.append(view)
    check_coverage(usable, marker_ids, reference, options)
    dimensions = {mid: (length, length) for mid in marker_ids}
    marker_poses, dimensions, geometry, poses, rigid_report = optimize(
        usable, marker_poses, poses, dimensions, reference, camera, options, False, rectangular)
    # Fit actual edge differences before classifying a marker observation as bad.
    marker_poses, dimensions, geometry, poses, scale_report = optimize(
        usable, marker_poses, poses, dimensions, reference, camera, options, True, rectangular)
    records = residual_records(usable, geometry, poses, camera)
    errors = np.array([record['rmse_px'] for record in records])
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    threshold = min(options['max_observation_rmse_px'], max(options['loss_scale_px'], median + 4 * 1.4826 * mad))
    rejected = [record for record in records if record['rmse_px'] > threshold or not record['positive_depth']]
    rejected_set = {(record['image'], record['marker_id']) for record in rejected}
    outlier_limit_ok = len(rejected) <= options['max_outlier_fraction'] * len(records)
    if rejected and outlier_limit_ok:
        filtered, retained_poses = [], []
        for view, pose in zip(usable, poses):
            corners = {mid: points for mid, points in view['corners'].items()
                       if (view['image'], mid) not in rejected_set}
            if corners:
                filtered.append(dict(view, corners=corners))
                retained_poses.append(pose)
        check_coverage(filtered, marker_ids, reference, options)
        usable, poses = filtered, retained_poses
    marker_poses, dimensions, geometry, poses, shape_report = optimize(
        usable, marker_poses, poses, dimensions, reference, camera, options, True, rectangular)
    final_records = residual_records(usable, geometry, poses, camera)
    validation_records, validation_failures = [], []
    for view in validation:
        pose = estimate_view(view, geometry, camera, options, rectangular=rectangular)
        if pose is None:
            validation_failures.append(view['image'])
        else:
            validation_records.extend(residual_records([view], geometry, [pose], camera))
    # Measure actual viewing direction diversity for every marker in its local frame.
    diversity = {}
    for mid in marker_ids:
        directions = []
        for view, pose in zip(usable, poses):
            if mid not in view['corners']:
                continue
            marker_to_camera = pose @ marker_poses[mid]
            direction = -marker_to_camera[:3, :3].T @ marker_to_camera[:3, 3]
            directions.append(direction / np.linalg.norm(direction))
        directions = np.asarray(directions)
        diversity[mid] = float(np.rad2deg(np.arccos(np.clip(np.min(directions @ directions.T), -1, 1))))
    ratios = [width / length for mid, (width, _) in dimensions.items() if mid != reference]
    if rectangular:
        ratios.extend(height / length for _, height in dimensions.values())
    lengths = {mid: (width + height) / 2 for mid, (width, height) in dimensions.items()}
    checks = {
        'solver_converged': rigid_report['success'] and scale_report['success'] and shape_report['success'],
        'outlier_fraction': outlier_limit_ok,
        'training_reprojection': all(r['rmse_px'] <= options['max_observation_rmse_px'] for r in final_records),
        'positive_depth': all(r['positive_depth'] for r in final_records + validation_records),
        'validation_complete': not validation_failures and bool(validation_records),
        'validation_marker_coverage': set(r['marker_id'] for r in validation_records) == set(marker_ids),
        'validation_reprojection': bool(validation_records) and all(r['rmse_px'] <= options['max_validation_rmse_px'] for r in validation_records),
        'viewing_diversity': all(angle >= options['min_view_angle_deg'] for angle in diversity.values()),
        'length_bounds_inactive': all(options['min_length_ratio'] * 1.001 < ratio < options['max_length_ratio'] / 1.001 for ratio in ratios),
    }
    report = {'passed': all(checks.values()), 'checks': checks, 'reference_marker_id': reference,
              'reference_marker_length_mm': length, 'scale_source': 'fixed_reference_marker_length',
              'geometry_model': model,
              'initialization': initialization, 'failed_training_initialization': failed_initialization,
              'training_image_count': len(usable), 'validation_image_count': len(validation),
              'stages': {'rigid_ba': rigid_report, f'initial_{model}_ba': scale_report, f'independent_{model}_ba': shape_report},
              'outlier_threshold_px': threshold, 'outlier_candidates': rejected,
              'outliers_removed': bool(rejected and outlier_limit_ok),
              'training_observations': final_records, 'validation_observations': validation_records,
              'failed_validation_images': validation_failures, 'view_angle_range_deg': diversity,
              'marker_lengths_mm': lengths}
    if rectangular:
        report['reference_marker_width_mm'] = length
        report['marker_dimensions_mm'] = {
            mid: {'width': width, 'height': height} for mid, (width, height) in dimensions.items()
        }
    return geometry, lengths, reference, report
