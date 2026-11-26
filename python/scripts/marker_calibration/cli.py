"""Configured ArUco capture, detection and instrument geometry calibration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

from ..session_storage import session_path, timestamp_directory, write_json
from ..remote import add_target_arguments, run_workflow
from ..workflow.config import PROJECT_ROOT
from .marker_io import DEFAULT_CONFIG, detect_session, load_camera, load_config, save_geometry, select_capture_camera


def build_parser():
    parser = argparse.ArgumentParser(description='ArUco 靶标标定：PnP 共视投票、正方形或矩形 BA、YAML 输出')
    parser.add_argument('command', nargs='?', default='run', choices=('run', 'capture', 'detect', 'calibrate'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='运行配置 JSON')
    parser.add_argument('--session', type=Path, help='detect/calibrate 读取图像；run/capture 续采现有靶标会话')
    parser.add_argument('--camera', choices=('galaxy', 'vcamera'))
    parser.add_argument('--camera-intrinsics', type=Path)
    add_target_arguments(parser, PROJECT_ROOT)
    return parser


def rectangle_failure_details(report, options):
    descriptions = {
        'solver_converged': '求解器未收敛',
        'validation_marker_coverage': '留出图未覆盖所有靶标码',
        'positive_depth': '有角点处于相机后方',
        'viewing_diversity': f'有靶标码的视角跨度小于 {options["min_view_angle_deg"]:g}°',
        'dimension_bounds_inactive': '有靶标码的宽或高触及尺寸约束',
    }
    details = []
    for check, observations, label, key in (
        ('training_reprojection', 'training_observations', '训练码', 'max_observation_rmse_px'),
        ('validation_reprojection', 'validation_observations', '留出码', 'max_validation_rmse_px'),
        ('checkerboard_reprojection', 'checkerboard_observations', '棋盘格内点', 'max_observation_rmse_px'),
    ):
        if report['checks'][check]:
            continue
        records = report[observations]
        worst = max(records, key=lambda observation: observation['rmse_px'])
        limit = options[key]
        count = sum(observation['rmse_px'] > limit for observation in records)
        location = f'第 {worst["group"] + 1} 组 {Path(worst["image"]).name}'
        if 'id' in worst:
            location += f' ID {worst["id"]}'
        details.append(f'{label} RMS 超限 {count}/{len(records)}，最大 {worst["rmse_px"]:.3f} px'
                       f'（上限 {limit:g} px，{location}）')
    details.extend(message for check, message in descriptions.items() if not report['checks'][check])
    return '。'.join(details)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in ('detect', 'calibrate') and args.session is None:
        parser.error('--session is required for detect/calibrate')
    if (args.camera_intrinsics and args.camera != 'galaxy') or (args.camera and args.command not in ('run', 'capture')):
        parser.error('--camera-intrinsics applies only to Galaxy capture')
    output = None
    try:
        if args.ssh:
            return run_workflow(args, PROJECT_ROOT, marker=True)
        config = load_config(args.config)
        select_capture_camera(config, args.camera, args.camera_intrinsics)
        marker = config['marker']
        model = marker['geometry_model']
        length = marker.get('reference_marker_length_mm')
        board = marker.get('checkerboard')
        if length is None and board is None and args.command in ('run', 'calibrate'):
            raise ValueError('Set marker.reference_marker_length_mm to the measured decoded width of the reference marker before calibration')
        if args.command in ('run', 'capture'):
            from .marker_session import capture_session

            session = session_path(args.session) if args.session is not None else timestamp_directory(PROJECT_ROOT / 'data')
            camera_path = session / 'camera_params.yml' if args.session is not None else config['camera_parameters']
            camera = load_camera(camera_path) if camera_path and (config['camera_parameters'] is not None or camera_path.is_file()) else None
            print(f'Capture session: {session}', flush=True)
            if board:
                print(f'Group 1: show top ID {marker["reference_marker_id"]} and a side ID. '
                      'Move the camera within each group; only reposition the target after Next group (N).', flush=True)
            print(f'Rotate and tilt the target; cover all {marker["expected_count"]} IDs with overlapping views. SPACE saves; Q/ESC finishes.', flush=True)
            captured = capture_session(config, session, camera, resume=args.session is not None)
            print(f'Saved {captured} images: {session}', flush=True)
            if args.command == 'capture':
                print(f'Calibrate: python python/scripts/calibrate_marker.py calibrate --config "{config["_config_path"]}" --session "{session}"')
                return 0
        else:
            session = session_path(args.session)
            if not session.is_dir():
                raise ValueError(f'Session does not exist: {session}')
        camera_path = config['camera_parameters']
        metadata_path = session / 'capture_metadata.json'
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
            if metadata.get('calibration_type') == 'marker_geometry':
                camera_path = session / 'camera_params.yml'
                if metadata.get('marker', {}).get('dictionary') != config['marker']['dictionary']:
                    raise ValueError('Configured dictionary differs from the captured marker dictionary')
                if board and 'capture_group' in metadata and args.command != 'detect' and not metadata.get('capture_complete'):
                    if args.command == 'run' and os.environ.get('CALIBRATION_EMBEDDED') == '1':
                        print('CALIBRATION_MARKER_PHASE=paused', flush=True)
                        print(f'Partial capture saved. Resume session: {session}', flush=True)
                        return 0
                    raise ValueError('Grouped capture is incomplete. Resume this session and follow the coverage prompts.')
        if args.command != 'detect' and os.environ.get('CALIBRATION_EMBEDDED') == '1':
            print('CALIBRATION_MARKER_PHASE=solve', flush=True)
        camera = load_camera(camera_path)
        output = timestamp_directory(PROJECT_ROOT / 'output')
        write_json(output / 'marker_workflow.json', {
            'config': str(config['_config_path']), 'session': str(session),
            'camera_parameters': str(camera_path), 'marker': config['marker'],
            'calibration': config['calibration'], 'output': config['output'],
        })
        print(f'Results: {output}', flush=True)
        views, ids, detections = detect_session(session, config, camera, output)
        print(f'Detected target IDs: {ids}', flush=True)
        if args.command == 'detect':
            return 0
        if board:
            from .checkerboard_geometry import calibrate_rectangles

            geometry, lengths, reference, report = calibrate_rectangles(
                views, ids, marker, camera, config['calibration'], output)
            method = 'checkerboard_joint_rectangle_ba'
        else:
            from .geometry import calibrate

            geometry, lengths, reference, report = calibrate(
                views, ids, marker['reference_marker_id'], length, camera, config['calibration'], model=model)
            method = f'pnp_voting_independent_{model}_ba'
        report['scale_source'] = config['marker']['length_source']
        report['image_count'] = len(detections['images'])
        write_json(output / 'marker_calibration_quality.json', report)
        sizes = report.get('marker_dimensions_mm', lengths)
        print(f'Reference ID: {reference}; {model} dimensions (mm): {sizes}', flush=True)
        print(f'Quality checks: {report["checks"]}', flush=True)
        if not report['passed']:
            if board:
                details = rectangle_failure_details(report, config['calibration'])
                print(f'Error: 几何标定质量检查未通过。{details}。详见 marker_calibration_quality.json。',
                      file=sys.stderr, flush=True)
            else:
                print('Calibration checks failed; quality report saved. No geometry YAML written.', file=sys.stderr)
            return 1
        destination = output / config['output']['filename']
        save_geometry(destination, geometry, lengths, reference, config,
                      optimization_method=method,
                      reference_marker_length_fixed=model == 'square',
                      dimensions=report.get('marker_dimensions_mm'))
        print(f'Saved geometry: {destination}', flush=True)
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError, cv2.error) as error:
        if output is not None:
            write_json(output / 'marker_calibration_error.json', {'passed': False, 'error': str(error)})
        print(f'Error: {error}', file=sys.stderr, flush=True)
        return 1
