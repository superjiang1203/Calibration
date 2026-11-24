"""Configured ArUco capture, detection and instrument geometry calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from ..session_storage import session_path, timestamp_directory, write_json
from ..remote import add_target_arguments, run_workflow
from ..workflow.config import PROJECT_ROOT
from .marker_io import DEFAULT_CONFIG, detect_session, load_camera, load_config, save_geometry


def build_parser():
    parser = argparse.ArgumentParser(description='ArUco 靶标标定：PnP 共视投票、正方形或矩形 BA、YAML 输出')
    parser.add_argument('command', nargs='?', default='run', choices=('run', 'capture', 'detect', 'calibrate'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='运行配置 JSON')
    parser.add_argument('--session', type=Path, help='detect/calibrate 读取图像；run/capture 续采现有靶标会话')
    add_target_arguments(parser, PROJECT_ROOT)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in ('detect', 'calibrate') and args.session is None:
        parser.error('--session is required for detect/calibrate')
    output = None
    try:
        if args.ssh:
            return run_workflow(args, PROJECT_ROOT, marker=True)
        config = load_config(args.config)
        marker = config['marker']
        model = marker['geometry_model']
        length = marker.get('reference_marker_length_mm')
        board = marker.get('checkerboard')
        if length is None and board is None and args.command in ('run', 'calibrate'):
            raise ValueError('Set marker.reference_marker_length_mm to the measured decoded width of the reference marker before calibration')
        if args.command in ('run', 'capture'):
            from .marker_session import capture_session

            session = session_path(args.session) if args.session is not None else timestamp_directory(PROJECT_ROOT / 'data')
            camera = load_camera(session / 'camera_params.yml' if args.session is not None else config['camera_parameters'])
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
                    raise ValueError('Grouped capture is incomplete. Resume this session and follow the coverage prompts.')
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
            print('Calibration checks failed; quality report saved. No geometry YAML written.', file=sys.stderr)
            return 1
        destination = output / config['output']['filename']
        save_geometry(destination, geometry, lengths, reference, config,
                      optimization_method=method,
                      reference_marker_length_fixed=model == 'square')
        print(f'Saved geometry: {destination}', flush=True)
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError, cv2.error) as error:
        if output is not None:
            write_json(output / 'marker_calibration_error.json', {'passed': False, 'error': str(error)})
        print(f'Error: {error}', file=sys.stderr, flush=True)
        return 1
