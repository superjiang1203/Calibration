"""Public workflow entry: configured capture, independent intrinsics, and extrinsics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..session_storage import session_path, timestamp_directory
from ..remote import add_target_arguments, run_workflow
from .config import DEFAULT_CONFIG, PROJECT_ROOT, load_config


def _check_session_geometry(session: Path, config: dict) -> None:
    metadata_path = session / "capture_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"No capture_metadata.json in {session}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Capture metadata must be an object: {metadata_path}")
    source_board = metadata["board"]
    board_keys = ("pattern_cols", "pattern_rows", "square_size_mm")
    if (not isinstance(source_board, dict) or
            any(source_board.get(key) != config["board"].get(key) for key in board_keys)):
        print("Warning: configured board differs from capture metadata; calibration uses the captured board dimensions.")
        print(f"Captured board: {source_board}; configured board: {config['board']}")
    sensors = metadata.get("sensors")
    if metadata.get("schema_version") != 2 or not isinstance(sensors, list):
        raise ValueError("Capture metadata must use schema_version 2 and a sensor list")
    source_labels = [entry["label"] for entry in sensors]
    configured_labels = [sensor["label"] for sensor in config["sensors"]]
    if source_labels != configured_labels:
        raise ValueError(f"Session sensors {source_labels} do not match configured sensors {configured_labels}. Select the matching config and capture session; IR pairs cannot calibrate Galaxy/color extrinsics.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        epilog="Vcamera diagnostics: inspect-vcamera --help",
        description="通过配置选择传感器内外参或靶标几何标定；数据和结果自动保存在时间戳目录。")
    parser.add_argument("command", nargs="?", default="run", choices=("run", "capture", "calibrate", "verify", "detect"),
                        help="run=采集后标定；capture=只采集；calibrate=使用现有数据；verify=传感器参数验证；detect=靶标角点识别")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"标定配置 JSON，默认 {DEFAULT_CONFIG.relative_to(PROJECT_ROOT).as_posix()}")
    parser.add_argument("--session", type=Path,
                        help="calibrate/verify/detect 使用的数据目录；相对路径以工程根目录为基准")
    parser.add_argument("--calibration-dir", type=Path,
                        help="verify 读取内参和外参 YAML 的结果目录")
    add_target_arguments(parser, PROJECT_ROOT)
    return parser


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "inspect-vcamera":
        from ..sensor_capture.vcamera import inspect_main

        return inspect_main(arguments[1:])
    if arguments and arguments[0] == "studio":
        from .studio import main as studio_main

        return studio_main(arguments[1:])
    camera_only = bool(arguments and arguments[0] == "camera")
    if camera_only:
        arguments[0] = "capture"
        if not any(arg == "--config" or arg.startswith("--config=") for arg in arguments):
            arguments.extend(("--config", "configs/intrinsics/galaxy.json"))
    if arguments and arguments[0] == "pivot":
        from ..marker_calibration.pivot import main as pivot_main

        return pivot_main(arguments[1:])
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.command in ("calibrate", "verify", "detect") and args.session is None:
        parser.error(f"{args.command} requires --session <capture directory>")
    if args.command == "verify" and args.calibration_dir is None:
        parser.error("verify requires --calibration-dir <saved YAML directory>")
    if args.command != "verify" and args.calibration_dir is not None:
        parser.error("--calibration-dir is only used by verify")
    if args.ssh:
        try:
            return run_workflow(args, PROJECT_ROOT)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
    if camera_only:
        os.environ["CALIBRATION_CAPTURE_WEB"] = "1"
        os.environ["CALIBRATION_CAPTURE_LOCAL_BROWSER"] = "1"
    try:
        import cv2
    except ImportError as error:
        print(f"Error: OpenCV is required: {error}", file=sys.stderr)
        return 1
    try:
        configured = json.loads(session_path(args.config).read_text(encoding="utf-8-sig"))
        if not isinstance(configured, dict):
            raise ValueError("Workflow configuration must be a JSON object")
        configured_mode = configured.get("mode")
        if configured_mode == "marker_pivot":
            from ..marker_calibration.pivot import run_configured

            return run_configured(
                args.command, session_path(args.config),
                session_path(args.session) if args.session is not None else None,
            )
        if configured_mode == "marker_geometry":
            if args.command == "verify":
                raise ValueError("Marker geometry uses held-out validation during calibrate; verify is for sensor calibration")
            from ..marker_calibration.cli import main as marker_main

            marker_args = [args.command, "--config", str(args.config)]
            if args.session is not None:
                marker_args.extend(["--session", str(args.session)])
            return marker_main(marker_args)
        if args.command not in ("calibrate", "verify", "detect") and args.session is not None:
            raise ValueError("Sensor --session is only used by calibrate or verify")
        if args.command == "detect":
            raise ValueError("detect requires a marker_geometry configuration")
        config = load_config(args.config)
        if args.command in ("calibrate", "verify"):
            session = session_path(args.session)
            if not session.is_dir():
                raise ValueError(f"Capture session does not exist: {session}")
            if not (session / "capture_metadata.json").is_file():
                raise ValueError(f"No capture_metadata.json in {session}")
        else:
            from ..sensor_intrinsics.checkerboard_session import capture_session

            session = timestamp_directory(PROJECT_ROOT / "data")
            print(f"Capture session: {session}")
            captured = capture_session(config, session)
            if isinstance(captured, int) and captured != 0:
                return captured
            if args.command == "capture":
                print(f"Saved capture session: {session}")
                print(f'Calibrate this session: python python/scripts/calibrate.py calibrate --session "{session}" --config "{config["_config_path"]}"')
                return 0
        _check_session_geometry(session, config)
        options = dict(config["calibration"], max_pair_delta_ms=config["capture"]["max_pair_delta_ms"])
        output = None
        if args.command == "verify":
            from ..stereo_calibration.verification import verify_session

            calibration_dir = session_path(args.calibration_dir)
            print(f"Fixed calibration parameters: {calibration_dir}")
            print(f"Validation images: {session}; K/D/R/T are not refitted.")
            result = verify_session(session, calibration_dir, options, config["mode"])
        else:
            from ..stereo_calibration.pairwise import calibrate_session

            output = timestamp_directory(PROJECT_ROOT / "output")
            print(f"Calibration results: {output}")
            result = calibrate_session(session, output, options, config["mode"])
        passed = bool(result["passed"])
        if isinstance(result, dict):
            for label, sensor in result.get("sensors", {}).items():
                if args.command != "verify":
                    print(f"{label}: valid={sensor.get('valid_image_count', 0)}, training={sensor.get('training_image_count', 0)}, validation={sensor.get('validation_image_count', 0)}, RMS={sensor.get('training_rms_px')} px")
                    rejected = sensor.get("rejected_training_views", [])
                    print(f"{label}: excluded training images={len(rejected)} (validated separately using final parameters)")
                    for view in rejected:
                        print(f"  {view['image']}: RMSE={view['rmse_px_before_removal']:.6f} px, threshold={view['threshold_px']:.6f} px")
                validation = sensor.get("validation", {})
                print(f"{label}: validation images={validation.get('count', 0)}, mean RMSE={validation.get('mean_rmse_px')}, max RMSE={validation.get('max_rmse_px')} px")
                if sensor.get("excluded_validation_views"):
                    for view in sensor["excluded_validation_views"]:
                        print(f"{label}: excluded-image validation {view['path']}: RMSE={view.get('view_rms_px')} px, status={view['status']}")
            for pair in result.get("pairs", []):
                title = f"{pair['left_sensor']} -> {pair['right_sensor']}"
                print(f"{title}: baseline={pair.get('baseline_mm')} mm")
                if args.command != "verify":
                    print(f"{title}: stereo RMS={pair.get('stereo_rms_px')} px")
                    for rejected in pair.get("rejected_pairs", []):
                        print(f"  Excluded pair: {rejected['left_image']} / {rejected['right_image']}: {rejected['reason']}")
                validation = pair.get("validation", {})
                print(f"{title}: validation pairs={validation.get('count', 0)}, mean RMSE={validation.get('mean_rmse_px')}, max RMSE={validation.get('max_rmse_px')} px")
                for view in pair.get("excluded_validation_views", []):
                    print(f"{title}: excluded-pair validation {view['left_image']} / {view['right_image']}: RMSE={view.get('right_reprojection_rmse_px')} px, status={view['status']}")
        print(f"Pixel residual checks: {'PASS' if passed else 'FAIL'}")
        for path in result.get("output_files", []):
            print(f"Saved OpenCV YAML: {path}")
        for path in result.get("report_files", []):
            print(f"Saved calibration quality report: {path}")
        if output is not None and not passed and not any(output.iterdir()):
            output.rmdir()
        print("Results are geometry candidates. Review coverage warnings and independently validate metric accuracy.")
        return 0 if passed else 1
    except (ValueError, KeyError, OSError, RuntimeError, ImportError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
