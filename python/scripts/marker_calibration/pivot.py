"""Estimate a fixed ruby sphere center from multi-view marker poses."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

from .marker_io import create_detector, detect, load_camera


def solve_pivot(rotations, translations, robust_mm):
    # R_i @ offset + t_i = center. Eliminate center to check observability.
    motion = (rotations - rotations.mean(axis=0)).reshape(-1, 3)
    singular = np.linalg.svd(motion, compute_uv=False) / np.sqrt(len(rotations))
    if singular[-1] < 0.10 or singular[0] / singular[-1] > 10:
        raise ValueError(
            f"Poor rotation coverage; normalized singular values: {singular.tolist()}"
        )
    design = np.concatenate(
        (rotations, -np.broadcast_to(np.eye(3), rotations.shape)), axis=2
    )
    design = design.reshape(-1, 6)
    target = -translations.ravel()
    initial = np.linalg.lstsq(design, target, rcond=None)[0]
    fit = least_squares(
        lambda x: design @ x - target,
        initial,
        jac=lambda x: design,
        loss="soft_l1",
        f_scale=robust_mm,
    )
    if not fit.success:
        raise ValueError(f"Pivot fit failed: {fit.message}")
    return fit.x[:3], fit.x[3:], singular


def capture_poses(paths, geometry, camera, detector, threshold):
    accepted, rejected = [], []
    k, d = camera["matrix"], camera["distortion"]
    for path in paths:
        if path.suffix.lower() not in (
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".jpg",
            ".jpeg",
        ):
            continue
        if not path.is_file():
            rejected.append({"image": path.name, "reason": "missing capture file"})
            continue
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or list(image.shape[:2][::-1]) != camera["image_size"]:
            raise ValueError(
                f"Image unreadable or size differs from intrinsics: {path}"
            )
        observed, _ = detect(image, detector)
        ids = sorted(geometry.keys() & observed.keys())
        if len(ids) < 2:
            rejected.append({"image": path.name, "reason": "fewer than two markers"})
            continue
        obj = np.concatenate([geometry[i] for i in ids])
        pix = np.concatenate([observed[i] for i in ids]).astype(np.float64)
        # A planar subset can produce two nearly equivalent PnP orientations.
        if np.linalg.svd(obj - obj.mean(axis=0), compute_uv=False)[-1] < 1:
            rejected.append(
                {"image": path.name, "reason": "nearly planar corner set", "ids": ids}
            )
            continue
        ok, rvec, tvec = cv2.solvePnP(obj, pix, k, d, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            rejected.append({"image": path.name, "reason": "PnP failed", "ids": ids})
            continue
        rotation = cv2.Rodrigues(rvec)[0]
        error = cv2.projectPoints(obj, rvec, tvec, k, d)[0].reshape(-1, 2) - pix
        rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
        if (
            not np.isfinite(rms)
            or rms > threshold
            or (obj @ rotation.T + tvec.ravel())[:, 2].min() <= 0
        ):
            rejected.append(
                {"image": path.name, "reason": "PnP quality", "ids": ids, "rms_px": rms}
            )
            continue
        accepted.append(
            (rotation, tvec.ravel(), {"image": path.name, "ids": ids, "rms_px": rms})
        )
    return accepted, rejected


def run_configured(command, config_path, session=None, *, geometry=None, camera_vendor=None, camera_intrinsics=None):
    from ..session_storage import timestamp_directory
    from ..workflow.config import PROJECT_ROOT
    from .marker_session import capture_session
    from .marker_io import load_config, select_capture_camera

    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if geometry is None:
        if not config.get("geometry"):
            raise ValueError("Specify the calibrated marker geometry YAML with --pivot-geometry.")
        geometry = (config_path.parent / config["geometry"]).resolve()
    if not geometry.is_file():
        raise ValueError(f"Marker geometry is missing: {geometry}")
    if command in ("run", "capture"):
        if session is not None:
            raise ValueError("Pivot capture starts a new session")
        capture_config = load_config(config_path.parent / config["capture_config"])
        capture_config["capture_title"] = "竖轴定位 · 相机采集"
        document = yaml.safe_load(geometry.read_text(encoding="utf-8-sig"))
        ids = sorted(int(mid) for mid in document["marker_positions"])
        capture_config["marker"] = {
            "dictionary": document["instrument"]["marker_dictionary"],
            "expected_count": len(ids),
            "ids": ids,
        }
        if "intrinsics" in config:
            capture_config["camera_parameters"] = (
                config_path.parent / config["intrinsics"]
            ).resolve()
        select_capture_camera(capture_config, camera_vendor, camera_intrinsics)
        camera_path = capture_config["camera_parameters"]
        camera = load_camera(camera_path) if camera_path else None
        session = timestamp_directory(PROJECT_ROOT / "data")
        print(f"Capture session: {session}", flush=True)
        if not capture_session(capture_config, session, camera):
            print("No pivot images saved.", flush=True)
            return 1
        if command == "capture":
            return 0
    elif command != "calibrate" or session is None:
        raise ValueError("Pivot supports run, capture, or calibrate --session")
    # main requires a new directory so a prior result cannot be overwritten.
    output = timestamp_directory(PROJECT_ROOT / "output")
    output.rmdir()
    print(f"Results: {output}", flush=True)
    return main(
        [
            "--geometry",
            str(geometry),
            "--session",
            str(session),
            "--output",
            str(output),
            "--max-validation-rms-mm",
            str(config["max_validation_rms_mm"]),
        ]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("geometry", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("session", "intrinsics", "train", "validate"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--max-validation-rms-mm", type=float, required=True)
    parser.add_argument("--max-reprojection-px", type=float, default=1.0)
    args = parser.parse_args(argv)
    for value in (args.max_validation_rms_mm, args.max_reprojection_px):
        if not np.isfinite(value) or value <= 0:
            parser.error("error limits must be finite and positive")
    if args.session:
        if args.train or args.validate:
            parser.error("--session cannot be combined with --train or --validate")
        metadata = json.loads(
            (args.session / "capture_metadata.json").read_text(encoding="utf-8-sig")
        )
        paths = [args.session / p for p in metadata["images"]["marker_camera"]]
        if len({p.resolve() for p in paths}) != len(paths):
            parser.error("duplicate capture paths in session")
        split = len(paths) * 3 // 4
        paths_train, paths_test = paths[:split], paths[split:]
        args.intrinsics = args.intrinsics or args.session / metadata.get(
            "camera_parameters", "camera_params.yml"
        )
        split_method = "same_session_chronological_last_quarter_held_out"
    else:
        if not all((args.train, args.validate, args.intrinsics)):
            parser.error("use --session or supply --train, --validate and --intrinsics")
        if args.train.resolve() == args.validate.resolve():
            parser.error("use separately captured training and validation directories")
        paths_train, paths_test = (
            sorted(args.train.iterdir()),
            sorted(args.validate.iterdir()),
        )
        split_method = "separate_capture_directories"
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}")
    document = yaml.safe_load(args.geometry.read_text(encoding="utf-8-sig"))
    geometry = {
        int(i): np.asarray(p, np.float64)
        for i, p in document["marker_positions"].items()
    }
    camera = load_camera(args.intrinsics)
    detector = create_detector(document["instrument"]["marker_dictionary"])
    train, train_rejected = capture_poses(
        paths_train, geometry, camera, detector, args.max_reprojection_px
    )
    test, test_rejected = capture_poses(
        paths_test, geometry, camera, detector, args.max_reprojection_px
    )
    if len(train) < 12 or len(test) < 6:
        parser.error(
            f"Need at least 12 training and 6 validation poses after filtering; got {len(train)} and {len(test)}"
        )
    rotations, translations, _ = zip(*train)
    try:
        offset, center, singular = solve_pivot(
            np.asarray(rotations),
            np.asarray(translations),
            args.max_validation_rms_mm,
        )
    except ValueError as error:
        parser.error(str(error))
    report = {
        "geometry": str(args.geometry),
        "session": str(args.session) if args.session else None,
        "validation_split": split_method,
        "intrinsics": str(args.intrinsics),
        "sphere_center_mm": offset.tolist(),
        "pivot_in_camera_mm": center.tolist(),
        "rotation_singular_values": singular.tolist(),
        "train_rejected": train_rejected,
        "validation_rejected": test_rejected,
        "max_validation_rms_mm": args.max_validation_rms_mm,
    }
    for label, rows in (("train", train), ("validation", test)):
        errors = np.array([np.linalg.norm(r @ offset + t - center) for r, t, _ in rows])
        report[label] = {
            "rms_mm": float(np.sqrt(np.mean(errors**2))),
            "p95_mm": float(np.quantile(errors, 0.95)),
            "max_mm": float(errors.max()),
            "frames": [
                dict(record, pivot_error_mm=float(e))
                for (_, _, record), e in zip(rows, errors)
            ],
        }
    report["passed"] = report["validation"]["rms_mm"] <= args.max_validation_rms_mm
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "pivot_calibration_quality.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not report["passed"]:
        print(f"Validation failed: {report['validation']['rms_mm']:.3f} mm RMS")
        return 1
    document.update(
        tip_offset=offset.tolist(),
        sphere_center=offset.tolist(),
        tip_offset_source="fixed_sphere_center_multi_pose_pivot",
    )
    (args.output / "double_cube_geometry.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(
        f"Saved {args.output / 'double_cube_geometry.yaml'}; validation RMS {report['validation']['rms_mm']:.3f} mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
