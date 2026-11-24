#!/usr/bin/env python3
"""Focused checkerboard-scale calibration and group validation for marker IDs 84..92."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..remote import add_target_arguments, remote_path, run_calibration
from ..workflow.config import PROJECT_ROOT
from . import checkerboard as b
from .marker_io import result_path

IDS = list(range(84, 93))
TOOLS = {"C": IDS}
REFERENCE = 84
MIN_AREA_RATIO = 0.25


def calibrate_rectangles(views, ids, marker, camera, options, output):
    """Redetect full-resolution checkerboard corners for every capture group."""
    from ..sensor_intrinsics.detection import detect_checkerboard_realtime, prepare_gray8

    spec = marker["checkerboard"]
    pattern = (spec["pattern_cols"], spec["pattern_rows"])
    board = b.object_board(pattern, spec["square_size_mm"])
    frames, missing_board = [], []
    for view in views:
        image = cv2.imdecode(np.fromfile(view["image"], np.uint8), cv2.IMREAD_UNCHANGED)
        found, pixels, *_ = detect_checkerboard_realtime(prepare_gray8(image), pattern)
        if found:
            frames.append(dict(view, board=pixels.reshape(-1, 2)))
        else:
            missing_board.append(view["image"])
    b.write_json(output / "checkerboard_observations.json", {
        "checkerboard": spec, "joint_image_count": len(frames),
        "group_image_counts": dict(Counter(frame.get("group", 0) for frame in frames)),
        "marker_images_without_checkerboard": missing_board,
    })
    return fit_rectangles(frames, ids, marker, board, camera, options, output)


def fit_rectangles(views, ids, marker, board, camera, options, output):
    from .geometry import check_coverage
    from .planar import PlanarJointProblem, errors, rectangle_shape

    reference = marker["reference_marker_id"]
    k, d = camera["matrix"], camera["distortion"]
    frames = []
    group_counts = Counter()
    for view in views:
        group = view.get("group", 0)
        group_counts[group] += 1
        default_split = "test" if group_counts[group] % options["validation_stride"] == 0 else "train"
        frames.append(dict(view, group=group, markers=view["corners"],
                           split=view.get("split", default_split)))
    train = [frame for frame in frames if frame["split"] == "train"]
    test = [frame for frame in frames if frame["split"] == "test"]
    if not test:
        raise ValueError("Capture more frames containing both the checkerboard and target")
    connectivity_views = []
    for group in group_counts:
        group_train = [frame for frame in train if frame["group"] == group]
        if len(group_train) < 3 or not any(frame["group"] == group for frame in test):
            raise ValueError(f"Group {group + 1} needs at least three training images and one held-out image with both target and checkerboard")
        counts = Counter(mid for frame in group_train for mid in frame["markers"])
        connectivity_views.append({"corners": {mid for mid, count in counts.items() if count >= 3}})
    check_coverage(train, ids, reference, options, connectivity_views=connectivity_views)
    b.orient_boards(frames, board, k, d)
    tools = {"C": ids}
    poses, lengths, placements, _ = b.init_geometry(train, tools, k, d, output)
    anchor = poses[reference]
    inverse = np.linalg.inv(anchor)
    poses = {mid: inverse @ pose for mid, pose in poses.items()}
    placements = {key: pose @ anchor for key, pose in placements.items()}
    shapes = {mid: np.log([length, length]) for mid, length in lengths.items()}
    problem = PlanarJointProblem(
        train, tools, {"C": reference}, poses, shapes, placements, board, k, d,
        True, rectangular=True,
    )
    (poses, shapes, placements, cameras), stage = problem.solve(options["max_nfev"])
    points = {mid: b.transform(poses[mid], rectangle_shape(shape)) for mid, shape in shapes.items()}
    dimensions = {mid: dict(zip(("width", "height"), np.exp(shape).tolist())) for mid, shape in shapes.items()}
    lengths = {mid: (size["width"] + size["height"]) / 2 for mid, size in dimensions.items()}
    camera_map = {frame["image"]: pose for frame, pose in zip(train, cameras)}
    training = errors(train, points, placements, k, d, camera_map)
    validation = errors(test, points, placements, k, d)
    board_residuals = []
    for frame in frames:
        pose = camera_map.get(frame["image"], frame["camera"])
        delta = b.project(board, pose, k, d) - frame["board"]
        board_residuals.append(float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))))
    angles = {}
    for mid in ids:
        directions = []
        for frame, pose in zip(train, cameras):
            if mid in frame["markers"]:
                marker_to_camera = pose @ placements[frame["group"], "C"] @ poses[mid]
                direction = -marker_to_camera[:3, :3].T @ marker_to_camera[:3, 3]
                directions.append(direction / np.linalg.norm(direction))
        directions = np.asarray(directions)
        angles[mid] = float(np.rad2deg(np.arccos(np.clip(np.min(directions @ directions.T), -1, 1))))
    checks = {
        "solver_converged": stage["success"],
        "training_reprojection": all(r["rmse_px"] <= options["max_observation_rmse_px"] for r in training),
        "validation_reprojection": all(r["rmse_px"] <= options["max_validation_rmse_px"] for r in validation),
        "validation_marker_coverage": {r["id"] for r in validation} == set(ids),
        "checkerboard_reprojection": max(board_residuals) <= options["max_observation_rmse_px"],
        "positive_depth": all(r["positive_depth"] for r in training + validation),
        "viewing_diversity": all(angle >= options["min_view_angle_deg"] for angle in angles.values()),
        "dimension_bounds_inactive": all(2.002 < value < 59.94 for size in dimensions.values() for value in size.values()),
    }
    report = {
        "passed": all(checks.values()), "checks": checks, "geometry_model": "rectangle",
        "reference_marker_id": reference, "reference_marker_length_fixed": False,
        "checkerboard": marker["checkerboard"], "marker_dimensions_mm": dimensions,
        "maximum_checkerboard_rmse_px": max(board_residuals), "view_angle_range_deg": angles,
        "training_image_count": len(train), "validation_image_count": len(test),
        "groups": {group: {"images": count, "target_to_board": placements[group, "C"].tolist()}
                   for group, count in group_counts.items()},
        "stages": {"rectangle_joint_ba": stage},
        "training_observations": training, "validation_observations": validation,
    }
    return points, lengths, reference, report


def detect_frame(record, root):
    gray = b.gray8(cv2.imread(str(root / record["image"]), cv2.IMREAD_UNCHANGED))
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.65)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    params.minMarkerPerimeterRate = 0.008
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 43
    params.adaptiveThreshWinSizeStep = 10
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100), params
    )
    corners, ids, _ = detector.detectMarkers(gray)
    counts = Counter([] if ids is None else ids.ravel().tolist())
    markers = {}
    all_markers = {}
    quality = {}
    for mid, p in zip([] if ids is None else ids.ravel(), corners):
        mid = int(mid)
        if mid not in IDS:
            continue
        refined, q = b.black_border_corners(smooth, p, return_quality=True)
        contour = refined.astype(np.float32)
        area = abs(cv2.contourArea(contour))
        longest = float(
            np.linalg.norm(refined - np.roll(refined, 1, axis=0), axis=1).max()
        )
        ratio = float(area / max(longest * longest, 1.0))
        q["area_ratio"] = ratio
        if counts[mid] != 1:
            q["valid"] = False
            q["reason"] = "duplicate_id"
        if ratio < MIN_AREA_RATIO:
            q["valid"] = False
            q["reason"] = "severe_foreshortening"
        all_markers[mid] = refined.tolist()
        quality[mid] = q
        if q["valid"]:
            markers[mid] = refined.tolist()
    return dict(
        record,
        markers=markers,
        all_c_markers=all_markers,
        marker_quality=quality,
        marker_corner_method="signed_edge_intersections_with_image_only_quality_checks",
    )


def detect(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = json.loads(
        (Path(args.source) / "detections.json").read_text(encoding="utf-8")
    )
    cv2.setNumThreads(1)
    result = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(detect_frame, r, Path(args.images)) for r in cache["frames"]
        ]
        for future in as_completed(futures):
            result.append(future.result())
            if len(result) % 20 == 0:
                print("MARKERS_DETECTED", len(result), flush=True)
    cache["frames"] = sorted(result, key=lambda r: r["index"])
    b.write_json(output / "detections.json", cache)
    b.write_json(
        output / "image_quality.json",
        {
            "minimum_area_over_longest_edge_squared": MIN_AREA_RATIO,
            "quality_uses_geometry_residual": False,
            "retained_counts": dict(
                Counter(int(mid) for r in result for mid in r["markers"])
            ),
            "excluded": [
                {"image": r["image"], "id": int(mid), **q}
                for r in result
                for mid, q in r["marker_quality"].items()
                if not q["valid"]
            ],
            "large_corner_corrections": [
                {"image": r["image"], "id": int(mid), **q}
                for r in result
                for mid, q in r["marker_quality"].items()
                if q["valid"] and q["maximum_corner_shift_px"] > 12.0
            ],
        },
    )


def prepared(cache, k, d, board):
    frames = b.prepare_frames(copy.deepcopy(cache))
    b.orient_boards(frames, board, k, d)
    return frames


def apply_image_quality(cache, output):
    """Apply the fixed pixel-only eligibility rule; keep rejected candidates."""
    for r in cache["frames"]:
        for mid, q in r["marker_quality"].items():
            if q["area_ratio"] < MIN_AREA_RATIO:
                q["valid"] = False
                q["reason"] = "severe_foreshortening"
        r["markers"] = {
            mid: points
            for mid, points in r["all_c_markers"].items()
            if r["marker_quality"][mid]["valid"]
        }
    b.write_json(output / "detections.json", cache)
    b.write_json(
        output / "image_quality.json",
        {
            "minimum_area_over_longest_edge_squared": MIN_AREA_RATIO,
            "quality_uses_geometry_residual": False,
            "retained_counts": dict(
                Counter(int(mid) for r in cache["frames"] for mid in r["markers"])
            ),
            "excluded": [
                {"image": r["image"], "id": int(mid), **q}
                for r in cache["frames"]
                for mid, q in r["marker_quality"].items()
                if not q["valid"]
            ],
            "large_corner_corrections": [
                {"image": r["image"], "id": int(mid), **q}
                for r in cache["frames"]
                for mid, q in r["marker_quality"].items()
                if q["valid"] and q["maximum_corner_shift_px"] > 12.0
            ],
        },
    )
    return cache


def rebase(geometry, placements):
    reference = geometry[REFERENCE].copy()
    inverse = np.linalg.inv(reference)
    return (
        {mid: inverse @ matrix for mid, matrix in geometry.items()},
        {key: matrix @ reference for key, matrix in placements.items()},
    )


def solve_training(train, geometry, lengths, placements, board, k, d, limit=2.0):
    problem = b.JointProblem(
        train,
        TOOLS,
        {"C": REFERENCE},
        geometry,
        lengths,
        placements,
        board,
        k,
        d,
        False,
    )
    (geometry, lengths, placements, _), first = problem.solve(120)
    problem = b.JointProblem(
        train, TOOLS, {"C": REFERENCE}, geometry, lengths, placements, board, k, d, True
    )
    (geometry, lengths, placements, cameras), joint = problem.solve(160)
    removed = []
    for iteration in range(2):
        changed = False
        for r, c in zip(train, cameras):
            for mid, pixels in list(r["markers"].items()):
                points = b.transform(
                    placements[r["group"], "C"],
                    b.transform(geometry[mid], b.square(lengths[mid])),
                )
                error = float(
                    np.sqrt(
                        np.mean(
                            np.sum((b.project(points, c, k, d) - pixels) ** 2, axis=1)
                        )
                    )
                )
                if error > limit:
                    removed.append(
                        {
                            "image": r["image"],
                            "id": mid,
                            "rmse_px": error,
                            "reason": "training_observation_rmse_over_configured_2px",
                        }
                    )
                    del r["markers"][mid]
                    changed = True
            r["camera"] = c
        if not changed:
            break
        counts = Counter(mid for r in train for mid in r["markers"])
        if any(counts[mid] < 5 for mid in IDS):
            raise ValueError("Coverage lost after rejecting training outliers")
        if len(removed) > 0.2 * (sum(len(r["markers"]) for r in train) + len(removed)):
            raise ValueError("Excessive training outliers")
        problem = b.JointProblem(
            train,
            TOOLS,
            {"C": REFERENCE},
            geometry,
            lengths,
            placements,
            board,
            k,
            d,
            True,
        )
        (geometry, lengths, placements, cameras), joint = problem.solve(120)
    return (
        geometry,
        lengths,
        placements,
        cameras,
        {"fixed_board_camera_poses": first, "joint_board_marker_ba": joint},
        removed,
    )


def sample_errors(
    frames, geometry, lengths, placements, k, d, cameras=None, all_candidates=False
):
    records = []
    for r in frames:
        c = (
            cameras[r["image"]]
            if cameras is not None and r["image"] in cameras
            else r["camera"]
        )
        observations = r.get("all_c_markers", {}) if all_candidates else r["markers"]
        for mid, pixels in observations.items():
            mid = int(mid)
            points = b.transform(
                placements[r["group"], "C"],
                b.transform(geometry[mid], b.square(lengths[mid])),
            )
            predicted = b.project(points, c, k, d)
            delta = np.linalg.norm(predicted - np.asarray(pixels), axis=1)
            records.append(
                {
                    "image": r["image"],
                    "group": r["group"],
                    "split": r["split"],
                    "tool": "C",
                    "id": mid,
                    "measured": np.asarray(pixels).tolist(),
                    "predicted": predicted.tolist(),
                    "rmse_px": float(np.sqrt(np.mean(delta**2))),
                    "corner_errors_px": delta.tolist(),
                    "positive_depth": bool(np.min(b.transform(c, points)[:, 2]) > 0),
                }
            )
    return records


def save_fit(
    output, cache, frames, k, d, geometry, lengths, placements, cameras, stages, removed
):
    b.write_json(
        output / "fit.json",
        {
            "tools": TOOLS,
            "references": {"C": REFERENCE},
            "lengths_mm": lengths,
            "marker_poses": {mid: m.tolist() for mid, m in geometry.items()},
            "group_poses": {
                f"{g}:{tool}": m.tolist() for (g, tool), m in placements.items()
            },
            "training_cameras": {
                r["image"]: c.tolist()
                for r, c in zip([r for r in frames if r["split"] == "train"], cameras)
            },
            "board_order_flipped_images": [],
            "camera_matrix": k.tolist(),
            "distortion": d.ravel().tolist(),
            "stages": stages,
            "split": {r["image"]: r["split"] for r in frames},
            "counts": dict(Counter(mid for r in frames for mid in r["markers"])),
            "training_observations_removed": removed,
            "source": cache["source"],
        },
    )


def fit(args):
    cv2.setNumThreads(1)
    output = Path(args.output)
    source = Path(args.source)
    cache = apply_image_quality(
        json.loads((output / "detections.json").read_text(encoding="utf-8")), output
    )
    previous = json.loads((source / "fit.json").read_text(encoding="utf-8"))
    k = np.asarray(previous["camera_matrix"])
    d = np.asarray(previous["distortion"])
    board = b.object_board(tuple(cache["board"]["pattern_size"]), 35.0)
    frames = prepared(cache, k, d, board)
    geometry, lengths, placements, _ = b.init_geometry(frames, TOOLS, k, d, output)
    geometry, placements = rebase(geometry, placements)
    train = [r for r in frames if r["split"] == "train"]
    geometry, lengths, placements, cameras, stages, removed = solve_training(
        train, geometry, lengths, placements, board, k, d
    )
    save_fit(
        output,
        cache,
        frames,
        k,
        d,
        geometry,
        lengths,
        placements,
        cameras,
        stages,
        removed,
    )
    for name in (
        "camera_intrinsics.yml",
        "intrinsics_report.json",
        "source_manifest.json",
    ):
        shutil.copyfile(source / name, output / name)
    # Preserve the core validation format for projection overlays and metrics.
    b.validate_all(argparse.Namespace(output=str(output)))
    extra_validation(
        output,
        cache,
        frames,
        geometry,
        lengths,
        placements,
        cameras,
        k,
        d,
        board,
        removed,
    )


def extra_validation(
    output, cache, frames, geometry, lengths, placements, cameras, k, d, board, removed
):
    independent = prepared(cache, k, d, board)
    camera_by_name = {
        r["image"]: c
        for r, c in zip([r for r in frames if r["split"] == "train"], cameras)
    }
    all_raw = sample_errors(
        independent, geometry, lengths, placements, k, d, camera_by_name, True
    )
    eligible = sample_errors(
        independent, geometry, lengths, placements, k, d, camera_by_name, False
    )
    test = [r for r in eligible if r["split"] == "test"]
    b.write_json(
        output / "marker_quality_summary.json",
        {
            "ids": IDS,
            "reference_marker_id": REFERENCE,
            "marker_count": 9,
            "scale_source": "11x8 checkerboard, 35 mm; all marker lengths free",
            "eligible_test_corner_errors_px": b.stats(
                [e for r in test for e in r["corner_errors_px"]]
            ),
            "eligible_test_per_id_px": {
                mid: b.stats(
                    [e for r in test if r["id"] == mid for e in r["corner_errors_px"]]
                )
                for mid in IDS
            },
            "all_candidates_test_corner_errors_px": b.stats(
                [
                    e
                    for r in all_raw
                    if r["split"] == "test"
                    for e in r["corner_errors_px"]
                ]
            ),
            "eligible_test_observations": len(test),
            "all_candidate_test_observations": sum(
                r["split"] == "test" for r in all_raw
            ),
            "training_outliers_removed": removed,
            "maximum_eligible_test_observation_rmse_px": max(
                r["rmse_px"] for r in test
            ),
            "all_eligible_test_depths_positive": all(r["positive_depth"] for r in test),
            "edge_lengths_mm": lengths,
            "statement": "Reprojection and split reconstruction consistency do not establish independent absolute metrology.",
        },
    )
    b.write_json(
        output / "marker_all_candidate_observations.json", {"observations": all_raw}
    )


def fit_placement(frames, geometry, lengths, k, d):
    def residual(x):
        e = b.pose(x)
        records = []
        for r in frames:
            for mid, pixels in r["markers"].items():
                points = b.transform(
                    e, b.transform(geometry[mid], b.square(lengths[mid]))
                )
                records.append((b.project(points, r["camera"], k, d) - pixels).ravel())
        return np.concatenate(records)

    first = max(frames, key=lambda r: len(r["markers"]))
    ids = sorted(first["markers"])
    objects = np.concatenate(
        [b.transform(geometry[mid], b.square(lengths[mid])) for mid in ids]
    )
    pixels = np.concatenate([first["markers"][mid] for mid in ids])
    camera_to_object = b.pnp(objects, pixels, k, d)
    initial = np.linalg.inv(first["camera"]) @ camera_to_object
    result = least_squares(
        residual,
        b.vector(initial),
        loss="soft_l1",
        f_scale=0.7,
        x_scale="jac",
        max_nfev=120,
    )
    return b.pose(result.x), {"success": bool(result.success), "nfev": int(result.nfev)}


def group_validation(args):
    cv2.setNumThreads(1)
    output = Path(args.output)
    cache = json.loads((output / "detections.json").read_text(encoding="utf-8"))
    final = json.loads((output / "fit.json").read_text(encoding="utf-8"))
    k = np.asarray(final["camera_matrix"])
    d = np.asarray(final["distortion"])
    board = b.object_board(tuple(cache["board"]["pattern_size"]), 35.0)
    final_geometry = {
        int(mid): np.asarray(m) for mid, m in final["marker_poses"].items()
    }
    final_lengths = {int(mid): v for mid, v in final["lengths_mm"].items()}
    folds = []
    for group in range(7):
        fold_dir = output / "group_validation" / f"without_group_{group}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        frames = prepared(cache, k, d, board)
        others = [r for r in frames if r["group"] != group]
        geometry, lengths, placements, _ = b.init_geometry(
            others, TOOLS, k, d, fold_dir
        )
        geometry, placements = rebase(geometry, placements)
        train = [r for r in others if r["split"] == "train"]
        geometry, lengths, placements, _, stages, removed = solve_training(
            train, geometry, lengths, placements, board, k, d
        )
        held_train = [
            r
            for r in frames
            if r["group"] == group and r["split"] == "train" and r["markers"]
        ]
        placement, placement_report = fit_placement(held_train, geometry, lengths, k, d)
        test = [r for r in frames if r["group"] == group and r["split"] == "test"]
        errors = sample_errors(test, geometry, lengths, {(group, "C"): placement}, k, d)
        differences = {
            mid: np.linalg.norm(
                b.transform(geometry[mid], b.square(lengths[mid]))
                - b.transform(final_geometry[mid], b.square(final_lengths[mid])),
                axis=1,
            ).tolist()
            for mid in IDS
        }
        fold = {
            "held_geometry_group": group,
            "test_corner_error_px": b.stats(
                [e for r in errors for e in r["corner_errors_px"]]
            ),
            "maximum_test_observation_rmse_px": max(r["rmse_px"] for r in errors),
            "corner_geometry_change_mm": b.stats(
                [e for values in differences.values() for e in values]
            ),
            "per_id_corner_geometry_change_mm": differences,
            "per_id_edge_change_mm": {
                mid: abs(lengths[mid] - final_lengths[mid]) for mid in IDS
            },
            "lengths_mm": lengths,
            "stages": stages,
            "placement_fit": placement_report,
            "training_outliers_removed": removed,
        }
        b.write_json(fold_dir / "result.json", fold)
        folds.append(fold)
        print(
            "GROUP_VALIDATION",
            group,
            json.dumps(fold["test_corner_error_px"]),
            json.dumps(fold["corner_geometry_change_mm"]),
            flush=True,
        )
        b.write_json(
            output / "marker_group_validation.json",
            {
                "method": "Leave one complete capture group out of marker geometry estimation; fixed camera intrinsics. Estimate only held-group placement using its training portion; evaluate its untouched final 20%. Reference ID 84 fixes coordinate frame; no similarity alignment.",
                "folds": folds,
            },
        )


def export(args):
    from .marker_io import save_geometry

    output = Path(args.output)
    fit = json.loads((output / "fit.json").read_text(encoding="utf-8"))
    report = json.loads(
        result_path(output, "marker_quality_summary.json").read_text(encoding="utf-8")
    )
    group_report = json.loads(
        result_path(output, "marker_group_validation.json").read_text(encoding="utf-8")
    )
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    geometry = {
        int(mid): b.transform(np.asarray(matrix), b.square(fit["lengths_mm"][mid]))
        for mid, matrix in fit["marker_poses"].items()
    }
    lengths = {int(mid): float(v) for mid, v in fit["lengths_mm"].items()}
    config["marker"]["length_source"] = "checkerboard_35_mm"
    thresholds = config["calibration"]
    cache = json.loads((output / "detections.json").read_text(encoding="utf-8"))
    board = b.object_board((11, 8), 35.0)
    k = np.asarray(fit["camera_matrix"])
    d = np.asarray(fit["distortion"])
    frames = prepared(cache, k, d, board)
    excluded = {(r["image"], r["id"]) for r in fit["training_observations_removed"]}
    train = [r for r in frames if r["split"] == "train"]
    for r in train:
        r["markers"] = {
            mid: p
            for mid, p in r["markers"].items()
            if (r["image"], mid) not in excluded
        }
    poses = {int(mid): np.asarray(m) for mid, m in fit["marker_poses"].items()}
    placements = {
        (int(key.split(":")[0]), key.split(":")[1]): np.asarray(m)
        for key, m in fit["group_poses"].items()
    }
    cameras = {name: np.asarray(m) for name, m in fit["training_cameras"].items()}
    training_errors = sample_errors(train, poses, lengths, placements, k, d, cameras)
    counts = Counter(r["id"] for r in training_errors)
    diversity = {}
    for mid in IDS:
        directions = []
        for r in train:
            if mid not in r["markers"]:
                continue
            matrix = cameras[r["image"]] @ placements[r["group"], "C"] @ poses[mid]
            direction = -matrix[:3, :3].T @ matrix[:3, 3]
            directions.append(direction / np.linalg.norm(direction))
        directions = np.asarray(directions)
        diversity[mid] = float(
            np.rad2deg(np.arccos(np.clip((directions @ directions.T).min(), -1, 1)))
        )
    checks = {
        "all_nine_C_ids": sorted(geometry) == IDS,
        "training_coverage": len(train) >= thresholds["min_views"]
        and all(counts[mid] >= thresholds["min_marker_views"] for mid in IDS),
        "solver_converged": all(stage["success"] for stage in fit["stages"].values()),
        "training_reprojection": all(
            r["rmse_px"] <= thresholds["max_observation_rmse_px"]
            for r in training_errors
        ),
        "validation_reprojection": report["maximum_eligible_test_observation_rmse_px"]
        <= thresholds["max_validation_rmse_px"],
        "validation_marker_coverage": all(
            report["eligible_test_per_id_px"][str(mid)]["count"] > 0 for mid in IDS
        ),
        "positive_depth": report["all_eligible_test_depths_positive"]
        and all(r["positive_depth"] for r in training_errors),
        "viewing_diversity": all(
            angle >= thresholds["min_view_angle_deg"] for angle in diversity.values()
        ),
        "group_validation_complete": len(group_report["folds"]) == 7,
        "group_validation_solvers": all(
            all(s["success"] for s in fold["stages"].values())
            and fold["placement_fit"]["success"]
            for fold in group_report["folds"]
        ),
        "group_validation_reprojection": all(
            fold["maximum_test_observation_rmse_px"]
            <= thresholds["max_validation_rmse_px"]
            for fold in group_report["folds"]
        ),
    }
    quality = {
        "passed": all(checks.values()),
        "checks": checks,
        "reference_marker_id": REFERENCE,
        "marker_lengths_mm": lengths,
        "scale_source": "checkerboard_35_mm",
        "reference_marker_length_fixed": False,
        "training_image_count": len(train),
        "validation_image_count": sum(r["split"] == "test" for r in frames),
        "training_marker_counts": dict(counts),
        "view_angle_range_deg": diversity,
        "training_observations": training_errors,
        "quality_summary": report,
        "group_validation": group_report,
        "scope": "Target marker geometry only; image-qualified observations; conditional on fixed 35 mm planar board",
        "absolute_accuracy_guaranteed": False,
        "tip_offset_source": "configured_not_calibrated",
        "thresholds_source": str(Path(args.config).resolve()),
        "thresholds": thresholds,
    }
    b.write_json(output / "marker_calibration_quality.json", quality)
    if not quality["passed"]:
        raise ValueError(
            f"Quality gates failed; formal geometry not exported: {checks}"
        )
    destination = output / config["output"]["filename"]
    save_geometry(
        destination,
        geometry,
        lengths,
        REFERENCE,
        config,
        optimization_method="checkerboard_metric_independent_square_joint_ba",
        reference_marker_length_fixed=False,
    )
    b.write_json(
        output / "marker_workflow.json",
        {
            "source": fit["source"],
            "target_ids": IDS,
            "reference_marker_id": REFERENCE,
            "board": {"pattern_size": [11, 8], "square_size_mm": 35},
            "output": config["output"],
            "coordinate_frame": "ID84 center, decoded +X right, +Y up, +Z outward",
            "training_and_validation": "Only target marker observations used in fitting",
            "reference_marker_length_fixed": False,
            "tip_offset_calibrated": False,
        },
    )
    print("SAVED", destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["detect", "fit", "groups", "export"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--source")
    parser.add_argument("--images")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--geometry-model",
        choices=["planar", "square"],
        default="planar",
        help="Exact planar quadrilaterals (default), or legacy ideal squares",
    )
    parser.add_argument(
        "--baseline",
        help="Existing square result directory; freezes observations for a fair comparison",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/markers/double_cube.json"),
    )
    add_target_arguments(parser, PROJECT_ROOT)
    args = parser.parse_args()
    if args.stage == "detect" and not args.source:
        parser.error("--source is required")
    if args.stage == "fit" and not args.source and not args.baseline:
        parser.error("--source or --baseline is required")
    if args.stage == "detect" and not args.images:
        parser.error("--images is required")
    if args.ssh:
        remote_args = [
            args.stage,
            "--workers",
            str(args.workers),
            "--geometry-model",
            args.geometry_model,
        ]
        try:
            for name in ("output", "source", "images", "baseline", "config"):
                value = getattr(args, name)
                if value is not None:
                    remote_args.extend(("--" + name, remote_path(value, PROJECT_ROOT)))
            return run_calibration(
                args,
                PROJECT_ROOT,
                remote_args,
                module="python.scripts.marker_calibration.checkerboard_geometry",
            )
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            RuntimeError,
            ImportError,
        ) as error:
            parser.error(str(error))
    if args.stage == "fit" and args.geometry_model == "planar":
        from . import planar as planar_marker_geometry

        planar_marker_geometry.fit(args)
        return
    if args.stage in ("groups", "export"):
        saved = json.loads((Path(args.output) / "fit.json").read_text(encoding="utf-8"))
        if saved.get("geometry_model") == "independent_planar_quadrilateral":
            from . import planar as planar_marker_geometry

            {
                "groups": planar_marker_geometry.group_validation,
                "export": planar_marker_geometry.export,
            }[args.stage](args)
            return
    if args.stage == "fit" and not args.source:
        parser.error("Square fit requires --source")
    {"detect": detect, "fit": fit, "groups": group_validation, "export": export}[
        args.stage
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
