"""Exact planar quadrilateral BA, with a centered reference-marker gauge.

Five local shape parameters describe each convex quad independently of its
six-dimensional rigid pose. The reference pose is fixed; its shape is free.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from . import checkerboard as b
from .marker_io import result_path

MODEL = "independent_planar_quadrilateral"


def square_shape(length):
    return np.array([np.log(length), length, np.log(length), 0.0, np.log(length)])


def rectangle_shape(shape, derivatives=False):
    width, height = np.exp(shape)
    points = b.square(1.0) * [width, height, 1.0]
    if not derivatives:
        return points
    jac = np.zeros((4, 3, 2))
    jac[:, 0, 0] = points[:, 0]
    jac[:, 1, 1] = points[:, 1]
    return points, jac


def quad(shape, derivatives=False):
    """Center at the four-corner centroid; decoded top edge defines +X.

    Raw points: (0,0), (a,0), (u,-h), (v,-k). Positive a,h,k remove
    reflections of the local frame. No right angle/equal-edge constraint.
    """
    a, h, k = np.exp(np.asarray(shape)[[0, 2, 4]])
    points = np.array(
        [[0.0, 0.0, 0.0], [a, 0.0, 0.0], [shape[1], -h, 0.0], [shape[3], -k, 0.0]]
    )
    points -= points.mean(axis=0)
    if not derivatives:
        return points
    jac = np.zeros((4, 3, 5))
    jac[1, 0, 0] = a
    jac[2, 0, 1] = 1.0
    jac[2, 1, 2] = -h
    jac[3, 0, 3] = 1.0
    jac[3, 1, 4] = -k
    jac -= jac.mean(axis=0, keepdims=True)
    return points, jac


def corners(poses, shapes):
    return {mid: b.transform(poses[mid], quad(shapes[mid])) for mid in poses}


def edge_lengths(points):
    # Edge order: decoded 0->1, 1->2, 2->3, 3->0.
    return np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)


def geometry_metrics(points):
    """Angles at decoded vertices; edge directions in the marker reference frame."""
    points = np.asarray(points, np.float64)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 1e-10):
        raise ValueError("Degenerate quadrilateral edge")
    incoming = np.roll(points, 1, axis=0) - points
    cosine = (
        np.sum(incoming * edges, axis=1) / np.linalg.norm(incoming, axis=1) / lengths
    )
    angles = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    normal = np.cross(points[3] - points[0], points[1] - points[0])
    if np.linalg.norm(normal) < 1e-10:
        raise ValueError("Degenerate quadrilateral plane")
    normal /= np.linalg.norm(normal)
    directions = edges / lengths[:, None]
    return {
        "edge_lengths_mm": lengths.tolist(),
        "interior_angles_deg": angles.tolist(),
        "interior_angle_sum_deg": float(angles.sum()),
        "edge_turn_angles_deg": (180.0 - angles).tolist(),
        "edge_unit_vectors": directions.tolist(),
        "edge_angles_to_positive_axes_deg": np.rad2deg(
            np.arccos(np.clip(directions, -1, 1))
        ).tolist(),
        "outward_normal": normal.tolist(),
        "center_mm": points.mean(axis=0).tolist(),
        "area_mm2": float(
            np.linalg.norm(
                np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
            )
            * 0.5
        ),
    }


def plane_angles(points_by_id):
    normals = {
        mid: np.asarray(geometry_metrics(q)["outward_normal"])
        for mid, q in points_by_id.items()
    }
    result = []
    ids = sorted(normals)
    for i, a in enumerate(ids):
        for other in ids[i + 1 :]:
            dot = float(np.clip(normals[a] @ normals[other], -1, 1))
            result.append(
                {
                    "ids": [a, other],
                    "directed_normal_angle_deg": float(np.rad2deg(np.arccos(dot))),
                    "unoriented_plane_angle_deg": float(
                        np.rad2deg(np.arccos(abs(dot)))
                    ),
                }
            )
    return result


class PlanarJointProblem(b.JointProblem):
    def __init__(
        self,
        frames,
        tools,
        references,
        geometry,
        shapes,
        placements,
        board,
        k,
        d,
        vary_cameras,
        *,
        rectangular=False,
    ):
        self.frames = frames
        self.tools = tools
        self.references = references
        self.board = board
        self.k = k
        self.d = d
        self.owner = {mid: tool for tool, ids in tools.items() for mid in ids}
        self.geometry0 = geometry
        self.shape_points = rectangle_shape if rectangular else quad
        length_offsets = (0, 1) if rectangular else (0, 2, 4)
        self.marker_slices = {}
        self.shape_slices = {}
        self.group_slices = {}
        self.camera_slices = {}
        x = []
        for mid in sorted(geometry):
            if mid not in references.values():
                self.marker_slices[mid] = slice(len(x), len(x) + 6)
                x.extend(b.vector(geometry[mid]))
            self.shape_slices[mid] = slice(len(x), len(x) + len(shapes[mid]))
            x.extend(shapes[mid])
        for key in sorted(placements):
            self.group_slices[key] = slice(len(x), len(x) + 6)
            x.extend(b.vector(placements[key]))
        if vary_cameras:
            for i, r in enumerate(frames):
                self.camera_slices[i] = slice(len(x), len(x) + 6)
                x.extend(b.vector(r["camera"]))
        self.x = np.asarray(x)
        self.lower = np.full(len(x), -np.inf)
        self.upper = np.full(len(x), np.inf)
        for section in self.shape_slices.values():
            for offset in length_offsets:
                self.lower[section.start + offset] = np.log(2.0)
                self.upper[section.start + offset] = np.log(60.0)
        self.cached_x = None
        self.cached = None

    def unpack(self, x):
        poses = {
            mid: b.pose(x[self.marker_slices[mid]])
            if mid in self.marker_slices
            else np.eye(4)
            for mid in self.geometry0
        }
        shapes = {mid: x[s].copy() for mid, s in self.shape_slices.items()}
        placements = {key: b.pose(x[s]) for key, s in self.group_slices.items()}
        cameras = [
            b.pose(x[self.camera_slices[i]]) if i in self.camera_slices else r["camera"]
            for i, r in enumerate(self.frames)
        ]
        return poses, shapes, placements, cameras

    def evaluate(self, x):
        if self.cached_x is not None and np.array_equal(x, self.cached_x):
            return self.cached
        poses, shapes, placements, cameras = self.unpack(x)
        local = {mid: self.shape_points(shape, True) for mid, shape in shapes.items()}
        world = {mid: b.transform(poses[mid], q[0]) for mid, q in local.items()}
        marker_dr = {
            mid: cv2.Rodrigues(x[s][:3])[1].reshape(3, 3, 3)
            for mid, s in self.marker_slices.items()
        }
        group_dr = {
            key: cv2.Rodrigues(x[s][:3])[1].reshape(3, 3, 3)
            for key, s in self.group_slices.items()
        }
        rows = []
        cols = []
        data = []
        residuals = []
        offset = 0

        def add(start, section, values):
            values = np.asarray(values).reshape(-1, section.stop - section.start)
            rows.append(
                np.repeat(np.arange(start, start + len(values)), values.shape[1])
            )
            cols.append(np.tile(np.arange(section.start, section.stop), len(values)))
            data.append(values.ravel())

        for i, (r, camera) in enumerate(zip(self.frames, cameras)):
            ids = sorted(mid for mid in r["markers"] if mid in poses)
            placed = {
                mid: b.transform(placements[r["group"], self.owner[mid]], world[mid])
                for mid in ids
            }
            board_inliers = r.get("board_inliers", slice(None))
            board_points = self.board[board_inliers]
            points = np.concatenate([board_points, *[placed[mid] for mid in ids]])
            measured = np.concatenate([r["board"][board_inliers], *[r["markers"][mid] for mid in ids]])
            pixels, jac = cv2.projectPoints(
                points, b.vector(camera)[:3], camera[:3, 3], self.k, self.d
            )
            residuals.append((pixels.reshape(-1, 2) - measured).ravel())
            if i in self.camera_slices:
                add(offset, self.camera_slices[i], jac[:, :6])
            jp = jac[:, 3:6].reshape(-1, 2, 3)
            begin = len(board_points)
            for mid in ids:
                key = (r["group"], self.owner[mid])
                placement = placements[key]
                jpc = jp[begin : begin + 4] @ camera[:3, :3]
                drg = np.einsum("arc,nc->nra", group_dr[key], world[mid])
                add(
                    offset + begin * 2,
                    self.group_slices[key],
                    np.concatenate([jpc @ drg, jpc], axis=2),
                )
                jpm = jpc @ placement[:3, :3]
                if mid in self.marker_slices:
                    drm = np.einsum("arc,nc->nra", marker_dr[mid], local[mid][0])
                    add(
                        offset + begin * 2,
                        self.marker_slices[mid],
                        np.concatenate([jpm @ drm, jpm], axis=2),
                    )
                add(
                    offset + begin * 2,
                    self.shape_slices[mid],
                    (jpm @ poses[mid][:3, :3]) @ local[mid][1],
                )
                begin += 4
            offset += len(points) * 2
        residual = np.concatenate(residuals)
        jacobian = coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(offset, len(x)),
        ).tocsr()
        self.cached_x = x.copy()
        self.cached = (residual, jacobian)
        return self.cached


def errors(frames, points, placements, k, d, cameras=None, all_candidates=False):
    result = []
    for r in frames:
        camera = (
            cameras[r["image"]]
            if cameras is not None and r["image"] in cameras
            else r["camera"]
        )
        observed = r.get("all_c_markers", {}) if all_candidates else r["markers"]
        for mid, pixels in observed.items():
            mid = int(mid)
            placed = b.transform(placements[r["group"], "C"], points[mid])
            prediction = b.project(placed, camera, k, d)
            delta = np.linalg.norm(prediction - np.asarray(pixels), axis=1)
            result.append(
                {
                    "image": r["image"],
                    "group": r["group"],
                    "split": r["split"],
                    "tool": "C",
                    "id": mid,
                    "measured": np.asarray(pixels).tolist(),
                    "predicted": prediction.tolist(),
                    "corner_errors_px": delta.tolist(),
                    "rmse_px": float(np.sqrt(np.mean(delta**2))),
                    "positive_depth": bool(
                        np.min(b.transform(camera, placed)[:, 2]) > 0
                    ),
                }
            )
    return result


def summarize(records):
    return {
        split: b.stats(
            [v for r in records if r["split"] == split for v in r["corner_errors_px"]]
        )
        for split in ("train", "test")
    }


def initial_data(args):
    from . import checkerboard_geometry as c

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = Path(args.baseline) if getattr(args, "baseline", None) else None
    if not (output / "detections.json").exists():
        source = baseline or Path(args.source)
        shutil.copyfile(source / "detections.json", output / "detections.json")
    cache = c.apply_image_quality(
        json.loads((output / "detections.json").read_text(encoding="utf-8")), output
    )
    source = baseline or Path(args.source)
    prior = json.loads((source / "fit.json").read_text(encoding="utf-8"))
    k = np.asarray(prior["camera_matrix"])
    d = np.asarray(prior["distortion"])
    board = b.object_board(tuple(cache["board"]["pattern_size"]), 35.0)
    frames = c.prepared(cache, k, d, board)
    train = [r for r in frames if r["split"] == "train"]
    if baseline:
        if prior.get("geometry_model", "square") != "square":
            raise ValueError("--baseline must be a square-model fit")
        poses = {int(mid): np.asarray(v) for mid, v in prior["marker_poses"].items()}
        lengths = {int(mid): v for mid, v in prior["lengths_mm"].items()}
        placements = {
            (int(key.split(":")[0]), key.split(":")[1]): np.asarray(v)
            for key, v in prior["group_poses"].items()
        }
        removed = prior["training_observations_removed"]
        excluded = {(r["image"], r["id"]) for r in removed}
        for r in train:
            r["camera"] = np.asarray(prior["training_cameras"][r["image"]])
            r["markers"] = {
                mid: p
                for mid, p in r["markers"].items()
                if (r["image"], mid) not in excluded
            }
    else:
        poses, lengths, placements, _ = b.init_geometry(frames, c.TOOLS, k, d, output)
        poses, placements = c.rebase(poses, placements)
        # Square geometry is only an initializer/control model. Never classify
        # a new-data observation as an outlier merely because a square cannot
        # explain a genuinely non-square engraving.
        square = b.JointProblem(
            train, c.TOOLS, {"C": 84}, poses, lengths, placements, board, k, d, False
        )
        (poses, lengths, placements, _), first = square.solve(120)
        square = b.JointProblem(
            train, c.TOOLS, {"C": 84}, poses, lengths, placements, board, k, d, True
        )
        (poses, lengths, placements, cameras), joint = square.solve(160)
        stages = {"fixed_cameras": first, "joint": joint}
        removed = []
        for r, matrix in zip(train, cameras):
            r["camera"] = matrix
        prior = {
            "marker_poses": {mid: m.tolist() for mid, m in poses.items()},
            "lengths_mm": lengths,
            "group_poses": {
                f"{g}:{tool}": m.tolist() for (g, tool), m in placements.items()
            },
            "training_cameras": {r["image"]: r["camera"].tolist() for r in train},
            "training_observations_removed": removed,
            "stages": stages,
        }
    prior["camera_matrix"] = k.tolist()
    prior["distortion"] = d.ravel().tolist()
    prior_dir = output / "square_baseline"
    prior_dir.mkdir(exist_ok=True)
    b.write_json(prior_dir / "fit.json", prior)
    if baseline and result_path(baseline, "marker_group_validation.json").exists():
        shutil.copyfile(
            result_path(baseline, "marker_group_validation.json"),
            prior_dir / "marker_group_validation.json",
        )
    for name in (
        "camera_intrinsics.yml",
        "intrinsics_report.json",
        "source_manifest.json",
    ):
        if (source / name).is_file() and (source / name).resolve() != (
            output / name
        ).resolve():
            shutil.copyfile(source / name, output / name)
    return cache, frames, train, k, d, board, poses, lengths, placements, removed


def fit(args):
    from . import checkerboard_geometry as c

    cv2.setNumThreads(1)
    output = Path(args.output)
    cache, frames, train, k, d, board, poses, lengths, placements, removed = (
        initial_data(args)
    )
    shapes = {mid: square_shape(length) for mid, length in lengths.items()}
    baseline_points = {
        mid: b.transform(poses[mid], b.square(lengths[mid])) for mid in poses
    }
    baseline_cameras = {r["image"]: r["camera"].copy() for r in train}
    baseline_errors = errors(
        frames, baseline_points, placements, k, d, baseline_cameras
    )
    problem = PlanarJointProblem(
        train, c.TOOLS, {"C": c.REFERENCE}, poses, shapes, placements, board, k, d, True
    )
    initial_residual = problem.evaluate(problem.x)[0]
    (poses, shapes, placements, cameras), stage = problem.solve(200)
    points = corners(poses, shapes)
    edge = {mid: edge_lengths(p).tolist() for mid, p in points.items()}
    fit_result = {
        "geometry_model": MODEL,
        "source": cache["source"],
        "tools": c.TOOLS,
        "references": {"C": 84},
        "marker_poses": {mid: m.tolist() for mid, m in poses.items()},
        "marker_shapes": {mid: s.tolist() for mid, s in shapes.items()},
        "marker_positions": {mid: p.tolist() for mid, p in points.items()},
        "marker_edge_lengths_mm": edge,
        "lengths_mm": {mid: float(np.mean(v)) for mid, v in edge.items()},
        "lengths_mm_semantics": "arithmetic mean of four edges; marker_positions is authoritative",
        "group_poses": {
            f"{g}:{tool}": m.tolist() for (g, tool), m in placements.items()
        },
        "training_cameras": {r["image"]: m.tolist() for r, m in zip(train, cameras)},
        "camera_matrix": k.tolist(),
        "distortion": d.ravel().tolist(),
        "training_observations_removed": removed,
        "training_mask_policy": (
            "frozen square-baseline mask for model comparison"
            if getattr(args, "baseline", None)
            else "image-qualified observations; no square-residual rejection"
        ),
        "stages": {"planar_joint_ba": stage},
        "split": {r["image"]: r["split"] for r in frames},
    }
    b.write_json(output / "fit.json", fit_result)
    camera_map = {r["image"]: m for r, m in zip(train, cameras)}
    planar_errors = errors(frames, points, placements, k, d, camera_map)
    # frames has the frozen training mask; test pixels and image quality unchanged.
    all_errors = errors(frames, points, placements, k, d, camera_map, True)
    test = [r for r in planar_errors if r["split"] == "test"]
    comparison = {
        "training_observations_identical": True,
        "test_observations_identical": True,
        "square": summarize(baseline_errors),
        "planar": summarize(planar_errors),
        "square_robust_cost": float(
            np.sum(0.7**2 * (np.sqrt(1 + (initial_residual / 0.7) ** 2) - 1))
        ),
        "planar_robust_cost": stage["cost"],
        "test_error_used_to_fit_parameters": False,
        "note": "Existing development holdout; no new blind-test acquisition.",
    }
    b.write_json(output / "model_comparison.json", comparison)
    b.write_json(
        output / "reprojection_observations.json", {"observations": planar_errors}
    )
    b.write_json(
        output / "marker_all_candidate_observations.json", {"observations": all_errors}
    )
    b.write_json(
        output / "marker_quality_summary.json",
        {
            "geometry_model": MODEL,
            "ids": c.IDS,
            "reference_marker_id": 84,
            "eligible_test_corner_errors_px": b.stats(
                [v for r in test for v in r["corner_errors_px"]]
            ),
            "all_candidates_test_corner_errors_px": b.stats(
                [
                    v
                    for r in all_errors
                    if r["split"] == "test"
                    for v in r["corner_errors_px"]
                ]
            ),
            "eligible_test_observations": len(test),
            "all_candidate_test_observations": sum(
                r["split"] == "test" for r in all_errors
            ),
            "eligible_test_per_id_px": {
                mid: b.stats(
                    [v for r in test if r["id"] == mid for v in r["corner_errors_px"]]
                )
                for mid in c.IDS
            },
            "maximum_eligible_test_observation_rmse_px": max(
                r["rmse_px"] for r in test
            ),
            "all_eligible_test_depths_positive": all(
                r["positive_depth"] for r in planar_errors
            ),
            "training_outliers_removed": removed,
            "marker_edge_lengths_mm": edge,
        },
    )
    print("MODEL_COMPARISON", json.dumps(comparison), flush=True)


def load(output):
    output = Path(output)
    f = json.loads((output / "fit.json").read_text(encoding="utf-8"))
    if f.get("geometry_model") != MODEL:
        raise ValueError("Expected planar quadrilateral fit")
    poses = {int(mid): np.asarray(m) for mid, m in f["marker_poses"].items()}
    shapes = {int(mid): np.asarray(s) for mid, s in f["marker_shapes"].items()}
    placements = {
        (int(key.split(":")[0]), key.split(":")[1]): np.asarray(m)
        for key, m in f["group_poses"].items()
    }
    return f, poses, shapes, placements


def fit_placement(frames, points, k, d):
    def residual(x):
        e = b.pose(x)
        return np.concatenate(
            [
                (
                    b.project(b.transform(e, points[mid]), r["camera"], k, d) - pixels
                ).ravel()
                for r in frames
                for mid, pixels in r["markers"].items()
            ]
        )

    first = max(frames, key=lambda r: len(r["markers"]))
    ids = sorted(first["markers"])
    initial = np.linalg.inv(first["camera"]) @ b.pnp(
        np.concatenate([points[mid] for mid in ids]),
        np.concatenate([first["markers"][mid] for mid in ids]),
        k,
        d,
    )
    solved = least_squares(
        residual,
        b.vector(initial),
        loss="soft_l1",
        f_scale=0.7,
        x_scale="jac",
        max_nfev=120,
    )
    return b.pose(solved.x), bool(solved.success)


def group_validation(args):
    from . import checkerboard_geometry as c

    cv2.setNumThreads(1)
    output = Path(args.output)
    final, poses, shapes, _ = load(output)
    final_points = corners(poses, shapes)
    cache = json.loads((output / "detections.json").read_text(encoding="utf-8"))
    k = np.asarray(final["camera_matrix"])
    d = np.asarray(final["distortion"])
    board = b.object_board((11, 8), 35.0)
    baseline_path = result_path(
        output / "square_baseline", "marker_group_validation.json"
    )
    baseline_folds = (
        json.loads(baseline_path.read_text())["folds"] if baseline_path.exists() else []
    )
    folds = []
    for group in range(7):
        directory = output / "group_validation" / f"without_group_{group}"
        directory.mkdir(parents=True, exist_ok=True)
        frames = c.prepared(cache, k, d, board)
        others = [r for r in frames if r["group"] != group]
        gp, lengths, placements, _ = b.init_geometry(others, c.TOOLS, k, d, directory)
        gp, placements = c.rebase(gp, placements)
        train = [r for r in others if r["split"] == "train"]
        old_fold = next(
            (f for f in baseline_folds if f["held_geometry_group"] == group), None
        )
        removed = (
            old_fold["training_outliers_removed"]
            if old_fold
            else final["training_observations_removed"]
        )
        excluded = {(r["image"], r["id"]) for r in removed}
        for r in train:
            r["markers"] = {
                mid: p
                for mid, p in r["markers"].items()
                if (r["image"], mid) not in excluded
            }
        gs = {mid: square_shape(v) for mid, v in lengths.items()}
        problem = PlanarJointProblem(
            train, c.TOOLS, {"C": 84}, gp, gs, placements, board, k, d, False
        )
        (gp, gs, placements, _), first = problem.solve(120)
        problem = PlanarJointProblem(
            train, c.TOOLS, {"C": 84}, gp, gs, placements, board, k, d, True
        )
        (gp, gs, placements, _), joint = problem.solve(180)
        points = corners(gp, gs)
        held_train = [
            r
            for r in frames
            if r["group"] == group and r["split"] == "train" and r["markers"]
        ]
        placement, success = fit_placement(held_train, points, k, d)
        test = [r for r in frames if r["group"] == group and r["split"] == "test"]
        records = errors(test, points, {(group, "C"): placement}, k, d)
        change = {
            mid: np.linalg.norm(points[mid] - final_points[mid], axis=1).tolist()
            for mid in c.IDS
        }
        fold = {
            "held_geometry_group": group,
            "test_corner_error_px": b.stats(
                [v for r in records for v in r["corner_errors_px"]]
            ),
            "maximum_test_observation_rmse_px": max(r["rmse_px"] for r in records),
            "corner_geometry_change_mm": b.stats(
                [v for values in change.values() for v in values]
            ),
            "per_id_corner_geometry_change_mm": change,
            "marker_edge_lengths_mm": {
                mid: edge_lengths(p).tolist() for mid, p in points.items()
            },
            "marker_geometry_metrics": {
                mid: geometry_metrics(q) for mid, q in points.items()
            },
            "stages": {"fixed_cameras": first, "joint": joint},
            "placement_fit_success": success,
            "baseline_training_mask_used": old_fold is not None,
        }
        b.write_json(directory / "result.json", fold)
        folds.append(fold)
        b.write_json(
            output / "marker_group_validation.json",
            {
                "geometry_model": MODEL,
                "folds": folds,
                "method": "Fixed intrinsics; exclude entire group from geometry; fit only held-group placement on its training frames; evaluate its held-out tail. No geometry alignment.",
            },
        )
        print(
            "PLANAR_GROUP",
            group,
            json.dumps(fold["test_corner_error_px"]),
            json.dumps(fold["corner_geometry_change_mm"]),
            flush=True,
        )


def export(args):
    from . import checkerboard_geometry as c
    from .marker_io import save_geometry

    output = Path(args.output)
    f, poses, shapes, placements = load(output)
    points = corners(poses, shapes)
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    summary = json.loads(result_path(output, "marker_quality_summary.json").read_text())
    groups = json.loads(result_path(output, "marker_group_validation.json").read_text())
    thresholds = config["calibration"]
    geometry_checks = {}
    for mid, q in points.items():
        local = quad(shapes[mid])
        edges = edge_lengths(q)
        geometry_checks[mid] = {
            **geometry_metrics(q),
            "coplanarity_error_mm": float(
                np.max(np.abs((q - q.mean(axis=0)) @ poses[mid][:3, 2]))
            ),
            "convex": bool(cv2.isContourConvex(local[:, :2].astype(np.float32))),
            "shape_bounds_inactive": bool(
                all(2.002 < v < 59.94 for v in np.exp(shapes[mid][[0, 2, 4]]))
            ),
        }
    records = json.loads((output / "reprojection_observations.json").read_text())[
        "observations"
    ]
    counts = {
        mid: sum(r["split"] == "train" and r["id"] == mid for r in records)
        for mid in c.IDS
    }
    cache = json.loads((output / "detections.json").read_text(encoding="utf-8"))
    frames = c.prepared(
        cache,
        np.asarray(f["camera_matrix"]),
        np.asarray(f["distortion"]),
        b.object_board((11, 8), 35.0),
    )
    excluded = {(r["image"], r["id"]) for r in f["training_observations_removed"]}
    diversity = {}
    for mid in c.IDS:
        directions = []
        for r in frames:
            if (
                r["split"] != "train"
                or mid not in r["markers"]
                or (r["image"], mid) in excluded
            ):
                continue
            matrix = (
                np.asarray(f["training_cameras"][r["image"]])
                @ placements[r["group"], "C"]
                @ poses[mid]
            )
            ray = -matrix[:3, :3].T @ matrix[:3, 3]
            directions.append(ray / np.linalg.norm(ray))
        rays = np.asarray(directions)
        diversity[mid] = (
            float(np.rad2deg(np.arccos(np.clip((rays @ rays.T).min(), -1, 1))))
            if len(rays)
            else 0.0
        )
    checks = {
        "all_nine_ids": sorted(points) == c.IDS,
        "all_quads_convex": all(q["convex"] for q in geometry_checks.values()),
        "all_quads_coplanar": all(
            q["coplanarity_error_mm"] < 1e-8 for q in geometry_checks.values()
        ),
        "shape_bounds_inactive": all(
            q["shape_bounds_inactive"] for q in geometry_checks.values()
        ),
        "solver_converged": all(s["success"] for s in f["stages"].values()),
        "positive_depth": all(r["positive_depth"] for r in records),
        "training_coverage": all(
            v >= thresholds["min_marker_views"] for v in counts.values()
        ),
        "minimum_training_views": len(f["training_cameras"]) >= thresholds["min_views"],
        "viewing_diversity": all(
            v >= thresholds["min_view_angle_deg"] for v in diversity.values()
        ),
        "training_reprojection": all(
            r["rmse_px"] <= thresholds["max_observation_rmse_px"]
            for r in records
            if r["split"] == "train"
        ),
        "validation_reprojection": summary["maximum_eligible_test_observation_rmse_px"]
        <= thresholds["max_validation_rmse_px"],
        "validation_marker_coverage": all(
            summary["eligible_test_per_id_px"][str(mid)]["count"] > 0 for mid in c.IDS
        ),
        "group_validation_complete": len(groups["folds"]) == 7,
        "group_validation_solvers": all(
            all(s["success"] for s in fold["stages"].values())
            and fold["placement_fit_success"]
            for fold in groups["folds"]
        ),
        "group_validation_reprojection": all(
            fold["maximum_test_observation_rmse_px"]
            <= thresholds["max_validation_rmse_px"]
            for fold in groups["folds"]
        ),
    }
    quality = {
        "passed": all(checks.values()),
        "checks": checks,
        "geometry_model": MODEL,
        "reference_marker_id": 84,
        "reference_frame": "Four-corner centroid of ID84; +X along decoded corner0->1; +Y toward decoded top; +Z outward",
        "marker_geometry": geometry_checks,
        "quality_summary": summary,
        "group_validation": groups,
        "view_angle_range_deg": diversity,
        "legacy_marker_lengths_semantics": "mean of four edges; use marker_positions for actual geometry",
        "absolute_accuracy_guaranteed": False,
    }
    b.write_json(output / "marker_calibration_quality.json", quality)
    if not quality["passed"]:
        raise ValueError(f"Planar quality gates failed: {checks}")
    config["marker"]["length_source"] = "checkerboard_35_mm"
    means = {mid: float(np.mean(edge_lengths(p))) for mid, p in points.items()}
    save_geometry(
        output / config["output"]["filename"],
        points,
        means,
        84,
        config,
        optimization_method="checkerboard_metric_planar_quadrilateral_joint_ba",
        reference_marker_length_fixed=False,
    )
    b.write_json(
        output / "marker_edge_lengths.json",
        {
            "units": "mm",
            "decoded_edge_order": ["0->1", "1->2", "2->3", "3->0"],
            "marker_edge_lengths": {
                mid: edge_lengths(q).tolist() for mid, q in points.items()
            },
            "legacy_marker_lengths": means,
            "legacy_semantics": "arithmetic mean only; use the four corners for geometry",
        },
    )
    angle_changes = {}
    for mid in points:
        final_angles = np.asarray(geometry_checks[mid]["interior_angles_deg"])
        fold_angles = [
            fold.get("marker_geometry_metrics", {})
            .get(str(mid), {})
            .get("interior_angles_deg")
            for fold in groups["folds"]
        ]
        if all(value is not None for value in fold_angles):
            values = np.asarray(fold_angles)
            angle_changes[mid] = {
                "fold_min_deg": values.min(axis=0).tolist(),
                "fold_max_deg": values.max(axis=0).tolist(),
                "max_abs_change_from_final_deg": np.max(
                    np.abs(values - final_angles), axis=0
                ).tolist(),
            }
    b.write_json(
        output / "marker_geometry_metrics.json",
        {
            "interior_angle_order": [
                "corner0: edge30 vs edge01",
                "corner1: edge01 vs edge12",
                "corner2: edge12 vs edge23",
                "corner3: edge23 vs edge30",
            ],
            "edge_direction_order": ["0->1", "1->2", "2->3", "3->0"],
            "coordinate_frame": "ID84 centroid; +X decoded 0->1; +Y toward top; +Z outward",
            "marker_geometry": geometry_checks,
            "marker_plane_angle_pairs": plane_angles(points),
            "leave_group_out_interior_angle_changes": angle_changes,
            "note": "Angles are derived from optimized coordinates, not constrained to 90 degrees. Planar convex angle sum 360 is structural, not independent accuracy evidence.",
        },
    )
    b.write_json(
        output / "marker_workflow.json",
        {
            "geometry_model": MODEL,
            "source": f["source"],
            "target_ids": c.IDS,
            "board": {"pattern_size": [11, 8], "square_size_mm": 35.0},
            "output": config["output"],
            "reference_marker_id": 84,
            "reference_marker_shape_fixed": False,
            "tip_offset_calibrated": False,
            "legacy_marker_lengths_semantics": "mean of four independently estimated edges; actual corners in marker_positions",
        },
    )
    print("PLANAR_EXPORTED", str(output / config["output"]["filename"]), flush=True)


def render_report(root):
    """Report both models without turning lower training error into a claim of accuracy."""
    import base64

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    f, poses, shapes, _ = load(root)
    comparison = json.loads((root / "model_comparison.json").read_text())
    quality = json.loads((root / "marker_calibration_quality.json").read_text())
    folds = quality["group_validation"]["folds"]
    baseline_path = result_path(
        root / "square_baseline", "marker_group_validation.json"
    )
    old = (
        json.loads(baseline_path.read_text())["folds"] if baseline_path.exists() else []
    )
    metrics = json.loads((root / "marker_geometry_metrics.json").read_text())
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    points = corners(poses, shapes)
    ids = sorted(points)
    distribution = plt.figure(figsize=(12, 5), layout="constrained")
    space = distribution.add_subplot(121, projection="3d")
    for mid, q in points.items():
        color = "#007f87" if mid == 84 else ("#4389c4" if mid <= 88 else "#d48a42")
        space.add_collection3d(
            Poly3DCollection([q], facecolors=color, edgecolors=color, alpha=0.35)
        )
        center = q.mean(axis=0)
        normal = np.asarray(geometry_metrics(q)["outward_normal"])
        space.quiver(*center, *(normal * 6.0), color=color, arrow_length_ratio=0.25)
        space.text(*center, str(mid), fontsize=9)
    cloud = np.concatenate(list(points.values()))
    center = (cloud.max(axis=0) + cloud.min(axis=0)) / 2
    radius = float(np.max(np.ptp(cloud, axis=0)) / 2 + 5)
    space.set(
        xlim=(center[0] - radius, center[0] + radius),
        ylim=(center[1] - radius, center[1] + radius),
        zlim=(center[2] - radius, center[2] + radius),
        xlabel="X (mm)",
        ylabel="Y (mm)",
        zlabel="Z (mm)",
        title="Estimated marker planes and outward normals",
    )
    space.set_box_aspect((1, 1, 1))
    space.view_init(elev=24, azim=35)
    heat = distribution.add_subplot(122)
    normals = np.array([geometry_metrics(points[mid])["outward_normal"] for mid in ids])
    angles = np.rad2deg(np.arccos(np.clip(normals @ normals.T, -1, 1)))
    bitmap = heat.imshow(angles, vmin=0, vmax=180, cmap="viridis")
    heat.set(
        xticks=range(9),
        xticklabels=ids,
        yticks=range(9),
        yticklabels=ids,
        title="Directed normal angles (degrees)",
    )
    for i in range(9):
        for j in range(9):
            heat.text(
                j,
                i,
                f"{angles[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if angles[i, j] < 100 else "black",
            )
    distribution.colorbar(bitmap, ax=heat, shrink=0.8)
    distribution.savefig(root / "marker_geometry_distribution.png", dpi=160)
    plt.close(distribution)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(2)
    axes[0].bar(
        x - 0.18,
        [comparison["square"][s]["rms"] for s in ("train", "test")],
        0.36,
        label="Square",
        color="#a4afb7",
    )
    axes[0].bar(
        x + 0.18,
        [comparison["planar"][s]["rms"] for s in ("train", "test")],
        0.36,
        label="Planar quad",
        color="#007f87",
    )
    axes[0].set(
        xticks=x,
        xticklabels=["Train", "Holdout"],
        ylabel="Marker corner RMS (px)",
        title="Identical observations and masks",
    )
    axes[0].legend()
    if old:
        axes[1].bar(
            np.arange(7) - 0.18,
            [r["corner_geometry_change_mm"]["max"] for r in old],
            0.36,
            label="Square",
            color="#a4afb7",
        )
    axes[1].bar(
        np.arange(7) + 0.18,
        [r["corner_geometry_change_mm"]["max"] for r in folds],
        0.36,
        label="Planar quad",
        color="#007f87",
    )
    axes[1].set(
        xlabel="Group excluded from geometry fit",
        ylabel="Max corner change (mm)",
        title="Group sensitivity in the ID84 frame",
    )
    axes[1].legend()
    fig.savefig(root / "planar_model_comparison.png", dpi=160)
    plt.close(fig)
    edge_rows = []
    for mid in sorted(poses):
        edges = edge_lengths(b.transform(poses[mid], quad(shapes[mid])))
        edge_rows.append(
            "<tr><td>"
            + str(mid)
            + "</td>"
            + "".join(f"<td>{v:.6f}</td>" for v in edges)
            + f"<td>{np.ptp(edges):.6f}</td></tr>"
        )
    cv_rows = "".join(
        f"<tr><td>{r['held_geometry_group']}</td><td>{r['test_corner_error_px']['rms']:.6f}</td><td>{r['corner_geometry_change_mm']['max']:.6f}</td></tr>"
        for r in folds
    )
    angle_rows = "".join(
        "<tr><td>"
        + str(mid)
        + "</td>"
        + "".join(
            f"<td>{v:.3f}</td>"
            for v in metrics["marker_geometry"][str(mid)]["interior_angles_deg"]
        )
        + "</tr>"
        for mid in sorted(poses)
    )
    chosen = {
        (85, 86),
        (86, 87),
        (87, 88),
        (85, 88),
        (89, 90),
        (90, 91),
        (91, 92),
        (89, 92),
        (85, 87),
        (86, 88),
        (89, 91),
        (90, 92),
    }
    plane_rows = "".join(
        f"<tr><td>{r['ids'][0]} / {r['ids'][1]}</td><td>{r['directed_normal_angle_deg']:.3f}</td><td>{r['unoriented_plane_angle_deg']:.3f}</td></tr>"
        for r in metrics["marker_plane_angle_pairs"]
        if r["ids"][0] == 84 or tuple(r["ids"]) in chosen
    )
    angle_values = np.array(
        [
            metrics["marker_geometry"][str(mid)]["interior_angles_deg"]
            for mid in sorted(poses)
        ]
    )
    changes = metrics["leave_group_out_interior_angle_changes"]
    max_angle_change = max(
        (v for r in changes.values() for v in r["max_abs_change_from_final_deg"]),
        default=0.0,
    )

    def metric(model, split):
        return comparison[model][split]["rms"]

    image = (
        "data:image/png;base64,"
        + base64.b64encode((root / "planar_model_comparison.png").read_bytes()).decode()
    )
    distribution_image = (
        "data:image/png;base64,"
        + base64.b64encode(
            (root / "marker_geometry_distribution.png").read_bytes()
        ).decode()
    )
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C 平面四边形几何标定</title>
<style>body{{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;background:#f0f5f6;color:#203540;margin:0}}main{{max-width:1050px;margin:auto;background:white;padding:32px}}h2{{color:#08757c}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:8px;border-bottom:1px solid #dce5e8;text-align:left}}th{{background:#edf4f5}}pre{{white-space:pre-wrap;background:#f1f5f6;padding:15px}}img{{width:100%;height:auto}}.note{{background:#fff7e5;padding:16px;border-left:4px solid #bd8a34}}a{{color:#08757c}}</style><main>
<h1>C：独立平面四边形 BA</h1><p>ID84–92；35 mm 棋盘尺度；参考码84。<a href="double_cube_geometry.yaml">工程格式几何 YAML</a> · <a href="marker_calibration_quality.json">质量检查</a> · <a href="model_comparison.json">完整模型对照</a></p>
<p>四角严格共面，各边长、对角线和夹角自由变化，不强制等边、直角或对边平行。参考码84同样有5个自由形状参数；仅固定其中心、码面法向和解码上边方向来定义坐标系。全部共93个几何自由度，加七组放置位姿和251个相机位姿，总计1641个变量。</p>
<h2>为什么有机会降低误差</h2><p>正方形是平面四边形模型中的一个特例，因此新增形状自由度能表达原模型不能表达的真实刻印偏差。优化仍最小化棋盘与码角点的像素残差，使用0.7 px尺度的soft-L1损失；更低的训练目标并不保证更低的留出误差。</p>
<h2>同一批观测的公平比较</h2><p>两种模型使用相同内参、同一训练/验证划分、完全相同的观测掩码。此次固定了正方形基线已移除的一个训练异常观测，未按新模型的验证误差筛选图像。66张留出图是开发验证，曾在此前调试中查看，不是新的盲测。</p>
<table><tr><th>指标</th><th>正方形</th><th>平面四边形</th></tr><tr><td>训练 marker 角点 RMS px</td><td>{metric("square", "train"):.6f}</td><td>{metric("planar", "train"):.6f}</td></tr><tr><td>留出 marker 角点 RMS px</td><td>{metric("square", "test"):.6f}</td><td>{metric("planar", "test"):.6f}</td></tr><tr><td>训练鲁棒目标（含棋盘）</td><td>{comparison["square_robust_cost"]:.6f}</td><td>{comparison["planar_robust_cost"]:.6f}</td></tr></table>
<div class="note">本批数据训练误差略降，留出 RMS 略升。新增模型已实现，但这些结果不支持宣称它比正方形模型更准确。估计出来的微小边长差异也可能包含角点噪声和模型误差。两种结果均保留，没有覆盖先前标定。</div>
<img src="{image}" alt="同观测模型误差及整组留出稳定性对比">
<h2>每码四条边长，单位 mm</h2><p>边顺序为解码角点 0→1、1→2、2→3、3→0。完整四角坐标在 YAML 的 marker_positions。为兼容旧格式，marker_lengths 和 marker_length 记录四边算术平均；它们不能替代实际四边形几何。</p>
<table><tr><th>ID</th><th>边01</th><th>边12</th><th>边23</th><th>边30</th><th>最大减最小</th></tr>{"".join(edge_rows)}</table>
<h2>四条边之间的夹角与整体空间分布</h2><p>角度在四个角点定义：θ0是边30与01的内角，θ1是边01与12的内角，θ2是边12与23的内角，θ3是边23与30的内角。当前范围为{angle_values.min():.3f}°～{angle_values.max():.3f}°。这些内角由坐标计算，未施加90°约束。</p>
<table><tr><th>ID</th><th>θ0 (°)</th><th>θ1 (°)</th><th>θ2 (°)</th><th>θ3 (°)</th></tr>{angle_rows}</table>
<p>七轮整组留出中，内角相对最终结果的最大变化为{max_angle_change:.3f}°。几百分之一度的偏离小于此处的最大跨组变化，不能直接当作真实制造偏差。凸平面四边形内角和360°是参数化的结构性质，并非额外的精度证明。</p>
<img src="{distribution_image}" alt="优化后的九个marker平面、外法向和码面法向夹角矩阵">
<p>左图只画实际优化的九个码面，没有拟合或强加立方体棱线。箭头为外法向，三轴采用相同长度比例。右图为有向法向夹角。下表另外给出无向平面锐角：相对面的法向可接近180°，而两个平面本身的无向夹角接近0°。</p>
<table><tr><th>码对</th><th>有向法向夹角 (°)</th><th>无向平面夹角 (°)</th></tr>{plane_rows}</table>
<p><a href="marker_geometry_metrics.json">完整角度与方向 JSON</a> 包括每条边在C坐标系中的单位方向向量、与三根正坐标轴的方向角、全部36对码面夹角以及内角的跨组变化范围。三维边的方向不能由单个角度唯一描述。</p>
<h2>七轮整组留出</h2><p>相机内参固定，整组 marker 观测不用于几何拟合；仅用该组训练部分估计其器械放置位姿，再预测该组留出部分。三维变化直接在定义好的参考码坐标系中比较，没有缩放或额外对齐。</p><table><tr><th>移除组</th><th>该组留出 RMS px</th><th>最大角点变化 mm</th></tr>{cv_rows}</table>
<p>全部项目数值门禁通过：{quality["passed"]}。严格共面由参数化保证，凸性、正深度、形状边界、观测覆盖、误差和分组求解均另行检查。这些检查不等于外部计量精度认证。</p>
<h2>复现</h2><pre>python -m python.scripts.marker_calibration.checkerboard_geometry fit --geometry-model planar --baseline output/20261116_marker_C_geometry --output output/marker_geometry
python -m python.scripts.marker_calibration.checkerboard_geometry groups --output output/marker_geometry
python -m python.scripts.marker_calibration.checkerboard_geometry export --output output/marker_geometry
python -m python.scripts.marker_calibration.report --output output/marker_geometry
</pre>
<p>新数据的默认 planar 模式不会仅因正方形初始模型误差较大而删除观测；显式提供 --baseline 时采用基线掩码用于模型对照。--geometry-model square 可重现旧的正方形流程。tip_offset仍沿用工程配置，未估计针尖。</p></main></html>'''
    (root / "marker_calibration_report.html").write_text(page, encoding="utf-8")
    (root / "README.md").write_text(
        """# C 平面四边形几何标定

最终几何：double_cube_geometry.yaml；图文报告：marker_calibration_report.html。
每个码严格共面，四条边和夹角自由；参考码84的形状也自由。
marker_positions为实际几何。兼容字段marker_lengths仅表示四边算术平均，四条边的值在marker_calibration_quality.json中。

本批数据训练RMS从0.237006降至0.235503 px，留出RMS从0.345356变为0.346521 px，未体现更好的泛化。新结果不覆盖原正方形结果。
""",
        encoding="utf-8",
    )
