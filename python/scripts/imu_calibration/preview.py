#!/usr/bin/env python3
"""Live planar-marker overlay, served as MJPEG; never modifies calibration.

Uses actual marker_positions with generic PnP, not IPPE_SQUARE or mean edge
lengths. With >=3 markers one complete marker is held out of pose estimation.
"""

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import threading
import time
from collections import deque
from concurrent.futures import Future, TimeoutError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
import yaml

from ...common.GalaxyCamera import GalaxyCamera
from ..marker_calibration.checkerboard import black_border_corners
from ..sensor_capture.galaxy_controls import (
    CAMERA_SETTINGS,
    apply_camera_setting,
    camera_settings,
)
from .attitude import ImuState, project_imu_axes


def detect(gray, detector, ids, maximum=1600):
    factor = min(1.0, maximum / max(gray.shape))
    small = (
        cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        if factor < 1
        else gray
    )
    polygons, decoded, _ = detector.detectMarkers(small)
    result = {}
    rejected = {}
    if decoded is None:
        return result, rejected
    scale = np.array([small.shape[1] / gray.shape[1], small.shape[0] / gray.shape[0]])
    duplicate = {
        int(mid) for mid in decoded.ravel() if np.count_nonzero(decoded == mid) > 1
    }
    for mid, points in zip(decoded.ravel(), polygons):
        mid = int(mid)
        if mid not in ids:
            continue
        if mid in duplicate:
            rejected[mid] = "duplicate"
            continue
        p = (points.reshape(4, 2) + 0.5) / scale - 0.5
        low = np.maximum(0, np.floor(p.min(axis=0) - 20).astype(int))
        high = np.minimum(
            [gray.shape[1], gray.shape[0]], np.ceil(p.max(axis=0) + 20).astype(int)
        )
        x, y = low
        xx, yy = high
        smooth = cv2.GaussianBlur(gray[y:yy, x:xx], (3, 3), 0.65)
        refined, q = black_border_corners(smooth, p - low, return_quality=True)
        refined += low
        longest = np.linalg.norm(refined - np.roll(refined, 1, axis=0), axis=1).max()
        ratio = abs(cv2.contourArea(refined.astype(np.float32))) / max(longest**2, 1)
        if not q["valid"] or ratio < 0.25:
            rejected[mid] = q["reason"] if not q["valid"] else "grazing angle"
            continue
        result[mid] = refined
    return result, rejected


def estimate_pose(points, measured, k, d):
    ids = sorted(measured)
    objects = np.concatenate([points[mid] for mid in ids]).astype(np.float64)
    pixels = np.concatenate([measured[mid] for mid in ids]).astype(np.float64)
    if len(ids) == 1:
        ok, rs, ts, _ = cv2.solvePnPGeneric(
            objects, pixels, k, d, flags=cv2.SOLVEPNP_IPPE
        )
        options = []
        if ok:
            for r, t in zip(rs, ts):
                transformed = objects @ cv2.Rodrigues(r)[0].T + t.reshape(3)
                if transformed[:, 2].min() <= 0:
                    continue
                prediction = cv2.projectPoints(objects, r, t, k, d)[0].reshape(-1, 2)
                options.append((np.mean((prediction - pixels) ** 2), r, t))
        if not options:
            return None
        _, r, t = min(options, key=lambda v: v[0])
        return r, t
    ok, r, t, inliers = cv2.solvePnPRansac(
        objects,
        pixels,
        k,
        d,
        iterationsCount=120,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 6:
        # Two small adjacent faces can make EPNP's minimal samples unstable.
        # Initialize from each planar face's IPPE alternatives, then fit ALL
        # observed corners. Never fall back to a nominal square geometry.
        candidates = []
        for mid in ids:
            success, rs, ts, _ = cv2.solvePnPGeneric(
                np.asarray(points[mid], np.float64),
                np.asarray(measured[mid], np.float64),
                k,
                d,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if not success:
                continue
            for seed_r, seed_t in zip(rs, ts):
                good, rr, tt = cv2.solvePnP(
                    objects,
                    pixels,
                    k,
                    d,
                    seed_r.copy(),
                    seed_t.copy(),
                    True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if (
                    not good
                    or (objects @ cv2.Rodrigues(rr)[0].T + tt.reshape(3))[:, 2].min()
                    <= 0
                ):
                    continue
                delta = np.linalg.norm(
                    cv2.projectPoints(objects, rr, tt, k, d)[0].reshape(-1, 2) - pixels,
                    axis=1,
                )
                error = float(np.sqrt(np.mean(delta**2)))
                if error <= 3.0:
                    candidates.append((error, rr, tt))
        if not candidates:
            return None
        _, r, t = min(candidates, key=lambda c: c[0])
        return r, t
    select = inliers.ravel()
    ok, r, t = cv2.solvePnP(
        objects[select], pixels[select], k, d, r, t, True, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok or np.min(objects @ cv2.Rodrigues(r)[0].T + t.reshape(3), axis=0)[2] <= 0:
        return None
    return r, t


def overlay(
    gray,
    observed,
    rejected,
    points,
    k,
    d,
    sequence,
    age_ms,
    fps,
    imu=None,
):
    ids = sorted(observed)
    held = None
    if len(ids) >= 3:
        held = ids[int(time.monotonic() // 3) % len(ids)]
    fitting = {mid: p for mid, p in observed.items() if mid != held}
    solved = estimate_pose(points, fitting, k, d) if fitting else None
    factor = min(1.0, 1280 / max(gray.shape))
    canvas = cv2.cvtColor(
        cv2.resize(gray, None, fx=factor, fy=factor), cv2.COLOR_GRAY2BGR
    )
    per_marker = {}
    fitting_errors = []
    held_errors = []
    imu_pixels = None

    def xy(p):
        return tuple(np.rint(np.asarray(p) * factor).astype(int))

    for mid, p in observed.items():
        cv2.polylines(
            canvas,
            [np.rint(p * factor).astype(np.int32)],
            True,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(mid) + (" HOLD" if mid == held else ""),
            xy(p.mean(axis=0)),
            0,
            0.5,
            (0, 220, 255) if mid == held else (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        for corner in p:
            cv2.circle(canvas, xy(corner), 2, (0, 255, 0), 1, cv2.LINE_AA)
    if solved:
        r, t = solved
        for mid, p in observed.items():
            predicted = cv2.projectPoints(points[mid], r, t, k, d)[0].reshape(-1, 2)
            delta = np.linalg.norm(predicted - p, axis=1)
            per_marker[mid] = float(np.sqrt(np.mean(delta**2)))
            (held_errors if mid == held else fitting_errors).extend(delta.tolist())
            cv2.polylines(
                canvas,
                [np.rint(predicted * factor).astype(np.int32)],
                True,
                (30, 70, 255),
                1,
                cv2.LINE_AA,
            )
            for a, b in zip(p, predicted):
                cv2.drawMarker(
                    canvas, xy(b), (0, 0, 255), cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA
                )
                cv2.line(canvas, xy(a), xy(b), (0, 255, 255), 1, cv2.LINE_AA)
        # Draw all geometry in the zoom view later; axes are at ID84 center.
        cv2.drawFrameAxes(
            canvas, k * factor + np.diag([0.0, 0.0, 1.0 - factor]), d, r, t, 20.0, 2
        )
        imu_pixels = project_imu_axes(imu, t, k, d)
        translation = t.ravel().tolist()
        rotation = r.ravel().tolist()
    else:
        translation = rotation = None
    rms = lambda v: float(np.sqrt(np.mean(np.asarray(v) ** 2))) if v else None
    status = {
        "sequence": sequence,
        "visible_ids": ids,
        "rejected": rejected,
        "pose_found": bool(solved),
        "imu_axes_at_marker_origin": imu_pixels is not None,
        "full_projection_scale": factor,
        "held_out_id": held,
        "fit_rmse_px": rms(fitting_errors),
        "heldout_rmse_px": rms(held_errors),
        "per_marker_rmse_px": per_marker,
        "translation_mm": translation,
        "rotation_rodrigues": rotation,
        "capture_to_processing_ms": age_ms,
        "processing_fps": fps,
        "pose_note": "single planar marker: ambiguous"
        if len(ids) == 1
        else "multi-marker",
        "measurement_note": "Live held-marker error is independent of pose fitting; fit RMS is not independent validation.",
    }
    text = f"Marker geometry | IDs {ids} | {fps:.1f} FPS | frame {sequence}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 80), (20, 20, 20), -1)
    cv2.putText(canvas, text, (12, 25), 0, 0.65, (235, 235, 235), 1, cv2.LINE_AA)
    fit_text = (
        "--" if status["fit_rmse_px"] is None else f"{status['fit_rmse_px']:.3f} px"
    )
    held_text = (
        "need 3 markers"
        if status["heldout_rmse_px"] is None
        else f"ID{held}: {status['heldout_rmse_px']:.3f} px"
    )
    cv2.putText(
        canvas,
        f"Fit RMS {fit_text} | Held-out {held_text} | axes X red Y green Z blue",
        (12, 53),
        0,
        0.55,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Green: measured corners   Red: geometry projection   Yellow: residual",
        (12, 74),
        0,
        0.45,
        (190, 210, 210),
        1,
        cv2.LINE_AA,
    )
    if imu_pixels is not None:
        label = (
            "IMU dashed 35mm | "
            + imu["source"]
            + " | "
            + imu["display_mode"]
            + " | NOT extrinsic calibrated"
        )
        cv2.rectangle(canvas, (0, 82), (canvas.shape[1], 108), (20, 20, 20), -1)
        cv2.putText(canvas, label, (12, 101), 0, 0.48, (100, 230, 255), 1, cv2.LINE_AA)
    if observed:
        all_points = np.concatenate(
            list(observed.values()) + ([] if imu_pixels is None else [imu_pixels])
        )
        lo = np.maximum(0, np.floor(all_points.min(axis=0) - 45).astype(int))
        hi = np.minimum(
            [gray.shape[1], gray.shape[0]],
            np.ceil(all_points.max(axis=0) + 45).astype(int),
        )
        x, y = lo
        xx, yy = hi
        zoom = cv2.cvtColor(gray[y:yy, x:xx], cv2.COLOR_GRAY2BGR)
        f = min(3.0, 1000 / max(zoom.shape[:2]))
        zoom = cv2.resize(zoom, None, fx=f, fy=f)
        for mid, p in observed.items():
            for a in p:
                cv2.circle(
                    zoom,
                    tuple(np.rint((a - lo) * f).astype(int)),
                    3,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                zoom,
                str(mid) + (" HOLD" if mid == held else ""),
                tuple(np.rint((p.mean(axis=0) - lo) * f).astype(int)),
                0,
                0.5,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
            if solved:
                pred = cv2.projectPoints(points[mid], *solved, k, d)[0].reshape(-1, 2)
                for a, b in zip(p, pred):
                    aa = tuple(np.rint((a - lo) * f).astype(int))
                    bb = tuple(np.rint((b - lo) * f).astype(int))
                    cv2.drawMarker(
                        zoom, bb, (0, 0, 255), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA
                    )
                    cv2.line(zoom, aa, bb, (0, 255, 255), 1, cv2.LINE_AA)
        if solved:
            zoom_k = k.copy()
            zoom_k[:2, :] *= f
            zoom_k[0, 2] -= lo[0] * f
            zoom_k[1, 2] -= lo[1] * f
            cv2.drawFrameAxes(zoom, zoom_k, d, *solved, 20.0, 2)
        status["zoom_projection"] = {
            "offset_px": lo.tolist(),
            "scale": f,
            "width": zoom.shape[1],
            "height": zoom.shape[0],
        }
    else:
        zoom = np.full((220, 700, 3), 25, np.uint8)
        cv2.putText(
            zoom,
            "No calibrated markers detected",
            (20, 100),
            0,
            0.8,
            (220, 220, 220),
            1,
        )
    return canvas, zoom, status


PAGE = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
IMAGE_ZOOM = (Path(__file__).resolve().parents[1] / "sensor_capture/image_zoom.js").read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--intrinsics")
    parser.add_argument("--serial", default="MAF26040019")
    parser.add_argument("--camera", choices=("galaxy", "vcamera"), default="galaxy")
    parser.add_argument("--imu-source", choices=("icm42670", "rion_imu406"), default="icm42670")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--exposure-us", type=float, default=None)
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Serve the page without acquiring the camera until Start is pressed",
    )
    args = parser.parse_args()
    cv2.setNumThreads(2)
    geometry = yaml.safe_load(Path(args.geometry).read_text(encoding="utf-8"))
    points = {
        int(mid): np.array(q, np.float64)
        for mid, q in geometry["marker_positions"].items()
    }
    if args.camera == "galaxy":
        if args.intrinsics is None:
            parser.error("Galaxy requires --intrinsics")
        storage = cv2.FileStorage(args.intrinsics, cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            parser.error(f"Cannot open camera intrinsics: {args.intrinsics}")
        galaxy_calibration = (
            storage.getNode("camera_matrix").mat(),
            storage.getNode("distortion_coefficients").mat(),
            [int(storage.getNode(name).real()) for name in ("image_width", "image_height")],
        )
        storage.release()
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    params.minMarkerPerimeterRate = 0.008
    params.adaptiveThreshWinSizeMax = 43
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100), params
    )
    imu_state = ImuState(args.imu_source)
    settings_requests = Queue()
    setting_pending = threading.Event()
    shared = {
        "state": {"state": "opening_camera"},
        "full": None,
        "zoom": None,
        "frame": None,
        "capture_sequence": 0,
        "camera_open": False,
        "camera_busy": False,
    }
    lock = threading.Lock()
    camera_logs = deque(maxlen=80)

    def camera_log(message):
        with lock:
            camera_logs.append({"time": time.time(), "message": message})
        print(message, flush=True)

    ready = threading.Condition(lock)
    stop = threading.Event()
    active = threading.Event()
    if not args.start_paused:
        active.set()

    def publish_paused():
        placeholder = np.full((300, 800, 3), 25, np.uint8)
        cv2.putText(
            placeholder,
            "Preview paused - camera released",
            (20, 110),
            0,
            0.8,
            (220, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            placeholder,
            "Press Start in the page to resume",
            (20, 160),
            0,
            0.7,
            (170, 200, 200),
            1,
            cv2.LINE_AA,
        )
        jpeg = cv2.imencode(".jpg", placeholder)[1].tobytes()
        with ready:
            if active.is_set() or shared["camera_busy"]:
                return
            shared.update(
                frame=None,
                full=jpeg,
                zoom=jpeg,
                state={"state": "paused", "pose_found": False},
                updated=time.monotonic(),
            )
            ready.notify_all()

    if args.start_paused:
        publish_paused()
    calibration_sha = hashlib.sha256(Path(args.geometry).read_bytes()).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/imu/stream":
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                previous = None
                last_send = 0
                try:
                    while not stop.is_set():
                        sample = imu_state.snapshot()
                        with lock:
                            visual = dict(shared["state"])
                            visual_age = time.monotonic() - shared.get("updated", 0)
                        sample["visual_anchor"] = (
                            dict(
                                translation_mm=visual["translation_mm"],
                                camera_matrix=visual["camera_matrix"],
                                distortion_coefficients=visual["distortion_coefficients"],
                                full_projection_scale=visual["full_projection_scale"],
                                zoom_projection=visual.get("zoom_projection"),
                            )
                            if active.is_set()
                            and visual.get("pose_found")
                            and visual_age < 0.3
                            else None
                        )
                        key = (
                            sample["received_samples"],
                            sample["display_mode"],
                            sample["source"],
                            sample["connected"],
                        )
                        now = time.monotonic()
                        if key != previous or now - last_send > 0.3:
                            self.wfile.write(
                                b"data: "
                                + json.dumps(sample, separators=(",", ":")).encode()
                                + b"\n\n"
                            )
                            self.wfile.flush()
                            previous = key
                            last_send = now
                        stop.wait(0.005)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(PAGE.encode())
                return
            if self.path == "/image_zoom.js":
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(IMAGE_ZOOM)
                return
            if self.path == "/status":
                with lock:
                    state = dict(shared["state"])
                    updated = shared.get("updated", time.monotonic())
                    state["camera_open"] = shared["camera_open"]
                    state["camera_active"] = active.is_set() or shared["camera_busy"] or shared["camera_open"]
                    state["settings_pending"] = setting_pending.is_set()
                    state["camera_logs"] = list(camera_logs)
                    state["acquisition_requested"] = active.is_set()
                    state["camera_settings"] = (
                        shared.get("camera_settings", {})
                        if shared["camera_open"]
                        else {}
                    )
                state["preview_age_ms"] = round((time.monotonic() - updated) * 1000)
                state["calibration_sha256"] = calibration_sha
                state["camera_type"] = args.camera
                if args.camera == "galaxy":
                    state["serial"] = args.serial
                state["imu"] = imu_state.snapshot()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(json.dumps(state).encode())
                return
            if self.path == "/latest.jpg":
                with lock:
                    frame = shared["full"]
                if frame is None:
                    self.send_error(503)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(frame)
                return
            if self.path not in ("/stream", "/zoom"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            key = "zoom" if self.path == "/zoom" else "full"
            previous = None
            try:
                while not stop.is_set():
                    with ready:
                        ready.wait_for(
                            lambda: shared[key] is not previous or stop.is_set(),
                            timeout=2,
                        )
                        frame = shared[key]
                    if frame is None or frame is previous:
                        continue
                    previous = frame
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode()
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if self.path in ("/start", "/stop", "/imu/reset", "/imu/align") and setting_pending.is_set():
                self.send_response(409)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "正在等待相机参数回读"}).encode())
                return
            if self.path == "/camera/settings":
                result = Future()
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 512:
                        raise ValueError("Invalid settings request")
                    request = json.loads(self.rfile.read(length))
                    key, value = request["setting"], float(request["value"])
                    if key not in CAMERA_SETTINGS or not math.isfinite(value):
                        raise ValueError("Invalid camera setting or value")
                    with lock:
                        if setting_pending.is_set():
                            raise ValueError("正在等待上一项相机参数回读")
                        if not active.is_set() or not shared["camera_open"]:
                            raise ValueError("Camera is disconnected")
                        setting_pending.set()
                        settings_requests.put((key, value, result))
                    response = {"camera_settings": result.result(timeout=20)}
                    status = 200
                except TimeoutError:
                    if result.cancel():
                        setting_pending.clear()
                    response, status = {"error": "Camera did not respond in time"}, 504
                except Exception as error:
                    response, status = {"error": str(error)}, 400
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                return
            if self.path == "/imu/ingest":
                if self.headers.get("Transfer-Encoding", "").lower() != "chunked":
                    self.send_error(400)
                    return
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.connection.settimeout(5.0)
                try:
                    while not stop.is_set():
                        line = self.rfile.readline(64)
                        if not line:
                            break
                        length = int(line.strip(), 16)
                        if length == 0:
                            break
                        if not 0 < length <= 4096:
                            raise ValueError("Invalid IMU stream chunk")
                        data = self.rfile.read(length)
                        if self.rfile.read(2) != b"\r\n":
                            raise ValueError("Invalid chunk terminator")
                        imu_state.feed(json.loads(data))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    print("IMU_STREAM_CLOSED", str(error), flush=True)
                self.close_connection = True
                return
            if self.path in ("/imu/reset", "/imu/align"):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 <= length <= 4096:
                        raise ValueError("Packet too large")
                    packet = json.loads(self.rfile.read(length) or b"{}")
                    if self.path == "/imu/reset":
                        imu_state.reset(packet.get("source"))
                    else:
                        with lock:
                            visual = dict(shared["state"])
                            updated = shared.get("updated", 0)
                        if (
                            not active.is_set()
                            or not visual.get("pose_found")
                            or time.monotonic() - updated > 0.3
                        ):
                            raise ValueError("Need a current visible marker pose")
                        if len(visual.get("visible_ids", [])) < 2:
                            raise ValueError(
                                "Need at least two visible markers for alignment"
                            )
                        imu_state.align(
                            cv2.Rodrigues(np.array(visual["rotation_rodrigues"]))[0]
                        )
                    result = imu_state.snapshot()
                    self.send_response(200)
                except (ValueError, KeyError, TypeError) as error:
                    result = {"error": str(error)}
                    self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                return
            if self.path not in ("/start", "/stop"):
                self.send_error(404)
                return
            if self.path == "/start":
                with lock:
                    if not active.is_set():
                        shared["state"] = {
                            "state": "opening_camera",
                            "pose_found": False,
                        }
                        active.set()
                camera_log("收到连接相机请求")
            else:
                with lock:
                    active.clear()
                    shared["frame"] = None
                    shared["state"] = {"state": "pausing", "pose_found": False}
                    busy = shared["camera_busy"]
                if not busy:
                    publish_paused()
                camera_log("收到断开相机请求")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "acquisition_requested": active.is_set(),
                        "web_service_running": True,
                    }
                ).encode()
            )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def shutdown(*reason):
        print("PROCESS_SHUTDOWN", reason, flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    def capture():
        camera = None
        while not stop.is_set():
            with lock:
                should_open = active.is_set()
                shared["camera_busy"] = should_open
            if not should_open:
                stop.wait(0.1)
                continue
            started = time.monotonic()
            stage = ""
            stage_started = started
            first_frame = False

            def next_stage(message):
                nonlocal stage, stage_started
                now = time.monotonic()
                if stage:
                    camera_log(f"{stage}完成（{now - stage_started:.3f} 秒）")
                stage, stage_started = message, now
                if message:
                    camera_log(f"正在{message}…")

            try:
                next_stage(f"加载并初始化 {args.camera} SDK")
                if args.camera == "vcamera":
                    from ...common.VcameraCamera import VcameraCamera

                    camera = VcameraCamera(
                        stream="color",
                        exposure_us=args.exposure_us,
                        exposure_mode="preserve" if args.exposure_us is None else "manual",
                    )
                    camera.open()
                    factory = camera.metadata().get("sdk_intrinsics")
                    if not factory:
                        raise ValueError("Vcamera SDK 未提供当前模式的出厂内参")
                    k = np.asarray(factory["camera_matrix"], np.float64)
                    d = np.asarray(factory["distortion_coefficients"], np.float64)
                    size = factory["image_size"]
                    camera_log(f"已读取 Vcamera SDK 出厂内参：{size[0]} × {size[1]}。")
                    read_frame = camera.read

                    def read_settings():
                        return {}

                    def set_setting(_key, _value):
                        raise ValueError("当前 Vcamera 预览不支持实时修改参数")
                elif os.name == "nt":
                    from ..sensor_capture.galaxy import GxipyGalaxySource

                    k, d, size = galaxy_calibration
                    camera = GxipyGalaxySource(
                        sensor_index=1,
                        width=size[0],
                        height=size[1],
                        offset_x=0,
                        offset_y=0,
                        exposure_us=args.exposure_us,
                        gain_db=args.gain,
                        acquisition_fps=0,
                        gxipy_path=None,
                        serial=args.serial,
                    )
                    next_stage(f"Windows Galaxy SDK (gxipy): {args.serial}")
                    camera.open()
                    read_frame = camera.read
                    read_settings = camera.live_settings
                    set_setting = camera.set_live_setting
                else:
                    k, d, size = galaxy_calibration
                    camera = GalaxyCamera()
                    camera.init()
                    next_stage(f"搜索并打开相机 {args.serial}")
                    camera.open_by_sn(args.serial)
                    if not active.is_set():
                        continue
                    next_stage("配置采集参数")
                    camera.set_enum("TriggerMode", "Off")
                    camera.set_enum("AcquisitionMode", "Continuous")
                    if [camera.get_int("Width"), camera.get_int("Height")] != size:
                        camera.set_roi(size[0], size[1], offset_x=0, offset_y=0)
                    if camera.get_int("OffsetX") != 0 or camera.get_int("OffsetY") != 0:
                        raise ValueError(
                            "ROI offset differs from full-sensor calibration"
                        )
                    if camera.feature_is_writable("PixelFormat"):
                        camera.set_enum("PixelFormat", "Mono8")
                    if args.exposure_us is not None:
                        camera.set_exposure_time_us(args.exposure_us)
                    if args.gain is not None:
                        camera.set_gain(args.gain)
                    camera.set_buffer_number(3)
                    try:
                        camera.set_stream_buffer_mode("newest_only")
                    except Exception:
                        pass
                    next_stage("启动采集流并读取参数")
                    camera.stream_on()

                    def read_frame(timeout_ms):
                        frame = camera.get_latest_frame(timeout_ms=timeout_ms)
                        return frame.to_numpy() if frame is not None else None

                    def read_settings():
                        return camera_settings(camera)

                    def set_setting(key, value):
                        apply_camera_setting(camera, key, value)

                if not active.is_set():
                    continue
                settings = read_settings()
                with lock:
                    shared.update(camera_open=True, camera_settings=settings)
                next_stage("等待首帧")
                while not stop.is_set() and active.is_set():
                    try:
                        key, value, result = settings_requests.get_nowait()
                    except Empty:
                        pass
                    else:
                        if result.set_running_or_notify_cancel():
                            try:
                                set_setting(key, value)
                                settings = read_settings()
                                with lock:
                                    shared["camera_settings"] = settings
                            except Exception as error:
                                setting_pending.clear()
                                result.set_exception(error)
                            else:
                                setting_pending.clear()
                                result.set_result(settings)
                    gray = read_frame(150)
                    if gray is None:
                        if not first_frame and time.monotonic() - stage_started >= 5:
                            raise TimeoutError("5 秒内未收到相机首帧")
                        continue
                    if gray.ndim == 3:
                        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                    if [gray.shape[1], gray.shape[0]] != size:
                        raise ValueError(
                            "Camera image size differs from calibrated size"
                        )
                    if not first_frame:
                        next_stage("")
                        camera_log(
                            f"收到首帧，连接总耗时 {time.monotonic() - started:.3f} 秒"
                        )
                        first_frame = True
                    with lock:
                        shared["frame"] = (gray.copy(), time.monotonic(), k, d)
                        shared["capture_sequence"] += 1
            except Exception as error:
                duration = (
                    f"（{time.monotonic() - stage_started:.3f} 秒）" if stage else ""
                )
                camera_log(f"{stage or '采集'}失败{duration}：{error}")
                with lock:
                    if active.is_set():
                        shared["state"] = {
                            "state": "camera_unavailable",
                            "error": str(error),
                            "retry_seconds": 3,
                        }
                    shared["frame"] = None
            finally:
                if camera is not None:
                    try:
                        camera.close()
                        if args.camera == "galaxy" and os.name != "nt":
                            camera.uninit()
                    except Exception as error:
                        camera_log(f"释放相机失败：{error}")
                    camera = None
                with lock:
                    shared["camera_open"] = False
                    shared["camera_busy"] = False
                    while not settings_requests.empty():
                        _, _, result = settings_requests.get_nowait()
                        if result.set_running_or_notify_cancel():
                            result.set_exception(ValueError("Camera disconnected"))
                camera_log("相机采集已停止")
                if not active.is_set() and not stop.is_set():
                    publish_paused()
            if active.is_set():
                camera_log("将在 3 秒后重试连接")
                stop.wait(3)

    threading.Thread(target=capture, daemon=True).start()
    print(f"LIVE_URL http://{args.host}:{server.server_port}", flush=True)
    processed = 0
    last_sequence = -1
    frame_times = deque(maxlen=30)
    try:
        while not stop.is_set():
            with lock:
                frame = shared["frame"]
                sequence = shared["capture_sequence"]
            if frame is None:
                frame_times.clear()
                stop.wait(0.015)
                continue
            if sequence == last_sequence:
                stop.wait(0.015)
                continue
            gray, captured, k, d = frame
            last_sequence = sequence
            observed, rejected = detect(gray, detector, points)
            processed += 1
            frame_times.append(time.monotonic())
            fps = (
                (len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 1e-6)
                if len(frame_times) > 1
                else 0.0
            )
            full, zoom, state = overlay(
                gray,
                observed,
                rejected,
                points,
                k,
                d,
                sequence,
                (time.monotonic() - captured) * 1000,
                fps,
                imu_state.snapshot(),
            )
            state["state"] = "tracking" if state["pose_found"] else "searching_markers"
            if state["pose_found"]:
                state["camera_matrix"] = k.tolist()
                state["distortion_coefficients"] = d.ravel().tolist()
            encoded = [
                cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
                for im in (full, zoom)
            ]
            with ready:
                if not active.is_set():
                    continue
                shared.update(
                    full=encoded[0],
                    zoom=encoded[1],
                    state=state,
                    updated=time.monotonic(),
                )
                ready.notify_all()
            if processed % 30 == 0:
                print("LIVE", json.dumps(state), flush=True)
    finally:
        stop.set()
        time.sleep(0.7)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
