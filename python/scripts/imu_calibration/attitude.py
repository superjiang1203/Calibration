"""Read-only IMU telemetry state and explicitly labelled display alignment."""

import math
import threading
import time
from collections import deque

import cv2
import numpy as np


def quaternion_matrix(wxyz):
    q = np.asarray(wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
        raise ValueError("Expected a finite nonzero WXYZ quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class ImuState:
    def __init__(self):
        self.lock = threading.Lock()
        self.samples = {}
        self.source = "icm42670"
        self.alignment = None
        self.connected = False
        self.received = 0.0
        self.jy_status = None
        self.count = 0
        self.arrivals = {}

    def feed(self, packet):
        kind = packet.get("type")
        now = time.monotonic()
        with self.lock:
            if kind == "attitude":
                source = packet.get("source")
                if source not in ("icm42670", "jy931"):
                    raise ValueError("Unknown IMU source")
                r = quaternion_matrix(packet["quaternion_wxyz"])
                q = np.asarray(packet["quaternion_wxyz"], float)
                lag = float(packet.get("relay_age_ms", 0))
                if not math.isfinite(lag) or not 0 <= lag <= 10000:
                    raise ValueError("Invalid sample age")
                timestamp = int(packet["device_timestamp_ms"])
                old = self.samples.get(source)
                # Reboot/wrapped device time invalidates a display reference.
                if (
                    old
                    and timestamp < old["device_timestamp_ms"]
                    and source == self.source
                ):
                    self.alignment = None
                self.samples[source] = dict(
                    quaternion_wxyz=(q / np.linalg.norm(q)).tolist(),
                    matrix=r.tolist(),
                    valid=bool(packet.get("valid", False)),
                    device_timestamp_ms=timestamp,
                    received=now,
                    relay_age_ms=lag,
                )
                self.count += 1
                self.arrivals.setdefault(source, deque(maxlen=150)).append(now)
                self.samples[source]["sdk_receive_hz"] = float(
                    packet.get("sdk_receive_hz", 0)
                )
                self.connected = True
            elif kind == "connection":
                self.connected = bool(packet.get("connected"))
                if not self.connected:
                    self.alignment = None
            elif kind == "jy931_status":
                self.jy_status = {
                    k: packet.get(k)
                    for k in (
                        "device_timestamp_ms",
                        "valid_flags",
                        "checksum_error_count",
                    )
                }
            else:
                raise ValueError("Unknown telemetry packet")
            self.received = now

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            sources = {}
            for source, p in self.samples.items():
                s = {k: v for k, v in p.items() if k != "received"}
                s["sample_age_ms"] = round(
                    (now - p["received"]) * 1000 + p["relay_age_ms"], 1
                )
                s["fresh"] = s["sample_age_ms"] < 500 and s["valid"]
                arrivals = self.arrivals[source]
                s["server_receive_hz"] = (
                    round((len(arrivals) - 1) / (arrivals[-1] - arrivals[0]), 1)
                    if len(arrivals) > 1
                    and arrivals[-1] > arrivals[0]
                    and now - arrivals[-1] < 1
                    else 0.0
                )
                # ZYX convention, R = Rz(yaw) Ry(pitch) Rx(roll).
                r = np.array(p["matrix"])
                s["roll_pitch_yaw_deg"] = np.rad2deg(
                    [
                        math.atan2(r[2, 1], r[2, 2]),
                        math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0])),
                        math.atan2(r[1, 0], r[0, 0]),
                    ]
                ).tolist()
                sources[source] = s
            selected = sources.get(self.source)
            registered = self.alignment is not None
            display = None
            if (
                selected
                and selected["fresh"]
                and self.connected
                and now - self.received < 2
            ):
                r = np.array(selected["matrix"])
                display = ((self.alignment @ r) if registered else r).tolist()
            return dict(
                source=self.source,
                connected=self.connected and now - self.received < 2,
                sources=sources,
                selected=selected,
                display_matrix=display,
                display_mode="single_pose_alignment"
                if registered
                else "raw_unregistered",
                absolute_orientation_comparison_valid=False,
                received_samples=self.count,
                jy931_status=self.jy_status,
                timing_note="Host-receive alignment; camera and IMU are not hardware synchronized",
                note="Single pose assumes coincident mounting axes; not an IMU-to-marker extrinsic calibration",
            )

    def align(self, visual_rotation):
        s = self.snapshot()
        if not s["display_matrix"]:
            raise ValueError("No fresh valid IMU attitude")
        with self.lock:
            if self.source != s["source"]:
                raise ValueError("IMU source changed; retry")
            self.alignment = (
                np.asarray(visual_rotation) @ np.array(s["selected"]["matrix"]).T
            )

    def reset(self, source=None):
        with self.lock:
            if source is not None:
                if source not in ("icm42670", "jy931"):
                    raise ValueError("Unknown IMU source")
                self.source = source
            self.alignment = None


def project_imu_axes(sample, t, k, d, length=35.0):
    if not sample or sample.get("display_matrix") is None:
        return None
    r = np.asarray(sample["display_matrix"], float)
    points = np.vstack([np.zeros(3), np.eye(3) * length]) @ r.T + np.asarray(t).reshape(
        3
    )
    if not np.isfinite(points).all() or points[:, 2].min() <= 1:
        return None
    return cv2.projectPoints(points, np.zeros(3), np.zeros(3), k, d)[0].reshape(-1, 2)
