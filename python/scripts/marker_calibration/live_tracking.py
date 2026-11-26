"""Project calibrated marker corners and the pivot tip onto a live preview."""

import cv2
import numpy as np
import yaml

from ..imu_calibration.preview import estimate_pose
from .checkerboard import black_border_corners
from .marker_io import create_detector, detect_gray, detection_gray, load_camera


def tracking_corners(image, detector, target_ids):
    gray = detection_gray(image)
    decoded, duplicates = detect_gray(gray, detector)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.65)
    measured = {}
    rejected = set(duplicates) & target_ids
    height, width = gray.shape
    for mid, corners in decoded.items():
        if mid not in target_ids:
            continue
        edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        area = abs(cv2.contourArea(corners.astype(np.float32)))
        if (edges.min() < 12 or area < 0.25 * edges.max()**2
                or np.any(corners < 7) or np.any(corners >= np.array([width, height]) - 7)):
            rejected.add(mid)
            continue
        refined, quality = black_border_corners(smooth, corners, return_quality=True)
        if quality["valid"]:
            measured[mid] = refined
        else:
            rejected.add(mid)
    return measured, sorted(rejected)


class LiveTracking:
    def __init__(self, geometry, intrinsics):
        document = yaml.safe_load(geometry.read_text(encoding="utf-8-sig"))
        self.points = {int(mid): np.asarray(points, np.float64)
                       for mid, points in document["marker_positions"].items()}
        if not self.points or any(p.shape != (4, 3) or not np.isfinite(p).all() for p in self.points.values()):
            raise ValueError("Geometry YAML must contain four finite 3D corners for each marker.")
        dictionary = document["instrument"]["marker_dictionary"]
        self.detector = create_detector(dictionary)
        self.tip = None
        if document.get("tip_offset_source") == "fixed_sphere_center_multi_pose_pivot":
            self.tip = np.asarray(document["tip_offset"], np.float64)
            if self.tip.shape != (3,) or not np.isfinite(self.tip).all():
                raise ValueError("Invalid calibrated tip_offset.")
        self.camera = load_camera(intrinsics) if intrinsics else None
        self.uses_yaml = intrinsics is not None
        self.adapter = None

    def overlay(self, image, preview, adapter):
        if not self.uses_yaml and self.adapter is not adapter:
            factory = adapter.metadata().get("sdk_intrinsics")
            if not factory:
                raise ValueError("Vcamera SDK did not provide RGB intrinsics.")
            self.camera = {
                "matrix": np.asarray(factory["camera_matrix"], np.float64),
                "distortion": np.asarray(factory["distortion_coefficients"], np.float64),
                "image_size": factory["image_size"],
            }
            self.adapter = adapter
        if list(image.shape[1::-1]) != list(self.camera["image_size"]):
            raise ValueError("Live image size differs from the tracking intrinsics.")
        display = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR) if preview.ndim == 2 else preview
        measured, rejected = tracking_corners(image, self.detector, self.points.keys())
        k, d = self.camera["matrix"], self.camera["distortion"]
        fit = estimate_pose(self.points, measured, k, d) if measured else None
        message = "Marker tracking: no reliable pose"
        lines = []
        if fit is not None:
            r, t = fit
            rotation = cv2.Rodrigues(r)[0]
            scale = np.asarray(display.shape[1::-1]) / np.asarray(image.shape[1::-1])

            def project(points):
                return cv2.projectPoints(points, r, t, k, d)[0].reshape(-1, 2)

            accepted = []
            residuals = []
            for mid, pixels in measured.items():
                observed = np.rint(pixels * scale).astype(np.int32)
                cv2.polylines(display, [observed], True, (255, 180, 30), 1, cv2.LINE_AA)
                projected = project(self.points[mid])
                rms = np.sqrt(np.mean(np.sum((projected - pixels)**2, axis=1)))
                residuals.append(rms)
                if rms <= 3 and np.all((self.points[mid] @ rotation.T + t.ravel())[:, 2] > 0):
                    accepted.append(mid)
                    polygon = np.rint(projected * scale).astype(np.int32)
                    cv2.polylines(display, [polygon], True, (50, 230, 80), 2, cv2.LINE_AA)
                    cv2.putText(display, str(mid), tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, .6, (50, 230, 80), 2)
            if accepted:
                message = "IDs: " + ", ".join(map(str, accepted)) + " | measured: cyan / projected: green"
                angles = cv2.RQDecomp3x3(rotation)[0]
                lines.append("Target XYZ mm: " + " ".join(f"{value:.2f}" for value in t.ravel()))
                lines.append("Target RPY deg: " + " ".join(f"{value:.2f}" for value in angles))
                lines.append(f"Fit RMS (all detected IDs): {np.sqrt(np.mean(np.square(residuals))):.2f} px")
                axes = np.vstack((np.zeros(3), 15 * np.eye(3)))
                if np.all((axes @ rotation.T + t.ravel())[:, 2] > 0):
                    pixels = np.rint(project(axes) * scale).astype(np.int32)
                    for end, color in zip(pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 120, 0))):
                        cv2.arrowedLine(display, tuple(pixels[0]), tuple(end), color, 2, cv2.LINE_AA)
                if len(measured) >= 3:
                    held = max(measured)
                    support = {mid: pixels for mid, pixels in measured.items() if mid != held}
                    check = estimate_pose(self.points, support, k, d)
                    held_line = f"Holdout ID {held}: pose unavailable"
                    if check is not None:
                        hr, ht = check
                        hold_rotation = cv2.Rodrigues(hr)[0]
                        valid = all(
                            np.all((self.points[mid] @ hold_rotation.T + ht.ravel())[:, 2] > 0)
                            and np.mean(np.sum((cv2.projectPoints(self.points[mid], hr, ht, k, d)[0].reshape(-1, 2)
                                                - pixels)**2, axis=1)) <= 9
                            for mid, pixels in support.items()
                        )
                        if valid and np.all((self.points[held] @ hold_rotation.T + ht.ravel())[:, 2] > 0):
                            predicted = cv2.projectPoints(self.points[held], hr, ht, k, d)[0].reshape(-1, 2)
                            delta = predicted - measured[held]
                            held_line = f"Holdout ID {held}: {np.sqrt(np.mean(np.sum(delta**2, axis=1))):.2f} px RMS"
                    lines.append(held_line)
                else:
                    lines.append("Holdout RMS: show at least 3 IDs")
                if self.tip is None:
                    lines.append("Tip not calibrated")
                else:
                    xyz = rotation @ self.tip + t.ravel()
                    if xyz[2] > 0:
                        xy = project(self.tip.reshape(1, 3))[0] * scale
                        if np.all(xy >= 0) and np.all(xy < display.shape[1::-1]):
                            cv2.drawMarker(display, tuple(np.rint(xy).astype(int)), (30, 90, 255),
                                           cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
                        lines.append("Tip camera XYZ mm: " + " ".join(f"{value:.1f}" for value in xyz))
        if rejected:
            lines.append("Rejected by corner quality: " + ", ".join(map(str, rejected)))
        return self.with_hud(display, [message, *lines])

    @staticmethod
    def with_hud(display, lines):
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        # Keep the HUD height fixed so tracking loss cannot change MP4 dimensions.
        hud = np.full((180, display.shape[1], 3), 20, np.uint8)
        for row, line in enumerate(lines):
            cv2.putText(hud, line, (12, 21 + row * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (40, 240, 255), 1, cv2.LINE_AA)
        return np.vstack((hud, display))
