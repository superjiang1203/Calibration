"""Keep factory K/D fixed while selecting Galaxy measured or fitted K/D."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.io import sensor_image_directories
from scripts.sensor_intrinsics.core import Observation, calibrate_once, create_object_points
from scripts.stereo_calibration.core import _fixed_camera, calibrate_session
from scripts.validation.stereo import _camera
from scripts.workflow.config import load_config


def write_camera(path, matrix, distortion, size=(1280, 960)):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    try:
        storage.write("image_width", size[0])
        storage.write("image_height", size[1])
        storage.write("camera_matrix", matrix)
        storage.write("distortion_coefficients", distortion.reshape(1, -1))
    finally:
        storage.release()


class KnownIntrinsicsWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.session = self.root / "session"
        self.session.mkdir()
        self.k = np.array([[900., 0., 640.], [0., 910., 480.], [0., 0., 1.]])
        self.d = np.array([-.04, .01, .0003, -.0004, .001])
        self.rotation = cv2.Rodrigues(np.array([.02, -.04, .01]))[0]
        self.translation = np.array([[80.], [-60.], [10.]])
        self.board = dict(pattern_cols=9, pattern_rows=6, square_size_mm=15.)
        self.options = dict(min_views=8, validation_stride=4, reject_outliers=False,
                            max_rms_px=1., max_validation_rmse_px=2.)
        self.sensors = [dict(label="galaxy", sensor_type="galaxy", stream="gray", image_size=[1280, 960]),
                        dict(label="vcamera", sensor_type="vcamera", stream="color", image_size=[1280, 960])]
        self.sensors[1]["capture_metadata"] = dict(sdk_intrinsics=dict(
            camera_matrix=self.k.tolist(), distortion_coefficients=self.d.tolist(), image_size=[1280, 960]))
        directories = sensor_image_directories(self.sensors)
        for sensor in self.sensors:
            sensor["image_directory"] = directories[sensor["label"]]
            (self.session / sensor["image_directory"]).mkdir()
        objects = create_object_points((9, 6), 15.)
        images = {sensor["label"]: [] for sensor in self.sensors}
        pairs, self.observations = [], {}
        for index in range(12):
            rv = np.array([.25 * np.sin(index), .3 * np.cos(index * .8), .1 * np.sin(index * .6)])
            tv = np.array([[-70. + 50. * np.sin(index * .9)], [-50. + 30. * np.cos(index)], [620. + 18. * index]])
            corners = [cv2.projectPoints(objects, rv, tv, self.k, self.d)[0],
                       cv2.projectPoints(objects, cv2.Rodrigues(self.rotation @ cv2.Rodrigues(rv)[0])[0],
                                         self.rotation @ tv + self.translation, self.k, self.d)[0]]
            pair = {}
            for sensor, points in zip(self.sensors, corners):
                label = sensor["label"]
                relative = f"{directories[label]}/{index:02d}.png"
                images[label].append(relative)
                pair[label] = relative
                path = str((self.session / relative).resolve())
                self.observations[path] = Observation(path, points, (1280, 960))
            pairs.append(dict(images=pair, capture_delta_ms=0., timestamp_basis="device"))
        metadata = dict(schema_version=2, sensors=self.sensors, images=images, pairs=pairs, board=self.board)
        (self.session / "capture_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        self.fixed = {}
        for sensor in self.sensors:
            path = self.root / f"{sensor['label']}.yml"
            write_camera(path, self.k, self.d)
            self.fixed[sensor["label"]] = str(path)

    def calibrate(self, known, **extra):
        options = dict(self.options, use_known_intrinsics=known, **extra)

        def detect(path, *_args):
            return self.observations[str(path)], dict(status="detected", detector="synthetic")

        with patch("scripts.stereo_calibration.core._detect_image", side_effect=detect), \
                patch("scripts.stereo_calibration.core.calibrate_once", wraps=calibrate_once) as fit, \
                contextlib.redirect_stdout(io.StringIO()):
            result = calibrate_session(self.session, self.root / "output", options, "sensor_extrinsics")
        return result, fit.call_count

    def test_true_preserves_known_intrinsics_and_recovers_full_extrinsics(self):
        before = {label: Path(path).read_bytes() for label, path in self.fixed.items()}
        result, fit_count = self.calibrate(True, fixed_intrinsics=dict(galaxy=self.fixed["galaxy"]))
        self.assertTrue(result["passed"], result)
        self.assertEqual(fit_count, 0)
        self.assertEqual(result["intrinsics_mode"], "factory_and_known_intrinsics")
        for label, path in self.fixed.items():
            self.assertEqual(Path(path).read_bytes(), before[label])
            camera = _camera(self.root / "output" / f"sensor_{label}_intrinsics.yml")
            np.testing.assert_array_equal(camera["camera_matrix"], self.k)
            np.testing.assert_array_equal(camera["distortion_coefficients"], self.d)
            self.assertTrue(result["sensors"][label]["intrinsics_fixed"])
        from scripts.validation.stereo import _transform
        transform, _ = _transform(self.root / "output" / "pair_galaxy_to_vcamera_extrinsics.yml")
        np.testing.assert_allclose(transform[:3, :3], self.rotation, atol=1e-5)
        np.testing.assert_allclose(transform[:3, 3:], self.translation, atol=.01)

    def test_true_missing_known_model_fails_without_fitting_a_fallback(self):
        result, fit_count = self.calibrate(True)
        self.assertFalse(result["passed"])
        self.assertEqual(fit_count, 0)
        self.assertIn("Known intrinsics required", result["sensors"]["galaxy"]["reason"])
        self.assertEqual(result["output_files"], [])
        self.assertFalse(list((self.root / "output").glob("*.yml")))

    def test_false_fits_only_galaxy_and_keeps_factory_sdk_model(self):
        # A different YAML must neither override SDK K/D nor prevent Galaxy fitting.
        other_k = self.k.copy()
        other_k[0, 0] += 40
        for path in self.fixed.values():
            write_camera(Path(path), other_k, np.zeros(5))
        result, fit_count = self.calibrate(False, fixed_intrinsics=self.fixed)
        self.assertTrue(result["passed"], result)
        self.assertEqual(fit_count, 1)
        self.assertEqual(result["intrinsics_mode"], "factory_and_estimated_intrinsics")
        self.assertFalse(result["sensors"]["galaxy"]["intrinsics_fixed"])
        self.assertEqual(result["sensors"]["galaxy"]["intrinsics_source"], "checkerboard_fit")
        self.assertTrue(result["sensors"]["vcamera"]["intrinsics_fixed"])
        self.assertIn("GetFactoryCalibInfo", result["sensors"]["vcamera"]["intrinsics_source"])
        camera = _camera(self.root / "output" / "sensor_vcamera_intrinsics.yml")
        np.testing.assert_array_equal(camera["camera_matrix"], self.k)
        np.testing.assert_array_equal(camera["distortion_coefficients"], self.d)

    def config(self, *, known=True, fixed=None, mode="sensor_extrinsics", factory=False):
        entries = []
        for index, sensor in enumerate(self.sensors):
            profile_path = self.root / f"profile{index}.json"
            # Generic sources suffice for validating file selections without SDKs.
            sensor_type = "vcamera" if factory and index == 1 else "opencv"
            profile_path.write_text(json.dumps(dict(schema_version=1, sensor_type=sensor_type, sensor_index=index,
                                                    stream="color", width=1280, height=960)), encoding="utf-8")
            entries.append(dict(label=sensor["label"], sensor_config=profile_path.name))
        calibration = dict(use_known_intrinsics=known)
        if fixed is not None:
            calibration["fixed_intrinsics"] = fixed
        config_path = self.root / "workflow.json"
        config_path.write_text(json.dumps(dict(schema_version=2, mode=mode, board=self.board,
                                              sensors=entries, calibration=calibration)), encoding="utf-8")
        return config_path

    def test_config_requires_boolean_and_rejects_known_mode_for_intrinsics(self):
        for value in ("false", 0, None):
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                load_config(self.config(known=value))
        with self.assertRaisesRegex(ValueError, "requires mode sensor_extrinsics"):
            load_config(self.config(mode="sensor_intrinsics"))

    def test_config_preflights_missing_file_and_model_resolution(self):
        with self.assertRaisesRegex(ValueError, "Known intrinsics required"):
            load_config(self.config())
        with self.assertRaisesRegex(FileNotFoundError, "Missing calibration"):
            load_config(self.config(fixed=dict(galaxy="missing.yml")))
        write_camera(Path(self.fixed["galaxy"]), self.k, self.d, (640, 480))
        with self.assertRaisesRegex(ValueError, "resolution"):
            load_config(self.config(fixed=self.fixed))
        # Unused models do not block explicit checkerboard fitting.
        config = load_config(self.config(known=False, fixed=dict(galaxy="missing.yml")))
        self.assertIs(config["calibration"]["use_known_intrinsics"], False)

    def test_omitted_switch_retains_previous_mixed_selection(self):
        self.assertIsNone(_fixed_camera(self.sensors[0], {}, (1280, 960)))
        camera, source = _fixed_camera(self.sensors[1], {}, (1280, 960))
        self.assertIn("GetFactoryCalibInfo", source)
        np.testing.assert_array_equal(camera["camera_matrix"], self.k)
        camera, _ = _fixed_camera(self.sensors[0], dict(fixed_intrinsics=self.fixed), (1280, 960))
        np.testing.assert_array_equal(camera["camera_matrix"], self.k)

    def test_factory_sdk_wins_over_yaml_for_every_switch_value(self):
        for vendor in ("vcamera", "orbbec"):
            sensor = dict(self.sensors[1], sensor_type=vendor)
            for known in (True, False, None):
                camera, source = _fixed_camera(sensor, dict(use_known_intrinsics=known,
                                                           fixed_intrinsics=dict(vcamera="missing.yml")), (1280, 960))
                np.testing.assert_array_equal(camera["camera_matrix"], self.k)
                np.testing.assert_array_equal(camera["distortion_coefficients"], self.d)
                self.assertTrue(source.startswith("capture_metadata."))

    def test_factory_export_fallback_requires_sdk_or_yaml_even_when_switch_false(self):
        for vendor in ("vcamera", "orbbec"):
            sensor = dict(label="vcamera", sensor_type=vendor)
            for known in (True, False, None):
                options = dict(use_known_intrinsics=known)
                with self.assertRaisesRegex(ValueError, "factory intrinsics missing"):
                    _fixed_camera(sensor, options, (1280, 960))
                camera, source = _fixed_camera(sensor, dict(options, fixed_intrinsics=self.fixed), (1280, 960))
                np.testing.assert_array_equal(camera["camera_matrix"], self.k)
                self.assertTrue(source.startswith("factory_sdk_export:"))

    def test_factory_intrinsics_cannot_be_refitted_as_standalone_workflow(self):
        with self.assertRaisesRegex(ValueError, "only for non-factory cameras"):
            load_config(self.config(known=False, mode="sensor_intrinsics", factory=True))
        with self.assertRaisesRegex(ValueError, "only for non-factory cameras"), \
                contextlib.redirect_stdout(io.StringIO()):
            calibrate_session(self.session, self.root / "output", self.options, "sensor_intrinsics")
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
