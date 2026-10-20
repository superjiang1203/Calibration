"""Offline self-checks for instrument geometry and Galaxy-to-RGB projection."""
import sys
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.validation.extrinsic_check import InstrumentTracker, project, transform_tip, matrix_file, export_adjustment, ROOT


class InstrumentProjectionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'configs/instruments/instrument_extrinsic_check_large_qr.json').read_text(encoding='utf-8'))
        self.config['single_marker']['id'] = 7
        self.tracker = InstrumentTracker(self.config)
        self.k = np.array([[1800., 0, 1280], [0, 1800, 960], [0, 0, 1.]])
        self.d = np.array([-.2, .05, .001, -.002, 0.])

    def test_synthetic_pose_and_tip_axes(self):
        objects = self.tracker.positions[7]
        rv, tv = np.array([.25, -.15, .07]), np.array([20., -10., 650.])
        corners = cv2.projectPoints(objects, rv, tv, self.k, self.d)[0].reshape(-1, 2)
        result = self.tracker.solve(objects, corners, self.k, self.d)
        self.assertIsNotNone(result)
        expected = tv + cv2.Rodrigues(rv)[0] @ np.diag([1., -1., -1.]) @ self.tracker.tip
        np.testing.assert_allclose(result['tip'], expected, atol=1e-5)
        self.assertLess(result['rms'], 1e-5)

    def test_detect_rendered_marker(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        marker = cv2.aruco.generateImageMarker(dictionary, 7, 300)
        image = np.full((800, 1000), 255, np.uint8)
        image[250:550, 350:650] = marker
        result = self.tracker.detect(image, self.k, self.d)
        self.assertIsNotNone(result)
        self.assertEqual(result['ids'], [7])
        self.tracker.min_markers = 2
        self.assertIsNone(self.tracker.detect(image, self.k, self.d))

    def test_export_preserves_R_and_applies_rgb_translation(self):
        rotation = cv2.Rodrigues(np.array([.1, .2, -.1]))[0]
        translation, delta = np.array([20., -60., -16.]), np.array([1., 2., 3.])
        with tempfile.TemporaryDirectory() as tmp:
            path = export_adjustment(Path(tmp)/'new', rotation, translation, delta, {})
            r, t = matrix_file(path, 'R', 'T')
            np.testing.assert_allclose(r, rotation)
            np.testing.assert_allclose(t.ravel(), translation+delta)
        point = np.array([30., 40., 650.])
        expected = cv2.projectPoints(point.reshape(1, 3), np.zeros(3), np.zeros(3), self.k, self.d)[0].ravel()
        np.testing.assert_allclose(project(point, self.k, self.d), expected)
        self.assertIsNone(project([0, 0, -1], self.k, self.d))

    def test_nonplanar_multimarker_geometry(self):
        import yaml
        raw = dict(instrument=dict(marker_dictionary='DICT_4X4_100'),
                   marker_positions={1: [[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]],
                                     2: [[15, -10, 0], [15, -10, 20], [15, 10, 20], [15, 10, 0]]},
                   tip_offset=[0, -150, -5])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'geometry.yaml'
            path.write_text(yaml.safe_dump(raw))
            tracker = InstrumentTracker(self.config, path)
        objects = np.concatenate(list(tracker.positions.values()))
        rv, tv = np.array([.25, -.15, .07]), np.array([20., -10., 650.])
        corners = cv2.projectPoints(objects, rv, tv, self.k, self.d)[0].reshape(-1, 2)
        result = tracker.solve(objects, corners, self.k, self.d)
        np.testing.assert_allclose(result['tip'], tv+cv2.Rodrigues(rv)[0]@tracker.tip, atol=1e-4)

    def test_galaxy_to_rgb_direction_and_translation_axes(self):
        rotation = cv2.Rodrigues(np.array([0., 0., np.pi/2]))[0]
        tip = np.array([10., 20., 600.])
        translation = np.array([30., -40., 5.])
        delta = np.array([1., 2., 3.])
        original, adjusted, p0, p1 = transform_tip(tip, rotation, translation, delta, self.k, self.d)
        np.testing.assert_allclose(original, [10., -30., 605.], atol=1e-8)
        np.testing.assert_allclose(adjusted, [11., -28., 608.], atol=1e-8)
        np.testing.assert_allclose(rotation.T@(original-translation), tip, atol=1e-8)
        np.testing.assert_allclose(p1, project(adjusted, self.k, self.d))

    def test_large_qr_unique_id_and_multiple_marker_rejection(self):
        config = json.loads((ROOT/'configs/instruments/instrument_extrinsic_check_large_qr.json').read_text(encoding='utf-8'))
        tracker = InstrumentTracker(config)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        image = np.full((800, 1000), 255, np.uint8)
        image[250:550, 350:650] = cv2.aruco.generateImageMarker(dictionary, 27, 300)
        pose = tracker.detect(image, self.k, self.d)
        self.assertIsNotNone(pose)
        self.assertEqual(pose['ids'], [27])
        expected = pose['center'] + pose['rotation'] @ tracker.fix @ tracker.tip
        np.testing.assert_allclose(pose['tip'], expected)
        self.assertAlmostEqual(np.linalg.norm(pose['tip']-pose['center']), np.linalg.norm(tracker.tip))
        image[50:200, 50:200] = cv2.aruco.generateImageMarker(dictionary, 12, 150)
        self.assertIsNone(tracker.detect(image, self.k, self.d))
        config['single_marker']['id'] = 27
        self.assertEqual(InstrumentTracker(config).detect(image, self.k, self.d)['ids'], [27])


if __name__ == '__main__':
    unittest.main()
