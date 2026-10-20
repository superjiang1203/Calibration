"""Offline checks for fixed SDK geometry in extrinsic calibration."""
import unittest
import cv2
import numpy as np
from scripts.sensor_intrinsics.core import Observation, create_object_points
from scripts.stereo_calibration.core import _fixed_camera, _fixed_fit, _fit_stereo


class FixedIntrinsicsTests(unittest.TestCase):
    def test_fixed_model_and_stereo_direction(self):
        k = np.array([[900., 0, 640], [0, 910, 480], [0, 0, 1]])
        d = np.array([-.1, .02, .001, -.002, 0, .003, 0, 0, 0, 0, 0, 0])
        camera = dict(camera_matrix=k, distortion_coefficients=d, image_size=(1280, 960))
        objects = create_object_points((11, 8), 15)
        rotation = cv2.Rodrigues(np.array([.02, -.04, .01]))[0]
        translation = np.array([[80.], [5.], [10.]])
        views, observations = [], []
        for i in range(8):
            rv = np.array([.05 * i, -.03 * i, .01 * i])
            tv = np.array([[i * 5. - 80], [-40.], [650. + i * 30]])
            left = cv2.projectPoints(objects, rv, tv, k, d)[0]
            rr = cv2.Rodrigues(rotation @ cv2.Rodrigues(rv)[0])[0]
            right = cv2.projectPoints(objects, rr, rotation @ tv + translation, k, d)[0]
            obs = Observation(str(i), left, (1280, 960))
            observations.append(obs)
            views.append(dict(left=obs, right_corners=right))
        fit = _fixed_fit(observations, (11, 8), 15, camera)
        np.testing.assert_array_equal(fit.camera_matrix, k)
        np.testing.assert_array_equal(fit.dist_coeffs, d)
        self.assertLess(fit.rms, .001)
        rms, transform = _fit_stereo(objects, views, [camera, camera])
        self.assertLess(rms, .001)
        np.testing.assert_allclose(transform[:3, :3], rotation, atol=1e-5)
        np.testing.assert_allclose(transform[:3, 3:], translation, atol=.01)

    def test_missing_factory_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'factory intrinsics missing'):
            _fixed_camera(dict(label='vcamera_color', sensor_type='vcamera'), {}, (1280, 960))

    def test_orbbec_sdk_model_is_fixed_and_resolution_checked(self):
        for stream in ('color', 'ir_left', 'ir_right'):
            data = dict(camera_matrix=[[900, 0, 640], [0, 910, 480], [0, 0, 1]],
                        distortion_coefficients=[-.1, .02, .001, -.002, 0, .003, 0, 0])
            sensor = dict(label='orbbec_' + stream, sensor_type='orbbec',
                          capture_metadata=dict(sdk_intrinsics=data, image_size=[1280, 960]))
            camera, source = _fixed_camera(sensor, {}, (1280, 960))
            np.testing.assert_array_equal(camera['camera_matrix'], data['camera_matrix'])
            np.testing.assert_array_equal(camera['distortion_coefficients'], data['distortion_coefficients'])
            self.assertIn('OrbbecStreamProfile', source)
            with self.assertRaisesRegex(ValueError, 'resolution'):
                _fixed_camera(sensor, {}, (640, 480))
        with self.assertRaisesRegex(ValueError, 'factory intrinsics missing'):
            _fixed_camera(dict(label='orbbec_color', sensor_type='orbbec'), {}, (1280, 960))
        self.assertIsNone(_fixed_camera(dict(label='galaxy_gray', sensor_type='galaxy'), {}, (1280, 960)))

    def test_orbbec_color_capture_exports_sdk_model(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from scripts.sensor_capture.orbbec import OrbbecSensorSource
        source = OrbbecSensorSource(SimpleNamespace(sensor_index=0))
        profile = Mock()
        profile.get_intrinsic.return_value = SimpleNamespace(fx=900, fy=910, cx=640, cy=480)
        profile.get_distortion.return_value = SimpleNamespace(
            k1=.1, k2=.2, p1=.01, p2=.02, k3=.3, k4=.4, k5=.5, k6=.6, model='brown')
        profile.get_width.return_value = 1280
        profile.get_height.return_value = 960
        source.camera = SimpleNamespace(color_profile=profile)
        metadata = source.metadata()
        camera, _ = _fixed_camera(dict(label='orbbec_color', sensor_type='orbbec',
                                      capture_metadata=metadata), {}, (1280, 960))
        np.testing.assert_array_equal(camera['distortion_coefficients'], [.1, .2, .01, .02, .3, .4, .5, .6])
        self.assertEqual(camera['image_size'], (1280, 960))

    def test_factory_resolution_must_match(self):
        sensor = dict(label='vcamera_color', sensor_type='vcamera', capture_metadata={
            'sdk_intrinsics': dict(camera_matrix=[[900, 0, 640], [0, 900, 480], [0, 0, 1]],
                                   distortion_coefficients=[0] * 12, image_size=[1280, 960])})
        with self.assertRaisesRegex(ValueError, 'resolution'):
            _fixed_camera(sensor, {}, (640, 480))


if __name__ == '__main__':
    unittest.main()
