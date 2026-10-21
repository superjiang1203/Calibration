"""Offline regression checks for Galaxy SDK device-open diagnostics."""
import contextlib
from enum import IntEnum
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from scripts.sensor_capture.galaxy import GxipyGalaxySource


class InvalidParameter(Exception):
    """Simulate the vendor exception without importing or initializing its SDK."""


InvalidParameter.__module__ = "gxipy.Exception"


class DeviceClass(IntEnum):
    GEV = 2
    U3V = 3


class GalaxySourceOpenTests(unittest.TestCase):
    def source(self):
        return GxipyGalaxySource(
            sensor_index=1, width=4504, height=4504,
            offset_x=None, offset_y=None, exposure_us=30000.0,
            gain_db=0.0, acquisition_fps=0.0, gxipy_path=None,
        )

    def manager(self, devices):
        manager = Mock()
        manager.update_all_device_list.return_value = (len(devices), devices)
        return manager

    def device(self, index=1):
        return dict(index=index, vendor_name="Daheng Imaging",
                    model_name="MER2-2000-19U3M", sn="TEST-SERIAL",
                    device_class=DeviceClass.U3V)

    def sdk(self, manager):
        return SimpleNamespace(DeviceManager=lambda: manager, GxDeviceClassList=DeviceClass)

    def xml_error(self):
        return InvalidParameter("DeviceManager.open_device_by_index:{-5} __ReadXML failed")

    def test_vendor_open_failure_reports_identity_phase_and_original_cause(self):
        source = self.source()
        manager = self.manager([self.device()])
        error = InvalidParameter("DeviceManager.open_device_by_index:{-5} invalid parameter")
        manager.open_device_by_index.side_effect = error
        with patch("scripts.sensor_capture.galaxy.import_gxipy",
                   return_value=self.sdk(manager)), \
             patch.object(source, "_set_roi") as set_roi, \
             patch("scripts.sensor_capture.galaxy.time.sleep") as sleep:
            with self.assertRaises(RuntimeError) as raised:
                source.open()
        message = str(raised.exception)
        self.assertIs(raised.exception.__cause__, error)
        for detail in ("MER2-2000-19U3M", "TEST-SERIAL", "native device index 1",
                       "before applying ROI/exposure settings", str(error)):
            self.assertIn(detail, message)
        self.assertNotIn("journalctl", message)
        sleep.assert_not_called()
        manager.open_device_by_index.assert_called_once_with(1)
        set_roi.assert_not_called()
        self.assertIsNone(source.camera)
        self.assertIsNone(source.feature_control)
        self.assertIsNone(source.manager)

    def test_mixed_vendors_preserve_selected_native_device_index(self):
        source = self.source()
        manager = self.manager([
            dict(index=1, vendor_name="Percipio", model_name="Vcamera"),
            self.device(index=2),
        ])
        manager.open_device_by_index.side_effect = InvalidParameter("native open failed")
        with patch("scripts.sensor_capture.galaxy.import_gxipy",
                   return_value=self.sdk(manager)):
            with self.assertRaisesRegex(RuntimeError, "native device index 2"):
                source.open()
        manager.open_device_by_index.assert_called_once_with(2)
        self.assertEqual(source.device_info["sn"], "TEST-SERIAL")

    def test_unexpected_python_open_error_is_not_relabelled(self):
        source = self.source()
        manager = self.manager([self.device()])
        error = TypeError("unexpected SDK wrapper programming error")
        manager.open_device_by_index.side_effect = error
        with patch("scripts.sensor_capture.galaxy.import_gxipy",
                   return_value=self.sdk(manager)):
            with self.assertRaises(TypeError) as raised:
                source.open()
        self.assertIs(raised.exception, error)
        self.assertIsNone(source.manager)

    def test_xml_error_has_usb_diagnostic_hint_only_on_linux(self):
        for platform in ("linux", "win32"):
            with self.subTest(platform=platform):
                source = self.source()
                manager = self.manager([dict(self.device(), device_class=DeviceClass.GEV)])
                manager.open_device_by_index.side_effect = self.xml_error()
                with patch("scripts.sensor_capture.galaxy.import_gxipy",
                           return_value=self.sdk(manager)), \
                     patch("scripts.sensor_capture.galaxy.sys.platform", platform):
                    with self.assertRaises(RuntimeError) as raised:
                        source.open()
                self.assertEqual("journalctl -k -b" in str(raised.exception), platform == "linux")

    def test_usb3_xml_recovery_rediscovers_same_serial_with_new_native_index(self):
        source = self.source()
        manager = self.manager([self.device()])
        other = dict(self.device(index=1), sn="OTHER-SERIAL")
        manager.update_all_device_list.side_effect = [
            (1, [self.device()]), (2, [other, self.device(index=2)]),
        ]
        camera = Mock()
        camera.data_stream = [Mock()]
        camera.get_remote_device_feature_control.return_value.is_writable.return_value = False
        camera.get_remote_device_feature_control.return_value.is_implemented.return_value = False
        error = self.xml_error()
        manager.open_device_by_index.side_effect = [error, camera]
        stderr = io.StringIO()
        with patch("scripts.sensor_capture.galaxy.import_gxipy", return_value=self.sdk(manager)), \
             patch("scripts.sensor_capture.galaxy.sys.platform", "linux"), \
             patch("scripts.sensor_capture.galaxy.time.sleep") as sleep, \
             patch.object(source, "_set_roi"), patch.object(source, "_verify_configuration"), \
             contextlib.redirect_stderr(stderr):
            source.open()
        self.addCleanup(source.close)
        self.assertEqual(manager.open_device_by_index.call_args_list, [call(1), call(2)])
        self.assertEqual(manager.update_all_device_list.call_count, 2)
        sleep.assert_called_once_with(1.0)
        self.assertEqual(source.device_info["sn"], "TEST-SERIAL")
        self.assertEqual(source.device_info["index"], 2)
        camera.stream_on.assert_called_once_with()
        warning = source.metadata()["configuration_warnings"][0]
        self.assertIn("TEST-SERIAL", warning)
        self.assertIn("__ReadXML", warning)
        self.assertIn("native device index 2", warning)
        self.assertIn(str(error), stderr.getvalue())
        self.assertIn("retry opening that same USB3 camera once", stderr.getvalue())
        self.assertIn("Galaxy recovery succeeded", stderr.getvalue())

    def test_recovery_never_opens_a_different_camera_when_original_serial_is_absent(self):
        source = self.source()
        manager = self.manager([self.device()])
        manager.update_all_device_list.side_effect = [
            (1, [self.device()]),
            (2, [dict(self.device(index=1), sn="OTHER-SERIAL"),
                 dict(self.device(index=2), vendor_name="Other vendor")]),
        ]
        error = self.xml_error()
        manager.open_device_by_index.side_effect = error
        with patch("scripts.sensor_capture.galaxy.import_gxipy", return_value=self.sdk(manager)), \
             patch("scripts.sensor_capture.galaxy.sys.platform", "linux"), \
             patch("scripts.sensor_capture.galaxy.time.sleep") as sleep, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "original camera SN 'TEST-SERIAL'.*not rediscovered") as raised:
                source.open()
        self.assertIs(raised.exception.__cause__, error)
        self.assertIn(str(error), str(raised.exception))
        manager.open_device_by_index.assert_called_once_with(1)
        sleep.assert_called_once_with(1.0)
        self.assertIsNone(source.manager)

    def test_recovery_refuses_ambiguous_serial_or_changed_device_class(self):
        cases = (
            ([self.device(index=2), self.device(index=3)], "matched multiple Daheng devices"),
            ([dict(self.device(index=2), device_class=DeviceClass.GEV)], "different device class"),
        )
        for rediscovered, diagnostic in cases:
            with self.subTest(rediscovered=rediscovered):
                source = self.source()
                manager = self.manager([self.device()])
                manager.update_all_device_list.side_effect = [
                    (1, [self.device()]), (len(rediscovered), rediscovered),
                ]
                error = self.xml_error()
                manager.open_device_by_index.side_effect = error
                with patch("scripts.sensor_capture.galaxy.import_gxipy", return_value=self.sdk(manager)), \
                     patch("scripts.sensor_capture.galaxy.sys.platform", "linux"), \
                     patch("scripts.sensor_capture.galaxy.time.sleep"), \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, diagnostic) as raised:
                        source.open()
                self.assertIs(raised.exception.__cause__, error)
                manager.open_device_by_index.assert_called_once_with(1)
                self.assertIsNone(source.manager)

    def test_failed_recovery_is_bounded_and_preserves_both_sdk_errors(self):
        source = self.source()
        manager = self.manager([self.device()])
        manager.update_all_device_list.side_effect = [
            (1, [self.device()]), (1, [self.device(index=4)]),
        ]
        initial = self.xml_error()
        retry_error = InvalidParameter("second open failed after reenumeration")
        manager.open_device_by_index.side_effect = [initial, retry_error]
        with patch("scripts.sensor_capture.galaxy.import_gxipy", return_value=self.sdk(manager)), \
             patch("scripts.sensor_capture.galaxy.sys.platform", "linux"), \
             patch("scripts.sensor_capture.galaxy.time.sleep") as sleep, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(RuntimeError) as raised:
                source.open()
        self.assertEqual(manager.open_device_by_index.call_args_list, [call(1), call(4)])
        self.assertEqual(manager.update_all_device_list.call_count, 2)
        sleep.assert_called_once_with(1.0)
        self.assertIn(str(initial), str(raised.exception))
        self.assertIn(str(retry_error), str(raised.exception))
        self.assertIs(raised.exception.__cause__, retry_error)
        self.assertIs(retry_error.__context__, initial)

    def test_recovery_is_skipped_outside_linux_usb3_xml_failure_with_known_serial(self):
        cases = (
            ("win32", self.device(), self.xml_error()),
            ("linux", dict(self.device(), device_class=DeviceClass.GEV), self.xml_error()),
            ("linux", dict(self.device(), sn=""), self.xml_error()),
            ("linux", self.device(), InvalidParameter("{-5} unrelated invalid parameter")),
        )
        for platform, device, error in cases:
            with self.subTest(platform=platform, device=device, error=str(error)):
                source = self.source()
                manager = self.manager([device])
                manager.open_device_by_index.side_effect = error
                with patch("scripts.sensor_capture.galaxy.import_gxipy", return_value=self.sdk(manager)), \
                     patch("scripts.sensor_capture.galaxy.sys.platform", platform), \
                     patch("scripts.sensor_capture.galaxy.time.sleep") as sleep:
                    with self.assertRaises(RuntimeError):
                        source.open()
                sleep.assert_not_called()
                manager.open_device_by_index.assert_called_once_with(1)
                manager.update_all_device_list.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
