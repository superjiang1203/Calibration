"""Calibration adapter for the reusable Vcamera Python camera wrapper."""
import sys
from pathlib import Path
from .base import FrameSource


class VcameraSource(FrameSource):
    backend_name = 'vcamera-sdk'

    def __init__(self, args):
        common_root = Path(__file__).resolve().parents[2] / 'common'
        if str(common_root) not in sys.path:
            sys.path.insert(0, str(common_root))
        from VcameraCamera import VcameraCamera

        self.camera = VcameraCamera(
            target_width=args.width, target_height=args.height,
            sensor_index=args.sensor_index, serial_number=args.serial_number,
            ip_address=args.ip_address)

    def open(self):
        self.camera.open()

    def read(self, timeout_ms):
        return self.camera.read(timeout_ms)

    def metadata(self):
        return self.camera.metadata()

    def close(self):
        self.camera.close()
