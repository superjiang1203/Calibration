"""Single configuration-driven entry point for Python camera adapters."""
from __future__ import annotations
import os
from .base import FrameSource, require_gui_support
from .galaxy import GxipyGalaxySource, CtypesGalaxySource, import_gxipy
from .orbbec import OrbbecSensorSource, disable_orbbec_logging
from .opencv import OpenCVSource
from .vcamera import VcameraSource

def create_frame_source(args) -> FrameSource:
    if args.source == "vcamera":
        return VcameraSource(args)
    if args.source == "orbbec":
        return OrbbecSensorSource(args)
    if args.source == "opencv":
        return OpenCVSource(
            args.opencv_sensor_index,
            args.width,
            args.height,
            args.acquisition_fps,
        )

    if args.source != "galaxy":
        raise ValueError(f"Unknown camera source: {args.source}")

    kwargs = {
        "sensor_index": args.sensor_index,
        "width": args.width,
        "height": args.height,
        "offset_x": args.offset_x,
        "offset_y": args.offset_y,
        "exposure_us": args.exposure_us,
        "gain_db": args.gain_db,
        "acquisition_fps": args.acquisition_fps,
    }
    if args.galaxy_backend == "gxipy":
        return GxipyGalaxySource(**kwargs, gxipy_path=args.gxipy_path)
    if args.galaxy_backend == "ctypes":
        return CtypesGalaxySource(**kwargs)
    if os.name == "nt":
        return GxipyGalaxySource(**kwargs, gxipy_path=args.gxipy_path)
    try:
        import_gxipy(args.gxipy_path)
    except Exception:
        return CtypesGalaxySource(**kwargs)
    return GxipyGalaxySource(**kwargs, gxipy_path=args.gxipy_path)
