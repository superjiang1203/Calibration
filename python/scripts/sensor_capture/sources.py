"""Single configuration-driven entry point for Python camera adapters."""
from __future__ import annotations
import os
import time
from .camera_options import sensor_options
from .frame_source import FrameSource
from .galaxy import GxipyGalaxySource, CtypesGalaxySource, import_gxipy
from .orbbec import OrbbecIRSource, OrbbecSensorSource
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
            gain_sdk=getattr(args, 'gain_sdk', None),
            exposure_sdk=getattr(args, 'exposure_sdk', None),
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


def open_sources(sensors):
    """Open adapters, grouping IR streams that share one Orbbec device."""
    adapters = []
    ir_groups = {}
    for sensor in sensors:
        if sensor['sensor_type'] == 'orbbec' and sensor['stream'].startswith('ir_'):
            ir_groups.setdefault(sensor['sensor_index'], []).append(sensor)
        else:
            adapters.append((create_frame_source(sensor_options(sensor)), [sensor]))
    adapters.extend((OrbbecIRSource(group), group) for group in ir_groups.values())
    try:
        for adapter, _ in adapters:
            adapter.open()
    except Exception:
        for adapter, _ in reversed(adapters):
            adapter.close()
        raise
    return adapters


def read_sources(adapters, timeout_ms):
    images, timestamps = {}, {}
    for adapter, sensors in adapters:
        if isinstance(adapter, OrbbecIRSource):
            received, device_times = adapter.read(timeout_ms)
            images.update(received)
            timestamps.update(device_times)
        else:
            image = adapter.read(timeout_ms)
            if image is not None:
                label = sensors[0]['label']
                images[label] = image
                timestamps[label] = time.monotonic_ns() // 1000
    synchronized = len(adapters) == 1 and isinstance(adapters[0][0], OrbbecIRSource) and len(adapters[0][1]) == 2
    return images, timestamps, synchronized
