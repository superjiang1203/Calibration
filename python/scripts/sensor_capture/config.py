"""Translate validated mode-independent sensor settings into SDK options."""
from argparse import Namespace


def sensor_options(sensor):
    source = sensor['sensor_type']
    index = sensor.get('sensor_index', 1 if source == 'galaxy' else 0)
    return Namespace(
        source=source, sensor_index=index,
        serial_number=sensor.get("serial_number", ""), ip_address=sensor.get("ip_address", ""), opencv_sensor_index=index,
        width=sensor.get('width', 0), height=sensor.get('height', 0),
        offset_x=sensor.get('offset_x'), offset_y=sensor.get('offset_y'),
        exposure_us=sensor.get('exposure_us'), exposure_sdk=sensor.get('exposure_sdk'),
        gain_db=sensor.get('gain_db', 0.0), acquisition_fps=sensor.get('acquisition_fps', 0),
        galaxy_backend=sensor.get('galaxy_backend', 'auto'), gxipy_path=sensor.get('gxipy_path'),
        white_balance_mode=sensor.get('white_balance_mode', 'auto_lock'),
        white_balance_kelvin=sensor.get('white_balance_kelvin'),
        white_balance_warmup_frames=sensor.get('white_balance_warmup_frames', 30),
    )
