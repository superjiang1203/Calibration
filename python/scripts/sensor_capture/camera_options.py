"""Normalize camera controls and translate profiles into SDK adapter options."""

from argparse import Namespace

CONTROL_FIELDS = {
    'galaxy': ('exposure_us', 'gain_db'),
    'vcamera': ('exposure_us', 'gain_sdk'),
    'orbbec': ('exposure_sdk', 'gain_sdk'),
    'opencv': ('exposure_sdk', 'gain_sdk'),
}


def normalize_controls(profile, label='camera'):
    """Accept common fields and retain compatibility with existing SDK-specific JSON."""
    result = dict(profile)
    fields = CONTROL_FIELDS.get(result.get('sensor_type'))
    if fields is None:
        return result
    for public, native, alternatives in (
            ('exposure', fields[0], ('exposure_us', 'exposure_sdk')),
            ('gain', fields[1], ('gain_db', 'gain_sdk'))):
        if public not in result:
            continue
        if any(key in result for key in alternatives):
            raise ValueError(f'{label}: use {public} once; do not also specify its SDK-specific field')
        value = result.pop(public)
        if result['sensor_type'] == 'orbbec' and type(value) is float and value.is_integer():
            value = int(value)
        result[native] = value
    return result


def sensor_options(sensor):
    sensor = normalize_controls(sensor)
    source = sensor['sensor_type']
    index = sensor.get('sensor_index', 1 if source == 'galaxy' else 0)
    return Namespace(
        source=source, sensor_index=index, stream=sensor.get('stream', 'color'),
        serial_number=sensor.get("serial_number", ""), ip_address=sensor.get("ip_address", ""), opencv_sensor_index=index,
        width=sensor.get('width', 0), height=sensor.get('height', 0),
        offset_x=sensor.get('offset_x'), offset_y=sensor.get('offset_y'),
        exposure_us=sensor.get('exposure_us'), exposure_sdk=sensor.get('exposure_sdk'),
        exposure_mode=sensor.get('exposure_mode'),
        exposure_warmup_frames=sensor.get('exposure_warmup_frames', 30),
        gain_db=sensor.get('gain_db', 0.0), gain_sdk=sensor.get('gain_sdk'),
        gain_feature=sensor.get('gain_feature'),
        acquisition_fps=sensor.get('acquisition_fps', 0),
        galaxy_backend=sensor.get('galaxy_backend', 'gxipy'), gxipy_path=sensor.get('gxipy_path'),
        white_balance_mode=sensor.get('white_balance_mode', 'auto_lock'),
        white_balance_kelvin=sensor.get('white_balance_kelvin'),
        white_balance_warmup_frames=sensor.get('white_balance_warmup_frames', 30),
    )
