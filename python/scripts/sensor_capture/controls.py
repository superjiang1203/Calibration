"""Common control names; SDK keys and units stay inside the camera adapters."""
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
