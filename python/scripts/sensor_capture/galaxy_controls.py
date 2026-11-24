"""Live Galaxy exposure and gain controls shared by both H5 previews."""

import math

CAMERA_SETTINGS = {
    "exposure_us": ("ExposureTime", "ExposureAuto", "set_exposure_time_us"),
    "gain": ("Gain", "GainAuto", "set_gain"),
}


def camera_settings(camera):
    return {
        key: camera.get_float_range(feature)
        for key, (feature, _, _) in CAMERA_SETTINGS.items()
        if camera.feature_is_available(feature)
    }


def validate_setting(feature, limits, value):
    if not math.isfinite(value) or not limits["min"] <= value <= limits["max"]:
        raise ValueError(
            f"{feature}: {limits['min']}–{limits['max']} {limits.get('unit', '')}"
        )


def apply_camera_setting(camera, key, value):
    feature, automatic, setter = CAMERA_SETTINGS[key]
    validate_setting(feature, camera.get_float_range(feature), value)
    manual = not camera.feature_is_available(automatic) or (
        camera.get_enum(automatic)["symbolic"] == "Off"
    )
    if manual and camera.feature_is_writable(feature):
        try:
            camera.set_float(feature, value, live=True)
            return
        except Exception:
            pass
    # Some camera modes lock these nodes during acquisition.
    camera.stream_off()
    try:
        getattr(camera, setter)(value)
    finally:
        camera.stream_on()
