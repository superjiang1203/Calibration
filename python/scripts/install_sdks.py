"""Install camera Python bindings without copying complete vendor SDK trees."""
from __future__ import annotations

import argparse
import ctypes
import importlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
ORBBEC_VERSION = "2.1.1"


def install(vendor: str, wheel_dir: Path, offline: bool) -> None:
    # Dependencies are supplied by requirements-calibration.txt. In particular,
    # Orbbec's sample-only open3d/pygame/PyAV requirements are not used here.
    command = [sys.executable, "-m", "pip", "install", "--no-deps",
               "--only-binary=:all:", "--find-links", str(wheel_dir)]
    if vendor == "orbbec":
        command += (["--no-index"] if offline else
                    ["--index-url", "https://pypi.org/simple"])
        requirement = f"pyorbbecsdk2=={ORBBEC_VERSION}"
    else:
        command += ["--no-index"]
        requirement = {"vcamera": "vcamera-python==26.1.15",
                       "galaxy": "gxipy==2.0.2512.9261"}[vendor]
    subprocess.run([*command, requirement], check=True)


def check(vendor: str) -> None:
    if vendor == "galaxy":
        if sys.platform.startswith("linux"):
            # gxipy's vendor wrapper hardcodes this system installation path.
            try:
                ctypes.CDLL("/usr/lib/libgxiapi.so")
            except OSError as error:
                raise RuntimeError(
                    "Install the official Galaxy Linux SDK runtime first "
                    "(/usr/lib/libgxiapi.so). The bundled gxipy wheel contains "
                    "Python bindings only; it does not contain Linux drivers."
                ) from error
        elif sys.platform == "win32" and not os.environ.get("GALAXY_GENICAM_ROOT"):
            raise RuntimeError("Install the official Galaxy Windows SDK, then "
                               "reopen the terminal to load GALAXY_GENICAM_ROOT.")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from python.scripts.sensor_capture.galaxy import import_gxipy
        sdk = import_gxipy()
        required = ("DeviceManager", "GxPixelFormatEntry")
    else:
        sdk = importlib.import_module({"vcamera": "vcamera", "orbbec": "pyorbbecsdk"}[vendor])
        required = {"vcamera": ("CameraUtils", "CameraFactory", "SensorType"),
                    "orbbec": ("Context", "Pipeline", "Config", "OBSensorType", "OBFormat")}[vendor]
    missing = [name for name in required if not hasattr(sdk, name)]
    if missing:
        raise RuntimeError(f"{vendor} SDK is missing required API: {', '.join(missing)}")
    print(f"{vendor}: import and required API OK ({sdk.__file__})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vendors", nargs="+", choices=("vcamera", "galaxy", "orbbec"))
    parser.add_argument("--check", action="store_true",
                        help="Check installed native bindings and API without opening cameras")
    parser.add_argument("--offline", action="store_true",
                        help="Use only local wheels, including for Orbbec")
    parser.add_argument("--wheel-dir", type=Path, default=ROOT / "thirdparty" / "wheels")
    args = parser.parse_args(argv)
    failed = False
    for vendor in dict.fromkeys(args.vendors):
        try:
            if args.check:
                check(vendor)
            else:
                install(vendor, args.wheel_dir.resolve(), args.offline)
        except (ImportError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f"{vendor}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
