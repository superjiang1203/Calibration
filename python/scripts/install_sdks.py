"""Install camera Python bindings without copying complete vendor SDK trees."""
from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vendors", nargs="+", choices=("vcamera", "galaxy", "orbbec"))
    parser.add_argument("--offline", action="store_true",
                        help="Use only local wheels, including for Orbbec")
    parser.add_argument("--wheel-dir", type=Path, default=ROOT / "thirdparty" / "wheels")
    args = parser.parse_args(argv)
    failed = False
    for vendor in dict.fromkeys(args.vendors):
        try:
            install(vendor, args.wheel_dir.resolve(), args.offline)
        except (ImportError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f"{vendor}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
