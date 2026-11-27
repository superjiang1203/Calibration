"""Start the camera preview over SSH and open its H5 page locally."""

import argparse
import io
import shlex
import subprocess
import sys
import time
import webbrowser
import zipfile
from getpass import getpass
from pathlib import Path
from threading import Event, Thread
from urllib.request import ProxyHandler, build_opener

from ..remote import connect, load_profile, run_remote, tunnel
from .relay import forward

ROOT = Path(__file__).resolve().parents[3]


def upload(ssh, directory, geometry, intrinsics, remote_python):
    files = {
        "geometry.yaml": geometry,
        "python/common/GalaxyCamera.py": ROOT / "python/common/GalaxyCamera.py",
        "python/common/VcameraCamera.py": ROOT / "python/common/VcameraCamera.py",
        "python/scripts/sensor_capture/galaxy_controls.py": ROOT
        / "python/scripts/sensor_capture/galaxy_controls.py",
        "python/scripts/sensor_capture/image_zoom.js": ROOT
        / "python/scripts/sensor_capture/image_zoom.js",
        "python/scripts/marker_calibration/checkerboard.py": ROOT
        / "python/scripts/marker_calibration/checkerboard.py",
    }
    if intrinsics is not None:
        files["intrinsics.yml"] = intrinsics
    for name in ("__init__.py", "attitude.py", "preview.py", "preview.html"):
        relative = f"python/scripts/imu_calibration/{name}"
        files[relative] = ROOT / relative
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, path in files.items():
            bundle.write(path, name)
    archive.seek(0)
    with ssh.open_sftp() as sftp:
        sftp.putfo(archive, f"{directory}/preview.zip")
    run_remote(
        ssh, remote_python, "-m", "zipfile", "-e", f"{directory}/preview.zip", directory
    )


def launch_local(args, producer):
    stopping = Event()
    with subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "python.scripts.imu_calibration.preview",
            "--geometry",
            str(args.geometry.resolve()),
            *(["--intrinsics", str(args.intrinsics.resolve())] if args.intrinsics else []),
            "--serial",
            args.serial,
            "--camera",
            args.camera,
            "--imu-source",
            args.imu_source,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--start-paused",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:

        def read_preview():
            for line in process.stdout:
                print(line.rstrip(), flush=True)
                if line.startswith("LIVE_URL "):
                    url = line.split()[1]
                    relay = Thread(target=forward, args=(producer, url, stopping))
                    workers.append(relay)
                    relay.start()
                    if not args.no_browser:
                        webbrowser.open(url)

        workers = []
        reader = Thread(target=read_preview)
        reader.start()
        try:
            return process.wait()
        except KeyboardInterrupt:
            return 0
        finally:
            stopping.set()
            if process.poll() is None:
                process.terminate()
            reader.join()
            for worker in workers:
                worker.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--ssh", metavar="PROFILE")
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "--ssh-config", type=Path, default=ROOT / "configs/ssh.local.json"
    )
    parser.add_argument("--host")
    parser.add_argument("--user")
    parser.add_argument("--remote-python")
    parser.add_argument("--serial", default="MAF26040019")
    parser.add_argument("--camera", choices=("galaxy", "vcamera"), default="galaxy")
    parser.add_argument("--imu-source", choices=("icm42670", "rion_imu406"), default="icm42670")
    parser.add_argument("--imu-port", default="auto")
    parser.add_argument("--imu-baud", type=int, default=115200)
    parser.add_argument("--imu-address", type=int, default=0)
    parser.add_argument("--port", type=int, default=8879)
    parser.add_argument("--remote-port", type=int, default=8789)
    parser.add_argument(
        "--bridge",
        type=Path,
        default=ROOT / "output/imu_attitude/build/Release/imu_attitude_bridge.exe",
    )
    parser.add_argument("--password-prompt", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.camera == "galaxy" and args.intrinsics is None:
        parser.error("Galaxy requires --intrinsics")
    if args.camera == "vcamera":
        args.intrinsics = None
    rion = args.imu_source == "rion_imu406"
    for path in (args.geometry, args.intrinsics, None if rion else args.bridge):
        if path is not None and not path.is_file():
            parser.error(f"File not found: {path}")
    if rion:
        from importlib.util import find_spec

        if find_spec("serial") is None:
            parser.error("IMU406 requires pyserial in the Studio Python environment.")
        if not args.imu_port.strip() or args.imu_baud <= 0 or not 0 <= args.imu_address <= 254:
            parser.error("Invalid IMU406 serial port, baud rate or address.")
        producer = [sys.executable, "-u", "-m", "python.scripts.imu_calibration.rion",
                    "--port", args.imu_port, "--baud", str(args.imu_baud),
                    "--address", str(args.imu_address)]
    else:
        producer = [str(args.bridge.resolve())]

    if args.local:
        if args.ssh:
            parser.error("--local and --ssh are mutually exclusive")
        return launch_local(args, producer)
    try:
        profile = load_profile(args.ssh_config, args.ssh) if args.ssh else {}
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    host = args.host or profile.get("host", "192.168.1.33")
    user = args.user or profile.get("user", "cair")
    args.remote_python = args.remote_python or profile.get(
        "python", "/home/cair/miniconda3/envs/mod_camera_v2/bin/python"
    )
    if profile and (host, user) != (profile["host"], profile["user"]):
        parser.error("Use a separate SSH profile for a different host or user")
    password = (
        getpass("SSH password: ") if args.password_prompt else profile.get("password")
    )
    with connect(host, user, password, port=profile.get("port", 22)) as ssh:
        directory = run_remote(ssh, "mktemp", "-d", "/tmp/imu-calibration.XXXXXXXX")
        server = worker = None
        stopping = Event()
        try:
            upload(ssh, directory, args.geometry, args.intrinsics, args.remote_python)
            command = shlex.join(
                [
                    args.remote_python,
                    "-u",
                    "-m",
                    "python.scripts.imu_calibration.preview",
                    "--geometry",
                    "geometry.yaml",
                    *(["--intrinsics", "intrinsics.yml"] if args.intrinsics else []),
                    "--serial",
                    args.serial,
                    "--camera",
                    args.camera,
                    "--imu-source",
                    args.imu_source,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.remote_port),
                    "--start-paused",
                ]
            )
            quoted = shlex.quote(directory)
            run_remote(
                ssh,
                "sh",
                "-c",
                f"cd {quoted} && (nohup {command} >preview.log 2>&1 </dev/null & echo $! >preview.pid)",
            )
            for _ in range(40):
                log = run_remote(ssh, "cat", f"{directory}/preview.log")
                port_line = next(
                    (line for line in log.splitlines() if line.startswith("LIVE_URL ")),
                    None,
                )
                if port_line:
                    break
                if "Traceback" in log:
                    raise RuntimeError(log)
                time.sleep(0.2)
            else:
                raise RuntimeError(log or "Remote IMU preview did not start")
            server = tunnel(ssh, args.port, int(port_line.rsplit(":", 1)[1]))
            url = f"http://127.0.0.1:{server.server_address[1]}"
            opener = build_opener(ProxyHandler({}))
            for _ in range(30):
                try:
                    with opener.open(f"{url}/status", timeout=1):
                        break
                except OSError:
                    time.sleep(0.2)
            else:
                raise RuntimeError(run_remote(ssh, "cat", f"{directory}/preview.log"))
            worker = Thread(target=forward, args=(producer, url, stopping))
            worker.start()
            print(url, flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            while not stopping.wait(0.5) and ssh.get_transport().is_active():
                pass
        except KeyboardInterrupt:
            pass
        finally:
            stopping.set()
            if worker is not None:
                worker.join()
            if server is not None:
                server.shutdown()
                server.server_close()
            if ssh.get_transport().is_active():
                run_remote(
                    ssh,
                    "sh",
                    "-c",
                    f"cd {shlex.quote(directory)} && "
                    'if [ -s preview.pid ]; then kill -TERM "$(cat preview.pid)" 2>/dev/null || true; fi',
                )


if __name__ == "__main__":
    main()
