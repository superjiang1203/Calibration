"""One local browser workspace for camera capture, calibration and IMU preview."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

from ..remote import add_target_arguments
from .config import PROJECT_ROOT


class Studio:
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.lock = Lock()
        self.logs = deque(maxlen=100)
        self.process = None
        self.mode = self.url = self.exit_code = None

    def snapshot(self):
        with self.lock:
            return dict(
                mode=self.mode,
                url=self.url,
                running=self.process is not None,
                exit_code=self.exit_code,
                logs=list(self.logs),
            )

    def start(self, mode):
        with self.lock:
            if self.process is not None:
                raise ValueError("请先结束当前采集或关闭 IMU 预览，等待相机释放。")
            if mode not in ("camera", "intrinsics", "stereo", "marker", "pivot", "imu"):
                raise ValueError("未知模式")
            location = (
                [
                    "--ssh",
                    self.args.ssh,
                    "--ssh-config",
                    str(self.args.ssh_config.resolve()),
                ]
                if self.args.ssh
                else ["--local"]
            )
            if mode == "imu":
                config = self.config[mode]
                entry = "python.scripts.imu_calibration.launch"
                arguments = [
                    "--no-browser",
                    "--port",
                    "0",
                    "--geometry",
                    config["geometry"],
                    "--intrinsics",
                    config["intrinsics"],
                    *location,
                ]
                if self.args.ssh:
                    arguments.extend(
                        ("--remote-port", str(config.get("remote_port", 0)))
                    )
                if config.get("serial"):
                    arguments.extend(("--serial", config["serial"]))
            else:
                entry = "python.scripts.calibrate"
                arguments = [
                    "capture" if mode == "camera" else "run",
                    "--config",
                    self.config[mode],
                    *location,
                ]
            code = (
                "import runpy,signal,sys\n"
                "def stop(*args): raise KeyboardInterrupt\n"
                "if hasattr(signal,'SIGBREAK'): signal.signal(signal.SIGBREAK,stop)\n"
                "runpy.run_module(sys.argv.pop(1),run_name='__main__')"
            )
            group = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                if os.name == "nt"
                else {"start_new_session": True}
            )
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", code, entry, *arguments],
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={
                    **os.environ,
                    "CALIBRATION_EMBEDDED": "1",
                    "CALIBRATION_CAPTURE_WEB": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
                **group,
            )
            self.process, self.mode, self.url, self.exit_code = (
                process,
                mode,
                None,
                None,
            )
            self.logs.clear()
            Thread(target=self.monitor, args=(process,), daemon=True).start()

    def monitor(self, process):
        with process.stdout:
            for line in process.stdout:
                line = line.strip()
                with self.lock:
                    if line.startswith("CALIBRATION_CAPTURE_PORT="):
                        self.url = (
                            "http://127.0.0.1:" + str(int(line.split("=")[1])) + "/"
                        )
                    elif match := re.fullmatch(
                        r"(?:Remote capture: |LIVE_URL )?(http://127\.0\.0\.1:\d+)/?",
                        line,
                    ):
                        self.url = match[1] + "/"
                    else:
                        self.logs.append(line)
        code = process.wait()
        with self.lock:
            self.process, self.url, self.exit_code = None, None, code

    def stop(self, *, shutdown=False):
        with self.lock:
            process = self.process
            if process is None:
                return
            if self.mode != "imu" and not shutdown:
                raise ValueError("在采集页点击「结束采集」，以保存会话并完成当前任务。")
            if process.poll() is None:
                process.send_signal(
                    signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
                )
        if shutdown:
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.terminate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_arguments(parser, PROJECT_ROOT)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/studio.json"
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    studio = Studio(args, json.loads(args.config.read_text(encoding="utf-8-sig")))
    page = Path(__file__).with_suffix(".html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, content, kind):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == "/":
                self.send(200, page, "text/html; charset=utf-8")
            elif self.path == "/status":
                self.send(
                    200, json.dumps(studio.snapshot()).encode(), "application/json"
                )
            else:
                self.send_error(404)

        def do_POST(self):
            if (
                self.headers.get("Origin")
                != f"http://127.0.0.1:{self.server.server_port}"
            ):
                self.send_error(403)
                return
            try:
                if self.path.startswith("/start/"):
                    studio.start(self.path.removeprefix("/start/"))
                elif self.path == "/stop":
                    studio.stop()
                else:
                    self.send_error(404)
                    return
            except (ValueError, KeyError, OSError) as error:
                self.send(
                    409, json.dumps({"error": str(error)}).encode(), "application/json"
                )
                return
            self.send(202, b"{}", "application/json")

    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Calibration workspace: {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            studio.stop(shutdown=True)
    return 0
