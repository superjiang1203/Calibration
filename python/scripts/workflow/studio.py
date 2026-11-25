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
from urllib.request import ProxyHandler, build_opener

from ..remote import add_target_arguments
from .config import PROJECT_ROOT


COMMANDS = {
    "galaxy": ("capture",),
    "intrinsics": ("run", "capture", "calibrate", "verify", "factory"),
    "stereo": ("run", "capture", "calibrate", "verify"),
    "marker": ("run", "capture", "calibrate", "detect"),
    "pivot": ("run", "capture", "calibrate"),
    "imu": ("preview",),
    "vcamera": ("capture", "devices", "inspect", "frame"),
}
PREVIEW_PROFILES = {"galaxy": "configs/cameras/galaxy.json",
                    "vcamera": "configs/cameras/vcamera_color.json"}
class Studio:
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.lock = Lock()
        self.logs = deque(maxlen=100)
        self.process = None
        self.mode = self.url = self.exit_code = None
        self.phase = None
        self.session = self.result = None
        self.closing = False

    def snapshot(self):
        with self.lock:
            snapshot = dict(
                mode=self.mode,
                url=self.url,
                running=self.process is not None,
                exit_code=self.exit_code,
                phase=self.phase,
                session=self.session,
                result=self.result,
                logs=list(self.logs),
            )
        snapshot.update(self.preview_state(snapshot["running"], snapshot["url"]))
        return snapshot

    @staticmethod
    def preview_state(running, url):
        if not running:
            return {"camera_locked": False, "settings_pending": False}
        if url:
            try:
                with build_opener(ProxyHandler({})).open(url + "status", timeout=0.8) as response:
                    data = json.load(response)
                pending = bool(data.get("settings_pending"))
                return {"camera_locked": bool(data["camera_active"]) or pending,
                        "settings_pending": pending}
            except (OSError, ValueError, KeyError):
                pass
        return {"camera_locked": True, "settings_pending": False}

    def switch_task(self):
        state = self.snapshot()
        if state["camera_locked"]:
            raise ValueError("请先断开当前任务的所有相机，并等待参数调整完成。")
        with self.lock:
            self.phase = "switched"
        self.stop(shutdown=True)

    def start(self, mode, *, command=None, session=None, calibration_dir=None,
              preset=None, geometry=None, intrinsics=None, frame=None, camera=None,
              camera_intrinsics=None, factory_output=None):
        with self.lock:
            if self.closing:
                raise ValueError("工作台正在退出。")
            if self.process is not None:
                raise ValueError("请先结束当前采集或关闭 IMU 预览，等待相机释放。")
            if mode not in COMMANDS:
                raise ValueError("未知模式")
            command = COMMANDS[mode][0] if command is None else command
            if command not in COMMANDS[mode]:
                raise ValueError("该模式不支持此操作")
            for path in (session, calibration_dir, preset, geometry, intrinsics, frame, camera_intrinsics, factory_output):
                if path is not None and (not isinstance(path, str) or not path.strip() or any(c in path for c in "\0\r\n")):
                    raise ValueError("请输入有效路径")
            offline = command in ("calibrate", "verify", "detect")
            if offline and not session:
                raise ValueError("请输入已有采集会话路径")
            if session and not (offline or mode == "marker"):
                raise ValueError("该操作不支持会话路径")
            if bool(calibration_dir) != (command == "verify"):
                raise ValueError("复核需要指定标定结果目录")
            if (mode != "imu" and (geometry or intrinsics)) or (mode == "imu" and preset):
                raise ValueError("参数与当前模式不匹配")
            if bool(frame) != (mode == "vcamera" and command == "frame"):
                raise ValueError("保存测试帧需要指定 PNG 路径")
            if camera is not None and (camera not in ("galaxy", "vcamera") or mode not in ("intrinsics", "marker", "pivot", "imu")):
                raise ValueError("此任务不支持该相机选项")
            capture_override = mode in ("marker", "pivot") and command in ("run", "capture") and camera is not None
            if bool(camera_intrinsics) != (capture_override and camera == "galaxy"):
                raise ValueError("显式选择 Galaxy 时请输入配套内参 YAML；Vcamera 从 SDK 读取")
            if (command == "factory" and camera != "vcamera") or bool(factory_output) != (command == "factory"):
                raise ValueError("导出图漾出厂内参需要指定 YAML 输出路径")
            if mode == "intrinsics" and camera == "vcamera" and command not in ("factory", "capture"):
                raise ValueError("Vcamera 普通内参使用 SDK 出厂参数")
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
                    geometry or config["geometry"],
                    "--camera",
                    camera or "galaxy",
                    *location,
                ]
                if camera != "vcamera":
                    arguments.extend(("--intrinsics", intrinsics or config["intrinsics"]))
                if self.args.ssh:
                    arguments.extend(
                        ("--remote-port", str(config.get("remote_port", 0)))
                    )
                if config.get("serial"):
                    arguments.extend(("--serial", config["serial"]))
            elif mode == "intrinsics" and command == "factory":
                entry = "python.scripts.calibrate"
                arguments = ["inspect-vcamera", "--config", preset or PREVIEW_PROFILES["vcamera"],
                             "--factory-intrinsics", factory_output, *location]
            elif (mode in PREVIEW_PROFILES or mode == "intrinsics" and camera == "vcamera") and command == "capture":
                entry = "python.scripts.calibrate"
                profile = PREVIEW_PROFILES["vcamera"] if mode == "intrinsics" else self.config.get(mode, PREVIEW_PROFILES[mode])
                arguments = ["capture", "--config", preset or profile, *location]
            elif mode == "vcamera":
                entry = "python.scripts.calibrate"
                arguments = ["inspect-vcamera", *location]
                if command == "devices":
                    arguments.append("--list-devices")
                else:
                    arguments.extend(("--config", preset or self.config.get(mode, PREVIEW_PROFILES[mode])))
                if frame:
                    arguments.extend(("--save-frame", frame))
            else:
                entry = "python.scripts.calibrate"
                arguments = [
                    command,
                    "--config",
                    preset or self.config[mode],
                    *location,
                ]
                if session is not None:
                    arguments.extend(("--session", session.strip()))
                if calibration_dir:
                    arguments.extend(("--calibration-dir", calibration_dir))
                if capture_override:
                    arguments.extend(("--camera", camera))
                    if camera_intrinsics:
                        arguments.extend(("--camera-intrinsics", camera_intrinsics))
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
            self.phase = "solve" if offline else None
            self.session, self.result = session, None
            Thread(target=self.monitor, args=(process,), daemon=True).start()

    def monitor(self, process):
        with process.stdout:
            for line in process.stdout:
                line = line.strip()
                with self.lock:
                    if self.mode == "marker" and line == "CALIBRATION_MARKER_PHASE=solve":
                        self.phase, self.url = "solve", None
                    elif self.mode == "marker" and line == "CALIBRATION_MARKER_PHASE=paused":
                        self.phase, self.url = "paused", None
                    elif line.startswith("CALIBRATION_CAPTURE_PORT="):
                        self.url = (
                            "http://127.0.0.1:" + str(int(line.split("=")[1])) + "/"
                        )
                    elif match := re.fullmatch(
                        r"(?:Remote capture: |LIVE_URL )?(http://127\.0\.0\.1:\d+)/?",
                        line,
                    ):
                        self.url = match[1] + "/"
                    else:
                        if line.startswith("Capture session: "):
                            self.session = line.partition(": ")[2]
                        if line.startswith(("Results: ", "Calibration results: ", "Saved geometry: ", "Saved OpenCV YAML: ")):
                            self.result = line.partition(": ")[2]
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
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def shutdown(self):
        with self.lock:
            if self.closing:
                raise ValueError("工作台正在退出。")
            self.closing = True
        self.stop(shutdown=True)


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
            elif self.path == "/presets":
                self.send(200, json.dumps({
                    "target": args.ssh or "本机", "presets": {**PREVIEW_PROFILES, **studio.config},
                }).encode(), "application/json")
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
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 <= size <= 8192:
                        raise ValueError("请求长度无效")
                    options = json.loads(self.rfile.read(size)) if size else {}
                    if not isinstance(options, dict) or options.keys() - {
                        "command", "session", "calibration_dir", "preset", "geometry", "intrinsics", "frame",
                        "camera", "camera_intrinsics", "factory_output",
                    }:
                        raise ValueError("任务参数无效")
                    studio.start(self.path.removeprefix("/start/"), **options)
                elif self.path == "/stop":
                    studio.stop()
                elif self.path == "/switch":
                    studio.switch_task()
                elif self.path == "/shutdown":
                    try:
                        studio.shutdown()
                        self.send(200, b"{}", "application/json")
                    finally:
                        if studio.closing:
                            Thread(target=self.server.shutdown, daemon=True).start()
                    return
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
