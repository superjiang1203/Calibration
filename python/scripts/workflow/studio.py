"""One local browser workspace for camera capture, calibration and IMU preview."""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from threading import Lock, Thread
from tempfile import mkdtemp
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

from ..remote import add_target_arguments, connect, load_profile
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
PATH_FIELDS = {"preset", "session", "calibration_dir", "geometry", "intrinsics",
               "frame", "camera_intrinsics", "factory_output", "imu_port"}


def launch_studio_browser(url, profile):
    # A private Chromium profile gives Studio an owned process, including on
    # browsers that refuse window.close() for tabs opened by the OS.
    candidates = [shutil.which(name) for name in
                  ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser")]
    if os.name == "nt":
        for key in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            root = os.environ.get(key)
            if root:
                candidates.extend(str(Path(root) / name) for name in
                                  ("Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe"))
    for candidate in dict.fromkeys(candidates):
        if candidate and Path(candidate).is_file():
            try:
                return subprocess.Popen([
                    candidate, f"--user-data-dir={profile}", "--no-first-run",
                    "--no-default-browser-check", f"--app={url}",
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                continue
    webbrowser.open(url)
    print("使用默认浏览器打开 Studio。浏览器可能要求手动关闭页签。", flush=True)
    return None


class RecentPaths:
    def __init__(self, path):
        self.path = path
        self.lock = Lock()

    def _read(self):
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return saved if isinstance(saved, dict) else {}

    def read(self):
        return {
            mode: {key: value for key, value in fields.items()
                   if key in PATH_FIELDS and isinstance(value, str)}
            for mode, fields in self._read().items() if mode in COMMANDS and isinstance(fields, dict)
        }

    def _write(self, saved):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, mode, key, value):
        if (not isinstance(mode, str) or mode not in COMMANDS
                or not isinstance(key, str) or key not in PATH_FIELDS or not isinstance(value, str)
                or len(value) > 4096 or any(c in value for c in "\0\r\n")):
            raise ValueError("文件路径设置无效")
        with self.lock:
            saved = self._read()
            fields = saved.get(mode, {})
            saved[mode] = dict(fields if isinstance(fields, dict) else {}, **{key: value})
            self._write(saved)

    def marker_resume(self):
        resume = self._read().get("marker_resume")
        if isinstance(resume, dict) and isinstance(resume.get("session"), str):
            return resume
        return None

    def save_marker_resume(self, resume, completed_sessions=()):
        with self.lock:
            saved = self._read()
            saved["marker_resume"] = resume
            marker = saved.get("marker")
            if isinstance(marker, dict) and marker.get("session") in completed_sessions:
                marker.pop("session")
            self._write(saved)


def find_marker_resume(root, list_names, read_metadata, saved=None):
    """Select only the latest marker session so completed runs clear the field."""
    if saved:
        path = root.joinpath(saved["session"])
        candidates = (path,)
    else:
        candidates = (root / "data" / name for name in sorted(list_names(root / "data"), reverse=True)
                      if re.fullmatch(r"\d{8}_\d{6}_\d{6}", name))
    for path in candidates:
        try:
            metadata = json.loads(read_metadata(path / "capture_metadata.json"))
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            if saved:
                return saved
            continue
        if (not isinstance(metadata, dict) or metadata.get("calibration_type") != "marker_geometry"
                or "capture_group" not in metadata):
            continue
        if metadata.get("calibration_complete") or (
            not saved and metadata.get("capture_complete") and "calibration_complete" not in metadata
        ):
            return None
        result = dict(saved or {}, session=str(path), capture_complete=bool(metadata.get("capture_complete")))
        if not saved:
            if isinstance(metadata.get("capture_config"), str):
                result["preset"] = metadata["capture_config"]
            sensors = metadata.get("sensors") or []
            if sensors and sensors[0].get("sensor_type") == "vcamera":
                result["camera"] = "vcamera"
        return result
    return None


class Studio:
    def __init__(self, args, config, recent=None):
        self.args, self.config = args, config
        self.lock = Lock()
        self.logs = deque(maxlen=100)
        self.process = None
        self.mode = self.url = self.exit_code = None
        self.phase = None
        self.session = self.result = None
        self.closing = False
        self.recent = recent
        self.marker_resume = recent.marker_resume() if recent else None
        self.marker_capture_options = None
        self.completed_marker_session = None
        self.marker_resume_revision = 0

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
                marker_resume=self.marker_resume,
                completed_marker_session=self.completed_marker_session,
            )
        snapshot.update(self.preview_state(snapshot["running"], snapshot["url"]))
        return snapshot

    def restore_marker_resume(self):
        with self.lock:
            revision, saved = self.marker_resume_revision, self.marker_resume
        try:
            if self.args.ssh:
                from paramiko import SSHException

                profile = load_profile(self.args.ssh_config, self.args.ssh)
                root = PurePosixPath(profile["project_dir"])
                try:
                    with connect(profile["host"], profile["user"], profile.get("password"),
                                 port=profile.get("port", 22)) as ssh, ssh.open_sftp() as sftp:
                        def read_metadata(path):
                            with sftp.open(str(path), "rb") as handle:
                                return handle.read().decode("utf-8-sig")
                        restored = find_marker_resume(root, lambda path: sftp.listdir(str(path)), read_metadata, saved)
                except SSHException:
                    return
            else:
                restored = find_marker_resume(
                    PROJECT_ROOT, lambda path: [entry.name for entry in path.iterdir()],
                    lambda path: path.read_text(encoding="utf-8-sig"), saved,
                )
        except (OSError, ValueError, KeyError, RuntimeError, EOFError, ImportError):
            return
        with self.lock:
            if self.marker_resume_revision == revision and restored != saved:
                completed = []
                if saved and restored is None:
                    completed.append(saved["session"])
                    manual = self.recent.read().get("marker", {}).get("session") if self.recent else None
                    if self.same_marker_session(manual, saved["session"]):
                        completed.append(manual)
                self.set_marker_resume(restored, completed)

    def set_marker_resume(self, resume, completed_sessions=()):
        self.marker_resume = resume
        self.marker_resume_revision += 1
        if self.recent:
            try:
                self.recent.save_marker_resume(resume, completed_sessions)
            except OSError as error:
                self.logs.append(f"保存 Marker 续采会话失败：{error}")

    @staticmethod
    def same_marker_session(a, b):
        if not a or not b:
            return False
        a, b = (PurePosixPath(value.replace("\\", "/")) for value in (a, b))
        return a == b or (not a.is_absolute() and a.parts == b.parts[-len(a.parts):]) or (
            not b.is_absolute() and b.parts == a.parts[-len(b.parts):])

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
              camera_intrinsics=None, factory_output=None, imu_source=None, imu_port=None):
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
            for path in (session, calibration_dir, preset, geometry, intrinsics, frame, camera_intrinsics, factory_output, imu_port):
                if path is not None and (not isinstance(path, str) or not path.strip() or any(c in path for c in "\0\r\n")):
                    raise ValueError("请输入有效路径")
            offline = command in ("calibrate", "verify", "detect")
            if offline and not session:
                raise ValueError("请输入已有采集会话路径")
            if session and not (offline or mode == "marker"):
                raise ValueError("该操作不支持会话路径")
            if bool(calibration_dir) != (command == "verify"):
                raise ValueError("复核需要指定标定结果目录")
            tracking = mode in PREVIEW_PROFILES and command == "capture"
            if ((mode not in ("imu", "pivot") and not tracking and (geometry or intrinsics))
                    or (mode == "imu" and preset) or (mode == "pivot" and intrinsics)):
                raise ValueError("参数与当前模式不匹配")
            if mode == "pivot" and not geometry:
                raise ValueError("请选择已经标定好的 Marker 几何 YAML。")
            if mode != "imu" and (imu_source is not None or imu_port is not None):
                raise ValueError("串口选项用于 IMU 预览。")
            if tracking and (bool(intrinsics) != (bool(geometry) and mode == "galaxy")):
                raise ValueError("Galaxy 追踪需要几何 YAML 和内参 YAML。RGB 预览自动读取 SDK 内参。")
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
                source = imu_source or config.get("source", "icm42670")
                if source not in ("icm42670", "rion_imu406") or (imu_port and source != "rion_imu406"):
                    raise ValueError("请选择有效的 IMU 数据源和串口。")
                entry = "python.scripts.imu_calibration.launch"
                arguments = [
                    "--no-browser",
                    "--port",
                    "0",
                    "--geometry",
                    geometry or config["geometry"],
                    "--camera",
                    camera or "galaxy",
                    "--imu-source",
                    source,
                    *location,
                ]
                if source == "rion_imu406":
                    rion = config.get("rion", {})
                    arguments.extend(("--imu-port", imu_port or rion.get("port") or "auto",
                                      "--imu-baud", str(rion.get("baud", 115200)),
                                      "--imu-address", str(rion.get("address", 0))))
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
                if geometry:
                    arguments.extend(("--tracking-geometry", geometry))
                    if intrinsics:
                        arguments.extend(("--tracking-intrinsics", intrinsics))
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
                if mode == "pivot":
                    arguments.extend(("--pivot-geometry", geometry.strip()))
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
            self.marker_capture_options = None
            if mode == "marker":
                self.completed_marker_session = None
                if command in ("run", "capture"):
                    self.marker_capture_options = {
                        "preset": preset or self.config[mode],
                        "camera": camera or "",
                        "camera_intrinsics": camera_intrinsics or "",
                    }
            Thread(target=self.monitor, args=(process,), daemon=True).start()

    def monitor(self, process):
        geometry_saved = False
        with process.stdout:
            for line in process.stdout:
                line = line.strip()
                with self.lock:
                    if self.mode == "marker" and line == "CALIBRATION_MARKER_PHASE=solve":
                        self.phase, self.url = "solve", None
                        if self.marker_resume and self.same_marker_session(self.marker_resume["session"], self.session):
                            self.set_marker_resume(dict(self.marker_resume, capture_complete=True))
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
                            if self.marker_capture_options is not None:
                                self.set_marker_resume(dict(self.marker_capture_options, session=self.session,
                                                            capture_complete=False))
                        if line.startswith(("Results: ", "Calibration results: ", "Saved geometry: ", "Saved OpenCV YAML: ")):
                            self.result = line.partition(": ")[2]
                        if self.mode == "marker" and line.startswith("Saved geometry: "):
                            geometry_saved = True
                        self.logs.append(line)
        code = process.wait()
        with self.lock:
            if code == 0 and geometry_saved and self.session:
                self.completed_marker_session = self.session
                resume = self.marker_resume
                completed = [self.session]
                if resume and self.same_marker_session(resume["session"], self.session):
                    completed.append(resume["session"])
                    resume = None
                if self.recent:
                    manual = self.recent.read().get("marker", {}).get("session")
                    if self.same_marker_session(manual, self.session):
                        completed.append(manual)
                self.set_marker_resume(resume, completed)
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
    scope = (str(args.config.resolve()), args.ssh, str(args.ssh_config) if args.ssh else None)
    scope_key = hashlib.sha256(json.dumps(scope).encode()).hexdigest()[:24]
    recent = RecentPaths(PROJECT_ROOT / "configs" / "studio.local" / f"{scope_key}.json")
    studio = Studio(args, json.loads(args.config.read_text(encoding="utf-8-sig")), recent)
    Thread(target=studio.restore_marker_resume, daemon=True).start()
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
                    "recent_paths": recent.read(),
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
                        "camera", "camera_intrinsics", "factory_output", "imu_source", "imu_port",
                    }:
                        raise ValueError("任务参数无效")
                    studio.start(self.path.removeprefix("/start/"), **options)
                elif self.path == "/recent-path":
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 8192:
                        raise ValueError("请求长度无效")
                    options = json.loads(self.rfile.read(size))
                    if not isinstance(options, dict) or options.keys() != {"mode", "key", "value"}:
                        raise ValueError("文件路径设置无效")
                    recent.save(**options)
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
        profile = None
        browser = None
        try:
            if not args.no_browser:
                profile = mkdtemp(prefix="calibration-studio-browser-")
                browser = launch_studio_browser(url, profile)
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                studio.stop(shutdown=True)
            finally:
                if browser is not None and browser.poll() is None:
                    browser.terminate()
                    try:
                        browser.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        browser.kill()
                        browser.wait()
                if profile is not None:
                    shutil.rmtree(profile, ignore_errors=True)
    return 0
