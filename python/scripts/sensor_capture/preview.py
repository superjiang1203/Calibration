"""Capture controls shared by local OpenCV windows and SSH browser capture."""

import json
import os
import sys
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread

import cv2


def require_gui_support():
    gui = next((line.split(':', 1)[1].strip() for line in
                cv2.getBuildInformation().splitlines() if line.strip().startswith('GUI:')), 'UNKNOWN')
    if gui in ('NONE', 'NO', ''):
        raise RuntimeError(f'OpenCV has no GUI support (GUI={gui}). Current Python: {sys.executable}. '
                           'Install GUI-enabled OpenCV in this environment using '
                           'python -m pip install -r requirements-calibration.txt.')


class CaptureView:
    def __init__(self):
        self.web = os.environ.get("CALIBRATION_CAPTURE_WEB") == "1"
        self.images = {}
        self.jpeg = None
        self.published = 0.0
        self.save = Event()
        self.record = Event()
        self.last_auto_save = float("-inf")
        self.finish = Event()
        self.next_group = Event()
        self.group_status = None
        self.server = None
        self.requests = Queue()
        self.lock = Lock()
        self.logs = deque(maxlen=60)
        self.state = "disconnected"
        self.controls = {}

    def log(self, message):
        with self.lock:
            self.logs.append({"time": time.time(), "message": message})
        print(message, flush=True)

    def __enter__(self):
        if not self.web:
            require_gui_support()
            return self

        page = Path(__file__).with_name("preview.html").read_bytes()
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.path == "/":
                    content, kind = page, "text/html; charset=utf-8"
                elif self.path.partition("?")[0] == "/frame.jpg":
                    content, kind = view.jpeg, "image/jpeg"
                elif self.path == "/status":
                    with view.lock:
                        status = dict(
                            state=view.state,
                            logs=list(view.logs),
                            controls=view.controls,
                            recording=view.record.is_set(),
                            has_frame=view.jpeg is not None,
                            group=view.group_status,
                        )
                    status["frame_revision"] = view.published
                    content, kind = json.dumps(status).encode(), "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200 if content else 204)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if content:
                    self.wfile.write(content)

            def do_POST(self):
                if self.path in ("/connect", "/disconnect", "/settings"):
                    payload = None
                    if self.path == "/settings":
                        try:
                            size = int(self.headers.get("Content-Length", "0"))
                            if not 0 < size <= 1024:
                                raise ValueError
                            payload = json.loads(self.rfile.read(size))
                        except (ValueError, TypeError):
                            self.send_error(400)
                            return
                    view.requests.put((self.path, payload))
                    self.send_response(202)
                    self.end_headers()
                    return
                if self.path not in (
                    "/save",
                    "/finish",
                    "/record/start",
                    "/record/stop",
                    "/group/next",
                ):
                    self.send_error(404)
                    return
                if self.path in ("/save", "/record/start") and view.jpeg is None:
                    self.send_error(409, "Waiting for camera frames")
                    return
                if self.path == "/group/next" and not (view.group_status and view.group_status["ready"]):
                    self.send_error(409, "Current group is incomplete")
                    return
                if self.path == "/record/start" and view.group_status is not None:
                    self.send_error(409, "Grouped capture uses individual shots")
                    return
                self.send_response(204)
                self.end_headers()
                if self.path == "/record/start":
                    view.record.set()
                elif self.path == "/record/stop":
                    view.record.clear()
                elif self.path == "/group/next":
                    view.next_group.set()
                else:
                    (view.save if self.path == "/save" else view.finish).set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"CALIBRATION_CAPTURE_PORT={self.server.server_port}", flush=True)
        if os.environ.get("CALIBRATION_CAPTURE_LOCAL_BROWSER") == "1":
            url = f"http://127.0.0.1:{self.server.server_port}/"
            print(f"Camera capture: {url}", flush=True)
            webbrowser.open(url)
        return self

    def prepare(self, adapters, opener):
        """Perform all device operations on the capture thread."""
        if not self.web:
            if not adapters:
                adapters.extend(opener())
            return bool(adapters)
        try:
            action, payload = self.requests.get_nowait()
        except Empty:
            return bool(adapters)
        if action == "/connect" and not adapters and not self.finish.is_set():
            started = time.monotonic()
            self.state = "connecting"
            self.log("正在初始化相机 SDK 并打开配置的设备…")
            try:
                adapters.extend(opener())
                self.state = "connected"
                self.log(f"相机已打开，等待首帧（{time.monotonic() - started:.2f} 秒）")
                self.refresh_controls(adapters)
            except Exception as error:
                self.state = "failed"
                self.log(
                    f"相机连接失败（{time.monotonic() - started:.2f} 秒）：{error}"
                )
        elif action == "/disconnect" and adapters:
            self.state = "disconnecting"
            self.record.clear()
            self.save.clear()
            self.jpeg = None
            self.images.clear()
            for adapter, _ in reversed(adapters):
                try:
                    adapter.close()
                except Exception as error:
                    self.log(f"关闭相机失败：{error}")
            adapters.clear()
            self.controls = {}
            self.state = "disconnected"
            self.log("已结束采集流并关闭相机")
        elif action == "/settings":
            try:
                label, field = payload["label"], payload["field"]
                if field not in ("exposure_us", "gain") or label not in self.controls:
                    raise ValueError("此设备或参数不支持实时调节")
                adapter = next(
                    adapter for adapter, group in adapters if group[0]["label"] == label
                )
                adapter.set_live_setting(field, float(payload["value"]))
                self.refresh_controls(adapters)
                self.log(
                    f"{label} {field} 回读值：{self.controls[label][field]['cur']}"
                )
            except Exception as error:
                self.log(f"修改相机参数失败：{error}")
        return bool(adapters)

    def refresh_controls(self, adapters):
        controls = {}
        for adapter, group in adapters:
            if not hasattr(adapter, "live_settings"):
                continue
            try:
                values = adapter.live_settings()
                if values:
                    controls[group[0]["label"]] = values
            except Exception as error:
                self.log(f"读取曝光和增益失败：{error}")
        self.controls = controls

    def __exit__(self, *_args):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        elif not self.web:
            cv2.destroyAllWindows()

    def open_window(self, title, width, height):
        if not self.web:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, width, height)

    def show(self, title, image):
        self.images[title] = image if self.web else None
        if not self.web:
            cv2.imshow(title, image)

    def key(self):
        if not self.web:
            key = cv2.waitKey(1) & 0xFF
            if any(
                cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
                for title in self.images
            ):
                return 27
            return key

        if self.finish.is_set():
            return 27
        if self.next_group.is_set():
            self.next_group.clear()
            return ord("n")
        now = time.monotonic()
        if self.images and now - self.published >= 0.15:
            previews = []
            images = list(self.images.values())
            height = min(600, *(image.shape[0] for image in images))
            for image in images:
                width = max(1, round(image.shape[1] * height / image.shape[0]))
                resized = cv2.resize(
                    image, (width, height), interpolation=cv2.INTER_AREA
                )
                previews.append(
                    cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                    if resized.ndim == 2
                    else resized
                )
            ok, jpeg = cv2.imencode(".jpg", cv2.hconcat(previews))
            if ok:
                first_preview = self.jpeg is None
                self.jpeg, self.published = jpeg.tobytes(), now
                if first_preview:
                    self.log("已收到首张预览图")
        if self.save.is_set():
            self.save.clear()
            self.last_auto_save = now
            return ord(" ")
        if self.record.is_set() and self.images and now - self.last_auto_save >= 1.25:
            self.last_auto_save = now
            return ord(" ")
        return -1
