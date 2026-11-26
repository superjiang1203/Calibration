"""Capture controls shared by local OpenCV windows and SSH browser capture."""

import json
import os
import sys
import time
from collections import deque
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

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
        self.frames = {}
        self.published = 0.0
        self.next_preview = 0.0
        self.preview_timing = None
        self.save = Event()
        self.video_requested = Event()
        self.video_directory = None
        self.video_writer = None
        self.video_path = None
        self.video_temporary = None
        self.video_previous = None
        self.video_start = 0.0
        self.video_frames = 0
        self.finish = Event()
        self.next_group = Event()
        self.group_status = None
        self.marker_distances = None
        self.marker_save_feedback = None
        self.server = None
        self.requests = Queue()
        self.lock = Lock()
        self.logs = deque(maxlen=60)
        self.state = "disconnected"
        self.controls = {}
        self.title = "相机采集"
        self.device_states = {}
        self.pair_ready = None
        self.required_frames = 1
        self.camera_slots = []
        self.settings_pending = False
        self.connection_pending = False

    def can_save(self):
        return len(self.frames) >= self.required_frames and self.pair_ready is not False

    def log(self, message):
        with self.lock:
            self.logs.append({"time": time.time(), "message": message})
        print(message, flush=True)

    def __enter__(self):
        if not self.web:
            require_gui_support()
            return self

        page = Path(__file__).with_name("preview.html").read_bytes()
        zoom_script = Path(__file__).with_name("image_zoom.js").read_bytes()
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                revision = None
                if self.path == "/":
                    content, kind = page, "text/html; charset=utf-8"
                elif self.path == "/image_zoom.js":
                    content, kind = zoom_script, "text/javascript; charset=utf-8"
                elif self.path.partition("?")[0] == "/frame.jpg":
                    query = parse_qs(urlsplit(self.path).query)
                    name = query.get("camera", [None])[0]
                    with view.lock:
                        frames, revision = view.frames, str(view.published)
                    content = frames.get(name) if name is not None else next(iter(frames.values()), None)
                    if query.get("after", [None])[0] == revision:
                        content = None
                    kind = "image/jpeg"
                elif self.path == "/status":
                    with view.lock:
                        status = dict(
                            state=view.state,
                            logs=list(view.logs),
                            controls=view.controls,
                            video_enabled=view.video_directory is not None,
                            video_recording=view.video_writer is not None,
                            video_requested=view.video_requested.is_set(),
                            has_frame=bool(view.frames),
                            cameras=list(view.frames),
                            camera_slots=view.camera_slots,
                            frame_revision=view.published,
                            preview_timing=view.preview_timing,
                            group=view.group_status,
                            marker_distances=view.marker_distances,
                            marker_save_feedback=view.marker_save_feedback,
                            title=view.title,
                            devices=dict(view.device_states),
                            can_save=view.can_save(),
                            settings_pending=view.settings_pending,
                            camera_active=view.connection_pending or (
                                any(value in ("connected", "connecting", "disconnecting") for value in view.device_states.values())
                                if view.device_states else view.state in ("connected", "connecting", "disconnecting")
                            ),
                        )
                    content, kind = json.dumps(status).encode(), "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200 if content else 204)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                if revision is not None:
                    self.send_header("X-Frame-Revision", revision)
                self.end_headers()
                if content:
                    self.wfile.write(content)

            def do_POST(self):
                if view.settings_pending:
                    self.send_error(409, "Waiting for camera setting readback")
                    return
                if self.path in ("/connect", "/disconnect", "/settings"):
                    payload = None
                    if self.path == "/settings" or self.headers.get("Content-Length", "0") != "0":
                        try:
                            size = int(self.headers.get("Content-Length", "0"))
                            if not 0 < size <= 1024:
                                raise ValueError
                            payload = json.loads(self.rfile.read(size))
                            if not isinstance(payload, dict):
                                raise ValueError("Expected an object")
                        except (ValueError, TypeError):
                            self.send_error(400)
                            return
                    if self.path == "/settings":
                        result = Future()
                        with view.lock:
                            if view.settings_pending or view.connection_pending:
                                self.send_error(409, "Camera operation in progress")
                                return
                            view.settings_pending = True
                        view.requests.put((self.path, (payload, result)))
                        try:
                            status, body = 200, {"settings": result.result(timeout=20)}
                        except Exception as error:
                            status, body = 409, {"error": str(error) or "等待设备回读超时"}
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(json.dumps(body).encode())
                        return
                    with view.lock:
                        if view.connection_pending or view.settings_pending:
                            self.send_error(409, "Camera operation in progress")
                            return
                        view.connection_pending = True
                    view.requests.put((self.path, payload))
                    self.send_response(202)
                    self.end_headers()
                    return
                if self.path not in (
                    "/save",
                    "/finish",
                    "/group/next",
                    "/video/start",
                    "/video/stop",
                ):
                    self.send_error(404)
                    return
                if self.path == "/save" and not view.can_save():
                    self.send_error(409, "Waiting for all required camera frames")
                    return
                if self.path == "/group/next" and not (view.group_status and view.group_status["ready"]):
                    self.send_error(409, "Current group is incomplete")
                    return
                if self.path.startswith("/video/"):
                    if view.video_directory is None or (self.path == "/video/start" and not view.can_save()):
                        self.send_error(409, "Video recording requires a ready camera preview")
                        return
                    if self.path == "/video/start":
                        view.video_requested.set()
                    else:
                        view.video_requested.clear()
                    self.send_response(204)
                    self.end_headers()
                    return
                self.send_response(204)
                self.end_headers()
                if self.path == "/group/next":
                    view.next_group.set()
                else:
                    (view.save if self.path == "/save" else view.finish).set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"CALIBRATION_CAPTURE_PORT={self.server.server_port}", flush=True)
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
            self.save.clear()
            with self.lock:
                self.frames = {}
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
            self.complete_setting(payload, adapters)
        if action in ("/connect", "/disconnect"):
            self.connection_pending = False
        return bool(adapters)

    def prepare_sources(self, adapters, groups):
        """Manage each independently openable device on the capture thread."""
        from .sources import open_sources

        if not self.web:
            return self.prepare(adapters, lambda: open_sources([s for group in groups.values() for s in group]))
        try:
            action, payload = self.requests.get_nowait()
        except Empty:
            return bool(adapters)
        if action == "/settings":
            self.complete_setting(payload, adapters)
            return bool(adapters)
        selected = (payload or {}).get("device")
        for key, sensors in groups.items():
            if selected is not None and key != selected:
                continue
            labels = {sensor["label"] for sensor in sensors}
            current = next((pair for pair in adapters if pair[1][0]["label"] in labels), None)
            if action == "/connect" and current is None:
                self.device_states[key] = "connecting"
                try:
                    adapters.extend(open_sources(sensors))
                    self.device_states[key] = "connected"
                    self.log(f"{key} 已连接，等待首帧。")
                except Exception as error:
                    self.device_states[key] = "failed"
                    self.log(f"{key} 连接失败：{error}")
            elif action == "/disconnect" and current is not None:
                self.stop_video()
                self.save.clear()
                self.pair_ready = False
                adapters.remove(current)
                self.device_states[key] = "disconnected"
                try:
                    current[0].close()
                except Exception as error:
                    self.log(f"{key} 关闭失败：{error}")
                for label in labels:
                    name = f"Sensor calibration: {label}"
                    self.images.pop(name, None)
                    with self.lock:
                        self.frames = {title: frame for title, frame in self.frames.items() if title != name}
        self.refresh_controls(adapters)
        self.state = ("connected" if adapters else
                      "failed" if "failed" in self.device_states.values() else "disconnected")
        self.connection_pending = False
        return bool(adapters)

    def complete_setting(self, request, adapters):
        payload, result = request
        try:
            label, field = payload["label"], payload["field"]
            if field not in ("exposure_us", "gain") or label not in self.controls:
                raise ValueError("此设备或参数不支持实时调节")
            adapter = next(adapter for adapter, group in adapters if group[0]["label"] == label)
            adapter.set_live_setting(field, float(payload["value"]))
            self.refresh_controls(adapters)
            self.log(f"{label} {field} 回读值：{self.controls[label][field]['cur']}")
        except Exception as error:
            self.log(f"修改相机参数失败：{error}")
            self.settings_pending = False
            result.set_exception(error)
        else:
            self.settings_pending = False
            result.set_result(self.controls)

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
        if self.video_requested.is_set():
            try:
                self.write_video_frame(image)
            except (OSError, RuntimeError, cv2.error) as error:
                self.stop_video(save=False)
                self.log(f"录像失败：{error}")
        self.images[title] = image if self.web else None
        if not self.web:
            cv2.imshow(title, image)

    def write_video_frame(self, image):
        fps = 20
        now = time.monotonic()
        width, height = image.shape[1::-1]
        size = (width - width % 2, height - height % 2)
        frame = image[:size[1], :size[0]]
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = frame.copy(order="C")
        if self.video_writer is None:
            self.video_directory.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"preview_{stamp}_{uuid4().hex[:8]}"
            self.video_path = self.video_directory / f"{filename}.mp4"
            self.video_temporary = self.video_directory / f"{filename}.partial.mp4"
            writer = cv2.VideoWriter(str(self.video_temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not writer.isOpened():
                writer.release()
                raise RuntimeError("无法创建 MP4 文件，请检查 OpenCV 视频后端和目录写入权限。")
            self.video_writer = writer
            self.video_start = now
            self.video_frames = 0
            self.video_previous = frame
            self.log("已开始录制相机预览。")
            print(f"Capture session: {self.video_directory}", flush=True)
        if frame.shape != self.video_previous.shape:
            self.stop_video()
            self.log("画面分辨率变化，当前录像已结束。")
            return
        target = int((now - self.video_start) * fps)
        while self.video_frames < target:
            self.video_writer.write(self.video_previous)
            self.video_frames += 1
        if self.video_frames == target:
            self.video_writer.write(frame)
            self.video_frames += 1
        self.video_previous = frame

    def stop_video(self, *, save=True):
        self.video_requested.clear()
        writer, self.video_writer = self.video_writer, None
        if writer is not None:
            writer.release()
        if self.video_temporary is None:
            return
        try:
            if save and writer is not None:
                self.video_temporary.replace(self.video_path)
                self.log(f"MP4 已保存：{self.video_path}")
            else:
                self.video_temporary.unlink(missing_ok=True)
        except OSError as error:
            self.log(f"MP4 文件处理失败：{error}")
        finally:
            self.video_temporary = self.video_path = self.video_previous = None

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
        if self.settings_pending:
            return -1
        if self.next_group.is_set():
            self.next_group.clear()
            return ord("n")
        now = time.monotonic()
        if self.images and now >= self.next_preview:
            interval = 1 / 30
            self.next_preview = max(now + interval / 2, self.next_preview + interval)
            frames = {}
            for name, image in self.images.items():
                ok, jpeg = cv2.imencode(".jpg", image)
                if ok:
                    frames[name] = jpeg.tobytes()
            if frames:
                first_preview = not self.frames
                with self.lock:
                    self.frames, self.published = frames, now
                if first_preview:
                    self.log("已收到首张预览图")
        if self.save.is_set():
            self.save.clear()
            return ord(" ")
        return -1
