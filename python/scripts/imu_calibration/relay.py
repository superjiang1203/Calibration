"""Forward attitude packets from a local producer to the live page."""

import json
import socket
import subprocess
import threading
import time
from collections import deque
from urllib.parse import urlsplit


def forward(command, url, stopping):
    address = urlsplit(url)
    if address.scheme != "http":
        raise ValueError("Use HTTP over the SSH tunnel")
    pending = deque(maxlen=256)
    lock = threading.Lock()
    changed = threading.Event()
    arrivals = {}
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1
    )

    def receive():
        for line in process.stdout:
            try:
                packet = json.loads(line)
                now = time.monotonic()
                with lock:
                    if packet["type"] == "attitude":
                        ticks = arrivals.setdefault(packet["source"], deque(maxlen=250))
                        ticks.append(now)
                        packet["sdk_receive_hz"] = (
                            (len(ticks) - 1) / (ticks[-1] - ticks[0])
                            if len(ticks) > 1 and ticks[-1] > ticks[0]
                            else 0.0
                        )
                    pending.append((packet, now))
                changed.set()
            except (ValueError, KeyError):
                pass
        stopping.set()

    threading.Thread(target=receive, daemon=True).start()
    sent = 0
    last_log = 0
    connection = None
    try:
        while not stopping.is_set():
            try:
                if connection is None:
                    connection = socket.create_connection(
                        (address.hostname, address.port or 80), timeout=2
                    )
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    connection.sendall(
                        (
                            "POST /imu/ingest HTTP/1.1\r\nHost: "
                            + address.netloc
                            + "\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\n\r\n"
                        ).encode()
                    )
                    print("IMU_STREAM_CONNECTED", flush=True)
                changed.wait(0.1)
                changed.clear()
                with lock:
                    batch = list(pending)
                    pending.clear()
                # Continuous chunked stream: no request/response round trip per sample.
                for packet, received in batch:
                    age = (time.monotonic() - received) * 1000
                    if age > 300:
                        continue
                    packet["relay_age_ms"] = round(age, 2)
                    data = json.dumps(packet, separators=(",", ":")).encode() + b"\n"
                    connection.sendall(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                    sent += 1
            except OSError as error:
                if connection is not None:
                    connection.close()
                    connection = None
                if time.monotonic() - last_log > 3:
                    print("IMU_FORWARD_WAIT", str(error), flush=True)
                    last_log = time.monotonic()
                stopping.wait(0.25)
            if sent and time.monotonic() - last_log > 5:
                print("IMU_FORWARDED", sent, flush=True)
                last_log = time.monotonic()
    finally:
        if connection is not None:
            connection.close()
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
