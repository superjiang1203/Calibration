"""Read IMU406 attitude over the manufacturer's RION 68 serial protocol."""

import argparse
import json
import math
import time


SOURCE = "rion_imu406"
FRAME_SIZE = 32


def decode_bcd(data, scale):
    digits = [nibble for byte in data for nibble in (byte >> 4, byte & 15)]
    sign, *magnitude = digits
    if sign not in (0, 1) or any(digit > 9 for digit in magnitude):
        raise ValueError("Invalid RION signed BCD")
    value = 0
    for digit in magnitude:
        value = value * 10 + digit
    return (-value if sign else value) / scale


def attitude_quaternion(angles):
    """Convert vendor roll, pitch, yaw to WXYZ using Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = (math.radians(angle) / 2 for angle in angles)
    cr, cp, cy = map(math.cos, (roll, pitch, yaw))
    sr, sp, sy = map(math.sin, (roll, pitch, yaw))
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


class FrameDecoder:
    def __init__(self, address):
        self.address = address
        self.buffer = bytearray()

    def feed(self, data):
        self.buffer.extend(data)
        while self.buffer:
            start = self.buffer.find(b"\x68")
            if start < 0:
                self.buffer.clear()
                break
            del self.buffer[:start]
            if len(self.buffer) < 4:
                break
            if self.buffer[1:4] != bytes((0x1F, self.address, 0x84)):
                del self.buffer[0]
                continue
            if len(self.buffer) < FRAME_SIZE:
                break
            frame = self.buffer[:FRAME_SIZE]
            if sum(frame[1:-1]) & 0xFF != frame[-1]:
                del self.buffer[0]
                continue
            del self.buffer[:FRAME_SIZE]
            try:
                values = [decode_bcd(frame[i:i + 3], 1000 if 13 <= i < 22 else 100)
                          for i in range(4, 31, 3)]
            except ValueError:
                continue
            yield {
                "type": "attitude", "source": SOURCE, "valid": True,
                "host_timestamp_ms": time.monotonic_ns() // 1_000_000,
                "quaternion_wxyz": attitude_quaternion(values[:3]),
                "acceleration_g": values[3:6], "angular_velocity_deg_s": values[6:],
            }


def select_port(requested, ports):
    if requested != "auto":
        return requested
    usb = [port.device for port in ports if port.vid is not None]
    if len(usb) == 1:
        return usb[0]
    if usb:
        raise ValueError("检测到多个 USB 串口，请指定 IMU406 的串口：" + ", ".join(usb))
    raise ValueError("未检测到 USB 串口，请连接 IMU406 的 USB 转串口适配器")


def read_stream(device, address, status, *, clock=time.monotonic):
    decoder = FrameDecoder(address)
    request = bytes((0x68, 4, address, 4, (8 + address) & 0xFF))
    last_frame = clock()
    next_poll = last_frame + 0.5
    next_status = last_frame + 1
    polling = False
    while True:
        data = device.read(min(2048, device.in_waiting or 1))
        now = clock()
        for packet in decoder.feed(data):
            last_frame = now
            yield packet
        # Allow automatic output to run. Query silent devices without
        # changing their configured baud rate, output mode or reference.
        if now >= next_poll and (polling or now - last_frame >= 0.5):
            polling = True
            device.write(request)
            next_poll = now + 0.02
        if now >= next_status:
            if now - last_frame >= 1:
                status("waiting", "串口已打开，等待有效的 RION 68 数据帧")
            next_status = now + 1


def main():
    import serial
    from serial.tools import list_ports

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="auto")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--address", type=int, choices=range(255), default=0, metavar="0–254")
    args = parser.parse_args()
    previous = None
    last_status = 0.0

    def status(state, message):
        nonlocal previous, last_status
        current = (state, message)
        now = time.monotonic()
        if current != previous or (state != "receiving" and now - last_status >= 1):
            print(json.dumps({"type": "rion_status", "state": state, "message": message}), flush=True)
            previous = current
            last_status = now

    while True:
        try:
            port = select_port(args.port, list_ports.comports())
            with serial.Serial(port, args.baud, timeout=0.02, write_timeout=0.2) as device:
                status("waiting", f"{port} 已打开，等待 IMU406 数据")
                for packet in read_stream(device, args.address, status):
                    status("receiving", f"{port} 正在接收 IMU406 数据")
                    print(json.dumps(packet), flush=True)
        except (ValueError, OSError, serial.SerialException) as error:
            status("disconnected", str(error))
            time.sleep(1)


if __name__ == "__main__":
    main()
