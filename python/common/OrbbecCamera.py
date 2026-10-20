# ******************************************************************************
# OrbbecCamera：封装 pyorbbecsdk 的最小相机采集类，不依赖 cylindertag_cpp。
#
# 用途：
#   1. 单独验证 Orbbec 相机 + SDK 这一层是否正常工作（与 C++ CylinderTag 处理解耦）；
#   2. 作为一个可复用的相机模块，被其它脚本（比如真正跑 CylinderTag 的脚本）import
#      之后实例化使用，而不用每次都重写 pipeline/config 的初始化逻辑。
#
# 用法示例：
#   with OrbbecCamera() as cam:
#       frame = cam.read()
#
# 或者：
#   cam = OrbbecCamera()
#   cam.open()
#   frame = cam.read()
#   cam.close()
# ******************************************************************************
import time
from typing import Optional

import cv2
import numpy as np

from pyorbbecsdk import (
    Config,
    FormatConvertFilter,
    OBConvertFormat,
    OBFormat,
    OBSensorType,
    Pipeline,
    OBError,
    VideoStreamProfile,
)

ESC_KEY = 27


class OrbbecCamera:
    """对 pyorbbecsdk 的 color stream 采集做最小封装。"""

    def __init__(
        self,
        target_width: int = 3840,
        target_height: int = 2160,
        target_fps: int = 30,
        enable_frame_sync: bool = True,
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.enable_frame_sync = enable_frame_sync

        self.pipeline: Optional[Pipeline] = None
        self.config: Optional[Config] = None
        self.color_profile: Optional[VideoStreamProfile] = None
        self._convert_filter: Optional[FormatConvertFilter] = None
        self._is_open = False

    # ------------------------------------------------------------------ #
    # 生命周期管理
    # ------------------------------------------------------------------ #
    def open(self) -> None:
        """初始化 pipeline、选择 color stream profile 并启动采集。"""
        if self._is_open:
            return

        self.pipeline = Pipeline()
        self.config = Config()

        self.color_profile = self._select_color_profile(self.pipeline)
        self.config.enable_stream(self.color_profile)

        try:
            if self.enable_frame_sync:
                self.pipeline.enable_frame_sync()
        except Exception as exc:
            print(f"[OrbbecCamera] enable_frame_sync 失败（忽略）: {exc}")

        self.pipeline.start(self.config)
        self._is_open = True

        intrinsic = self.get_intrinsic()
        if intrinsic is not None:
            print(
                f"[OrbbecCamera] fx={intrinsic.fx:.2f} fy={intrinsic.fy:.2f} "
                f"cx={intrinsic.cx:.2f} cy={intrinsic.cy:.2f}"
            )

    def close(self) -> None:
        """停止采集，释放 pipeline。"""
        if not self._is_open:
            return
        try:
            self.pipeline.stop()
        finally:
            self._is_open = False

    def __enter__(self) -> "OrbbecCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._is_open

    # ------------------------------------------------------------------ #
    # 采集接口
    # ------------------------------------------------------------------ #
    def read(self, timeout_ms: int = 100) -> Optional[np.ndarray]:
        """取一帧 color frame 并转换成 BGR ndarray，取不到则返回 None。"""
        if not self._is_open:
            raise RuntimeError("OrbbecCamera 尚未 open()，无法 read()")

        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None

        return self._frame_to_bgr_image(color_frame)

    def get_intrinsic(self):
        """返回当前 color profile 的内参对象（fx/fy/cx/cy 等），失败返回 None。"""
        if self.color_profile is None:
            return None
        try:
            return self.color_profile.get_intrinsic()
        except Exception as exc:
            print(f"[OrbbecCamera] 无法读取内参: {exc}")
            return None

    def get_distortion(self):
        """返回当前 color profile 的畸变系数对象，失败返回 None。"""
        if self.color_profile is None:
            return None
        try:
            return self.color_profile.get_distortion()
        except Exception as exc:
            print(f"[OrbbecCamera] 无法读取畸变系数: {exc}")
            return None

    def get_camera_matrix_and_dist_coeffs(self):
        """直接拼好 OpenCV 格式的 cameraMatrix(3x3) 和 distCoeffs(1x5)。"""
        intrinsic = self.get_intrinsic()
        distortion = self.get_distortion()
        if intrinsic is None or distortion is None:
            return None, None

        camera_matrix = np.array(
            [
                [intrinsic.fx, 0.0, intrinsic.cx],
                [0.0, intrinsic.fy, intrinsic.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.array(
            [[distortion.k1, distortion.k2, distortion.p1, distortion.p2, distortion.k3]],
            dtype=np.float64,
        )
        return camera_matrix, dist_coeffs

    # ------------------------------------------------------------------ #
    # 内部工具方法
    # ------------------------------------------------------------------ #
    def _select_color_profile(self, pipeline: Pipeline) -> VideoStreamProfile:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            profile = profile_list.get_video_stream_profile(
                self.target_width, self.target_height, OBFormat.RGB, self.target_fps
            )
            return profile
        except OBError as exc:
            print(
                f"[OrbbecCamera] 获取 {self.target_width}x{self.target_height}"
                f"@{self.target_fps} color profile 失败: {exc}"
            )
            profile = profile_list.get_default_video_stream_profile()
            print(f"[OrbbecCamera] 改用默认 color profile: {profile}")
            return profile

    @staticmethod
    def _determine_convert_format(color_format: OBFormat) -> Optional[OBConvertFormat]:
        mapping = {
            OBFormat.I420: OBConvertFormat.I420_TO_RGB888,
            OBFormat.MJPG: OBConvertFormat.MJPG_TO_RGB888,
            OBFormat.YUYV: OBConvertFormat.YUYV_TO_RGB888,
            OBFormat.NV21: OBConvertFormat.NV21_TO_RGB888,
            OBFormat.NV12: OBConvertFormat.NV12_TO_RGB888,
            OBFormat.UYVY: OBConvertFormat.UYVY_TO_RGB888,
        }
        return mapping.get(color_format)

    def _frame_to_bgr_image(self, frame) -> Optional[np.ndarray]:
        color_format = frame.get_format()
        width, height = frame.get_width(), frame.get_height()

        def reshape3(buf):
            img = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            return np.ascontiguousarray(img)

        if color_format == OBFormat.RGB:
            return cv2.cvtColor(reshape3(frame.get_data()), cv2.COLOR_RGB2BGR)
        if color_format == OBFormat.BGR:
            return reshape3(frame.get_data())

        if self._convert_filter is None:
            self._convert_filter = FormatConvertFilter()

        convert_format = self._determine_convert_format(color_format)
        if convert_format is None:
            print(f"[OrbbecCamera] Unsupported color format: {color_format}")
            return None

        self._convert_filter.set_format_convert_format(convert_format)
        rgb_frame = self._convert_filter.process(frame)
        if rgb_frame is None:
            print(f"[OrbbecCamera] Failed to convert frame from format {color_format}")
            return None

        rgb = reshape3(rgb_frame.get_data())
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------- #
# 测试用例：直接运行本文件即可验证相机是否能正常采集、显示、计算 FPS
#   python orbbec_camera.py
#   按 q 或 ESC 退出
# ---------------------------------------------------------------------- #
def _test_basic_capture_and_display():
    """测试1：open -> 循环read -> 显示画面 + FPS -> close"""
    print("\n[Test] 基本采集+显示测试，按 q / ESC 退出")
    DISPLAY_WIDTH, DISPLAY_HEIGHT = 1920, 1080

    with OrbbecCamera() as cam:
        fps_counter = 0
        fps_timer = time.time()
        fps_value: Optional[float] = None

        while True:
            frame = cam.read(timeout_ms=100)
            if frame is None:
                continue

            display = frame.copy()

            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_timer = now
                fps_counter = 0

            if fps_value is not None:
                cv2.putText(
                    display, f"FPS: {fps_value:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                )

            view = (
                display
                if (display.shape[1], display.shape[0]) == (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                else cv2.resize(display, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
            )
            cv2.imshow("OrbbecCamera Test", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ESC_KEY or key == ord("q"):
                break

    cv2.destroyAllWindows()
    print("[Test] 基本采集+显示测试结束")


def _test_intrinsics_and_single_frame():
    """测试2：只验证 open() 是否成功、内参是否能读到、能否取到至少一帧"""
    print("\n[Test] 内参 + 单帧采集测试")
    cam = OrbbecCamera()
    try:
        cam.open()
        assert cam.is_open, "open() 之后 is_open 应为 True"

        camera_matrix, dist_coeffs = cam.get_camera_matrix_and_dist_coeffs()
        assert camera_matrix is not None, "cameraMatrix 不应为空"
        print(f"[Test] cameraMatrix=\n{camera_matrix}")
        print(f"[Test] distCoeffs={dist_coeffs}")

        frame = None
        for _ in range(30):  # 最多重试30次，等相机出第一帧
            frame = cam.read(timeout_ms=200)
            if frame is not None:
                break
        assert frame is not None, "30次重试内未能取到任何帧，检查相机连接"
        print(f"[Test] 取到帧，shape={frame.shape}")
        print("[Test] PASSED")
    finally:
        cam.close()


def main():
    _test_intrinsics_and_single_frame()
    _test_basic_capture_and_display()


if __name__ == "__main__":
    main()