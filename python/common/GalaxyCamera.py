#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Galaxy Camera SDK Python Wrapper (Daheng Imaging / 大恒图像相机)

This module provides a Python interface to the Galaxy Camera SDK (libgxiapi.so)
using ctypes. It wraps the core functions for device enumeration, parameter
control, and image acquisition.

Usage Example:
    from galaxy_camera import GalaxyCamera

    cam = GalaxyCamera()
    cam.init()
    cam.open_by_index(1)
    cam.set_enum("AcquisitionMode", "Continuous")
    cam.set_enum("TriggerMode", "Off")
    cam.stream_on()

    for i in range(10):
        frame = cam.get_frame(timeout_ms=1000)
        if frame:
            print(f"Frame {frame.frame_id}: {frame.width}x{frame.height}, {frame.pixel_format}")

    cam.stream_off()
    cam.close()
    cam.uninit()

Requirements:
    - libgxiapi.so (in lib/x86_64/ or system library path)
    - GxGVTL.cti and GxU3VTL.cti transport layer files
    - LD_LIBRARY_PATH must include the lib directory

====================================================================
本版改动说明（相对上一版）：
====================================================================
新增以下方法，用于解决实时预览卡顿/延迟问题（之前完全没有配置曝光、
增益、采集帧率、流缓冲策略，可能导致自动曝光拖累帧率、或缓冲区里
积压旧帧造成画面滞后）：

  - set_exposure_time_us(exposure_us): 关闭自动曝光并设置曝光时间(us)
  - set_gain(gain): 关闭自动增益并设置增益值
  - set_acquisition_fps(fps): 开启帧率限制模式并设置采集帧率
  - set_stream_buffer_mode(mode_name): 设置数据流缓冲处理模式
    (newest_only / oldest_first / oldest_first_overwrite)
  - get_stream_counters(): 读取丢帧/不完整帧计数，用于诊断卡顿来源

注意：这些方法和 set_int/set_enum 等一样，必须在 stream_on() 之前调用
（受 _check_stream_off 保护），否则会抛 RuntimeError。

StreamBufferHandlingMode 是数据流(DataStream)级别的属性，不是设备级别
的 GenICam 节点。Galaxy SDK 的 GXGetIntValue/GXGetEnumValue/
GXSetEnumValueByString 接收通用 GX_PORT_HANDLE，因此取得 DataStream handle
后应继续使用这些通用接口；SDK 并不存在 GXDSGetIntValue 之类的独立符号。

实时读取新增 get_latest_frame()：调用 GalaxyView 同样使用的
GXDQAllBufs/GXQAllBufs，一次排空当前队列、只复制最新有效帧并立即归还
所有 SDK buffer。算法来不及处理时会跳过旧帧，避免延迟不断累积。
"""

import ctypes as ct
import os
import sys
import threading
import time
import struct
from ctypes import (
    c_void_p, c_char_p, c_int32, c_uint32, c_int64, c_uint64,
    c_size_t, c_bool, c_double, c_ubyte, c_char, POINTER, byref,
    create_string_buffer, sizeof, Structure, Union, Array, pointer, cast
)

# ============================================================================
# Constants
# ============================================================================

GX_INFO_LENGTH_8_BYTE   = 8
GX_INFO_LENGTH_32_BYTE  = 32
GX_INFO_LENGTH_64_BYTE  = 64
GX_INFO_LENGTH_128_BYTE = 128
GX_INFO_LENGTH_256_BYTE = 256

# ---- Status Codes ----
class GX_STATUS_LIST:
    GX_STATUS_SUCCESS           =  0
    GX_STATUS_ERROR             = -1
    GX_STATUS_NOT_FOUND_TL      = -2
    GX_STATUS_NOT_FOUND_DEVICE  = -3
    GX_STATUS_OFFLINE           = -4
    GX_STATUS_INVALID_PARAMETER = -5
    GX_STATUS_INVALID_HANDLE    = -6
    GX_STATUS_INVALID_CALL      = -7
    GX_STATUS_INVALID_ACCESS    = -8
    GX_STATUS_NEED_MORE_BUFFER  = -9
    GX_STATUS_ERROR_TYPE        = -10
    GX_STATUS_OUT_OF_RANGE      = -11
    GX_STATUS_NOT_IMPLEMENTED   = -12
    GX_STATUS_NOT_INIT_API      = -13
    GX_STATUS_TIMEOUT           = -14

STATUS_MESSAGES = {
    -1: "GX_STATUS_ERROR: Unspecified internal error",
    -2: "GX_STATUS_NOT_FOUND_TL: TL library not found",
    -3: "GX_STATUS_NOT_FOUND_DEVICE: Device not found",
    -4: "GX_STATUS_OFFLINE: Device offline",
    -5: "GX_STATUS_INVALID_PARAMETER: Invalid parameter",
    -6: "GX_STATUS_INVALID_HANDLE: Invalid handle",
    -7: "GX_STATUS_INVALID_CALL: Invalid interface call",
    -8: "GX_STATUS_INVALID_ACCESS: Access denied or wrong mode",
    -9: "GX_STATUS_NEED_MORE_BUFFER: Buffer too small",
    -10: "GX_STATUS_ERROR_TYPE: Feature type mismatch",
    -11: "GX_STATUS_OUT_OF_RANGE: Value out of range",
    -12: "GX_STATUS_NOT_IMPLEMENTED: Not supported",
    -13: "GX_STATUS_NOT_INIT_API: GXInitLib not called",
    -14: "GX_STATUS_TIMEOUT: Operation timed out",
}

# ---- TL Types ----
class GX_TL_TYPE:
    UNKNOWN = 0
    USB     = 1
    GEV     = 2
    U3V     = 4
    CXP     = 8

# ---- Access Mode ----
class GX_ACCESS_MODE:
    READONLY  = 2
    CONTROL   = 3
    EXCLUSIVE = 4

# ---- Open Mode ----
class GX_OPEN_MODE:
    SN      = 0
    IP      = 1
    MAC     = 2
    INDEX   = 3
    USERID  = 4

# ---- Device Class ----
class GX_DEVICE_CLASS:
    UNKNOWN = 0
    USB2    = 1
    GEV     = 2
    U3V     = 3
    SMART   = 4
    CXP     = 5

# ---- Frame Status ----
class GX_FRAME_STATUS:
    SUCCESS    = 0
    INCOMPLETE = -1

# ---- Node Access Mode ----
class GX_NODE_ACCESS_MODE:
    NI    = 0   # Not implemented
    NA    = 1   # Not available
    WO    = 2   # Write only
    RO    = 3   # Read only
    RW    = 4   # Read/Write
    UNDEF = 5   # Undefined

# ---- Pixel Format Enum (common values) ----
class GX_PIXEL_FORMAT:
    MONO8          = 0x01080001
    MONO8_SIGNED   = 0x01080002
    MONO10         = 0x01100003
    MONO12         = 0x01100005
    MONO14         = 0x01100025
    MONO16         = 0x01100007
    BAYER_GR8      = 0x01080008
    BAYER_RG8      = 0x01080009
    BAYER_GB8      = 0x0108000A
    BAYER_BG8      = 0x0108000B
    BAYER_GR10     = 0x0110000C
    BAYER_RG10     = 0x0110000D
    BAYER_GB10     = 0x0110000E
    BAYER_BG10     = 0x0110000F
    BAYER_GR12     = 0x01100010
    BAYER_RG12     = 0x01100011
    BAYER_GB12     = 0x01100012
    BAYER_BG12     = 0x01100013
    RGB8           = 0x02180014
    BGR8           = 0x02180015
    YUV422_8       = 0x02100032
    YUV422_8_UYVY  = 0x0210001F


_SINGLE_CHANNEL_8_FORMATS = frozenset((
    GX_PIXEL_FORMAT.MONO8,
    GX_PIXEL_FORMAT.MONO8_SIGNED,
    GX_PIXEL_FORMAT.BAYER_GR8,
    GX_PIXEL_FORMAT.BAYER_RG8,
    GX_PIXEL_FORMAT.BAYER_GB8,
    GX_PIXEL_FORMAT.BAYER_BG8,
))

_SINGLE_CHANNEL_16_FORMATS = frozenset((
    GX_PIXEL_FORMAT.MONO10,
    GX_PIXEL_FORMAT.MONO12,
    GX_PIXEL_FORMAT.MONO14,
    GX_PIXEL_FORMAT.MONO16,
    GX_PIXEL_FORMAT.BAYER_GR10,
    GX_PIXEL_FORMAT.BAYER_RG10,
    GX_PIXEL_FORMAT.BAYER_GB10,
    GX_PIXEL_FORMAT.BAYER_BG10,
    GX_PIXEL_FORMAT.BAYER_GR12,
    GX_PIXEL_FORMAT.BAYER_RG12,
    GX_PIXEL_FORMAT.BAYER_GB12,
    GX_PIXEL_FORMAT.BAYER_BG12,
))

_BAYER_TO_BGR_CODES = {
    GX_PIXEL_FORMAT.BAYER_GR8: "COLOR_BAYER_GR2BGR",
    GX_PIXEL_FORMAT.BAYER_RG8: "COLOR_BAYER_RG2BGR",
    GX_PIXEL_FORMAT.BAYER_GB8: "COLOR_BAYER_GB2BGR",
    GX_PIXEL_FORMAT.BAYER_BG8: "COLOR_BAYER_BG2BGR",
    GX_PIXEL_FORMAT.BAYER_GR10: "COLOR_BAYER_GR2BGR",
    GX_PIXEL_FORMAT.BAYER_RG10: "COLOR_BAYER_RG2BGR",
    GX_PIXEL_FORMAT.BAYER_GB10: "COLOR_BAYER_GB2BGR",
    GX_PIXEL_FORMAT.BAYER_BG10: "COLOR_BAYER_BG2BGR",
    GX_PIXEL_FORMAT.BAYER_GR12: "COLOR_BAYER_GR2BGR",
    GX_PIXEL_FORMAT.BAYER_RG12: "COLOR_BAYER_RG2BGR",
    GX_PIXEL_FORMAT.BAYER_GB12: "COLOR_BAYER_GB2BGR",
    GX_PIXEL_FORMAT.BAYER_BG12: "COLOR_BAYER_BG2BGR",
}

# ---- IP Configure Mode ----
class GX_IP_CONFIGURE_MODE:
    DHCP      = 0x6
    LLA       = 0x4
    STATIC_IP = 0x5
    DEFAULT   = 0x7

# ---- Log Type ----
class GX_LOG_TYPE:
    OFF   = 0x00000000
    FATAL = 0x00000001
    ERROR = 0x00000010
    WARN  = 0x00000100
    INFO  = 0x00001000
    DEBUG = 0x00010000
    TRACE = 0x00100000

# ---- Reset Device Mode ----
class GX_RESET_DEVICE_MODE:
    RECONNECT = 0x1
    RESET     = 0x2

# ---- DataStream Buffer Handling Mode ----
# 大恒/标准GenTL 数据流缓冲处理模式的符号名称（用于 DSSetEnumValueByString）。
# 不同SDK版本可能略有出入，这里给出最常见的三种。
class GX_DS_STREAM_BUFFER_HANDLING_MODE:
    OLDEST_FIRST            = "OldestFirst"
    OLDEST_FIRST_OVERWRITE  = "OldestFirstOverwrite"
    NEWEST_ONLY             = "NewestOnly"


# ============================================================================
# Typedefs (ctypes types)
# ============================================================================

GX_STATUS = c_int32
GX_TL_TYPE_CT = c_int32
GX_ACCESS_MODE_CMD = c_int32
GX_OPEN_MODE_CMD = c_int32
GX_DEVICE_CLASS_CT = c_int32
GX_FRAME_STATUS_CT = c_int32
GX_DEV_HANDLE = c_void_p
GX_IF_HANDLE = c_void_p
GX_LOCAL_DEV_HANDLE = c_void_p
GX_DS_HANDLE = c_void_p
GX_PORT_HANDLE = c_void_p
GX_EVENT_CALLBACK_HANDLE = c_void_p
GX_FEATURE_CALLBACK_HANDLE = c_void_p


# ============================================================================
# Structures
# ============================================================================

class GX_STRING_VALUE(Structure):
    """String-type feature value."""
    _fields_ = [
        ("strCurValue", c_char * GX_INFO_LENGTH_256_BYTE),
        ("nMaxLength",  c_int64),
        ("reserved",    c_int32 * 4),
    ]

class GX_INT_VALUE(Structure):
    """Integer-type feature value (with range info)."""
    _fields_ = [
        ("nCurValue", c_int64),
        ("nMin",      c_int64),
        ("nMax",      c_int64),
        ("nInc",      c_int64),
        ("reserved",  c_int32 * 16),
    ]

class GX_FLOAT_VALUE(Structure):
    """Float-type feature value (with range info)."""
    _fields_ = [
        ("dCurValue",  c_double),
        ("dMin",       c_double),
        ("dMax",       c_double),
        ("dInc",       c_double),
        ("bIncIsValid", c_bool),
        ("szUnit",     c_char * GX_INFO_LENGTH_8_BYTE),
        ("reserved",   c_int32 * 16),
    ]

class GX_ENUM_VALUE_DES(Structure):
    """Single enum entry description."""
    _fields_ = [
        ("nCurValue",      c_int64),
        ("strCurSymbolic", c_char * GX_INFO_LENGTH_128_BYTE),
        ("reserved",       c_int32 * 4),
    ]

class GX_ENUM_VALUE(Structure):
    """Enum-type feature value (with all supported entries)."""
    _fields_ = [
        ("stCurValue",           GX_ENUM_VALUE_DES),
        ("nSupportedNum",        c_uint32),
        ("nArrySupportedValue",  GX_ENUM_VALUE_DES * 128),
        ("reserved",             c_int32 * 16),
    ]

class GX_OPEN_PARAM(Structure):
    """Parameters for opening a device."""
    _fields_ = [
        ("pszContent", c_char_p),
        ("openMode",   GX_OPEN_MODE_CMD),
        ("accessMode", GX_ACCESS_MODE_CMD),
    ]

class GX_FRAME_BUFFER(Structure):
    """Frame buffer returned by GXDQBuf (Linux layout)."""
    _fields_ = [
        ("nStatus",      GX_FRAME_STATUS_CT),
        ("pImgBuf",      c_void_p),
        ("nWidth",       c_int32),
        ("nHeight",      c_int32),
        ("nPixelFormat", c_int32),
        ("nImgSize",     c_int32),
        ("nFrameID",     c_uint64),
        ("nTimestamp",   c_uint64),
        ("nBufID",       c_uint64),
        ("nOffsetX",     c_int32),
        ("nOffsetY",     c_int32),
        ("pUserParam",   c_void_p),
        ("reserved",     c_int32 * 14),
    ]

PGX_FRAME_BUFFER = POINTER(GX_FRAME_BUFFER)

class GX_FRAME_CALLBACK_PARAM(Structure):
    """Frame callback parameter (Linux layout)."""
    _fields_ = [
        ("pUserParam",   c_void_p),
        ("status",       GX_FRAME_STATUS_CT),
        ("pImgBuf",      c_void_p),
        ("nImgSize",     c_int32),
        ("nWidth",       c_int32),
        ("nHeight",      c_int32),
        ("nPixelFormat", c_int32),
        ("nFrameID",     c_uint64),
        ("nTimestamp",   c_uint64),
        ("nOffsetX",     c_int32),
        ("nOffsetY",     c_int32),
        ("reserved",     c_int32 * 1),
    ]

class GX_FRAME_DATA(Structure):
    """Frame data structure (Linux layout)."""
    _fields_ = [
        ("nStatus",      GX_FRAME_STATUS_CT),
        ("pImgBuf",      c_void_p),
        ("nWidth",       c_int32),
        ("nHeight",      c_int32),
        ("nPixelFormat", c_int32),
        ("nImgSize",     c_int32),
        ("nFrameID",     c_uint64),
        ("nTimestamp",   c_uint64),
        ("nOffsetX",     c_int32),
        ("nOffsetY",     c_int32),
        ("reserved",     c_int32 * 1),
    ]

# Device info structures (simplified)
class GX_GEV_DEVICE_INFO(Structure):
    _fields_ = [
        ("nCurrentIp",        c_uint32),
        ("nCurrentSubNetMask", c_uint32),
        ("nDefaultGateWay",   c_uint32),
        ("nNetExport",        c_uint32),
        ("nMacAddress",       c_uint64),
        ("chVendorName",      c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chModelName",       c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chManufacturerInfo",c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chDeviceVersion",   c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chSerialNumber",    c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chUserDefinedName", c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("nReserved",         c_uint32 * 64),
    ]

class GX_U3V_DEVICE_INFO(Structure):
    _fields_ = [
        ("chVendorName",      c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chModelName",       c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chManufacturerInfo",c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chDeviceVersion",   c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chSerialNumber",    c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chUserDefinedName", c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("nReserved",         c_uint32 * 64),
    ]

class GX_USB_DEVICE_INFO(Structure):
    _fields_ = [
        ("chVendorName",      c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chModelName",       c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chManufacturerInfo",c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chDeviceVersion",   c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chSerialNumber",    c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("chUserDefinedName", c_ubyte * GX_INFO_LENGTH_64_BYTE),
        ("nReserved",         c_uint32 * 64),
    ]

class _DevInfoUnion(Union):
    _fields_ = [
        ("stGEVDevInfo", GX_GEV_DEVICE_INFO),
        ("stU3VDevInfo", GX_U3V_DEVICE_INFO),
        ("stUSBDevInfo", GX_USB_DEVICE_INFO),
        ("nReserved",    c_uint32 * 256),
    ]

class GX_DEVICE_INFO(Structure):
    _fields_ = [
        ("emDevType",  GX_DEVICE_CLASS_CT),
        ("nReserved",  c_uint32 * 4),
        ("DevInfo",    _DevInfoUnion),
    ]


# ============================================================================
# SDK Library Loader
# ============================================================================

def _find_library():
    """Locate the libgxiapi.so shared library.

    Search priority:
      1. GALAXY_SDK_LIB env var (exact path to libgxiapi.so)
      2. GALAXY_SDK_DIR env var (SDK root, looks for lib/x86_64/libgxiapi.so inside)
      3. Relative to this script: ../../../Galaxy_camera/lib/x86_64/libgxiapi.so
      4. Relative to this script: lib/x86_64/libgxiapi.so
      5. System paths (/usr/lib, /usr/local/lib)
      6. Fallback to "libgxiapi.so" (relies on LD_LIBRARY_PATH)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = []

    # 1. Exact library path via env var
    env_lib = os.environ.get("GALAXY_SDK_LIB", "")
    if env_lib:
        search_paths.append(env_lib)

    # 2. SDK root directory via env var
    env_sdk_dir = os.environ.get("GALAXY_SDK_DIR", "")
    if env_sdk_dir:
        search_paths.append(os.path.join(env_sdk_dir, "lib", "x86_64"))

    # 3. Common relative path: script is in <project>/python/common/, SDK is at <project>/Galaxy_camera/
    search_paths.append(os.path.normpath(os.path.join(script_dir, "..", "..", "Galaxy_camera", "lib", "x86_64")))

    # 4. Traditional: SDK lib alongside the script
    search_paths.append(os.path.join(script_dir, "lib", "x86_64"))

    # 5. System paths
    search_paths.append("/opt/galaxy/lib/x86_64")
    search_paths.append("/usr/lib")
    search_paths.append("/usr/local/lib")

    for p in search_paths:
        if os.path.isfile(p):
            return p
        # If directory, try libgxiapi.so inside
        if os.path.isdir(p):
            candidate = os.path.join(p, "libgxiapi.so")
            if os.path.isfile(candidate):
                return candidate

    # Fallback: try loading by name (relies on LD_LIBRARY_PATH)
    return "libgxiapi.so"


def _find_cti_path(lib_path):
    """
    Find the directory containing transport layer files (.cti).
    The Galaxy SDK needs GxU3VTL.cti / GxGVTL.cti to stream images.
    These should be in the same directory as libgxiapi.so.
    """
    lib_dir = os.path.dirname(os.path.abspath(lib_path))
    # Check if .cti files exist in the same directory as the .so
    if os.path.isfile(os.path.join(lib_dir, "GxU3VTL.cti")) or \
       os.path.isfile(os.path.join(lib_dir, "GxGVTL.cti")):
        return lib_dir
    # Check if .cti files are in a sibling 'lib/x86_64' relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.normpath(os.path.join(script_dir, "..", "..", "Galaxy_camera", "lib", "x86_64")),
        os.path.join(script_dir, "lib", "x86_64"),
    ]:
        if os.path.isfile(os.path.join(candidate, "GxU3VTL.cti")):
            return candidate
    return lib_dir


# ============================================================================
# Main Camera API Wrapper Class
# ============================================================================

class GalaxyCameraError(Exception):
    """Exception raised for Galaxy Camera SDK errors."""
    def __init__(self, status, message=None):
        self.status = status
        self.message = message or STATUS_MESSAGES.get(status, f"Unknown error: {status}")
        super().__init__(f"GX error {status}: {self.message}")


class FrameData:
    """Python-friendly frame data container."""
    def __init__(self, frame_buffer_ptr):
        fb = frame_buffer_ptr.contents
        self.status      = fb.nStatus
        self.width       = fb.nWidth
        self.height      = fb.nHeight
        self.pixel_format = fb.nPixelFormat
        self.frame_id    = fb.nFrameID
        self.timestamp   = fb.nTimestamp
        self.buf_id      = fb.nBufID
        self.offset_x    = fb.nOffsetX
        self.offset_y    = fb.nOffsetY
        self.data_size   = fb.nImgSize

        # Copy image data into a Python bytes object
        if fb.pImgBuf and fb.nImgSize > 0:
            buf = (c_ubyte * fb.nImgSize).from_address(fb.pImgBuf)
            self.image_data = bytes(buf)
        else:
            self.image_data = b""

    def save(self, filepath):
        """Save raw image data to file."""
        with open(filepath, "wb") as f:
            f.write(self.image_data)

    def to_numpy(self):
        """Convert image data to a numpy array (requires numpy)."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for to_numpy(). Install with: pip install numpy")

        arr = np.frombuffer(self.image_data, dtype=np.uint8)
        pixel_count = self.width * self.height
        if self.pixel_format in _SINGLE_CHANNEL_8_FORMATS:
            if len(arr) == pixel_count:
                return arr.reshape((self.height, self.width))
        elif self.pixel_format in _SINGLE_CHANNEL_16_FORMATS:
            arr16 = np.frombuffer(self.image_data, dtype=np.uint16)
            if len(arr16) == pixel_count:
                return arr16.reshape((self.height, self.width))
        elif self.pixel_format in (GX_PIXEL_FORMAT.RGB8, GX_PIXEL_FORMAT.BGR8):
            if len(arr) == pixel_count * 3:
                return arr.reshape((self.height, self.width, 3))
        elif self.pixel_format in (
            GX_PIXEL_FORMAT.YUV422_8,
            GX_PIXEL_FORMAT.YUV422_8_UYVY,
        ):
            if len(arr) == pixel_count * 2:
                return arr.reshape((self.height, self.width, 2))

        return arr

    def to_bgr(self):
        """Convert supported Mono/Bayer/RGB/YUV formats to OpenCV BGR."""
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ImportError("opencv-python and numpy are required for to_bgr()") from exc

        img = self.to_numpy()
        if img.ndim == 1:
            return None
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        bayer_code_name = _BAYER_TO_BGR_CODES.get(self.pixel_format)
        if bayer_code_name is not None:
            return cv2.cvtColor(img, getattr(cv2, bayer_code_name))
        if self.pixel_format == GX_PIXEL_FORMAT.RGB8:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.pixel_format == GX_PIXEL_FORMAT.BGR8:
            return img
        if self.pixel_format == GX_PIXEL_FORMAT.YUV422_8:
            return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUY2)
        if self.pixel_format == GX_PIXEL_FORMAT.YUV422_8_UYVY:
            return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        return None

    def __repr__(self):
        return (f"FrameData(frame_id={self.frame_id}, {self.width}x{self.height}, "
                f"pixel_format=0x{self.pixel_format:08X}, "
                f"size={self.data_size})")


class GalaxyCamera:
    """
    Python wrapper for the Galaxy Camera SDK (Daheng Imaging).

    Usage:
        cam = GalaxyCamera()
        cam.init()
        cam.open_by_index(1)

        # Read device name
        name = cam.get_string("DeviceModelName")

        # Configure acquisition
        cam.set_enum("AcquisitionMode", "Continuous")
        cam.set_enum("TriggerMode", "Off")

        # Optional: exposure / gain / fps / buffer mode (must be set BEFORE stream_on)
        cam.set_exposure_time_us(10000)
        cam.set_gain(10)
        cam.set_acquisition_fps(20)
        cam.set_stream_buffer_mode("newest_only")

        # Start streaming
        cam.stream_on()

        # Grab frames
        for i in range(10):
            frame = cam.get_frame(timeout_ms=2000)
            print(frame)

        print(cam.get_stream_counters())

        cam.stream_off()
        cam.close()
        cam.uninit()
    """

    def __init__(self, lib_path=None):
        """
        Initialize the GalaxyCamera wrapper.

        Args:
            lib_path: Path to libgxiapi.so. If None, auto-detect.
        """
        self._lib_path = lib_path or _find_library()
        self._dll = None          # ctypes CDLL handle
        self._device_handle = None
        self._streaming = False
        self._initialized = False
        self._acquisition_buffer_number = None
        self._dq_all_usable = None
        self._latest_fallback_warning = None
        # Serialize acquisition and lifecycle changes. RLock keeps cleanup
        # helpers safe if a caller nests them in one thread.
        self._acquisition_lock = threading.RLock()
        self._last_batch_size = 0
        self._host_dequeued_frames = 0
        self._host_discarded_frames = 0
        self._host_invalid_frames = 0

    # ---- Library Management ----

    def init(self):
        """
        Load the shared library and initialize the SDK.
        Must be called before any other operation.
        """
        if self._initialized:
            return

        # Resolve the real path (in case of symlinks or relative names)
        if os.path.isfile(self._lib_path):
            self._lib_path = os.path.realpath(self._lib_path)

        # Set up transport layer path BEFORE loading the library.
        # The Galaxy SDK looks for .cti files in GENICAM_GENTL64_PATH (or
        # GENICAM_GENTL32_PATH), and also in the same directory as the .so.
        cti_dir = _find_cti_path(self._lib_path)
        if cti_dir and 'GENICAM_GENTL64_PATH' not in os.environ:
            os.environ['GENICAM_GENTL64_PATH'] = cti_dir

        # Also add the library directory to LD_LIBRARY_PATH at runtime
        lib_dir = os.path.dirname(self._lib_path)
        if lib_dir and os.path.isdir(lib_dir):
            current_ld = os.environ.get('LD_LIBRARY_PATH', '')
            if lib_dir not in current_ld.split(os.pathsep):
                os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(
                    [p for p in [lib_dir, current_ld] if p]
                )

        # Load library (use default RTLD_LAZY to avoid early symbol resolution errors)
        self._dll = ct.CDLL(self._lib_path)

        # ---- Set up function signatures (gracefully skip missing symbols) ----
        self._setup_signatures()

        # Check that essential functions exist
        if not hasattr(self._dll, 'GXInitLib'):
            raise RuntimeError(
                f"Failed to load Galaxy SDK from {self._lib_path}: "
                f"GXInitLib symbol not found. Please check that libgxiapi.so "
                f"and its transport layer files (.cti) are properly installed."
            )

        # Initialize SDK
        status = self._dll.GXInitLib()
        self._check_status(status, "GXInitLib")
        self._initialized = True

        # Log where the library was loaded from (useful for debugging)
        print(f"[GalaxyCamera] Loaded library: {self._lib_path}")
        cti_dir_final = os.environ.get('GENICAM_GENTL64_PATH', '')
        if cti_dir_final:
            print(f"[GalaxyCamera] CTI (transport layer) path: {cti_dir_final}")

    def uninit(self):
        """Close the SDK library and release all resources."""
        with self._acquisition_lock:
            if not self._initialized:
                return
            status = self._dll.GXCloseLib()
            self._check_status(status, "GXCloseLib")
            self._initialized = False
            self._dll = None

    def get_lib_version(self):
        """Get the SDK library version string."""
        if not self._dll:
            raise RuntimeError("Library not loaded. Call init() first.")
        if not hasattr(self._dll, 'GXGetLibVersion'):
            return "<unknown>"
        func = self._dll.GXGetLibVersion
        func.restype = c_char_p
        return func().decode("utf-8") if func() else ""

    # ---- Device Enumeration ----

    def update_device_list(self, timeout_ms=1000):
        """
        Enumerate all devices on the network/system.

        Args:
            timeout_ms: Enumeration timeout in milliseconds.

        Returns:
            Number of devices found.
        """
        num_devices = c_uint32(0)
        status = self._dll.GXUpdateAllDeviceList(byref(num_devices), c_uint32(timeout_ms))
        self._check_status(status, "GXUpdateAllDeviceList")
        return num_devices.value

    def get_device_count_by_type(self, tl_type=GX_TL_TYPE.U3V, timeout_ms=1000):
        """
        Get the number of devices of a specific transport layer type.

        Args:
            tl_type: Transport layer type (GX_TL_TYPE values).
            timeout_ms: Timeout in milliseconds.

        Returns:
            Number of devices.
        """
        num_devices = c_uint32(0)
        status = self._dll.GXUpdateAllDeviceListEx(c_uint64(tl_type), byref(num_devices), c_uint32(timeout_ms))
        self._check_status(status, "GXUpdateAllDeviceListEx")
        return num_devices.value

    def get_device_info(self, index):
        """
        Get info for the device at the given index (1-based).

        Args:
            index: Device index (starting from 1).

        Returns:
            GX_DEVICE_INFO structure.
        """
        info = GX_DEVICE_INFO()
        status = self._dll.GXGetDeviceInfo(c_uint32(index), byref(info))
        self._check_status(status, "GXGetDeviceInfo")
        return info

    def list_devices(self):
        """
        List all enumerated devices with basic information.

        Returns:
            List of dicts with device information.
        """
        count = self.update_device_list()
        devices = []
        for i in range(1, count + 1):
            info = self.get_device_info(i)
            dev_class = info.emDevType

            d = {"index": i, "device_class": dev_class}

            if dev_class == GX_DEVICE_CLASS.GEV:
                gev = info.DevInfo.stGEVDevInfo
                d["vendor"]  = bytes(gev.chVendorName).rstrip(b"\x00").decode("utf-8", errors="replace")
                d["model"]   = bytes(gev.chModelName).rstrip(b"\x00").decode("utf-8", errors="replace")
                d["serial"]  = bytes(gev.chSerialNumber).rstrip(b"\x00").decode("utf-8", errors="replace")
                d["version"] = bytes(gev.chDeviceVersion).rstrip(b"\x00").decode("utf-8", errors="replace")
                d["mac"]     = f"{gev.nMacAddress:012X}"
                d["ip"]      = self._int_to_ip(gev.nCurrentIp)
            elif dev_class in (GX_DEVICE_CLASS.U3V, GX_DEVICE_CLASS.USB2):
                if dev_class == GX_DEVICE_CLASS.U3V:
                    usb = info.DevInfo.stU3VDevInfo
                else:
                    usb = info.DevInfo.stUSBDevInfo
                d["vendor"]  = bytes(usb.chVendorName).rstrip(b"\x00").decode("utf-8", errors="replace")
                d["model"]   = bytes(usb.chModelName).rstrip(b"\x00").decode("utf-8", errors="replace")
                d["serial"]  = bytes(usb.chSerialNumber).rstrip(b"\x00").decode("utf-8", errors="replace")
                d["version"] = bytes(usb.chDeviceVersion).rstrip(b"\x00").decode("utf-8", errors="replace")

            devices.append(d)
        return devices

    # ---- Device Open / Close ----

    def open_by_index(self, index, access_mode=GX_ACCESS_MODE.EXCLUSIVE):
        """
        Open a device by its 1-based index.

        Args:
            index: Device index (1-based).
            access_mode: Access mode (EXCLUSIVE, CONTROL, or READONLY).
        """
        open_param = GX_OPEN_PARAM()
        open_param.pszContent = ct.c_char_p(f"{index}".encode("utf-8"))
        open_param.openMode   = GX_OPEN_MODE.INDEX
        open_param.accessMode = access_mode

        handle = GX_DEV_HANDLE()
        status = self._dll.GXOpenDevice(byref(open_param), byref(handle))
        self._check_status(status, f"GXOpenDevice (index={index})")
        self._device_handle = handle
        self._reset_acquisition_state()

    def open_by_sn(self, serial_number, access_mode=GX_ACCESS_MODE.EXCLUSIVE):
        """
        Open a device by its serial number.

        Args:
            serial_number: Device serial number string.
            access_mode: Access mode.
        """
        open_param = GX_OPEN_PARAM()
        open_param.pszContent = ct.c_char_p(serial_number.encode("utf-8"))
        open_param.openMode   = GX_OPEN_MODE.SN
        open_param.accessMode = access_mode

        handle = GX_DEV_HANDLE()
        status = self._dll.GXOpenDevice(byref(open_param), byref(handle))
        self._check_status(status, f"GXOpenDevice (SN={serial_number})")
        self._device_handle = handle
        self._reset_acquisition_state()

    def open_by_ip(self, ip_address, access_mode=GX_ACCESS_MODE.EXCLUSIVE):
        """
        Open a device by its IP address (GigE cameras only).

        Args:
            ip_address: IP address string.
            access_mode: Access mode.
        """
        open_param = GX_OPEN_PARAM()
        open_param.pszContent = ct.c_char_p(ip_address.encode("utf-8"))
        open_param.openMode   = GX_OPEN_MODE.IP
        open_param.accessMode = access_mode

        handle = GX_DEV_HANDLE()
        status = self._dll.GXOpenDevice(byref(open_param), byref(handle))
        self._check_status(status, f"GXOpenDevice (IP={ip_address})")
        self._device_handle = handle
        self._reset_acquisition_state()

    def open_by_mac(self, mac_address, access_mode=GX_ACCESS_MODE.EXCLUSIVE):
        """
        Open a device by its MAC address (GigE cameras only).

        Args:
            mac_address: MAC address string (e.g. "00:11:22:33:44:55").
            access_mode: Access mode.
        """
        open_param = GX_OPEN_PARAM()
        open_param.pszContent = ct.c_char_p(mac_address.encode("utf-8"))
        open_param.openMode   = GX_OPEN_MODE.MAC
        open_param.accessMode = access_mode

        handle = GX_DEV_HANDLE()
        status = self._dll.GXOpenDevice(byref(open_param), byref(handle))
        self._check_status(status, f"GXOpenDevice (MAC={mac_address})")
        self._device_handle = handle
        self._reset_acquisition_state()

    def close(self):
        """Close the currently opened device."""
        with self._acquisition_lock:
            if self._device_handle:
                status = self._dll.GXCloseDevice(self._device_handle)
                self._check_status(status, "GXCloseDevice")
                self._device_handle = None
                self._reset_acquisition_state()

    # ---- Feature Read/Write (by string node name) ----

    def get_string(self, feature_name):
        """
        Get a string-type feature value.

        Args:
            feature_name: The GenICam feature node name (e.g., "DeviceVendorName").

        Returns:
            String value.
        """
        val = GX_STRING_VALUE()
        status = self._dll.GXGetStringValue(self._handle, self._encode(feature_name), byref(val))
        self._check_status(status, f"GXGetStringValue({feature_name})")
        return val.strCurValue.decode("utf-8", errors="replace")

    def set_string(self, feature_name, value):
        """Set a string-type feature value."""
        self._check_stream_off("set_string")
        status = self._dll.GXSetStringValue(self._handle, self._encode(feature_name), self._encode(value))
        self._check_status(status, f"GXSetStringValue({feature_name})")

    def get_int(self, feature_name):
        """
        Get an integer-type feature value (current value only).

        Args:
            feature_name: The GenICam feature node name (e.g., "Width", "Height").

        Returns:
            Current integer value.
        """
        val = GX_INT_VALUE()
        status = self._dll.GXGetIntValue(self._handle, self._encode(feature_name), byref(val))
        self._check_status(status, f"GXGetIntValue({feature_name})")
        return val.nCurValue

    def get_int_range(self, feature_name):
        """
        Get an integer-type feature with full range info.

        Returns:
            dict with keys: cur, min, max, inc.
        """
        val = GX_INT_VALUE()
        status = self._dll.GXGetIntValue(self._handle, self._encode(feature_name), byref(val))
        self._check_status(status, f"GXGetIntValue({feature_name})")
        return {"cur": val.nCurValue, "min": val.nMin, "max": val.nMax, "inc": val.nInc}

    def set_int(self, feature_name, value):
        """Set an integer-type feature value."""
        self._check_stream_off("set_int")
        status = self._dll.GXSetIntValue(self._handle, self._encode(feature_name), c_int64(value))
        self._check_status(status, f"GXSetIntValue({feature_name})")

    def _align_int_feature_value(self, feature_name, requested):
        info = self.get_int_range(feature_name)
        min_value = int(info["min"])
        max_value = int(info["max"])
        inc = max(1, int(info["inc"]))
        value = max(min_value, min(max_value, int(requested)))
        return min_value + ((value - min_value) // inc) * inc

    def set_int_aligned(self, feature_name, value):
        """
        Set an integer feature after clamping to its supported range and increment.

        This is useful for ROI nodes such as Width, Height, OffsetX and OffsetY,
        where cameras often require values to be multiples of a node-specific
        increment.
        """
        aligned = self._align_int_feature_value(feature_name, value)
        self.set_int(feature_name, aligned)
        return aligned

    def set_roi(self, width, height, offset_x=0, offset_y=0, center=False):
        """
        Configure camera acquisition ROI.

        Width/Height in GenICam are sensor ROI dimensions, not image scaling.
        The safe order is:
          1. clear offsets to zero,
          2. set Width/Height,
          3. set final OffsetX/OffsetY.

        Must be called before stream_on().
        """
        self._check_stream_off("set_roi")
        sensor_width = self.get_int("WidthMax") if self.feature_is_available("WidthMax") else None
        sensor_height = self.get_int("HeightMax") if self.feature_is_available("HeightMax") else None

        self.set_int_aligned("OffsetX", 0)
        self.set_int_aligned("OffsetY", 0)
        actual_width = self.set_int_aligned("Width", width)
        actual_height = self.set_int_aligned("Height", height)

        if center:
            if sensor_width is None:
                sensor_width = self.get_int("Width")
            if sensor_height is None:
                sensor_height = self.get_int("Height")
            offset_x = max(0, (int(sensor_width) - int(actual_width)) // 2)
            offset_y = max(0, (int(sensor_height) - int(actual_height)) // 2)

        actual_offset_x = self.set_int_aligned("OffsetX", offset_x)
        actual_offset_y = self.set_int_aligned("OffsetY", offset_y)

        print(
            f"[GalaxyCamera] ROI 设置为 {actual_width}x{actual_height} "
            f"offset=({actual_offset_x}, {actual_offset_y})"
        )
        return {
            "width": actual_width,
            "height": actual_height,
            "offset_x": actual_offset_x,
            "offset_y": actual_offset_y,
            "sensor_width": sensor_width,
            "sensor_height": sensor_height,
        }

    def set_center_roi(self, width=1024, height=1024):
        """Configure a centered acquisition ROI, commonly used to raise FPS."""
        return self.set_roi(width, height, center=True)

    def get_float(self, feature_name):
        """
        Get a float-type feature value.

        Returns:
            Current float value.
        """
        val = GX_FLOAT_VALUE()
        status = self._dll.GXGetFloatValue(self._handle, self._encode(feature_name), byref(val))
        self._check_status(status, f"GXGetFloatValue({feature_name})")
        return val.dCurValue

    def get_float_range(self, feature_name):
        """
        Get a float-type feature with full range info.

        Returns:
            dict with keys: cur, min, max, inc, unit.
        """
        val = GX_FLOAT_VALUE()
        status = self._dll.GXGetFloatValue(self._handle, self._encode(feature_name), byref(val))
        self._check_status(status, f"GXGetFloatValue({feature_name})")
        return {
            "cur":  val.dCurValue,
            "min":  val.dMin,
            "max":  val.dMax,
            "inc":  val.dInc,
            "unit": val.szUnit.decode("utf-8", errors="replace") if val.szUnit else "",
        }

    def set_float(self, feature_name, value):
        """Set a float-type feature value."""
        self._check_stream_off("set_float")
        status = self._dll.GXSetFloatValue(self._handle, self._encode(feature_name), c_double(value))
        self._check_status(status, f"GXSetFloatValue({feature_name})")

    def get_enum(self, feature_name):
        """
        Get an enum-type feature value.

        Returns:
            dict with keys: symbolic (current symbolic name), value (current int value),
            entries (list of {symbolic, value} dicts).
        """
        val = GX_ENUM_VALUE()
        status = self._dll.GXGetEnumValue(self._handle, self._encode(feature_name), byref(val))
        self._check_status(status, f"GXGetEnumValue({feature_name})")

        entries = []
        for i in range(val.nSupportedNum):
            entry = val.nArrySupportedValue[i]
            entries.append({
                "symbolic": entry.strCurSymbolic.decode("utf-8", errors="replace"),
                "value":    entry.nCurValue,
            })
        return {
            "symbolic": val.stCurValue.strCurSymbolic.decode("utf-8", errors="replace"),
            "value":    val.stCurValue.nCurValue,
            "entries":  entries,
        }

    def set_enum(self, feature_name, value):
        """
        Set an enum-type feature by symbolic string name.

        Args:
            feature_name: The GenICam feature node name.
            value: The enum symbolic string (e.g., "Continuous" for AcquisitionMode).
        """
        self._check_stream_off("set_enum")
        status = self._dll.GXSetEnumValueByString(self._handle,
                                                   self._encode(feature_name),
                                                   self._encode(str(value)))
        self._check_status(status, f"GXSetEnumValueByString({feature_name}, {value})")

    def get_bool(self, feature_name):
        """
        Get a boolean-type feature value.

        Returns:
            True or False.
        """
        val = c_bool()
        status = self._dll.GXGetBoolValue(self._handle, self._encode(feature_name), byref(val))
        self._check_status(status, f"GXGetBoolValue({feature_name})")
        return val.value

    def set_bool(self, feature_name, value):
        """Set a boolean-type feature value."""
        self._check_stream_off("set_bool")
        status = self._dll.GXSetBoolValue(self._handle, self._encode(feature_name), c_bool(value))
        self._check_status(status, f"GXSetBoolValue({feature_name})")

    def get_node_access_mode(self, feature_name):
        """
        Check whether a feature node is readable/writable.

        Args:
            feature_name: The GenICam feature node name.

        Returns:
            GX_NODE_ACCESS_MODE value.
        """
        mode = c_int32()
        status = self._dll.GXGetNodeAccessMode(self._handle, self._encode(feature_name), byref(mode))
        self._check_status(status, f"GXGetNodeAccessMode({feature_name})")
        return mode.value

    def feature_is_available(self, feature_name):
        """Check if a feature node is available (exists and is readable or writable)."""
        mode = self.get_node_access_mode(feature_name)
        return mode in (GX_NODE_ACCESS_MODE.RO, GX_NODE_ACCESS_MODE.WO, GX_NODE_ACCESS_MODE.RW)

    def feature_is_writable(self, feature_name):
        """Return whether a feature node currently accepts writes."""
        mode = self.get_node_access_mode(feature_name)
        return mode in (GX_NODE_ACCESS_MODE.WO, GX_NODE_ACCESS_MODE.RW)

    def send_command(self, feature_name):
        """
        Send a command to the device (for command-type features).

        Args:
            feature_name: Command feature name (e.g., "AcquisitionStart").
        """
        status = self._dll.GXSendCommand(self._handle, self._encode(feature_name))
        self._check_status(status, f"GXSendCommand({feature_name})")

    # ---- NEW: Exposure / Gain / Acquisition FPS convenience methods ----

    def set_exposure_time_us(self, exposure_us):
        """
        关闭自动曝光并设置曝光时间（单位：微秒）。

        必须在 stream_on() 之前调用（受 set_enum/set_float 内部的
        _check_stream_off 保护）。

        如果相机不支持 ExposureAuto 节点（部分型号可能没有），
        会打印警告并跳过关闭自动曝光这一步，直接尝试设置 ExposureTime。
        """
        if self.feature_is_available("ExposureAuto"):
            try:
                self.set_enum("ExposureAuto", "Off")
            except GalaxyCameraError as exc:
                print(f"[GalaxyCamera] 关闭 ExposureAuto 失败（可能本就是手动模式）: {exc}")
        else:
            print("[GalaxyCamera] 当前设备不支持 ExposureAuto 节点，跳过关闭自动曝光")

        if self.feature_is_available("ExposureTime"):
            self.set_float("ExposureTime", float(exposure_us))
            print(f"[GalaxyCamera] ExposureTime 设置为 {exposure_us} us")
        else:
            print("[GalaxyCamera] 当前设备不支持 ExposureTime 节点，跳过曝光设置")

    def set_gain(self, gain):
        """
        关闭自动增益并设置增益值。

        必须在 stream_on() 之前调用。
        """
        if self.feature_is_available("GainAuto"):
            try:
                self.set_enum("GainAuto", "Off")
            except GalaxyCameraError as exc:
                print(f"[GalaxyCamera] 关闭 GainAuto 失败（可能本就是手动模式）: {exc}")
        else:
            print("[GalaxyCamera] 当前设备不支持 GainAuto 节点，跳过关闭自动增益")

        if self.feature_is_available("Gain"):
            self.set_float("Gain", float(gain))
            print(f"[GalaxyCamera] Gain 设置为 {gain}")
        else:
            print("[GalaxyCamera] 当前设备不支持 Gain 节点，跳过增益设置")

    def set_acquisition_fps(self, fps):
        """
        开启帧率限制模式（AcquisitionFrameRateMode=On）并设置采集帧率。

        注意：若曝光时间设置过长（例如 > 1/fps），实际帧率会被曝光时间
        限制住，达不到这里设置的目标值——这种情况下应优先调小曝光时间。

        必须在 stream_on() 之前调用。
        """
        if self.feature_is_available("AcquisitionFrameRateMode"):
            try:
                self.set_enum("AcquisitionFrameRateMode", "On")
            except GalaxyCameraError as exc:
                print(f"[GalaxyCamera] 开启 AcquisitionFrameRateMode 失败: {exc}")
        else:
            print("[GalaxyCamera] 当前设备不支持 AcquisitionFrameRateMode 节点，跳过")

        if self.feature_is_available("AcquisitionFrameRate"):
            self.set_float("AcquisitionFrameRate", float(fps))
            print(f"[GalaxyCamera] AcquisitionFrameRate 设置为 {fps}")
        else:
            print("[GalaxyCamera] 当前设备不支持 AcquisitionFrameRate 节点，跳过帧率设置")

    # ---- NEW: DataStream-level buffer handling mode ----

    def set_stream_buffer_mode(self, mode_name="newest_only"):
        """
        设置数据流的缓冲处理模式。

        这是解决"画面滞后/卡顿越来越严重"的关键设置之一：如果不设置，
        很多 SDK 默认是"先进先出"（OldestFirst），算法处理跟不上采集速度时，
        显示出来的画面会是几帧甚至几十帧之前的内容，并且延迟会越积越多。
        设为 newest_only 后，SDK 会优先交付最新帧，旧帧可被丢弃。

        Args:
            mode_name: "newest_only" / "oldest_first" / "oldest_first_overwrite"

        必须在 stream_on() 之前调用，且需要先拿到 DataStream handle。
        若该 SDK 版本/相机型号不支持此节点，会打印警告并跳过（不抛异常），
        不影响其余流程继续运行。
        """
        self._check_stream_off("set_stream_buffer_mode")

        name_map = {
            "oldest_first": GX_DS_STREAM_BUFFER_HANDLING_MODE.OLDEST_FIRST,
            "oldest_first_overwrite": GX_DS_STREAM_BUFFER_HANDLING_MODE.OLDEST_FIRST_OVERWRITE,
            "newest_only": GX_DS_STREAM_BUFFER_HANDLING_MODE.NEWEST_ONLY,
        }
        symbolic = name_map.get(mode_name)
        if symbolic is None:
            print(f"[GalaxyCamera] 未知的 buffer 模式: {mode_name}，跳过设置")
            return False

        try:
            ds_handle = self._get_ds_handle()
        except Exception as exc:
            print(f"[GalaxyCamera] 获取 DataStream handle 失败，跳过 buffer 模式设置: {exc}")
            return False

        access_mode = c_int32()
        status = self._dll.GXGetNodeAccessMode(
            ds_handle,
            self._encode("StreamBufferHandlingMode"),
            byref(access_mode),
        )
        if status != GX_STATUS_LIST.GX_STATUS_SUCCESS or access_mode.value not in (
            GX_NODE_ACCESS_MODE.WO,
            GX_NODE_ACCESS_MODE.RW,
        ):
            print("[GalaxyCamera] 当前 DataStream 不支持写入 "
                  "StreamBufferHandlingMode，跳过设置")
            return False

        # GXSetEnumValueByString receives a generic GX_PORT_HANDLE. A
        # GX_DS_HANDLE is a valid port handle; the SDK has no GXDS* variant.
        status = self._dll.GXSetEnumValueByString(
            ds_handle,
            self._encode("StreamBufferHandlingMode"),
            self._encode(symbolic),
        )
        if status != GX_STATUS_LIST.GX_STATUS_SUCCESS:
            print(f"[GalaxyCamera] 设置 StreamBufferHandlingMode={symbolic} 失败 "
                  f"(status={status})，可能该设备不支持此节点，跳过。")
            return False

        print(f"[GalaxyCamera] StreamBufferHandlingMode 设置为 {symbolic}")
        return True

    def get_stream_counters(self):
        """
        读取数据流的丢帧/不完整帧/已交付帧计数，用于诊断卡顿来源。

        lost 表示 SDK 输入 buffer 不足，可能是应用处理不及时；
        incomplete 表示收到残帧，通常更接近链路丢包问题。

        Returns:
            dict，键可能包括 delivered / lost / incomplete，
            具体取决于 SDK 是否暴露对应节点；读取失败的项会被跳过。
        """
        result = {}
        try:
            ds_handle = self._get_ds_handle()
        except Exception as exc:
            print(f"[GalaxyCamera] 获取 DataStream handle 失败，无法读取计数器: {exc}")
            return result

        for feature_name, key in (
            ("StreamDeliveredFrameCount", "delivered"),
            ("StreamLostFrameCount", "lost"),
            ("StreamIncompleteFrameCount", "incomplete"),
        ):
            val = GX_INT_VALUE()
            status = self._dll.GXGetIntValue(
                ds_handle,
                self._encode(feature_name),
                byref(val),
            )
            if status == GX_STATUS_LIST.GX_STATUS_SUCCESS:
                result[key] = val.nCurValue

        return result

    # ---- Streaming ----

    def stream_on(self):
        """Start image streaming."""
        with self._acquisition_lock:
            if self._streaming:
                return
            status = self._dll.GXStreamOn(self._handle)
            if status != GX_STATUS_LIST.GX_STATUS_SUCCESS:
                # Provide system-level diagnostic hints for common GenTL errors
                if status <= -1001:
                    self._print_stream_on_help(status)
                self._check_status(status, "GXStreamOn")
            self._streaming = True

    def _print_stream_on_help(self, status):
        """Print troubleshooting hints when GXStreamOn fails."""
        import platform
        print("\n" + "=" * 60)
        print("GXStreamOn FAILED — Troubleshooting Tips")
        print("=" * 60)
        if platform.system() == "Linux":
            print("""
1. USB Permissions (most common cause):
   Install the Galaxy udev rules:
     sudo cp <SDK>/config/99-galaxy-u3v.rules /etc/udev/rules.d/
     sudo udevadm control --reload-rules
     sudo udevadm trigger
   Then re-plug the camera (or reboot).

2. USBFS Buffer Size:
   For high-resolution cameras (e.g. 20MP+), increase USB buffer:
     sudo bash -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'
   Or make it permanent:
     echo 'options usbcore usbfs_memory_mb=1000' | sudo tee /etc/modprobe.d/usbcore.conf

3. Verify the camera is accessible:
     lsusb | grep -i daheng
     ls -la /dev/bus/usb/*/*

4. Try running with sudo to isolate permission issues:
     sudo python3 GalaxyCamera.py
""")
        print(f"Original error code: {status}")
        print("=" * 60 + "\n")

    def stream_off(self):
        """Stop image streaming."""
        with self._acquisition_lock:
            if not self._streaming:
                return
            status = self._dll.GXStreamOff(self._handle)
            self._check_status(status, "GXStreamOff")
            self._streaming = False

    def get_payload_size(self):
        """
        Get the payload size (image data size in bytes).

        Returns:
            Payload size in bytes.
        """
        ds_handle = self._get_ds_handle()
        payload = c_uint32(0)
        status = self._dll.GXGetPayLoadSize(ds_handle, byref(payload))
        self._check_status(status, "GXGetPayLoadSize")
        return payload.value

    def set_buffer_number(self, n):
        """Set the number of acquisition buffers in the queue."""
        self._check_stream_off("set_buffer_number")
        n = int(n)
        if n < 1:
            raise ValueError("Acquisition buffer number must be at least 1")
        status = self._dll.GXSetAcqusitionBufferNumber(self._handle, c_uint64(n))
        self._check_status(status, "GXSetAcqusitionBufferNumber")
        self._acquisition_buffer_number = n

    def get_buffer_number(self):
        """Return the SDK acquisition-buffer count when the API is available."""
        if hasattr(self._dll, "GXGetAcqusitionBufferNumber"):
            value = c_uint64(0)
            status = self._dll.GXGetAcqusitionBufferNumber(
                self._handle,
                byref(value),
            )
            if status == GX_STATUS_LIST.GX_STATUS_SUCCESS and value.value > 0:
                self._acquisition_buffer_number = int(value.value)
        return self._acquisition_buffer_number

    def get_frame(self, timeout_ms=1000):
        """
        Get a single frame from the acquisition queue (blocking).

        This uses the low-level DQBuf/QBuf API. You must call stream_on() first.

        Args:
            timeout_ms: Timeout in milliseconds.

        Returns:
            FrameData object or None on timeout.
        """
        with self._acquisition_lock:
            if not self._streaming:
                raise RuntimeError("Stream is not started. Call stream_on() first.")
            return self._get_frame_single(timeout_ms)

    def get_latest_frame(self, timeout_ms=1000):
        """
        Return the newest valid frame currently available from the SDK queue.

        This follows GalaxyView's real-time strategy: GXDQAllBufs returns all
        queued buffers from oldest to newest, only the newest valid frame is
        copied, and GXQAllBufs immediately returns every SDK buffer. Older
        frames are intentionally discarded so an algorithm always starts from
        fresh data instead of processing an ever-growing backlog.

        The returned FrameData owns a Python ``bytes`` copy, so it remains
        valid after the SDK buffers are returned. Call this method from only
        one acquisition thread. If this SDK lacks the batch APIs, the method
        safely falls back to get_frame() semantics.
        """
        with self._acquisition_lock:
            if not self._streaming:
                raise RuntimeError("Stream is not started. Call stream_on() first.")
            if not self._batch_acquisition_available():
                return self._get_frame_single(timeout_ms)

            capacity = self.get_buffer_number()
            if capacity is None or capacity < 1:
                self._warn_latest_fallback(
                    "无法读取 acquisition buffer 数量，回退到逐帧 GXDQBuf"
                )
                return self._get_frame_single(timeout_ms)
            if capacity > 0xFFFFFFFF:
                raise RuntimeError(
                    f"Acquisition buffer count {capacity} exceeds uint32 range"
                )

            frame_array = (PGX_FRAME_BUFFER * capacity)()
            frame_count = c_uint32(0)
            status = self._dll.GXDQAllBufs(
                self._handle,
                frame_array,
                c_uint32(capacity),
                byref(frame_count),
                c_uint32(timeout_ms),
            )

            if status == GX_STATUS_LIST.GX_STATUS_TIMEOUT:
                self._last_batch_size = 0
                return None
            if status == GX_STATUS_LIST.GX_STATUS_NOT_IMPLEMENTED:
                self._dq_all_usable = False
                self._warn_latest_fallback(
                    "当前 SDK 不实现 GXDQAllBufs，回退到逐帧 GXDQBuf"
                )
                return self._get_frame_single(timeout_ms)

            self._check_status(status, "GXDQAllBufs")

            frame = None
            processing_error = None
            try:
                count = int(frame_count.value)
                if count > capacity:
                    # The SDK promised not to write beyond the supplied array.
                    # Do not inspect out-of-range pointers; still return any
                    # buffers owned by this successful DQAll call.
                    raise RuntimeError(
                        f"GXDQAllBufs returned {count} frames for "
                        f"capacity {capacity}"
                    )

                self._last_batch_size = count
                self._host_dequeued_frames += count
                selected_frame = None
                valid_count = 0
                invalid_count = 0

                # The SDK documents the array order as old -> new. Inspect all
                # metadata so diagnostics account for every invalid buffer,
                # while retaining the last valid pointer as the newest frame.
                for index in range(count):
                    p_frame = frame_array[index]
                    if not p_frame:
                        invalid_count += 1
                        continue
                    raw = p_frame.contents
                    if (
                        raw.nStatus != GX_FRAME_STATUS.SUCCESS
                        or not raw.pImgBuf
                        or raw.nImgSize <= 0
                        or raw.nWidth <= 0
                        or raw.nHeight <= 0
                    ):
                        invalid_count += 1
                        continue
                    valid_count += 1
                    selected_frame = p_frame

                self._host_invalid_frames += invalid_count
                # "discarded_stale" means complete frames intentionally
                # skipped; damaged frames count only as "invalid", never twice.
                self._host_discarded_frames += max(0, valid_count - 1)

                if selected_frame is not None:
                    frame = FrameData(selected_frame)
            except BaseException as exc:
                processing_error = exc
            finally:
                # A successful DQAll transfers ownership even when count is 0
                # or processing raises. Always return the batch in this block.
                q_status = self._dll.GXQAllBufs(self._handle)
            self._raise_after_requeue(q_status, "GXQAllBufs", processing_error)
            return frame

    def get_host_acquisition_stats(self):
        """Return application-side batch/drop statistics for latest-frame reads."""
        with self._acquisition_lock:
            return {
                "last_batch_size": self._last_batch_size,
                "dequeued": self._host_dequeued_frames,
                "discarded_stale": self._host_discarded_frames,
                "invalid": self._host_invalid_frames,
            }

    def _get_frame_single(self, timeout_ms):
        """Internal FIFO GXDQBuf/GXQBuf implementation; caller holds the lock."""
        p_frame = PGX_FRAME_BUFFER()
        status = self._dll.GXDQBuf(
            self._handle,
            byref(p_frame),
            c_uint32(timeout_ms),
        )

        if status == GX_STATUS_LIST.GX_STATUS_TIMEOUT:
            self._last_batch_size = 0
            return None
        self._check_status(status, "GXDQBuf")

        frame = None
        processing_error = None
        try:
            # SUCCESS with a null output pointer violates the SDK contract.
            # There is no owned buffer to return, so never call GXQBuf(NULL).
            if not p_frame:
                raise RuntimeError(
                    "GXDQBuf succeeded but returned a null frame pointer"
                )

            self._last_batch_size = 1
            self._host_dequeued_frames += 1
            raw = p_frame.contents
            if (
                raw.pImgBuf
                and raw.nImgSize > 0
                and raw.nWidth > 0
                and raw.nHeight > 0
            ):
                frame = FrameData(p_frame)
                if raw.nStatus != GX_FRAME_STATUS.SUCCESS:
                    # Preserve get_frame()'s historical contract: callers may
                    # inspect FrameData.status even for an incomplete frame.
                    self._host_invalid_frames += 1
            else:
                self._host_invalid_frames += 1
        except BaseException as exc:
            processing_error = exc
        finally:
            q_status = (
                self._dll.GXQBuf(self._handle, p_frame)
                if p_frame
                else GX_STATUS_LIST.GX_STATUS_SUCCESS
            )

        if p_frame:
            self._raise_after_requeue(q_status, "GXQBuf", processing_error)
        elif processing_error is not None:
            raise processing_error.with_traceback(processing_error.__traceback__)
        return frame

    def get_image(self, timeout_ms=1000):
        """
        Get a single image (simpler API, legacy-style).

        Returns:
            GX_FRAME_DATA structure or None on timeout.
        """
        if not self._streaming:
            raise RuntimeError("Stream is not started. Call stream_on() first.")

        frame_data = GX_FRAME_DATA()
        status = self._dll.GXGetImage(self._handle, byref(frame_data), c_uint32(timeout_ms))

        if status == GX_STATUS_LIST.GX_STATUS_TIMEOUT:
            return None

        self._check_status(status, "GXGetImage")
        return frame_data

    def set_transfer_params(self, transfer_size=64*1024, transfer_number_urb=64):
        """
        Set stream transfer parameters (if supported by the device).

        Args:
            transfer_size: Size of each data transfer block in bytes.
            transfer_number_urb: Number of transfer blocks.
        """
        self._check_stream_off("set_transfer_params")
        if self.feature_is_available("StreamTransferSize"):
            self.set_int("StreamTransferSize", transfer_size)
        if self.feature_is_available("StreamTransferNumberUrb"):
            self.set_int("StreamTransferNumberUrb", transfer_number_urb)

    # ---- GigE Specific ----

    def get_persistent_ip(self):
        """Get the persistent (permanent) IP configuration for GigE devices."""
        if not hasattr(self._dll, 'GXGetDevicePersistentIpAddress'):
            raise RuntimeError("GXGetDevicePersistentIpAddress is not available in this SDK version.")
        ip_buf       = create_string_buffer(64)
        mask_buf     = create_string_buffer(64)
        gateway_buf  = create_string_buffer(64)
        ip_len       = c_size_t(64)
        mask_len     = c_size_t(64)
        gateway_len  = c_size_t(64)

        status = self._dll.GXGetDevicePersistentIpAddress(
            self._handle,
            ip_buf, byref(ip_len),
            mask_buf, byref(mask_len),
            gateway_buf, byref(gateway_len)
        )
        self._check_status(status, "GXGetDevicePersistentIpAddress")
        return {
            "ip":      ip_buf.value.decode("utf-8") if ip_buf.value else "",
            "mask":    mask_buf.value.decode("utf-8") if mask_buf.value else "",
            "gateway": gateway_buf.value.decode("utf-8") if gateway_buf.value else "",
        }

    def set_persistent_ip(self, ip, subnet_mask, gateway):
        """Set the persistent IP configuration for GigE devices."""
        if not hasattr(self._dll, 'GXSetDevicePersistentIpAddress'):
            raise RuntimeError("GXSetDevicePersistentIpAddress is not available in this SDK version.")
        status = self._dll.GXSetDevicePersistentIpAddress(
            self._handle,
            ct.c_char_p(ip.encode("utf-8")),
            ct.c_char_p(subnet_mask.encode("utf-8")),
            ct.c_char_p(gateway.encode("utf-8"))
        )
        self._check_status(status, "GXSetDevicePersistentIpAddress")

    # ---- Utility ----

    def get_last_error(self):
        """Get the last error code and description."""
        error_code = c_int32()
        size = c_size_t(256)
        buf = create_string_buffer(256)
        status = self._dll.GXGetLastError(byref(error_code), buf, byref(size))
        # Don't check status here to avoid recursion
        return {
            "code": error_code.value,
            "text": buf.value.decode("utf-8", errors="replace") if buf.value else "",
        }

    def reset_device(self, mode=GX_RESET_DEVICE_MODE.RECONNECT):
        """Reset/reconnect the device (may not be available in older SDK versions)."""
        if not hasattr(self._dll, 'GXResetDevice'):
            raise RuntimeError("GXResetDevice is not available in this SDK version.")
        status = self._dll.GXResetDevice(self._handle, c_int32(mode))
        self._check_status(status, "GXResetDevice")

    # ---- Convenience: Auto-open first device ----

    def open_first(self, access_mode=GX_ACCESS_MODE.EXCLUSIVE):
        """
        Enumerate devices and open the first one found.

        Returns:
            Device info dict.
        """
        count = self.update_device_list()
        if count == 0:
            raise RuntimeError("No Galaxy cameras found.")
        info = self.get_device_info(1)
        self.open_by_index(1, access_mode)
        return info

    # ---- Context Manager Support ----

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._streaming:
                self.stream_off()
        except Exception:
            pass
        try:
            if self._device_handle:
                self.close()
        except Exception:
            pass
        try:
            self.uninit()
        except Exception:
            pass
        return False

    # ---- Private Helpers ----

    @property
    def _handle(self):
        if not self._device_handle:
            raise RuntimeError("No device opened. Call open_by_index() or similar first.")
        return self._device_handle

    def _get_ds_handle(self):
        """Get the first data stream handle from the device."""
        n_streams = c_uint32(0)
        status = self._dll.GXGetDataStreamNumFromDev(self._handle, byref(n_streams))
        self._check_status(status, "GXGetDataStreamNumFromDev")

        if n_streams.value < 1:
            raise RuntimeError("No data streams available on this device.")

        ds_handle = GX_DS_HANDLE()
        status = self._dll.GXGetDataStreamHandleFromDev(self._handle, c_uint32(1), byref(ds_handle))
        self._check_status(status, "GXGetDataStreamHandleFromDev")
        return ds_handle

    def _reset_acquisition_state(self):
        """Reset state that belongs to one opened device/stream."""
        self._acquisition_buffer_number = None
        self._dq_all_usable = None
        self._latest_fallback_warning = None
        self._last_batch_size = 0
        self._host_dequeued_frames = 0
        self._host_discarded_frames = 0
        self._host_invalid_frames = 0

    def _batch_acquisition_available(self):
        """Return whether this SDK exposes the paired Linux batch APIs."""
        if self._dq_all_usable is None:
            self._dq_all_usable = bool(
                hasattr(self._dll, "GXDQAllBufs")
                and hasattr(self._dll, "GXQAllBufs")
            )
            if not self._dq_all_usable:
                self._warn_latest_fallback(
                    "当前 SDK 不提供 GXDQAllBufs/GXQAllBufs，回退到逐帧 GXDQBuf"
                )
        return self._dq_all_usable

    def _warn_latest_fallback(self, message):
        """Print each latest-frame fallback reason only once."""
        if self._latest_fallback_warning != message:
            print(f"[GalaxyCamera] {message}")
            self._latest_fallback_warning = message

    def _raise_after_requeue(self, status, operation, processing_error=None):
        """Validate QBuf/QAllBufs without leaking buffers or hiding failures."""
        if status != GX_STATUS_LIST.GX_STATUS_SUCCESS:
            try:
                self._check_status(status, operation)
            except BaseException as requeue_error:
                if processing_error is not None:
                    raise requeue_error from processing_error
                raise
        if processing_error is not None:
            raise processing_error.with_traceback(processing_error.__traceback__)

    def _check_stream_off(self, operation_name):
        """Ensure stream is off before performing a write operation."""
        if self._streaming:
            raise RuntimeError(
                f"Cannot call {operation_name} while stream is running. Call stream_off() first."
            )

    def _check_status(self, status, operation):
        """Check the return status and raise an exception if not success."""
        if status != GX_STATUS_LIST.GX_STATUS_SUCCESS:
            msg = STATUS_MESSAGES.get(status, '')
            if not msg:
                # Handle GenTL-level error codes (typically -1001 to -1020)
                if -1020 <= status <= -1001:
                    msg = f"GenTL transport layer error (code {status}). " \
                          f"Check that transport layer .cti files are accessible " \
                          f"and GENICAM_GENTL64_PATH is set correctly."
                else:
                    msg = f"Unknown error code: {status}"
            raise GalaxyCameraError(status, f"{operation}: {msg}")

    @staticmethod
    def _encode(s):
        """Encode a Python string to bytes for the SDK."""
        if isinstance(s, bytes):
            return s
        return s.encode("utf-8")

    @staticmethod
    def _int_to_ip(ip_int):
        """Convert a 32-bit integer IP to dotted string notation."""
        return f"{(ip_int >> 24) & 0xFF}.{(ip_int >> 16) & 0xFF}.{(ip_int >> 8) & 0xFF}.{ip_int & 0xFF}"

    def _setup_signatures(self):
        """
        Configure all ctypes function signatures.
        Each function is wrapped in try/except to be compatible with
        different SDK versions that may not export all symbols.
        """
        dll = self._dll

        def _safe_set(fn_name, restype, argtypes=None):
            """Safely set function signature if the symbol exists."""
            try:
                fn = getattr(dll, fn_name)
                fn.restype = restype
                if argtypes is not None:
                    fn.argtypes = argtypes
            except AttributeError:
                pass  # Symbol not available in this SDK version

        # Init / Close (essential)
        _safe_set('GXInitLib', GX_STATUS)
        _safe_set('GXCloseLib', GX_STATUS)

        # Version (Linux only)
        _safe_set('GXGetLibVersion', c_char_p)

        # Device enumeration
        _safe_set('GXUpdateAllDeviceList', GX_STATUS, [POINTER(c_uint32), c_uint32])
        _safe_set('GXUpdateAllDeviceListEx', GX_STATUS, [c_uint64, POINTER(c_uint32), c_uint32])

        # Device info / interface
        _safe_set('GXGetDeviceInfo', GX_STATUS, [c_uint32, POINTER(GX_DEVICE_INFO)])
        _safe_set('GXGetInterfaceNum', GX_STATUS, [POINTER(c_uint32)])

        # Open / Close device
        _safe_set('GXOpenDevice', GX_STATUS, [POINTER(GX_OPEN_PARAM), POINTER(GX_DEV_HANDLE)])
        _safe_set('GXCloseDevice', GX_STATUS, [GX_DEV_HANDLE])

        # Feature access - String
        _safe_set('GXGetStringValue', GX_STATUS, [GX_PORT_HANDLE, c_char_p, POINTER(GX_STRING_VALUE)])
        _safe_set('GXSetStringValue', GX_STATUS, [GX_PORT_HANDLE, c_char_p, c_char_p])

        # Feature access - Integer
        _safe_set('GXGetIntValue', GX_STATUS, [GX_PORT_HANDLE, c_char_p, POINTER(GX_INT_VALUE)])
        _safe_set('GXSetIntValue', GX_STATUS, [GX_PORT_HANDLE, c_char_p, c_int64])

        # Feature access - Float
        _safe_set('GXGetFloatValue', GX_STATUS, [GX_PORT_HANDLE, c_char_p, POINTER(GX_FLOAT_VALUE)])
        _safe_set('GXSetFloatValue', GX_STATUS, [GX_PORT_HANDLE, c_char_p, c_double])

        # Feature access - Enum
        _safe_set('GXGetEnumValue', GX_STATUS, [GX_PORT_HANDLE, c_char_p, POINTER(GX_ENUM_VALUE)])
        _safe_set('GXSetEnumValue', GX_STATUS, [GX_PORT_HANDLE, c_char_p, c_int64])
        _safe_set('GXSetEnumValueByString', GX_STATUS, [GX_PORT_HANDLE, c_char_p, c_char_p])

        # Feature access - Bool
        _safe_set('GXGetBoolValue', GX_STATUS, [GX_PORT_HANDLE, c_char_p, POINTER(c_bool)])
        _safe_set('GXSetBoolValue', GX_STATUS, [GX_PORT_HANDLE, c_char_p, c_bool])

        # Feature access - Command
        _safe_set('GXSendCommand', GX_STATUS, [GX_PORT_HANDLE, c_char_p])

        # Node access mode
        _safe_set('GXGetNodeAccessMode', GX_STATUS, [GX_PORT_HANDLE, c_char_p, POINTER(c_int32)])

        # Streaming
        _safe_set('GXStreamOn', GX_STATUS, [GX_DEV_HANDLE])
        _safe_set('GXStreamOff', GX_STATUS, [GX_DEV_HANDLE])

        # Data stream
        _safe_set('GXGetDataStreamNumFromDev', GX_STATUS, [GX_DEV_HANDLE, POINTER(c_uint32)])
        _safe_set('GXGetDataStreamHandleFromDev', GX_STATUS, [GX_DEV_HANDLE, c_uint32, POINTER(GX_DS_HANDLE)])
        _safe_set('GXGetPayLoadSize', GX_STATUS, [GX_DS_HANDLE, POINTER(c_uint32)])

        # Image acquisition (queue-based)
        _safe_set('GXDQBuf', GX_STATUS, [GX_DEV_HANDLE, POINTER(PGX_FRAME_BUFFER), c_uint32])
        _safe_set('GXQBuf', GX_STATUS, [GX_DEV_HANDLE, PGX_FRAME_BUFFER])
        _safe_set('GXDQAllBufs', GX_STATUS, [
            GX_DEV_HANDLE,
            POINTER(PGX_FRAME_BUFFER),
            c_uint32,
            POINTER(c_uint32),
            c_uint32,
        ])
        _safe_set('GXQAllBufs', GX_STATUS, [GX_DEV_HANDLE])

        # Image acquisition (simple)
        _safe_set('GXGetImage', GX_STATUS, [GX_DEV_HANDLE, POINTER(GX_FRAME_DATA), c_uint32])

        # Buffer number
        _safe_set('GXSetAcqusitionBufferNumber', GX_STATUS, [GX_DEV_HANDLE, c_uint64])
        _safe_set('GXGetAcqusitionBufferNumber', GX_STATUS, [GX_DEV_HANDLE, POINTER(c_uint64)])

        # Error
        _safe_set('GXGetLastError', GX_STATUS, [POINTER(c_int32), c_char_p, POINTER(c_size_t)])

        # GigE IP (may not be available on all SDK builds)
        _safe_set('GXGetDevicePersistentIpAddress', GX_STATUS, [
            GX_DEV_HANDLE, c_char_p, POINTER(c_size_t),
            c_char_p, POINTER(c_size_t),
            c_char_p, POINTER(c_size_t),
        ])
        _safe_set('GXSetDevicePersistentIpAddress', GX_STATUS, [
            GX_DEV_HANDLE, c_char_p, c_char_p, c_char_p,
        ])

        # Reset device (not available in older SDK versions)
        _safe_set('GXResetDevice', GX_STATUS, [GX_DEV_HANDLE, c_int32])


class LatestFrameReader:
    """Continuously acquire the newest camera frame on a worker thread.

    The worker owns all calls to ``GalaxyCamera.get_latest_frame``.  A single
    latest-frame slot decouples SDK buffer draining from a slower display or
    algorithm consumer: publishing a new frame replaces an unread old frame
    instead of blocking acquisition.

    ``FrameData`` owns a Python ``bytes`` copy, so frames remain valid after
    ``GXQAllBufs`` returns the SDK buffers.
    """

    def __init__(self, camera, timeout_ms=100):
        timeout_ms = int(timeout_ms)
        if timeout_ms < 1:
            raise ValueError("timeout_ms must be at least 1")

        self._camera = camera
        self._timeout_ms = timeout_ms
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread = None
        self._running = False
        self._error = None

        self._latest_frame = None
        self._latest_sequence = 0
        self._last_consumed_sequence = 0
        self._acquired_frames = 0
        self._overwritten_frames = 0
        self._timeouts = 0
        self._started_at = None
        self._last_frame_at = None

    def start(self):
        """Start the acquisition worker; the camera stream must already be on."""
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._running = True
            self._error = None
            self._latest_frame = None
            self._latest_sequence = 0
            self._last_consumed_sequence = 0
            self._acquired_frames = 0
            self._overwritten_frames = 0
            self._timeouts = 0
            self._started_at = time.perf_counter()
            self._last_frame_at = None
            self._thread = threading.Thread(
                target=self._acquisition_loop,
                name="GalaxyAcquisition",
                daemon=False,
            )
            self._thread.start()

    def stop(self, join_timeout=2.0):
        """Request worker shutdown and wait for the blocking SDK read to end."""
        if join_timeout is not None and join_timeout <= 0:
            raise ValueError("join_timeout must be positive or None")

        self._stop_event.set()
        with self._condition:
            thread = self._thread
            self._condition.notify_all()

        if thread is None or thread is threading.current_thread():
            return
        thread.join(join_timeout)
        if thread.is_alive():
            raise RuntimeError(
                "Galaxy acquisition thread did not stop; the SDK read may be stuck"
            )

    def wait_for_frame(self, after_sequence=0, timeout=None):
        """Return ``(sequence, FrameData)`` newer than ``after_sequence``.

        There is only one frame slot.  If the consumer is slower than the
        acquisition worker, this returns the newest frame and intermediate
        frames are counted as ``slot_overwritten`` in :meth:`get_stats`.
        """
        after_sequence = int(after_sequence)
        if after_sequence < 0:
            raise ValueError("after_sequence must not be negative")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must not be negative")

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while (
                self._latest_sequence <= after_sequence
                and self._error is None
                and self._running
            ):
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)

            if self._error is not None:
                raise RuntimeError("Galaxy acquisition thread failed") from self._error
            if self._latest_sequence <= after_sequence:
                return None

            sequence = self._latest_sequence
            frame = self._latest_frame
            self._last_consumed_sequence = max(
                self._last_consumed_sequence,
                sequence,
            )
            return sequence, frame

    def get_stats(self):
        """Return a thread-safe snapshot of background acquisition statistics."""
        with self._condition:
            now = self._last_frame_at
            if now is None and self._running:
                now = time.perf_counter()
            elapsed = (
                max(0.0, now - self._started_at)
                if now is not None and self._started_at is not None
                else 0.0
            )
            average_fps = (
                self._acquired_frames / elapsed
                if elapsed > 0.0
                else 0.0
            )
            return {
                "running": self._running,
                "acquired": self._acquired_frames,
                "slot_overwritten": self._overwritten_frames,
                "timeouts": self._timeouts,
                "latest_sequence": self._latest_sequence,
                "latest_frame_id": (
                    self._latest_frame.frame_id
                    if self._latest_frame is not None
                    else None
                ),
                "average_fps": average_fps,
                "error": repr(self._error) if self._error is not None else None,
            }

    def _acquisition_loop(self):
        try:
            while not self._stop_event.is_set():
                frame = self._camera.get_latest_frame(
                    timeout_ms=self._timeout_ms,
                )
                if frame is None:
                    with self._condition:
                        self._timeouts += 1
                    continue

                frame_time = time.perf_counter()
                with self._condition:
                    if self._latest_sequence > self._last_consumed_sequence:
                        self._overwritten_frames += 1
                    self._latest_sequence += 1
                    self._acquired_frames += 1
                    self._latest_frame = frame
                    self._last_frame_at = frame_time
                    self._condition.notify_all()
        except BaseException as exc:
            self._stop_event.set()
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            with self._condition:
                self._running = False
                self._condition.notify_all()

# ============================================================================
# Demo / Quick Test
# ============================================================================

"""
Galaxy Camera 实时预览脚本
--------------------------------
持续从大恒相机采集画面并实时显示。
相机按 --resolution 指定的 ROI 采集，预览在送入 GUI 前按需缩小。

按 q 或 ESC 退出。
"""
ESC_KEY = 27
PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 900
UINT64_MASK = (1 << 64) - 1
MAX_REASONABLE_FRAME_ID_DELTA = 1_000_000
DEFAULT_DISPLAY_STEP = 1
DEFAULT_READER_TIMEOUT_MS = 100
DISPLAY_WAIT_TIMEOUT_SECONDS = 0.02
GALAXY_RESOLUTION_PRESETS = {
    "4504x4504": (4504, 4504),
    "2048x2048": (2048, 2048),
    "1024x1024": (1024, 1024),
}
DEFAULT_GALAXY_RESOLUTION = "4504x4504"


def should_display_frame(sequence, last_displayed_sequence, display_step):
    """Return whether the newest acquired frame is due for display.

    The comparison uses the sequence distance instead of ``sequence % step``.
    This still displays the newest image when the one-slot handoff skipped over
    an exact multiple of ``display_step``.
    """
    display_step = int(display_step)
    if display_step < 1:
        raise ValueError("display_step must be at least 1")
    return int(sequence) - int(last_displayed_sequence) >= display_step


def normalize_resolution_preset(value):
    """Normalize CLI/programmatic preset spelling to the canonical key."""
    return str(value).strip().lower().replace("×", "x")


def set_resolution_preset(camera, resolution):
    """Apply one of the supported centered, camera-side ROI presets."""
    resolution = normalize_resolution_preset(resolution)
    try:
        requested_width, requested_height = GALAXY_RESOLUTION_PRESETS[resolution]
    except KeyError as exc:
        choices = ", ".join(GALAXY_RESOLUTION_PRESETS)
        raise ValueError(
            f"unsupported Galaxy resolution {resolution!r}; choose from {choices}"
        ) from exc

    width_max = int(camera.get_int("WidthMax"))
    height_max = int(camera.get_int("HeightMax"))
    if requested_width > width_max or requested_height > height_max:
        raise RuntimeError(
            f"requested ROI {requested_width}x{requested_height} exceeds the "
            f"camera maximum {width_max}x{height_max}"
        )

    roi_info = camera.set_roi(requested_width, requested_height, center=True)
    actual_width = int(roi_info["width"])
    actual_height = int(roi_info["height"])
    if (actual_width, actual_height) != (requested_width, requested_height):
        raise RuntimeError(
            f"camera aligned {requested_width}x{requested_height} to "
            f"{actual_width}x{actual_height}; the selected preset is not "
            "supported exactly by this device"
        )
    return roi_info


def main(argv=None):
    import argparse
    import cv2

    parser = argparse.ArgumentParser(
        description="Galaxy camera threaded live preview",
    )
    parser.add_argument(
        "--display-step",
        type=int,
        default=DEFAULT_DISPLAY_STEP,
        help="display one newest frame after every N acquired frames (default: 1)",
    )
    parser.add_argument(
        "--reader-timeout-ms",
        type=int,
        default=DEFAULT_READER_TIMEOUT_MS,
        help="SDK read timeout used by the acquisition thread (default: 100)",
    )
    parser.add_argument(
        "--resolution",
        type=normalize_resolution_preset,
        choices=tuple(GALAXY_RESOLUTION_PRESETS),
        default=DEFAULT_GALAXY_RESOLUTION,
        help=(
            "camera-side centered ROI preset; smaller values reduce transfer "
            f"bandwidth (default: {DEFAULT_GALAXY_RESOLUTION})"
        ),
    )
    args = parser.parse_args(argv)
    if args.display_step < 1:
        parser.error("--display-step must be at least 1")
    if args.reader_timeout_ms < 1:
        parser.error("--reader-timeout-ms must be at least 1")

    cam = GalaxyCamera()
    reader = None

    try:
        cam.init()
        print(f"SDK Version: {cam.get_lib_version()}")

        devices = cam.list_devices()
        print(f"Found {len(devices)} device(s).")
        if not devices:
            print("No devices found. Exiting.")
            return

        cam.open_by_index(1)
        print("Device opened successfully.")

        # 配置采集
        cam.set_enum("AcquisitionMode", "Continuous")
        cam.set_enum("TriggerMode", "Off")
        try:
            cam.set_enum("PixelFormat", "Mono8")  # 带宽占用最低的格式
        except GalaxyCameraError as exc:
            print(f"[GalaxyCamera] Mono8 unavailable; using camera format: {exc}")

        roi_info = set_resolution_preset(cam, args.resolution)
        width = roi_info["width"]
        height = roi_info["height"]

        # 曝光 / 增益 / 帧率 / 缓冲模式：必须在 stream_on() 之前设置
        cam.set_exposure_time_us(10000)
        cam.set_gain(10)

        # 注释 set_acquisition_fps(20) 不会清除相机残留的 On/20 状态。
        if cam.feature_is_writable("AcquisitionFrameRateMode"):
            cam.set_enum("AcquisitionFrameRateMode", "Off")
        elif cam.feature_is_available("AcquisitionFrameRateMode"):
            frame_rate_mode = cam.get_enum("AcquisitionFrameRateMode")["symbolic"]
            print(
                "[GalaxyCamera] AcquisitionFrameRateMode is read-only: "
                f"{frame_rate_mode}"
            )

        cam.set_buffer_number(5)
        cam.set_transfer_params(transfer_size=64 * 1024, transfer_number_urb=64)
        cam.set_stream_buffer_mode("newest_only")

        print(f"[GalaxyCamera] Acquisition ROI: {width}x{height}")
        if cam.feature_is_available("AcquisitionFrameRate"):
            frame_rate_range = cam.get_float_range("AcquisitionFrameRate")
            print(
                "[GalaxyCamera] AcquisitionFrameRate supported range: "
                f"{frame_rate_range['min']:.2f} .. "
                f"{frame_rate_range['max']:.2f} FPS (limit disabled)"
            )
        window_name = "Galaxy Camera Live Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print(
            "Starting threaded stream... "
            f"(display_step={args.display_step}, 按 q 或 ESC 退出窗口)"
        )
        cam.stream_on()
        reader = LatestFrameReader(cam, timeout_ms=args.reader_timeout_ms)
        reader.start()

        last_seen_sequence = 0
        last_displayed_sequence = 0
        preview_size = None
        capture_fps_value = 0.0
        display_fps_value = 0.0
        frame_id_fps_value = None
        displayed_since_report = 0
        step_skipped = 0
        metrics_timer = time.perf_counter()
        metrics_acquired = 0
        metrics_frame_id = None

        while True:
            result = reader.wait_for_frame(
                after_sequence=last_seen_sequence,
                timeout=DISPLAY_WAIT_TIMEOUT_SECONDS,
            )
            reader_stats = reader.get_stats()
            now = time.perf_counter()

            if metrics_frame_id is None and reader_stats["latest_frame_id"] is not None:
                metrics_timer = now
                metrics_acquired = reader_stats["acquired"]
                metrics_frame_id = int(reader_stats["latest_frame_id"])
            else:
                metrics_elapsed = now - metrics_timer
                if metrics_elapsed >= 1.0:
                    capture_fps_value = (
                        reader_stats["acquired"] - metrics_acquired
                    ) / metrics_elapsed
                    display_fps_value = displayed_since_report / metrics_elapsed

                    latest_frame_id = reader_stats["latest_frame_id"]
                    if latest_frame_id is not None and metrics_frame_id is not None:
                        frame_id_delta = (
                            int(latest_frame_id) - metrics_frame_id
                        ) & UINT64_MASK
                        if 0 < frame_id_delta <= MAX_REASONABLE_FRAME_ID_DELTA:
                            frame_id_fps_value = frame_id_delta / metrics_elapsed
                        else:
                            frame_id_fps_value = None
                        metrics_frame_id = int(latest_frame_id)

                    metrics_timer = now
                    metrics_acquired = reader_stats["acquired"]
                    displayed_since_report = 0

            if result is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ESC_KEY or key == ord("q"):
                    break
                continue

            frame_sequence, frame = result
            last_seen_sequence = frame_sequence

            # Display only every N acquired frames.  The acquisition worker is
            # unaffected and keeps draining/requeueing SDK buffers continuously.
            if not should_display_frame(
                frame_sequence,
                last_displayed_sequence,
                args.display_step,
            ):
                step_skipped += 1
                key = cv2.waitKey(1) & 0xFF
                if key == ESC_KEY or key == ord("q"):
                    break
                continue
            last_displayed_sequence = frame_sequence

            if frame.pixel_format == GX_PIXEL_FORMAT.MONO8:
                img = frame.to_numpy()
            else:
                img = frame.to_bgr()
            if img is None:
                print(f"[Warning] 不支持的像素格式: {frame.pixel_format:#x}，跳过该帧")
                continue

            if preview_size is None:
                scale = min(
                    1.0,
                    PREVIEW_MAX_WIDTH / frame.width,
                    PREVIEW_MAX_HEIGHT / frame.height,
                )
                preview_size = (
                    max(1, int(round(frame.width * scale))),
                    max(1, int(round(frame.height * scale))),
                )
                cv2.resizeWindow(window_name, *preview_size)

            # Keep the selected camera ROI unchanged for algorithms, but
            # copy/scale only a small grayscale preview for WSL/OpenCV GUI.
            if (frame.width, frame.height) == preview_size:
                preview = img.copy()
            else:
                preview = cv2.resize(
                    img,
                    preview_size,
                    interpolation=cv2.INTER_AREA,
                )

            frame_id_fps_text = (
                f"{frame_id_fps_value:.1f}"
                if frame_id_fps_value is not None
                else "n/a"
            )

            cv2.putText(
                preview,
                f"Acq: {capture_fps_value:.1f}  Display: {display_fps_value:.1f}  "
                f"Frame IDs: {frame_id_fps_text}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                255 if preview.ndim == 2 else (0, 255, 0),
                2,
            )
            cv2.putText(
                preview,
                f"ROI: {width}x{height}  Step: {args.display_step}  "
                f"Slot drop: {reader_stats['slot_overwritten']}  "
                f"Step skip: {step_skipped}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                255 if preview.ndim == 2 else (0, 255, 0),
                2,
            )

            cv2.imshow(window_name, preview)
            displayed_since_report += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ESC_KEY or key == ord("q"):
                break

    except GalaxyCameraError as e:
        print(f"\nGalaxy Camera Error: {e}")
    except RuntimeError as e:
        print(f"\nGalaxy camera runtime error: {e}")
    except KeyboardInterrupt:
        print("\n用户中断 (Ctrl+C)")
    finally:
        if reader is not None:
            try:
                reader.stop()
                print(f"[Galaxy] Background reader: {reader.get_stats()}")
            except Exception as exc:
                print(f"[Galaxy] Failed to stop acquisition thread cleanly: {exc}")
        cv2.destroyAllWindows()
        try:
            print(f"[Galaxy] Stream counters: {cam.get_stream_counters()}")
        except Exception:
            pass
        try:
            print(f"[Galaxy] Host acquisition: {cam.get_host_acquisition_stats()}")
        except Exception:
            pass
        try:
            cam.stream_off()
        except Exception:
            pass
        try:
            cam.close()
        except Exception:
            pass
        cam.uninit()
        print("SDK uninitialized. Done.")


if __name__ == "__main__":
    main()
