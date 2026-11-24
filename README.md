# 相机标定

一个 H5 工作台，用于相机拍摄、普通内参标定、双目外参标定、Marker 标定、竖轴定位和 IMU 预览。支持本机相机，也支持通过 SSH 使用远端 Linux 主机上的相机。

## 启动

在工程根目录运行。下面两条任选其一，打开同一个 H5 工作台。

```powershell
python python/scripts/calibrate.py studio --local
python python/scripts/calibrate.py studio --ssh camera-lab
```

在页面里选模式，再点「连接相机」。**相机拍摄**只保存原图。选择**普通标定、双目标定、Marker 标定或竖轴定位**后，点击「结束采集」会自动求解并在任务日志中报告输出目录。**IMU 预览**显示视觉和 IMU 姿态。切换模式前先结束采集或关闭 IMU 预览，等待相机释放。

## 直接使用命令行

只开相机拍摄，用同一个 `camera` 命令选择本机或远端。

两种方式都会打开 H5。接入大恒相机后，普通拍摄和工作台的采集页都可实时调曝光（µs）和增益（dB）。提交后在相机日志查看设备回读值。

```powershell
python python/scripts/calibrate.py camera --local
python python/scripts/calibrate.py camera --ssh camera-lab
```

直接做标定，使用同一个入口并选一份 `--config`。采集结束后自动求解。本机直接命令使用 OpenCV 窗口，SSH 使用 H5。

```powershell
python python/scripts/calibrate.py --ssh camera-lab --config configs/markers/pivot.json
```

| 标定 | 配置 |
| --- | --- |
| 普通内参 | [configs/intrinsics/galaxy.json](configs/intrinsics/galaxy.json) |
| 双目外参 | [configs/extrinsics/galaxy_vcamera_color.json](configs/extrinsics/galaxy_vcamera_color.json)，其他组合见同目录 |
| Marker 几何 | [configs/markers/double_cube.json](configs/markers/double_cube.json) |
| 竖轴定位 | [configs/markers/pivot.json](configs/markers/pivot.json)，填写已标好的 marker 几何和对应内参 |

选择 `--local` 即可在本机执行。已有原图时，在同一命令中加子命令 `calibrate` 和 `--session data/SESSION`。直接命令的离线检测、校验等选项见 `calibrate.py --help`。

## 配置和结果

- 工作台各页签使用的配置在 [configs/studio.json](configs/studio.json)。传 `studio --config 路径` 可切换整套预设。
- 相机分辨率、曝光、增益和设备选择在 `configs/cameras/`。棋盘格数和格长在标定任务的配置里。同一双目任务的两台相机须能被同一执行主机访问。
- SSH 账号在 `configs/ssh.local.json`，参考 [模板](configs/ssh.example.json)。该文件可保存密码，已被 Git 忽略。使用 SSH 的工作台时，配置已保存的密码或可直接使用的 SSH 密钥。远端须安装相机 SDK 和依赖，并在远端工程目录准备对应的配置及输入文件。SSH 主机公钥须加入 `known_hosts`。
- 原图写入执行主机的 `data/`，结果写入 `output/`。远端任务的文件留在远端，路径会显示在日志里。
- 竖轴定位时固定红宝石球心，沿不同方向倾斜并改变滚转。建议采 45–70 个停稳姿态，程序按采集顺序留出最后四分之一，按配置中的毫米阈值验收。分开采集拟合与验证图时，可用 `calibrate.py pivot --help` 查看选项。

## 首次安装

建议使用 Conda 环境 `cylmarker`，Ubuntu 22.04 使用 Python 3.10，Windows x64 使用 Python 3.13。在实际连接相机的主机安装对应厂商 SDK。

```powershell
conda activate cylmarker
python -m pip install -r requirements-calibration.txt
python -m python.scripts.install_sdks galaxy vcamera
```

SDK 选项可用 `galaxy`、`vcamera`、`orbbec`，按实际设备安装。IMU 页需要先编译 C++ 桥接程序，见 [IMU 安装和坐标约定](python/scripts/imu_calibration/README.md)。

常规 marker 采集、检测和标定使用 `python python/scripts/calibrate_marker.py --help`。以棋盘格为尺度标定器械 C 时，使用 `python -m python.scripts.marker_calibration.checkerboard_geometry --help`。多器械棋盘标定使用 `python -m python.scripts.marker_calibration.checkerboard --help`。

使用 `python -m python.scripts.marker_calibration.report --output <结果目录>` 生成器械 C 的报告。生成多器械棋盘标定报告时，追加 `--checkerboard`。

#### ID 82–90 矩形靶标

使用现有的 [double_cube.json](configs/markers/double_cube.json)。配置沿用 Galaxy 内参和 `DICT_4X4_100`。棋盘配置为 **11×8 内角点、相邻内角点间距 35 mm**，请按实际棋盘核对。联合 BA 用棋盘角点固定毫米尺度，ID 90 只定义靶标坐标系，其宽和高均参与优化。保持 `marker.reference_marker_length_mm` 为 `null`。

棋盘可以一直留在原处。每组内保持靶标不动，移动相机，从不同倾角拍到完整棋盘和靶标。只有切换到下一组后才重新摆放靶标。

1. 第一组从斜上方拍，常见组合是顶部 ID 90 加一至三个侧码。每张至少看见顶部码和一个侧码。用空格保存，画面显示每个 ID 的训练张数、留出张数和视差。
2. 默认每个合格 ID 需要 **5 张训练图、1 张留出图、至少 10° 训练视差**，所以每组至少 **6 张有效照片**。太接近已拍视角、棋盘不全或可见码不足的照片会被跳过。第一组需拍足顶部码和至少一个侧码。程序提示本组就绪后，按 **N** 或点 **下一组**，再换靶标姿态。
3. 后续每组至少拍足两个码，其中至少一个是前面组已拍足的共视码。画面会列出共视码和全局缺失码。覆盖全部 9 个 ID 且训练图总数至少 15 张后，结束采集。未完成时退出会保留原图；续采未完成组时要保持退出前的摆放。

BA 为每组估计独立的靶标到棋盘变换，所有组共享一套矩形 marker 几何。留出图像只用棋盘确定相机位姿，再检查 marker 重投影误差。新会话保存 2026-11-10 版 Galaxy 内参，续采和标定已有会话时使用会话里的内参副本。

```powershell
python python/scripts/calibrate_marker.py capture
python python/scripts/calibrate_marker.py calibrate --session "data/<会话目录>"
```

BA 独立拟合每个 marker 的宽和高。上下边共用宽度，左右边共用高度，四角共面且相邻边垂直。质量报告的 `marker_dimensions_mm` 记录各码的宽高。YAML 的 `marker_positions` 保存矩形四角，兼容字段 `marker_lengths` 保存宽高平均值。

### Python 模块分工

| 位置 | 职责 |
| --- | --- |
| `python/scripts/workflow/` | 读取工作流配置，调度命令行和 Studio |
| `python/scripts/sensor_capture/` | 设备适配、相机选项、帧读取和实时预览 |
| `python/scripts/sensor_intrinsics/` | 棋盘采集会话、角点测量、内参拟合与验证 |
| `python/scripts/stereo_calibration/` | 传感器配对标定、固定参数复核和质量报告 |
| `python/scripts/marker_calibration/` | ArUco 采集会话、marker 几何和尖端标定 |
| `python/scripts/session_storage.py` | 会话路径、图像与标定结果落盘 |

查看 Vcamera 设备、图像模式和控制参数，或按相机配置保存一帧用于诊断，可运行 `python python/scripts/calibrate.py inspect-vcamera --help`。
