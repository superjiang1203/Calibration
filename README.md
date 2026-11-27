# 相机标定

一个 H5 工作台，用于普通内参标定、双目外参标定、Marker 标定、竖轴定位和 IMU 预览。支持本机相机，也支持通过 SSH 使用远端 Linux 主机上的相机。

## 启动

在工程根目录运行。下面两条任选其一，打开同一个 H5 工作台。

```powershell
python python/scripts/calibrate.py studio --local
python python/scripts/calibrate.py studio --ssh camera-lab
```

## 工作台操作

在左侧选择功能，在右侧填写或保留默认配置，再启动任务。采集页显示当前任务名，实时采集和 IMU 预览都在工作台内。完成后结束采集并等待相机释放，再启动其他任务。

双目任务按配置顺序左右显示两路相机，缺帧时在对应位置提示。将鼠标放在画面上滚动，或点该画面上方的「＋」「－」，即可单独缩放。放大后可拖动查看局部，「适应画面」恢复初始比例。IMU 的实时图和角点调试图同样左右显示，坐标选项和日志可在画面下方展开。

使用结束后点右上角「退出工作台」。确认后，Studio 会停止当前任务、释放相机和 SSH 连接，并尝试关闭该页签。若浏览器限制自动关闭，页面会显示工作台已退出。也可以在启动终端按 `Ctrl+C` 退出。

相机连接中或已连接时，左侧页签会变灰。断开当前任务的全部相机后可切换页签，已保存的采集记录会保留。调整曝光或增益时，界面会等待设备回读并显示实际值，期间暂停其他相机操作。

| 模式 | 页面操作 |
| --- | --- |
| Galaxy 灰度、Vcamera RGB | 在工作台内打开对应预览，再点「连接相机」。支持画面缩放和保存原图。Vcamera 页还可枚举设备和查看控制项。 |
| 普通内参 | Galaxy 可用棋盘图像拟合和复核；Vcamera RGB 可预览并导出 SDK 出厂内参。 |
| 双目外参 | 按任务配置分别连接两台相机。一台失败时另一台仍可预览，连接和取帧错误显示在相机日志。两路都取得有效帧后才允许保存双目配对。 |
| Marker 几何标定 | 在页面分组采集，覆盖完成后运行矩形 BA。也可以续采、标定已有会话或单独检查角点。 |
| 竖轴定位 | 填写执行主机上已标定的 Marker 几何 YAML，可只采集、采集并求解，或用已有会话求解。 |
| IMU 预览 | 填写本机几何 YAML，查看视觉和 IMU 姿态。用 Galaxy 时填写内参 YAML，用 Vcamera 时由 SDK 读取内参。完成后点「关闭 IMU 预览」。 |

Marker、竖轴定位和 IMU 页可以选择 Galaxy 或 Vcamera。选 Vcamera 时，连接设备后由 SDK 读取当前采集模式的出厂内参，页面无需填写内参路径。Marker 和竖轴会将内参写入会话的 `camera_params.yml`，供续采或离线求解。选 Galaxy 时使用对应设备和分辨率的标定 YAML。SSH 模式下 Marker 和竖轴的 Galaxy 内参路径以远端工程为准。IMU 预览支持本机 YAML 路径，也支持以 `/data2/…` 开头的远端 Linux 绝对路径。

使用瑞芬 IMU406 时，在 IMU 预览选择「瑞芬 IMU406」，填写 **运行 Studio 的主机**上的串口，例如 Windows 的 `COM5` 或 Linux 的 `/dev/ttyUSB0`。只有一个 USB 串口时可保留 `auto`。通过 SSH 使用远端相机时，IMU 仍接在启动 Studio 的主机上。串口默认使用 RION 68 协议、115200 波特、8N1、设备地址 0，可在 `configs/studio.json` 的 `imu.rion` 中调整。页面显示姿态、三轴加速度和三轴角速度。模块的 TTL 接口为 3.3 V，接电脑时使用配套 USB 转接器或兼容的 3.3 V USB–TTL 转接器。

Galaxy 和 Vcamera 配对时，程序在同一采集轮次读取两路图像。采集时保持棋盘和相机静止，两路的曝光时刻由各自设备决定。

Marker 页默认使用 [double_cube_82_90.json](configs/markers/double_cube_82_90.json) 中的 **ID 82–90**、**11×8 内角点**和 **35 mm 棋盘间距**。布局为顶部 **90**、上层 **89/88/87/86**、下层 **85/84/83/82**，两层按相同方向排列。第一组从上方斜拍，每张有效图拍到完整棋盘、顶部 **90** 和至少一个侧码。程序从已经拍足的侧码中选择一个交接码并显示出来。

第 2–5 组的目标依次为 **89、88、87、86**。上下层交错，侧码按 **89 → 83 → 88 → 82 → 87 → 85 → 86 → 84 → 89** 排列，每个目标码和两侧相邻的下层码一起纳入本组。第 2 组从交接码可见的一侧开始拍，页面会列出沿途用于共视的码。同一组保持靶标和棋盘的相对位姿固定，移动相机分帧拍摄。每组至少保存 **10 张有效图**，默认每第 5 张作为留出图。必拍码分别达到 **5 张训练图、1 张留出图和至少 10° 视差**，且在训练图中形成连通的共视关系后，才可点「下一组」或按 **N**。

每次保存都要求当前原图识别出棋盘格的全部 **88 个内角点**。棋盘位姿用 PnP-RANSAC 求解并用内点精化，随后用全部内角点计算重投影 RMS，默认上限为 **2 px**（`calibration.max_board_pnp_rmse_px`）。求解失败、内点不足或整帧 RMS 超限时取消保存，采集页显示原因，有效图计数保持不变。默认至少需要 **5 个达标组**（`calibration.min_groups`），且所有靶标码都要覆盖，才能结束采集并运行 BA。中断后重新打开 Studio，Marker 页会自动填入上次未完成的会话。续采当前组时恢复中断前的靶标与棋盘相对摆位。成功导出几何 YAML 后，会话的自动选择会清空。

预览显示当前可见的各靶标码到相机光心的估计距离（mm）。每个码在本组至少有 3 张已保存的训练图且达到视差要求后，系统用 35 mm 棋盘定尺度并三角化码的四角。实时画面也要看到完整棋盘，才能用当前相机位置计算码心距离。

离线标定会先用同一规则重检原图，棋盘 PnP 筛选结果保存在 `checkerboard_observations.json`。首次 BA 收敛后，按「图片 + ID」剔除训练帧中重投影 RMS 超过 **2 px**（`calibration.max_observation_rmse_px`）或角点深度无效的码。剩余至少 **2 个靶标码**的帧继续使用，少于 2 个则整帧移出训练。系统检查组数、有效帧数和码覆盖，再用余下的数据拟合。`marker_filter.json` 记录剔除的码、帧和重投影误差。留出帧保留独立验证。

BA 用棋盘 RANSAC 内点定毫米尺度，配置指定的参考码定义靶标坐标系，各码宽高独立拟合。求解得到几何后，每个 ID 至少有 **5 张有效训练图**即可导出 YAML。若原训练图覆盖不足，会保留首轮独立留出报告 `marker_holdout_quality.json`，再纳入原留出图重新筛选和拟合。留出误差及其他质量指标作为诊断记录。最终报告是 `marker_calibration_quality.json`，几何 YAML 的 `marker_positions` 保存四角。新会话使用配置指定的 Galaxy 内参，续采和重新标定读取会话内的副本。

Galaxy 灰度和 Vcamera RGB 预览支持加载 Marker 几何 YAML。填写执行主机上的路径再打开预览，画面会叠加定位到的码框和 ID。Galaxy 需填写对应分辨率的内参 YAML，Vcamera RGB 从 SDK 读取内参。加载末端标定生成的几何 YAML 后，画面还会显示尖端十字和相机坐标系下的 XYZ（mm）。

Galaxy 灰度和 Vcamera RGB 预览也可点击「开始录像」。录像保存预览画面及当前叠加层，单击「停止录像并保存 MP4」或断开相机后完成写入。MP4 位于执行主机本次会话的 `data/…` 目录，完整路径显示在相机日志里。

## 配置和结果

- 工作台各页签使用的配置在 [configs/studio.json](configs/studio.json)。传 `studio --config 路径` 可切换整套预设。
- 相机分辨率、曝光、增益和设备选择在 `configs/cameras/`。棋盘格数和格长在标定任务的配置里。同一双目任务的两台相机须能被同一执行主机访问。
- SSH 账号在 `configs/ssh.local.json`，参考 [模板](configs/ssh.example.json)。该文件可保存密码，已被 Git 忽略。使用 SSH 的工作台时，配置已保存的密码或可直接使用的 SSH 密钥。远端须安装相机 SDK 和依赖，并在远端工程目录准备对应的配置及输入文件。SSH 主机公钥须加入 `known_hosts`。
- 原图写入执行主机的 `data/`，结果写入 `output/`。当前任务会显示会话和结果路径，详细输出在任务日志。SSH 模式下，会话、标定配置和结果路径指向远端工程，IMU 页明确标出的 YAML 输入来自本机。
- 竖轴定位时固定红宝石球心，沿不同方向倾斜并改变滚转。建议采 45–70 个停稳姿态，程序按采集顺序留出最后四分之一，按配置中的毫米阈值验收。分开采集拟合与验证图时，可用 `calibrate.py pivot --help` 查看选项。

## 首次安装

建议使用 Conda 环境 `cylmarker`，Ubuntu 22.04 使用 Python 3.10，Windows x64 使用 Python 3.13。在实际连接相机的主机安装对应厂商 SDK。

```powershell
conda activate cylmarker
python -m pip install -r requirements-calibration.txt
python -m python.scripts.install_sdks galaxy vcamera
```

SDK 选项可用 `galaxy`、`vcamera`、`orbbec`，按实际设备安装。IMU 页需要先编译 C++ 桥接程序，见 [IMU 安装和坐标约定](python/scripts/imu_calibration/README.md)。

### Python 模块分工

| 位置 | 职责 |
| --- | --- |
| `python/scripts/workflow/` | 读取工作流配置，调度命令行和 Studio |
| `python/scripts/sensor_capture/` | 设备适配、相机选项、帧读取和实时预览 |
| `python/scripts/sensor_intrinsics/` | 棋盘采集会话、角点测量、内参拟合与验证 |
| `python/scripts/stereo_calibration/` | 传感器配对标定、固定参数复核和质量报告 |
| `python/scripts/marker_calibration/` | ArUco 采集会话、marker 几何和尖端标定 |
| `python/scripts/session_storage.py` | 会话路径、图像与标定结果落盘 |
