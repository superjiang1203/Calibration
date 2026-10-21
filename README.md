# 相机标定

图漾、奥比使用 SDK 内参；大恒支持单独标内参，或先估计内参再求双机外参。

## 平台区别

下表以仓库提供的图漾 wheel 为基准。分别按照对应系统的完整步骤部署。

| 项目 | Linux | Windows |
| --- | --- | --- |
| 系统 | Ubuntu 22.04 x86_64 | Windows x64 |
| 终端 | Bash | PowerShell |
| Python | 3.10 | 3.13 |
| 本文环境方案 | Conda 环境 `camera` | 工程内的 venv `.venv` |
| 激活命令 | `conda activate camera` | `.\.venv\Scripts\Activate.ps1` |
| 图漾 wheel | `cp310-cp310-linux_x86_64` | `cp313-cp313-win_amd64` |
| Galaxy 原生 SDK | Linux 运行库 `/usr/lib/libgxiapi.so` | Windows 运行库及 `GALAXY_GENICAM_ROOT` 环境变量 |
| 本文工程路径示例 | `/data2/project/Calibration` | `D:\jlc\calibration` |

`data/`、`output/` 和配置文件可以在两种系统之间复制；Python 环境及 SDK 二进制文件须在目标系统重新安装。两种系统的图漾 wheel 不能互换。

## Linux：Ubuntu 22.04 / Bash

以下命令仅在 Linux Bash 中执行，以大恒与奥比为例。

### 1. 准备系统 SDK

先按厂商安装说明安装 **Galaxy Linux SDK**，包括原生运行库和相机 USB 权限规则。仓库中的 gxipy wheel 仅提供 Python 绑定，不能替代原生 SDK。安装后确认运行库存在：

```bash
ls -l /usr/lib/libgxiapi.so
```

实时采集需要 OpenCV 图形窗口，应在可用的 Linux 图形会话中运行。纯 SSH 终端可执行后面的离线计算命令；实时采集还需要配置可用的图形显示。

### 2. 首次部署

以下使用 `/data2/project/Calibration` 作为工程目录。已有工程时跳过克隆；已有 Python 3.10 的 `camera` 环境时跳过环境创建。

```bash
git clone https://github.com/superjiang1203/Calibration.git /data2/project/Calibration
cd /data2/project/Calibration
conda create -n camera python=3.10 -y
conda activate camera
which python
python --version
python -m pip install -r requirements-calibration.txt
python -m python.scripts.install_sdks galaxy orbbec
python -m python.scripts.install_sdks galaxy orbbec --check
```

使用图漾时，在同一环境中追加安装 Linux 图漾绑定并检查：

```bash
python -m python.scripts.install_sdks vcamera
python -m python.scripts.install_sdks vcamera --check
```

`--check` 只检查 SDK 能否导入及必要 API 是否存在，不打开相机。相机连接与取帧情况通过实际采集确认。

### 3. 日常运行

每次打开 Bash 后，进入工程并激活环境。下面两条标定命令按用途选择执行。

```bash
cd /data2/project/Calibration
conda activate camera

# 大恒单独标内参
python python/scripts/calibrate.py --config configs/intrinsics/galaxy.json

# 大恒与奥比：采集后计算内参、外参
python python/scripts/calibrate.py --config configs/extrinsics/galaxy_orbbec_color.json
```

其他相机组合按后面的配置表替换 `--config`。采集窗口中按空格保存图像，Q/Esc 结束。

### 4. 离线计算与验证

在已激活的 `camera` 环境、工程根目录中执行。以下分别是只采集、使用已有图像标定、固定已有参数验证，按用途选择；将 `SESSION`、`RESULT` 替换成实际时间戳目录。

```bash
python python/scripts/calibrate.py capture --config configs/extrinsics/galaxy_orbbec_color.json
python python/scripts/calibrate.py calibrate --config configs/extrinsics/galaxy_orbbec_color.json --session data/SESSION
python python/scripts/calibrate.py verify --config configs/extrinsics/galaxy_orbbec_color.json --session data/SESSION --calibration-dir output/RESULT
```

离线自检不连接相机：

```bash
python python/scripts/validation/self_check.py -v
```

导航棒验证仅支持大恒与图漾，需先完成本节的图漾 SDK 安装，并更新对应配置中的参数目录及几何路径，再执行：

```bash
python python/scripts/validation/check_instrument_extrinsics.py --config configs/instruments/instrument_extrinsic_check_13mm.json
```

### 5. Linux 故障排查

- **Galaxy 日志无法写入**：`log4cplus:ERROR Unable to open file: /var/log/Galaxy/SDK.log` 表示日志文件写入失败。文件由 root 创建时，普通用户可能没有追加权限。备份 `/etc/Galaxy/cfg/log4cplus.properties` 后，将 `log4cplus.appender.SHARED_FILE.File` 改为当前用户可写的绝对路径，并确保该目录存在。相机是否连接成功需看后续设备打开和取帧结果。
- **Galaxy 打开时报 `{-5}` 且包含 `__ReadXML`**：失败发生在 SDK 打开设备、读取相机 XML 的阶段，尚未设置分辨率和曝光。本机 Linux SDK 曾在重置 USB 后出现超时、掉线和重新枚举。程序对 Linux USB3 的这一特定错误等待一秒，按原序列号重新查找相机，只重试一次；仍失败时保留完整错误。结合 Galaxy 日志及以下内核日志检查通信情况：

```bash
journalctl -k -b --no-pager | tail -n 80
```

## Windows：Windows x64 / PowerShell

以下命令仅在 Windows PowerShell 中执行，以大恒与奥比为例。

### 1. 准备系统 SDK

安装 Python 3.13 x64 和 **Galaxy Windows SDK** 原生运行库及相机驱动。安装 SDK 后重新打开 PowerShell，确认安装器设置的环境变量已加载：

```powershell
$env:GALAXY_GENICAM_ROOT
```

若输出为空，先检查 Galaxy Windows SDK 的安装和环境变量。Linux 的 `.so` 文件不能用于 Windows。

### 2. 首次部署

以下使用 `D:\jlc\calibration` 作为工程目录。已有工程时跳过克隆；已有 Python 3.13 的 `.venv` 时跳过环境创建。

```powershell
git clone https://github.com/superjiang1203/Calibration.git D:\jlc\calibration
Set-Location D:\jlc\calibration
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
Get-Command python
python --version
python -m pip install -r requirements-calibration.txt
python -m python.scripts.install_sdks galaxy orbbec
python -m python.scripts.install_sdks galaxy orbbec --check
```

使用图漾时，在同一环境中追加安装 Windows 图漾绑定并检查：

```powershell
python -m python.scripts.install_sdks vcamera
python -m python.scripts.install_sdks vcamera --check
```

`--check` 只检查 SDK 能否导入及必要 API 是否存在，不打开相机。相机连接与取帧情况通过实际采集确认。

### 3. 日常运行

每次打开 PowerShell 后，进入工程并激活环境。下面两条标定命令按用途选择执行。

```powershell
Set-Location D:\jlc\calibration
.\.venv\Scripts\Activate.ps1

# 大恒单独标内参
python .\python\scripts\calibrate.py --config .\configs\intrinsics\galaxy.json

# 大恒与奥比：采集后计算内参、外参
python .\python\scripts\calibrate.py --config .\configs\extrinsics\galaxy_orbbec_color.json
```

其他相机组合按后面的配置表替换 `--config`。采集窗口中按空格保存图像，Q/Esc 结束。

若 PowerShell 阻止运行激活脚本，可在工程根目录直接用 `.\.venv\Scripts\python.exe` 替代各命令中的 `python`，安装依赖和运行脚本都使用该解释器。

### 4. 离线计算与验证

在已激活的 `.venv` 环境、工程根目录中执行。以下分别是只采集、使用已有图像标定、固定已有参数验证，按用途选择；将 `SESSION`、`RESULT` 替换成实际时间戳目录。

```powershell
python .\python\scripts\calibrate.py capture --config .\configs\extrinsics\galaxy_orbbec_color.json
python .\python\scripts\calibrate.py calibrate --config .\configs\extrinsics\galaxy_orbbec_color.json --session .\data\SESSION
python .\python\scripts\calibrate.py verify --config .\configs\extrinsics\galaxy_orbbec_color.json --session .\data\SESSION --calibration-dir .\output\RESULT
```

离线自检不连接相机：

```powershell
python .\python\scripts\validation\self_check.py -v
```

导航棒验证仅支持大恒与图漾，需先完成本节的图漾 SDK 安装，并更新对应配置中的参数目录及几何路径。Windows 还需安装 `tzdata`，供工具导出结果时读取 `Asia/Shanghai` 时区：

```powershell
python -m pip install tzdata
python .\python\scripts\validation\check_instrument_extrinsics.py --config .\configs\instruments\instrument_extrinsic_check_13mm.json
```

## 两种系统共用的配置与标定规则

### 配置选择

不带 `--config` 时默认使用大恒与图漾配置。大恒与奥比需显式选择 `galaxy_orbbec_color.json`。

| 用途 | `--config` 对应的配置文件 |
| --- | --- |
| 大恒单独标内参 | `configs/intrinsics/galaxy.json` |
| 大恒 → 图漾彩色 | `configs/extrinsics/galaxy_vcamera_color.json` |
| 大恒 → 奥比彩色 | `configs/extrinsics/galaxy_orbbec_color.json` |
| 奥比彩色 → 图漾彩色 | `configs/extrinsics/orbbec_vcamera_color.json` |
| 奥比左红外 → 右红外 | `configs/extrinsics/orbbec_ir.json` |

`configs/cameras/` 保存相机设置；棋盘尺寸在工作流 JSON 的 `board` 中设置。导航棒配置在 `configs/instruments/`，按实物选择 `instrument_extrinsic_check_13mm.json`、`instrument_extrinsic_check_15mm.json` 或 `instrument_extrinsic_check_large_qr.json`；当前导航棒工具面向大恒与图漾，使用前需填写实际标定结果目录及几何路径。

### 内参与外参的计算顺序

- `calibration.use_known_intrinsics=false`：先用本批训练图像估计大恒内参，再固定两台相机的内参求外参。
- `calibration.use_known_intrinsics=true`：大恒使用已有内参，填写 `calibration.fixed_intrinsics.galaxy_gray`，例如 `../../output/RESULT/sensor_galaxy_gray_intrinsics.yml`。
- 图漾、奥比始终使用采集记录中的 SDK 内参。外参计算固定内参，只优化两台相机之间的旋转和平移。

### 路径与依赖约定

- `data/` 保存时间戳采集目录，`output/` 保存时间戳结果目录，两者均位于工程根目录。
- `--session`、`--calibration-dir` 的相对路径以工程根目录为基准；工作流 JSON 中的 `sensor_config`、固定内参路径以该 JSON 所在目录为基准。
- 跨系统共用 JSON 时建议使用 `/` 写相对路径；若在 JSON 字符串中填写 Windows 绝对路径，反斜杠须写成 `\\`。
- 奥比绑定通过工程的 `install_sdks` 安装器安装。安装器固定 SDK 版本并使用 `--no-deps`，基础依赖由 `requirements-calibration.txt` 提供，以处理奥比与图漾的 NumPy 依赖声明冲突。奥比 wheel 默认从官方 PyPI 下载，Linux 与 Windows 均需下载对应 Python 版本和平台的包，详见 [第三方 SDK 说明](thirdparty/README.md)。
