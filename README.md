# 相机标定

支持大恒、图漾和奥比相机。图漾、奥比使用 SDK 内参；大恒支持单独标内参及双机外参标定。

## 安装

| 平台 | Python | 额外依赖 |
| --- | --- | --- |
| Ubuntu 22.04 x86_64 | 3.10 | 使用大恒时安装 Galaxy Linux SDK |
| Windows x64 | 3.13 | 使用大恒时安装 Galaxy Windows SDK 和驱动 |

在工程根目录创建并激活环境。

**Linux**

```bash
conda create -n camera python=3.10 -y
conda activate camera
```

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

然后安装依赖和所用相机的绑定，以下以大恒与图漾为例：

```bash
python -m pip install -r requirements-calibration.txt
python -m python.scripts.install_sdks galaxy vcamera
```

相机名称可选 `galaxy`、`vcamera`、`orbbec`，按需组合。SDK 包说明见 [thirdparty/README.md](thirdparty/README.md)。

## 运行

以下命令适用于两种系统，均在已激活环境的工程根目录执行。

```bash
# 大恒单独标内参
python python/scripts/calibrate.py --config configs/intrinsics/galaxy.json

# 大恒与图漾左灰度：采集后计算内参、外参
python python/scripts/calibrate.py --config configs/extrinsics/galaxy_vcamera_ir_left.json
```

采集窗口按 **空格**保存，**Q / Esc** 结束。图像保存在 `data/时间戳/`，结果保存在 `output/时间戳/`。实时采集需图形桌面。

需要分步执行时：

```bash
# 只采集
python python/scripts/calibrate.py capture --config configs/extrinsics/galaxy_vcamera_ir_left.json

# 使用已有图像标定
python python/scripts/calibrate.py calibrate --config configs/extrinsics/galaxy_vcamera_ir_left.json --session data/SESSION

# 使用已保存参数核验棋盘重投影误差
python python/scripts/calibrate.py verify --config configs/extrinsics/galaxy_vcamera_ir_left.json --session data/SESSION --calibration-dir output/RESULT
```

将 `SESSION`、`RESULT` 替换为实际目录名，其他相机组合替换 `--config`。

## 配置

| 用途 | 配置文件 |
| --- | --- |
| 大恒内参 | [configs/intrinsics/galaxy.json](configs/intrinsics/galaxy.json) |
| 大恒 → 图漾左灰度（VMD02-5011） | [configs/extrinsics/galaxy_vcamera_ir_left.json](configs/extrinsics/galaxy_vcamera_ir_left.json) |
| 大恒 → 图漾彩色 | [configs/extrinsics/galaxy_vcamera_color.json](configs/extrinsics/galaxy_vcamera_color.json) |
| 大恒 → 奥比彩色 | [configs/extrinsics/galaxy_orbbec_color.json](configs/extrinsics/galaxy_orbbec_color.json) |
| 奥比彩色 → 图漾彩色 | [configs/extrinsics/orbbec_vcamera_color.json](configs/extrinsics/orbbec_vcamera_color.json) |
| 奥比左红外 → 右红外 | [configs/extrinsics/orbbec_ir.json](configs/extrinsics/orbbec_ir.json) |

相机设置在 `configs/cameras/`，修改后重启采集生效。棋盘在工作流配置的 `board` 中设置，默认 **11×8 内角点、格长 15 mm**，按实物修改。

| 相机 | `exposure` 单位 | `gain` 单位 |
| --- | --- | --- |
| 大恒 | 微秒 | dB |
| 图漾 | 微秒 | 设备原生单位 |
| 奥比 | 设备原生整数 | 设备原生整数 |
| OpenCV | 后端原生单位 | 后端原生单位 |

`null` 表示保留设备设置；奥比同机左右红外配对采集须指定曝光，且分辨率、曝光、增益一致。

VMD02-5011 左灰度使用 `stream: "ir_left"`，分辨率 2048×1536，参考 [vcamera_ir_left.json](configs/cameras/vcamera_ir_left.json)。`gray` 对应 Texture 通道，不适用于这台相机。检查通道与参数范围可运行：

```bash
python -m python.scripts.inspect_vcamera --config configs/cameras/vcamera_ir_left.json
```

双机标定中的大恒内参由 `calibration.use_known_intrinsics` 控制：

- `false`：先用本批图像估计大恒内参，再固定两台相机的内参求外参。
- `true`：使用已有大恒内参，在 `calibration.fixed_intrinsics.galaxy_gray` 填写内参 YAML 路径。

命令行相对路径以工程根目录为基准；工作流中的相机配置、固定内参路径以该 JSON 所在目录为基准。
