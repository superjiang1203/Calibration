# 相机标定

Ubuntu 22.04 使用 Python 3.10；Windows 使用 Python 3.13。图漾、奥比使用 SDK 内参；大恒支持单独标内参或与外参一起求解。

## 安装

```bash
git clone https://github.com/superjiang1203/Calibration.git
cd Calibration
# Ubuntu 22.04
python3 -m venv .venv
source .venv/bin/activate
```

Windows 创建环境：`py -3.13 -m venv .venv`，激活：`.venv\Scripts\Activate.ps1`。在已激活环境中安装：

```bash
python -m pip install -r requirements-calibration.txt
python -m python.scripts.install_sdks vcamera galaxy orbbec
python -m python.scripts.install_sdks vcamera galaxy orbbec --check
```

大恒须先安装对应系统的官方 Galaxy SDK 原生运行库。无需安装未使用的相机 SDK。

配置目录：`configs/cameras/` 相机设置；`configs/intrinsics/` 大恒内参；`configs/extrinsics/` 双机外参；`configs/instruments/` 导航棒验证。棋盘尺寸在工作流配置的 `board` 中设置。

## 大恒单独标内参

```bash
python python/scripts/calibrate.py --config configs/intrinsics/galaxy.json
```

## 双机外参

```bash
# 大恒 → 图漾彩色
python python/scripts/calibrate.py --config configs/extrinsics/galaxy_vcamera_color.json
# 大恒 → 奥比彩色
python python/scripts/calibrate.py --config configs/extrinsics/galaxy_orbbec_color.json
# 奥比彩色 → 图漾彩色
python python/scripts/calibrate.py --config configs/extrinsics/orbbec_vcamera_color.json
# 奥比左红外 → 右红外
python python/scripts/calibrate.py --config configs/extrinsics/orbbec_ir.json
```

大恒配对配置：`calibration.use_known_intrinsics=false` 同批估计大恒内参并求外参；`true` 固定已有大恒内参，仅求外参。
选 `true` 时填写 `calibration.fixed_intrinsics.galaxy_gray` 为真实 YAML 路径（相对工作流 JSON 目录），如 `../../output/RESULT/sensor_galaxy_gray_intrinsics.yml`。图漾、奥比始终使用采集记录中的 SDK 内参。

空格采集，Q/Esc 结束；图像在 `data/`，结果在 `output/`。

## 离线计算与验证

替换 `SESSION`、`RESULT` 为实际目录；可换用上面任一配置。

```bash
python python/scripts/calibrate.py capture --config configs/extrinsics/galaxy_vcamera_color.json
python python/scripts/calibrate.py calibrate --config configs/extrinsics/galaxy_vcamera_color.json --session data/SESSION
python python/scripts/calibrate.py verify --config configs/extrinsics/galaxy_vcamera_color.json --session data/SESSION --calibration-dir output/RESULT
python python/scripts/validation/self_check.py -v
```

## 导航棒验证

先更新对应配置的参数目录及几何路径。

```bash
python python/scripts/validation/check_instrument_extrinsics.py --config configs/instruments/instrument_extrinsic_check_13mm.json
python python/scripts/validation/check_instrument_extrinsics.py --config configs/instruments/instrument_extrinsic_check_15mm.json
python python/scripts/validation/check_instrument_extrinsics.py --config configs/instruments/instrument_extrinsic_check_large_qr.json
```
