# 相机标定

支持大恒（Galaxy）灰度、图漾（Vcamera）原始彩色、奥比（Orbbec）彩色及左右红外。

## 安装

Ubuntu 22.04 使用 Python 3.10；按顺序执行：

```bash
git clone https://github.com/superjiang1203/Calibration.git
cd Calibration
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-calibration.txt
python -m python.scripts.install_sdks vcamera galaxy orbbec
python -m python.scripts.install_sdks vcamera galaxy orbbec --check
```

SDK 平台支持及原生驱动安装见 [thirdparty/README.md](thirdparty/README.md)。无需安装未使用的相机 SDK。

配置棋盘尺寸：`board.pattern_cols/rows` 为内角点数，`square_size_mm` 为格长。相机设置见 `configs/sensor_*.json`。空格采集，Q/Esc 结束；图像保存在 `data/`，参数保存在 `output/`。

## 单独标定内参

```bash
# 大恒灰度
python python/scripts/calibrate.py --config configs/calibration_galaxy_intrinsics.json
# 图漾彩色
python python/scripts/calibrate.py --config configs/calibration_vcamera_intrinsics.json
# 奥比彩色
python python/scripts/calibrate.py --config configs/calibration_orbbec_color_intrinsics.json
# 奥比左红外
python python/scripts/calibrate.py --config configs/calibration_orbbec_ir_left_intrinsics.json
# 奥比右红外
python python/scripts/calibrate.py --config configs/calibration_orbbec_ir_right_intrinsics.json
```

## 使用已知内参，只标定外参

配置设为 `calibration.use_known_intrinsics: true`。大恒须将已有 `sensor_galaxy_gray_intrinsics.yml` 复制为 `configs/known_intrinsics/galaxy_gray.yml`，或修改配置中的路径。图漾、奥比默认读取采集记录中的 SDK 内参；如需使用单独标定结果，在 `calibration.fixed_intrinsics` 中填写对应标签和 YAML 路径，详见 [已知内参配置](configs/known_intrinsics/README.md)。

```bash
# 大恒 → 图漾彩色
python python/scripts/calibrate.py --config configs/calibration_galaxy_vcamera_color_known_intrinsics.json
# 大恒 → 奥比彩色
python python/scripts/calibrate.py --config configs/calibration_galaxy_orbbec_color_known_intrinsics.json
# 奥比彩色 → 图漾彩色
python python/scripts/calibrate.py --config configs/calibration_orbbec_vcamera_color_known_intrinsics.json
# 奥比左红外 → 右红外
python python/scripts/calibrate.py --config configs/calibration_orbbec_ir_known_intrinsics.json
```

## 同批图像标定内参＋外参

配置设为 `calibration.use_known_intrinsics: false`：重新估计两端内参，再计算外参。

```bash
# 大恒 → 图漾彩色
python python/scripts/calibrate.py --config configs/calibration_galaxy_vcamera_color.json
# 大恒 → 奥比彩色
python python/scripts/calibrate.py --config configs/calibration_galaxy_orbbec_color.json
# 奥比彩色 → 图漾彩色
python python/scripts/calibrate.py --config configs/calibration_orbbec_vcamera_color.json
# 奥比左红外 → 右红外
python python/scripts/calibrate.py --config configs/calibration_orbbec_ir.json
```

## 分开采集、离线计算、独立验证

以下示例可替换为上面任一配置；`SESSION` 和 `RESULT` 替换为实际目录。

```bash
# 仅采集
python python/scripts/calibrate.py capture --config configs/calibration_galaxy_vcamera_color_known_intrinsics.json
# 使用已有图像重新计算
python python/scripts/calibrate.py calibrate --config configs/calibration_galaxy_vcamera_color_known_intrinsics.json --session data/SESSION
# 用新图像验证已有参数
python python/scripts/calibrate.py verify --config configs/calibration_galaxy_vcamera_color_known_intrinsics.json --session data/SESSION --calibration-dir output/RESULT
# 无相机离线自检
python python/scripts/validation/self_check.py -v
```

## 导航棒验证

先在对应配置中更新参数目录和导航棒几何路径。

```bash
# 13 mm 导航棒
python python/scripts/validation/check_instrument_extrinsics.py --config configs/instruments/instrument_extrinsic_check_13mm.json
# 15 mm 导航棒
python python/scripts/validation/check_instrument_extrinsics.py --config configs/instruments/instrument_extrinsic_check_15mm.json
# 大号 AprilTag 末端
python python/scripts/validation/check_instrument_extrinsics.py --config configs/instruments/instrument_extrinsic_check_large_qr.json
```

图漾外参的目标是 Texture 彩色坐标系；转换到 Depth 点云坐标系的说明见 [FACTORY_EXTRINSICS.md](FACTORY_EXTRINSICS.md)。
