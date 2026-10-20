# 大恒到图漾外参

## 选择内参来源

仅计算外参：

```bash
python python/scripts/calibrate.py --config configs/calibration_galaxy_vcamera_color_known_intrinsics.json
```

此配置使用 `calibration.use_known_intrinsics: true`，固定两端内参。大恒默认读取 `configs/known_intrinsics/galaxy_gray.yml`；图漾默认读取本次采集记录中的 SDK 厂家 Texture K/D。`calibration.fixed_intrinsics` 中的 YAML 路径优先于 SDK 参数，路径相对配置文件目录。

同批图像重新估计两端内参并求外参：

```bash
python python/scripts/calibrate.py --config configs/calibration_galaxy_vcamera_color.json
```

此配置使用 `calibration.use_known_intrinsics: false`。原有未设置此开关的配置保持历史行为：图漾/奥比使用厂家内参，大恒从棋盘估计内参；有 `fixed_intrinsics` 时优先读取指定文件。

## 原始图像与参数

采集关闭 Texture 去畸变和 D2C，并在 `capture_metadata.json` 中保存 `GetFactoryCalibInfo(Texture)` 的 K/D。内参尺寸必须与原图一致，不自动缩放。

旧数据若无 SDK 内参，可在 `fixed_intrinsics.vcamera_color` 指定导航所用 SDK 导出的原始 Texture OpenCV YAML，包含 `image_width`、`image_height`、3×3 的 `camera_matrix` 和行向量 `distortion_coefficients`。畸变顺序为 SDK 定义的 k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4。原始彩色图像应匹配原始 Texture 内参。

## 结果坐标系

输出的 R/T 是大恒 → 图漾 Texture 彩色坐标系的刚体转换，T 单位为 mm。若点云处于 Depth 坐标系，还须按 SDK 定义的方向衔接 Texture → Depth。

导航与验证应加载同一结果目录中的两端内参和外参。重新验证触点时关闭历史人工补偿，并用独立、多位置、多深度的数据检查三维误差。
