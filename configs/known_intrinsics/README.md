# 已知内参

将大恒单独标定结果中的 `sensor_galaxy_gray_intrinsics.yml` 复制为本目录的 `galaxy_gray.yml`，或修改双机 `*_known_intrinsics.json` 的 `calibration.fixed_intrinsics.galaxy_gray` 路径。

图漾和奥比默认使用采集时保存的 SDK 内参；如需改用单独标定结果，在 `fixed_intrinsics` 中添加对应标签及 YAML 路径，例如 `orbbec_color`、`vcamera_color`。

所有路径相对工作流 JSON 所在目录；内参须对应实际分辨率、ROI 和镜头设置。本目录不提供虚构内参。
