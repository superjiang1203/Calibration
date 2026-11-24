# IMU 与相机实时预览

在本机编译 SDK 桥接程序，再从[标定工作台](../../../README.md#启动)打开 IMU 页签。可在 `configs/studio.json` 的 `imu` 下指定几何、相机内参和相机序列号。

```powershell
cmake -S python/scripts/imu_calibration/sdk_bridge -B output/imu_attitude/build -G "Visual Studio 17 2022" -A x64
cmake --build output/imu_attitude/build --config Release --target imu_attitude_bridge
```

页面启动时相机处于断开状态。连接进度、各阶段耗时和 SDK 原始错误显示在相机画面下方。曝光和增益由采集线程写入并回读。手动模式下优先在采集中写入，设备锁定参数时重启采集流。

C++ 桥接程序通过 `steerable_instrument_driver` 订阅 ICM42670 和 JY931。H5 通过 SSE 接收姿态，右侧显示原始姿态。图像上的虚线坐标轴以视觉估计的 ID84 原点为起点，独立于相机帧率刷新。

四元数采用 SDK 的 WXYZ 顺序，欧拉角满足 `R = Rz(yaw) Ry(pitch) Rx(roll)`。单次显示对齐需要至少两个可见 marker 和有效的 IMU 样本。它假设安装轴一致，保存 `A = R_camera_marker(t0) R_imu(t0)^T`，随后显示 `A R_imu(t)`。定量比较需要标定安装外参并同步两路数据。
