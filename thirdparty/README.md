# 相机 SDK 运行包

先安装项目依赖，再安装需要的相机绑定：

```bash
python -m pip install -r requirements-calibration.txt
python -m python.scripts.install_sdks vcamera galaxy orbbec
python -m python.scripts.install_sdks vcamera galaxy orbbec --check
```

可只选择一台，例如 `python -m python.scripts.install_sdks vcamera`。
`--check` 检查原生库导入及必要 API，不打开相机。

| 相机 | 仓库中的文件 | 系统要求 |
| --- | --- | --- |
| 图漾 | Vcamera 26.1.15 wheel，含 Release 运行库 | Windows x64 / Python 3.13；Ubuntu 22.04 x64 / Python 3.10 |
| 大恒 | gxipy 2.0.2512.9261 wheel，仅 Python 封装 | 另装对应系统的 [Galaxy SDK](https://en.daheng-imaging.com/list-58-1.html) 原生运行库 |
| 奥比 | 从 [官方 PyPI](https://pypi.org/project/pyorbbecsdk2/2.1.1/) 安装固定版本 2.1.1 | 使用官方提供匹配当前 Python / 系统的二进制 wheel |

`wheels/` 只保留以上预编译包，不需要完整 SDK 源码、示例、Debug 库或构建目录。
图漾 Linux wheel 不含 RPC 功能；本项目使用 `CameraUtils.Init(True)` 直连模式。
其他 Python 版本需要自行提供匹配的图漾 wheel，再用 `--wheel-dir <目录>` 安装。

安装器使用 `--no-deps`，项目实际使用的依赖由 `requirements-calibration.txt` 安装。
奥比 wheel 的示例依赖（Open3D、pygame、PyAV 等）不在本项目使用范围内，不随安装器下载。
奥比官方 Python 3.10 包声明 NumPy <2，图漾声明 NumPy >=2；本项目采用 NumPy 2.2，
仅验证原生绑定导入和本项目 API，不代表奥比全部示例依赖兼容。

Ubuntu 上先按 Galaxy SDK 安装说明安装原生运行库，确认 `/usr/lib/libgxiapi.so` 可加载。
仅安装 gxipy wheel 不能替代 Galaxy 驱动。USB 设备权限及硬件连接按厂商说明设置。

完全离线安装奥比时，先在联网的同平台环境下载到 `wheels/`，再使用 `--offline`：

```bash
python -m pip download --no-deps --only-binary=:all: pyorbbecsdk2==2.1.1 -d thirdparty/wheels
python -m python.scripts.install_sdks orbbec --offline
```

## 来源及许可

- Vcamera wheel 基于厂商提供的 VcameraSDK 26.1.15；Windows wheel 复用现有成品，Linux wheel 在 Ubuntu 22.04 / Python 3.10 编译，附带 Release 运行库。厂商元数据标记为 Proprietary，适用厂商许可；原始 SDK 未附单独的 Vcamera 再分发许可文件。
- gxipy wheel 打包自本机安装的官方 Galaxy SDK Python 文件，文件未修改；只增加 wheel 元数据，保留原版权头及 `galaxy_3rd_party_licenses.txt`。这些第三方声明不替代 Galaxy SDK 自身许可。
- `licenses/vcamera-pybind11-LICENSE.txt` 保留图漾绑定所用 pybind11 的 BSD 许可。
- 奥比直接下载官方包，保留包内 LICENSE；不在本仓库重复保存 SDK 源码或大体积 wheel。

以上厂商包不因放入本仓库而变更许可。校验和见 `SHA256SUMS`。
