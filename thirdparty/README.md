# 相机 SDK 运行包

| wheel 文件 | 对应相机与内容 | 适用环境 |
| --- | --- | --- |
| `vcamera_python-26.1.15-cp313-cp313-win_amd64.whl` | 图漾 Python 绑定及运行库 | Windows x64 / CPython 3.13 |
| `vcamera_python-26.1.15-cp310-cp310-linux_x86_64.whl` | 图漾 Python 绑定及运行库 | Ubuntu 22.04 x86_64 / CPython 3.10 |
| `gxipy-2.0.2512.9261-py3-none-any.whl` | 大恒 Python 封装 | Python ≥3.10，另需对应系统的官方 Galaxy SDK 原生运行库 |

两份图漾 wheel 按平台二选一，不能跨系统使用；这属于平台限制，不是依赖冲突。
gxipy 另需 Galaxy 原生运行库；其 NumPy 2 旧接口兼容由项目适配器处理。

奥比不在仓库重复保存 wheel，安装器从官方 PyPI 获取 `pyorbbecsdk2==2.1.1`。
该版本的 Python 3.10 包声明 `numpy<2`，图漾声明 `numpy>=2,<2.3`，两者存在依赖声明冲突。
项目实际采用 NumPy 2.2，安装器用 `--no-deps` 安装 SDK，基础依赖由项目 requirements 提供。
已在 Ubuntu 22.04 / CPython 3.10 / NumPy 2.2.6 验证绑定导入和必要 API，未验证真实相机采集或全部奥比示例。

厂商包适用各自原有许可；保留的第三方许可见 [licenses/](licenses/)，校验和见 [SHA256SUMS](SHA256SUMS)。
