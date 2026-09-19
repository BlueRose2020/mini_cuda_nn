# mini_cuda_nn

一个从零实现的 C++17 深度学习框架原型，包含 CPU 和 CUDA 单 GPU 后端，以及一个浏览器端 DQN 贪吃蛇示例。

## 框架结构

```text
include/mininn.hpp             统一入口
include/mininn/                Tensor、模块、优化器接口
src/tensor.cpp                 Tensor 生命周期和设备内存
src/ops.cpp                   add、matmul、mse_loss
src/autograd.cpp              反向传播
src/optimizer.cpp             SGD 和混合 SGD
src/device.cpp                CUDA 可用性检测
cuda/kernels.cu               CUDA 显存、矩阵乘法、SGD kernel
examples/train.cpp            CPU/CUDA 线性回归示例
examples/snake_dqn_ui/        DQN 贪吃蛇浏览器 UI
```

## C++/CUDA 构建

CPU 版本：

```powershell
cmake -S . -B build -DMINI_CUDA=OFF
cmake --build build --config Release
.\build\Release\train.exe
```

CUDA 版本需要 NVIDIA CUDA Toolkit、MSVC 和驱动：

```powershell
cmake -S . -B build-cuda -DMINI_CUDA=ON
cmake --build build-cuda --config Release
.\build-cuda\Release\train.exe cuda
```

`hybrid_sgd_step` 在 CPU 参数上执行主机更新，在 CUDA 参数上调用 GPU SGD kernel；工程只针对单 GPU，不包含多 GPU 通信。

当前 C++ API 已包含 `Tensor` 工厂函数、`add`/`matmul`/`add_bias`/`relu`/`mse_loss`、反向传播、`Linear`/`ReLU`/`Sequential`，以及带 momentum 的 `SGD` 和 `Adam`。CPU 回归测试通过 CTest 执行：

```powershell
ctest --test-dir build -C Release --output-on-failure
```

## DQN 贪吃蛇 UI

进入目录并启动本地静态服务器：

```powershell
cd examples/snake_dqn_ui
powershell -ExecutionPolicy Bypass -File .\serve.ps1 -Port 8090
```

打开终端打印的实际地址。页面提供训练、暂停、逐帧播放当前策略、最佳评估快照、训练曲线和诊断指标。代码更新后重启服务并按 `Ctrl+F5`。

DQN 训练核心使用 `11 → 32(ReLU) → 3` 网络、Double DQN、经验回放、Huber loss、Adam、梯度裁剪和平滑目标网络。详细指标定义和设计说明见 [examples/snake_dqn_ui/README.md](examples/snake_dqn_ui/README.md)。
