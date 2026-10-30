从零手写的 C++20 深度学习框架，CPU / CUDA 双后端，不依赖任何第三方深度学习库。

## 目录结构

```
include/mini_cuda_nn/
    mini_cuda_nn.h          伞头，一把梭
    core/                   Tensor、Device、DataType、kernel 声明
    autograd/               计算图接口（Edge / GradFn）、no_grad
    nn/                     Module、Linear、ReLU、Sequential、MSELoss
    optim/                  Optimizer、SGD
    detail/                 实现细节，不是公开 API
src/
    core/                   tensor.cpp + tensor.cu
    autograd/               反传引擎 + 反向节点
    nn/  optim/
tests/                      三套单元测试
examples/dqn_snake/         DQN 贪吃蛇
```

## 构建

需要 C++20 编译器、CMake ≥ 3.18、CUDA Toolkit。Windows 下先进入 x64 开发者命令行。

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

产物：`build/tests/*.exe`、`build/bin/*.exe`。

## 示例：DQN 贪吃蛇

```bash
build/bin/dqn_snake.exe --episodes 400 --eval 10 --probe
```
