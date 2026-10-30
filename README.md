# mini_cuda_nn

从零手写的 C++20 深度学习框架，CPU / CUDA 双后端，不依赖任何第三方深度学习库。

```cpp
mcn::nn::Sequential net;
net.add<mcn::nn::Linear>(2, 8).add<mcn::nn::ReLU>().add<mcn::nn::Linear>(8, 1);

mcn::optim::SGD opt(net.parameters(), 0.05, /*momentum=*/0.9);
mcn::nn::MSELoss loss_fn;

for (int it = 0; it < 600; ++it) {
    opt.zero_grad();
    mcn::Tensor loss = loss_fn(net(x), y);
    loss.backward();
    opt.step();
}
```

## 特性

| 模块 | 内容 |
|---|---|
| Tensor | 视图 / stride 模型、`shared_ptr` 共享存储、7 种 dtype |
| 算子 | `+ - *`（广播）、取负、标量乘、`matmul`（1-D/2-D）、`sum` / `mean`、`relu` |
| 自动微分 | 拓扑排序反传、梯度累加、广播梯度自动归约、`no_grad` |
| 网络层 | `Module` / `Linear` / `ReLU` / `Sequential` / `MSELoss` |
| 优化器 | `Optimizer` 基类（NVI）、`SGD`（momentum + weight decay） |
| 后端 | CPU 主机循环 + 手写 CUDA kernel，同一套宿主端接口 |

## 构建

需要 C++20 编译器、CMake ≥ 3.18、CUDA Toolkit。Windows 下先进入 x64 开发者命令行。

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

产物：`build/tests/*.exe`、`build/bin/*.exe`。

## 测试

三套独立测试，共 **635** 条断言：

| 目标 | 覆盖 |
|---|---|
| `test_tensor` | 构造、视图、reshape、clone、广播、matmul、autograd、CUDA |
| `test_optim` | SGD 数值、动量、weight decay、分批加参数、CUDA、float64 |
| `test_nn` | Linear 前后向、MSE、Sequential、数值梯度校验、监督学习基线 |

其中 `test_gradient_check` 用中心差分比对 autograd 的梯度（实测误差 ~1.6e-5）。

## 示例：DQN 贪吃蛇

```bash
build/bin/dqn_snake.exe --episodes 400 --eval 10 --probe
```

| 参数 | 说明 |
|---|---|
| `--episodes N` | 训练局数 |
| `--eval N` | 贪心评估局数 |
| `--probe` | 行为诊断：短周期循环占比、Q 值范围、终止原因 |
| `--render` | 训练后演示一局 |
| `--device cuda` | 在 GPU 上算 |
| `--hidden N` | 隐藏层宽度 |

400 局后贪心评估平均 **44 个食物/局**（棋盘 12×12）。诊断显示短周期循环占比 0%，终止原因全部是撞死而非饿死。

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

约定：`include/` 顶层目录是公开 API，`detail/` 是实现细节。

## 设计取舍

| 决定 | 理由 |
|---|---|
| `Tensor` 拷贝是浅拷贝，深拷贝走 `clone()` | 与 `operator=` 一致；之前两者语义相反，容易踩坑 |
| 反向图只在 `require_grad` 且未处于 `no_grad` 时构建 | 少写一堆 `if (track)` |
| 广播的梯度归约集中在引擎里做 | 每个反向节点不用各自处理 `sum_to` |
| `transpose` 是唯一带 `grad_fn` 的视图算子 | 反向可精确表达；否则 `x.t()` 临时视图会留下悬垂 `sink` |
| `Optimizer::step()` 用 NVI 不开放重写 | 遍历与校验不该被子类改掉 |
| `dtype` 分派放宿主端 | device 侧每个 kernel 只有单一表达式 |

## 限制

- `matmul` 只支持 1-D / 2-D，批维广播待实现
- 反向只在 2-D @ 2-D 时可用
- 非连续 CUDA 视图的 `clone()` 支持，但 `t()` 后立刻做原地操作会报错
- 编译期裁剪（`MCN_ENABLE_CUDA`）尚未实现，构建始终需要 CUDA
