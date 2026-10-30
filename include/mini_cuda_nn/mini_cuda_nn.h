#pragma once

// 一把梭：包含框架的全部公开 API。
// 想要更细的依赖控制就单独 include 对应的头文件。
//
//   core/      Tensor、Device、DataType
//   autograd/  计算图（Edge / GradFn）、no_grad
//   nn/        Module、Linear、ReLU、Sequential、MSELoss
//   optim/     Optimizer、SGD

#include "mini_cuda_nn/core/cuda_check.h"
#include "mini_cuda_nn/core/device.h"
#include "mini_cuda_nn/core/dtype.h"
#include "mini_cuda_nn/core/tensor.h"

#include "mini_cuda_nn/autograd/grad_fn.h"
#include "mini_cuda_nn/autograd/grad_mode.h"

#include "mini_cuda_nn/nn/activation.h"
#include "mini_cuda_nn/nn/linear.h"
#include "mini_cuda_nn/nn/loss.h"
#include "mini_cuda_nn/nn/module.h"
#include "mini_cuda_nn/nn/sequential.h"

#include "mini_cuda_nn/optim/optimizer.h"
#include "mini_cuda_nn/optim/sgd.h"
