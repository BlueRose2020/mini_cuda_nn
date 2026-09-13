#pragma once
#include "mininn/tensor.hpp"
#include <vector>
namespace mininn { void sgd_step(const std::vector<TensorPtr>& parameters, float learning_rate); void hybrid_sgd_step(const std::vector<TensorPtr>& parameters, float learning_rate); }
