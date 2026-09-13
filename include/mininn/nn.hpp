#pragma once
#include "mininn/tensor.hpp"
#include <vector>
namespace mininn { class Module { public: virtual ~Module()=default; virtual TensorPtr forward(TensorPtr input)=0; virtual std::vector<TensorPtr> parameters()=0; }; }
