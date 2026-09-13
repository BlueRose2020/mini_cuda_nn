#include "internal.hpp"
namespace mininn
{
    void backward(TensorPtr x)
    {
        x->grad.assign(x->numel(), 1);
        for (auto it = tape.rbegin(); it != tape.rend(); ++it)
            (*it)();
        tape.clear();
    }
}

