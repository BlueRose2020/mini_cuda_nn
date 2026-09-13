#include "mininn.hpp"
#include <iostream>
int main(int argc, char **argv)
{
    using namespace mininn;
    Device d = (argc > 1 && std::string(argv[1]) == "cuda") ? Device::CUDA : Device::CPU;
    if (d == Device::CUDA && !cuda_available())
    {
        std::cerr << "CUDA unavailable\n";
        return 2;
    }
    auto w = tensor({1, 1}, d, true);
    auto x = tensor({1, 1}, d);
    auto y = tensor({1, 1}, d);
    x->data[0] = 2;
    y->data[0] = 4;
    for (int i = 0; i < 60; i++)
    {
        w->zero_grad();
        auto l = mse_loss(matmul(x, w), y);
        backward(l);
        hybrid_sgd_step({w}, .1f);
    }
    std::cout << "device=" << (d == Device::CUDA ? "cuda" : "cpu") << " w=" << w->data[0] << "\n";
    return (w->data[0] > 1.9f && w->data[0] < 2.1f) ? 0 : 1;
}
