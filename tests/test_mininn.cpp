#include "mininn.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace mininn;
int main() {
    auto x=tensor({2,1},{2.0f,3.0f},Device::CPU,true); auto w=tensor({1,1},{1.0f},Device::CPU,true); auto y=tensor({2,1},{4.0f,6.0f});
    auto loss=mse_loss(matmul(x,w),y); backward(loss); if (std::abs(w->grad[0] - (-13.0f)) > 1e-5f) return 1;
    auto model=Sequential{{std::make_shared<Linear>(1,4),std::make_shared<ReLU>(),std::make_shared<Linear>(4,1)}}; Adam opt(0.03f); for(int i=0;i<100;++i){auto p=model.forward(x);auto l=mse_loss(p,y);model.zero_grad();backward(l);opt.step(model.parameters());} assert(model.forward(x)->data[0] > 2.5f);
    auto prediction = model.forward(x); if (prediction->data[0] <= 2.5f) return 1;
    std::cout << "mininn tests passed\n"; return 0;
}
