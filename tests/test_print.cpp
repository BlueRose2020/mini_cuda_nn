#include "mini_cuda_nn/core/tensor.h"
#include <iostream>
int main()
{
    mcn::Tensor t1{3, 2, 2, 3};
    mcn::Tensor t2{3};
    mcn::Tensor t3(1);
    mcn::Tensor t4({{2.0f, 3.1f},{1.0f,2.0f}});
    mcn::Tensor t_empty1 = mcn::Tensor::empty({2, 3});
    std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;
    std::cout << t3 << std::endl;
    std::cout << t4 << std::endl;
    std::cout << t_empty1 << "\tshape=" << mcn::format_shape(t_empty1.shape()) << std::endl;

    t1.to(mcn::Device::CUDA);
    t2.to(mcn::Device::CUDA);
    t3.to(mcn::Device::CUDA);
    t4.to(mcn::Device::CUDA);
    t_empty1.to(mcn::Device::CUDA);

    std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;
    std::cout << t3 << std::endl;
    std::cout << t4 << std::endl;
    std::cout << t_empty1 << "\tshape=" << mcn::format_shape(t_empty1.shape()) << std::endl;

    // 视图验证：不拷贝数据，只移动 offset_
    std::cout << "--- view ---" << std::endl;
    std::cout << t4[0] << std::endl; // 期望 [2, 3.1]
    std::cout << t4[1] << std::endl; // 期望 [1, 2]
    std::cout << t1[2] << "\tshape=" << mcn::format_shape(t1[2].shape()) << std::endl; // 期望 2，shape=[]
    std::cout << t4[1][0] << std::endl; // 期望 1（0 维标量）
    std::cout << t4[1].clone() << std::endl; // 期望与 t4[1] 相同

    return 0;
}