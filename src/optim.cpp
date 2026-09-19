#include "mininn/optim.hpp"
#include <algorithm>
#include <cmath>
namespace mininn {
void Optimizer::zero_grad(const std::vector<TensorPtr>& parameters) { for(auto& p:parameters) p->zero_grad(); }
SGD::SGD(float learning_rate, float momentum) : lr_(learning_rate), momentum_(momentum) {}
void SGD::step(const std::vector<TensorPtr>& parameters) {
    for(auto& p:parameters) { p->to_host(); auto& v=velocity_[p.get()]; if(v.size()!=p->numel()) v.assign(p->numel(),0); for(size_t i=0;i<p->numel();++i) { v[i]=momentum_*v[i]+p->grad[i]; p->data[i]-=lr_*v[i]; } p->to_device(); }
}
Adam::Adam(float learning_rate,float beta1,float beta2,float eps):lr_(learning_rate),beta1_(beta1),beta2_(beta2),eps_(eps){}
void Adam::step(const std::vector<TensorPtr>& parameters) {
    ++step_; const float b1=1-std::pow(beta1_,static_cast<float>(step_)), b2=1-std::pow(beta2_,static_cast<float>(step_));
    for(auto& p:parameters) { p->to_host(); auto& s=state_[p.get()]; if(s.m.size()!=p->numel()){s.m.assign(p->numel(),0);s.v.assign(p->numel(),0);} for(size_t i=0;i<p->numel();++i){s.m[i]=beta1_*s.m[i]+(1-beta1_)*p->grad[i];s.v[i]=beta2_*s.v[i]+(1-beta2_)*p->grad[i]*p->grad[i];p->data[i]-=lr_*(s.m[i]/b1)/(std::sqrt(s.v[i]/b2)+eps_);} p->to_device(); }
}
}
