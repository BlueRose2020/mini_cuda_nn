#include "mini_cuda_nn/autograd/backward_nodes.h"
#include "mini_cuda_nn/autograd/grad_mode.h"

#include <memory>
#include <utility>
#include <vector>

namespace
{
    using mcn::Edge;
    using mcn::GradFn;
    using mcn::Tensor;

    Edge make_edge(const Tensor &t, bool save)
    {
        Edge e;
        e.fn = t.grad_fn();
        if (!e.fn && t.require_grad())
            e.sink = &t;        e.shape = t.shape();        if (save)
            e.saved = std::make_shared<Tensor>(t.alias());
        return e;
    }

    bool needs_grad(const Edge &e) { return e.sink != nullptr || e.fn != nullptr; }

    // ---------------- 逐元素二元算子 ----------------
    class AddBackward : public GradFn
    {
    public:
        const char *name() const override { return "add_backward"; }

        std::vector<Tensor> backward(const Tensor &g) override
        {
            std::vector<Tensor> out;
            out.reserve(edges.size());
            for (const auto &e : edges)
                out.push_back(needs_grad(e) ? g.alias() : Tensor());
            return out;
        }
    };

    class SubBackward : public GradFn
    {
    public:
        const char *name() const override { return "sub_backward"; }

        std::vector<Tensor> backward(const Tensor &g) override
        {
            std::vector<Tensor> out;
            out.reserve(edges.size());
            for (size_t i = 0; i < edges.size(); ++i)
            {
                if (!needs_grad(edges[i]))
                    out.push_back(Tensor());
                else if (i == 0)
                    out.push_back(g.alias());
                else
                    out.push_back(-g);
            }
            return out;
        }
    };

    class MulBackward : public GradFn
    {
    public:
        const char *name() const override { return "mul_backward"; }

        std::vector<Tensor> backward(const Tensor &g) override
        {
            MCN_CHECK(edges.size() == 2 && edges[0].saved && edges[1].saved,
                      "mul_backward: missing saved inputs");

            const Tensor &a = *edges[0].saved;
            const Tensor &b = *edges[1].saved;

            std::vector<Tensor> out;
            out.reserve(2);
            out.push_back(needs_grad(edges[0]) ? g * b : Tensor());
            out.push_back(needs_grad(edges[1]) ? g * a : Tensor());
            return out;
        }
    };

    // ---------------- 一元算子 ----------------
    class NegBackward : public GradFn
    {
    public:
        const char *name() const override { return "neg_backward"; }

        std::vector<Tensor> backward(const Tensor &g) override
        {
            MCN_CHECK(edges.size() == 1, "neg_backward: expected 1 edge, got {}", edges.size());

            std::vector<Tensor> out;
            out.reserve(1);
            out.push_back(needs_grad(edges[0]) ? -g : Tensor());
            return out;
        }
    };

    class ScalarMulBackward : public GradFn
    {
    public:
        explicit ScalarMulBackward(double factor) : factor_(factor) {}

        const char *name() const override { return "scalar_mul_backward"; }

        std::vector<Tensor> backward(const Tensor &g) override
        {
            MCN_CHECK(edges.size() == 1, "scalar_mul_backward: expected 1 edge, got {}", edges.size());

            std::vector<Tensor> out;
            out.reserve(1);
            out.push_back(needs_grad(edges[0]) ? g.mul_scalar(factor_) : Tensor());
            return out;
        }

    private:
        double factor_;
    };

    // ---------------- 归约 ----------------

    // y = sum(x)（全归约或沿某一维） =>  dx = g 广播回 x 的形状
    //
    // 不能直接 g.broadcast_to(x.shape)：broadcast_to 是右对齐的，
    // 而 sum(dim=1) 的 [2] 对应的是第 0 维。所以先用 keepdim 形状
    // （被归约的维置 1）把 g 放回正确位置，再广播。
    class SumBackward : public GradFn
    {
    public:
        explicit SumBackward(std::vector<int> keepdim_shape)
            : keepdim_shape_(std::move(keepdim_shape)) {}

        const char *name() const override { return "sum_backward"; }

        std::vector<Tensor> backward(const Tensor &g) override
        {
            MCN_CHECK(edges.size() == 1, "sum_backward: expected 1 edge, got {}", edges.size());

            std::vector<Tensor> out;
            out.reserve(1);
            if (!needs_grad(edges[0]))
            {
                out.push_back(Tensor());
                return out;
            }

            Tensor gg = g.reshape(keepdim_shape_);
            out.push_back(gg.broadcast_to(edges[0].shape));
            return out;
        }

    private:
        std::vector<int> keepdim_shape_;
    };

    // ---------------- 矩阵乘法 ----------------
    // y = a @ b  =>  da = g @ b^T,  db = a^T @ g
    // 先只支持 2-D，1-D 的形状还原留到后面
    class MatmulBackward : public GradFn
    {
    public:
        const char *name() const override { return "matmul_backward"; }

        std::vector<Tensor> backward(const Tensor &g) override
        {
            MCN_CHECK(edges.size() == 2 && edges[0].saved && edges[1].saved,
                      "matmul_backward: missing saved inputs");

            const Tensor &a = *edges[0].saved;
            const Tensor &b = *edges[1].saved;

            std::vector<Tensor> out;
            out.reserve(2);
            out.push_back(needs_grad(edges[0]) ? g.matmul(b.t()) : Tensor());
            out.push_back(needs_grad(edges[1]) ? a.t().matmul(g) : Tensor());
            return out;
        }
    };

    void attach(GradFn &node, const Tensor &lhs, const Tensor &rhs, bool save)
    {
        node.edges.push_back(make_edge(lhs, save));
        node.edges.push_back(make_edge(rhs, save));
    }
}

namespace mcn::detail
{
    void attach_binary(const Tensor &lhs, const Tensor &rhs, BinaryOp op, Tensor &out)
    {
        std::shared_ptr<GradFn> node;
        bool save = false;

        switch (op)
        {
        case BinaryOp::Add:
            node = std::make_shared<AddBackward>();
            break;
        case BinaryOp::Sub:
            node = std::make_shared<SubBackward>();
            break;
        case BinaryOp::Mul:
            node = std::make_shared<MulBackward>();
            save = true;
            break;
        }

        MCN_CHECK(node != nullptr, "attach_binary: unsupported op");
        attach(*node, lhs, rhs, save);
        out.set_grad_fn(std::move(node));
    }

    void attach_matmul(const Tensor &lhs, const Tensor &rhs, Tensor &out)
    {
        auto node = std::make_shared<MatmulBackward>();
        attach(*node, lhs, rhs, true);
        out.set_grad_fn(std::move(node));
    }

    void attach_neg(const Tensor &in, Tensor &out)
    {
        auto node = std::make_shared<NegBackward>();
        node->edges.push_back(make_edge(in, false));
        out.set_grad_fn(std::move(node));
    }
    void attach_sum(const Tensor &in, std::vector<int> keepdim_shape, Tensor &out)
    {
        auto node = std::make_shared<SumBackward>(std::move(keepdim_shape));
        node->edges.push_back(make_edge(in, /*save=*/false)); // 只需要输入形状
        out.set_grad_fn(std::move(node));
    }
    void attach_scalar_mul(const Tensor &in, double factor, Tensor &out)
    {
        auto node = std::make_shared<ScalarMulBackward>(factor);
        node->edges.push_back(make_edge(in, false));
        out.set_grad_fn(std::move(node));
    }} // namespace mcn::detail
