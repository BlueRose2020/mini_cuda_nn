#include <cuda_runtime.h>
extern "C" __global__ void mininn_sgd_kernel(float *p, const float *g, float lr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p[i] -= lr * g[i];
}
extern "C" void *mininn_cuda_alloc(size_t n)
{
    void *p = nullptr;
    cudaMalloc(&p, n * sizeof(float));
    return p;
}
extern "C" void mininn_cuda_free(void *p) { cudaFree(p); }
extern "C" void mininn_cuda_upload(void *p, const float *x, size_t n) { cudaMemcpy(p, x, n * sizeof(float), cudaMemcpyHostToDevice); }
extern "C" void mininn_cuda_download(float *x, const void *p, size_t n) { cudaMemcpy(x, p, n * sizeof(float), cudaMemcpyDeviceToHost); }
__global__ void mm(const float *a, const float *b, float *c, int m, int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n)
    {
        float s = 0;
        for (int q = 0; q < k; q++)
            s += a[i * k + q] * b[q * n + j];
        c[i * n + j] = s;
    }
}
extern "C" void mininn_cuda_matmul(const float *a, const float *b, float *c, int m, int n, int k)
{
    mm<<<dim3((m + 15) / 16, (n + 15) / 16), dim3(16, 16)>>>(a, b, c, m, n, k);
    cudaDeviceSynchronize();
}
extern "C" void mininn_cuda_sgd(float *p, const float *g, float lr, int n)
{
    mininn_sgd_kernel<<<(n + 255) / 256, 256>>>(p, g, lr, n);
    cudaDeviceSynchronize();
}
extern "C" bool mininn_cuda_available()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
