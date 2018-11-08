#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel declaration
__global__ void cuda_kernel_forward(float* __restrict__ output, 
                                    const float* __restrict__ input){
    //const int i = blockIdx.x*blockDim.x + threadIdx.x;
    return;
}

// Kernel launcher declaration
at::Tensor expm_cuda_forward(at::Tensor input){
    auto output = at::zeros_like(input);    
    /*
    const int blockSize = 256;
    const int numBlocks = ceil((N + blockSize - 1) / blockSize);
    square_kernel_forward<<<numBlocks, blockSize>>>(output.data<float>(),
                                                    input.data<float>(),
                                                    N);
    */                                                    
    return output;
}