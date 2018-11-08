#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}

__global__ void pade13(const double* A, double* expmA, const int N, const int row, const int matsize){
    int b = blockIdx.x*blockDim.x+threadIdx.x;
    int i = blockIdx.z*blockDim.z+threadIdx.z;
    int j = blockIdx.y*blockDim.y+threadIdx.y;
    
    if(b < N && i < row && j < row){
    
    }
}


// Kernel launcher declaration
at::Tensor expm_cuda_forward(at::Tensor input){
    // Problem size
    const auto N = input.size(0);
    const auto row = input.size(1);
    const auto col = input.size(2);
    const auto matsize = row*col;
    
    // Allocate output
    auto output = at::zeros_like(input);
    auto A = input.data<double>();
    auto expmA = output.data<double>();
    
    // Strategy
    // - compute n_squaring on cpu
    // - find pade matrices with own kernel
    // - use cusolverDnDgetrs to solve the matrix system (probably multiple gpu streams)
    // - use unsquaring, probably own kernel for efficiency
                                                       
    return output;
}