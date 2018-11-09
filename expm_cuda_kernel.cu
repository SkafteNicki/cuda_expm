#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

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

__global__ void pade13(const double* A, double* U, double* V, const int N, const int row, const int matsize){
    // Allocate the shared memory
    extern __shared__ int s[];
    double *norm = (double*)&s;
    double *tmpsum = (double*)&norm[N];
    
    // Get threads index
    int b = blockIdx.x*blockDim.x+threadIdx.x;
    int i = blockIdx.y*blockDim.y+threadIdx.y;
    int j = blockIdx.z*blockDim.z+threadIdx.z;
    
    // Calculate within domain
    if(b < N && i < row && j < row){
        // Calcuate norm
        norm[b] += powf(abs(A[b * row * row + i * row + j]), 2.0);
    
        tmpsum[b * row * row + i * row + j] = 0;
        for(int k = 0; k < row; k++){
            tmpsum[b * row * row + i * row + j] += A[b * row * row + i * row + k] * A[b * row * row + k * row + j];
        }
        U[b * row * row + i * row + j] = tmpsum[b * row * row + i * row + j];
    }
    return;
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
    auto U = output.data<double>();
    
    // Kernel configuration
    dim3 tpb = dim3(std::min((int)N, 64), std::min((int)row, 32), std::min((int)col, 32));
    dim3 bc = dim3(DIV_UP(N, tpb.x), DIV_UP(row, tpb.y), DIV_UP(col, tpb.z));
    dim3 vtc = dim3(N, row, col);
    
    // Allocate numerator and denomerator and launch pade13 kernel
    //auto U = at::zeros_like(input).data<double>();
    auto V = at::zeros_like(input).data<double>();
    
    // N for norm calc, N*row*row for matrix mult calc
    auto shared_size = N*sizeof(double) + N*row*row*sizeof(double);
    pade13<<<bc, tpb, shared_size>>>(A, U, V, N, row, matsize);
    
    // Strategy
    // - compute n_squaring on cpu
    // - find pade matrices with own kernel
    // - use cusolverDnDgetrs to solve the matrix system (probably multiple gpu streams)
    // - use unsquaring, probably own kernel for efficiency
                                                       
    return output;
}