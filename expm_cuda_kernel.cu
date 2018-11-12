#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

__global__ void pade13(const double* A, double* U, double* V, int* n_squaring, const int N, const int row, const int matsize){
    // Allocate the shared memory
    extern __shared__ int s[];
    double *norm = (double*)&s;
    double *iden = (double*)&norm[N];
    double *A2   = (double*)&norm[N + N*matsize];
    double *A4   = (double*)&norm[N + 2*N*matsize];
    double *A6   = (double*)&norm[N + 3*N*matsize];
    double *temp = (double*)&norm[N + 4*N*matsize];
    
    // Get threads index
    int b = blockIdx.x*blockDim.x+threadIdx.x;
    int i = blockIdx.y*blockDim.y+threadIdx.y;
    int j = blockIdx.z*blockDim.z+threadIdx.z;
    
    // Calculate within domain
    if(b < N && i < row && j < row){
        int idx = b * matsize + i * row + j;
    
        // Calcuate norm
        norm[b] += powf(abs(A[idx]), 2.0);
        
        //synchronize so norm calculation is done
        __syncthreads();
        
        // Calculate n_squaring        
        n_squaring[b] = int(max(0.0, ceil(log2(sqrt(norm[b]) / 5.371920351148152))));
    
        // Fill identity matrix
        if(i == j){
            iden[idx] = 1.0;
        } else {
            iden[idx] = 0.0;
        }
        
        // Compute A2
        temp[idx] = 0;
        for(int k = 0; k < row; k++){
            temp[idx] += A[b * matsize + i * row + k] * A[b * matsize + k * row + j];
        }
        A2[idx] = tmpsum[idx];
        __syncthreads();
        
        // Compute A4
        temp[idx] = 0;
        for(int k = 0; k < row; k++){
            temp[idx] += A2[b * matsize + i * row + k] * A2[b * matsize + k * row + j];
        }
        A4[idx] = tmpsum[idx];
        __syncthreads();
        
        // Compute A6
        temp[idx] = 0;
        for(int k = 0; k < row; k++){
            temp[idx] += A4[b * matsize + i * row + k] * A2[b * matsize + k * row + j];
        }
        A6[idx] = tmpsum[idx];
        __syncthreads();
        
        // Compute U
        temp[idx] = beta[13]*A6[idx]*beta[11]*A4[idx]+beta[9]*A2[idx];
        __syncthreads();
        // matrix mul temp = A6*temp
        // update temp[i*row+j] += beta[7]*mat6[row*i+j] + beta[5]*mat4[row*i+j] + beta[3]*mat2[row*i+j] + beta[1]*iden[row*i+j];
        // matrix mul U = A*temp
        
        // Compute V
        
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
    auto n_squaring = at::zeros(N).data<int>();
    
    // Shared memory
    // - N double for norm calc, 
    // - 5*N*row*row for matrix mult calculations (A^2, A^4, A^6, ident, temp)
    auto shared_size = N*sizeof(double) + 5*N*row*row*sizeof(double);
    pade13<<<bc, tpb, shared_size>>>(A, U, V, n_squaring, N, row, matsize);
    
    // Strategy
    // - compute n_squaring on cpu
    // - find pade matrices with own kernel
    // - use cusolverDnDgetrs to solve the matrix system (probably multiple gpu streams)
    // - use unsquaring, probably own kernel for efficiency
                                                       
    return output;
}