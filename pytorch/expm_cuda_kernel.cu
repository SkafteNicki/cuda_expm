#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cublas_v2.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ void matmul(float* A, const float* B, const float* C, const int idx,
                       const int i, const int j, const int row){
    // Calculate A[idx] = B * C
    float temp = 0;
    for(int k = 0; k < row; k++){
        temp += B[i * row + k] * C[k * row + j];
    }
    A[idx] = temp;
    __syncthreads();
    return;
}

__global__ void pade_13(float* P, float* Q, int* n_squaring, 
                        const float* Abatch, const int N, 
                        const int row, const int matsize){
    // Allocate the shared memory
    extern __shared__ int s[];
    float* norm  = (float*)&s;
    float* A     = (float*)&norm[1];
    float* iden  = (float*)&norm[1 + 1*matsize];
    float* temp  = (float*)&norm[1 + 2*matsize];
    float* temp2 = (float*)&norm[1 + 3*matsize];
    float* A2    = (float*)&norm[1 + 4*matsize];
    float* A4    = (float*)&norm[1 + 5*matsize];
    float* A6    = (float*)&norm[1 + 6*matsize];
    float* U     = (float*)&norm[1 + 7*matsize];
    float* V     = (float*)&norm[1 + 8*matsize];
    
    // Pade constants
    double beta[14] = {64764752532480000., 32382376266240000., 7771770303897600.,
                       1187353796428800., 129060195264000., 10559470521600.,
                       670442572800., 33522128640., 1323241920., 40840800.,
                       960960., 16380., 182., 1.};
    
    // Get threads index
    int b = blockIdx.x*blockDim.x+threadIdx.x;
    int i = blockIdx.y*blockDim.y+threadIdx.y;
    int j = blockIdx.z*blockDim.z+threadIdx.z;
    
    // Calculate within domain
    if(b < N && i < row && j < row){
        // Place matrix in shared memory
        int batch_idx = b * matsize + i * row + j;
        int idx = i * row + j;
        A[idx] = Abatch[batch_idx];
    
        // Calcuate norm
        norm[0] += powf(abs(A[idx]), 2.0);
        __syncthreads();
        
        // Calculate n_squaring        
        n_squaring[b] = int(max(0.0, ceil(log2(sqrt(norm[0]) / 5.371920351148152))));
        A[idx] /= powf(2.0, n_squaring[b]);
        __syncthreads();
        
        // Fill identity matrix
        if(i == j){
            iden[idx] = 1.0;
        } else {
            iden[idx] = 0.0;
        }
        
        // Compute A2, A4, A6
        matmul(A2, A, A, idx, i, j, row);
        matmul(A4, A2, A2, idx, i, j, row);
        matmul(A6, A4, A2, idx, i, j, row);
        
        // Calculate U
        temp[idx] = beta[13]*A6[idx]*beta[11]*A4[idx]+beta[9]*A2[idx];
        matmul(temp2, A6, temp, idx, i, j, row);
        temp2[idx] += beta[7]*A6[idx] + beta[5]*A4[idx] + beta[3]*A2[idx] + beta[1]*iden[idx];
        matmul(U, A, temp, idx, i, j, row);
        
        // Calculate V
        temp[idx] = beta[12]*A6[idx] + beta[10]*A4[idx] + beta[8]*A2[idx];
        matmul(temp, A6, temp, idx, i, j, row);
        V[idx] = temp[idx] + beta[6]*A6[idx] + beta[4]*A4[idx] + beta[2]*A2[idx] + beta[0]*iden[idx];
        
        // Calculate nominator and denominator
        P[batch_idx] = U[idx] + V[idx];
        Q[batch_idx] = -U[idx] + V[idx];
    }
    return;
}

__global__ void unsquaring(float* expmA, const float* P, const float* invQ, 
                           const int* n_squaring, const int N, const int row,
                           const int matsize){
    // Allocate the shared memory                           
    extern __shared__ int s[];
    float* R    = (float*)&s;
    float* temp = (float*)&R[matsize];
    
    // Get threads index
    int b = blockIdx.x*blockDim.x+threadIdx.x;
    int i = blockIdx.y*blockDim.y+threadIdx.y;
    int j = blockIdx.z*blockDim.z+threadIdx.z;
    
    // Calculate within domain
    if(b < N && i < row && j < row){
        int batch_idx = b * matsize + i * row + j;
        int idx = i * row + j;
        
        // Calculate R = inv(Q)*P
        matmul(R, &invQ[b*matsize], &P[b*matsize], idx, i, j, row);
        
        // Unsquaring
        for(int u = 0; u < n_squaring[b]; u++){
            matmul(temp, R, R, idx, i, j, row);
            R[idx] = temp[idx];
            __syncthreads();
        }
        expmA[batch_idx] = R[idx];
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
    auto A = input.data<float>();
    auto expmA = output.data<float>();
    
    // Kernel configuration
    dim3 tpb = dim3(1, row, col);
    dim3 bc = dim3(N, 1, 1);
    
    // Shared memory
    // - 2 double for norm calc, n_squaring
    // - 6*matsize for matrix mult calculations (A, A^2, A^4, A^6, ident, temp1, temp2)
    auto shared_size = 1*sizeof(float) + 9*matsize*sizeof(float);
    
    // Find Pade-13 factors
    auto P = at::zeros_like(input).data<float>();
    auto Q = at::zeros_like(input).data<float>();
    auto Qinv = at::zeros_like(input).data<float>();
    auto n_squaring = at::zeros(at::CUDA(at::kInt), {N}).data<int>();
    pade_13<<<bc, tpb, shared_size>>>(P, Q, n_squaring, A, N, row, matsize);
    cudaDeviceSynchronize();
    
    // Solve linear system Q * R = P for R
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    int info;
    auto pivotarray = at::zeros(at::CUDA(at::kInt), {row, N}).data<int>(); // support array
    auto infoarray = at::zeros(at::CUDA(at::kInt), {N}).data<int>(); // support array
    float **Q_h    = (float **)malloc(N * sizeof(float *));
    float **Qinv_h = (float **)malloc(N * sizeof(float *));
    for(int b = 0; b < N; b++){ Q_h[b] = Q + b * matsize; }
    float** Q_d; gpuErrchk(cudaMalloc(&Q_d, N * sizeof(float *)));
    float** Qinv_d; gpuErrchk(cudaMalloc(&Qinv_d, N * sizeof(float *)));
    gpuErrchk(cudaMemcpy(Q_d, Q_h, N * sizeof(float *), cudaMemcpyHostToDevice));
    cublasSmatinvBatched(cublas_handle, row, (const float **)Q_d, row, Qinv_d, row, &info, N);
    cublasSgetrfBatched(cublas_handle, row, Q_d, row, pivotarray, infoarray, N);
    //cudaDeviceSynchronize();
    //cublasSgetriBatched(cublas_handle, row, (const float **)Q_as_pointers_d, row, 
    //                    pivotarray, Qinv_as_pointers_d, row, infoarray, N);
    free(Q_h);
    gpuErrchk(cudaFree(Q_d));
    gpuErrchk(cudaFree(Qinv_d));
    
    // Calculate R = inv(Q)*P, and unsquare R
    // shared_size = 2*matsize*sizeof(float);
    // unsquaring<<<bc, tpb, shared_size>>>(expmA, P, invQ, n_squaring, N, row, matsize);

    return output;
}
