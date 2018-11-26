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

__device__ void inverse2x2(float* Ainv, const float* A, const int idx,
                           const int b, const int i, const int j) {
    const int bidx = b*4;
    if(i == 0 & j == 0){Ainv[idx] = A[bidx + 3];}
    if(i == 1 & j == 0){Ainv[idx] = -A[idx];}
    if(i == 0 & j == 1){Ainv[idx] = -A[idx];}
    if(i == 1 & j == 1){Ainv[idx] = A[bidx];}
    Ainv[idx] /= (A[bidx]*A[bidx+3] - A[bidx+1]*A[bidx+2]);
    __syncthreads();
    return;
}

__device__ void inverse3x3(float* Ainv, const float* A, const int idx,
                           const int b, const int i, const int j) {
    const int bidx = b*9;
    if(i == 0 & j == 0){Ainv[idx] = (A[bidx+4]*A[bidx+8] - A[bidx+5]*A[bidx+7]);}
    if(i == 1 & j == 0){Ainv[idx] = -(A[bidx+3]*A[bidx+8] - A[bidx+5]*A[bidx+6]);}
    if(i == 2 & j == 0){Ainv[idx] = (A[bidx+3]*A[bidx+7] - A[bidx+4]*A[bidx+6]);}
    if(i == 0 & j == 1){Ainv[idx] = -(A[bidx+1]*A[bidx+8] - A[bidx+2]*A[bidx+4]);}
    if(i == 1 & j == 1){Ainv[idx] = (A[bidx+0]*A[bidx+8] - A[bidx+2]*A[bidx+6]);}
    if(i == 2 & j == 1){Ainv[idx] = -(A[bidx+0]*A[bidx+7] - A[bidx+1]*A[bidx+6]);}
    if(i == 0 & j == 2){Ainv[idx] = (A[bidx+1]*A[bidx+5] - A[bidx+2]*A[bidx+4]);}
    if(i == 1 & j == 2){Ainv[idx] = -(A[bidx+0]*A[bidx+5] - A[bidx+2]*A[bidx+3]);}
    if(i == 2 & j == 2){Ainv[idx] = (A[bidx+0]*A[bidx+4] - A[bidx+1]*A[bidx+3]);}
    Ainv[idx] /= A[bidx]*(A[bidx+4]*A[bidx+8] - A[bidx+5]*A[bidx+7]) +
                 A[bidx+1]*(A[bidx+5]*A[bidx+6] - A[bidx+3]*A[bidx+8]) + 
                 A[bidx+2]*(A[bidx+3]*A[bidx+7] - A[bidx+4]*A[bidx+6]);
    __syncthreads();
    return;
}

__device__ void inverse4x4(float* Ainv, const float* A, const int idx,
                           const int b, const int i, const int j){
    const int bidx = b*9;                           
    float a11 = A[bidx+0];    float a12 = A[bidx+1];    float a13 = A[bidx+2];    float a14 = A[bidx+3];
    float a21 = A[bidx+4];    float a22 = A[bidx+5];    float a23 = A[bidx+6];    float a24 = A[bidx+7];
    float a31 = A[bidx+8];    float a32 = A[bidx+9];    float a33 = A[bidx+10];   float a34 = A[bidx+11];
    float a41 = A[bidx+12];   float a42 = A[bidx+13];   float a43 = A[bidx+14];   float a44 = A[bidx+15];
    
    __syncthreads();
    return;                           
}

__global__ void expm_cuda_kernel(float* expmA, const float* Abatch, 
                                 const int N, const int row, const int matsize){
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
    /*
    double beta[14] = {64764752532480000., 32382376266240000., 7771770303897600.,
                       1187353796428800., 129060195264000., 10559470521600.,
                       670442572800., 33522128640., 1323241920., 40840800.,
                       960960., 16380., 182., 1.};
    */
    double beta[14] = {1., .5000000000, .1200000000, 0.1833333333e-1, 0.1992753623e-2, 
                      0.1630434783e-3, 0.1035196687e-4, 5.175983437e-7, 
                      2.043151357e-8, 6.306022706e-10, 1.483770048e-11, 
                      2.529153492e-13, 2.810170546e-15, 1.544049751e-17};
                      
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
        int n_squaring = int(max(0.0, ceil(log2(sqrt(norm[0]) / 5.371920351148152))));
        A[idx] /= powf(2.0, n_squaring);
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
        __syncthreads();
        matmul(temp2, A6, temp, idx, i, j, row);
        temp2[idx] += beta[7]*A6[idx] + beta[5]*A4[idx] + beta[3]*A2[idx] + beta[1]*iden[idx];
        __syncthreads();
        matmul(U, A, temp2, idx, i, j, row);
        
        
        // Calculate V
        temp[idx] = beta[12]*A6[idx] + beta[10]*A4[idx] + beta[8]*A2[idx];
        __syncthreads();
        matmul(temp, A6, temp, idx, i, j, row);
        V[idx] = temp[idx] + beta[6]*A6[idx] + beta[4]*A4[idx] + beta[2]*A2[idx] + beta[0]*iden[idx];
        __syncthreads();
        
        // Calculate nominator and denominator
        temp[batch_idx] = U[idx] + V[idx];
        temp2[batch_idx] = -U[idx] + V[idx];
        __syncthreads();

        // Calculate inverse of denominator
        if(row == 2){inverse2x2(U, temp2, idx, b, i, j);}
        else if(row == 3){inverse3x3(U, temp2, idx, b, i, j);}
        else if(row == 4){inverse4x4(U, temp2, idx, b, i, j);}
        
        // Calculate squared matrix exponential
        matmul(V, U, temp, idx, i, j, row);
        
        // Unsquaring
        for(int k = 0; k < n_squaring; k ++){
            matmul(temp, V, V, idx, i, j, row);
            V[idx] = temp[idx];
            __syncthreads();
        }
        expmA[idx] = V[idx];
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
    
    // Launch kernel
    expm_cuda_kernel<<<bc, tpb, shared_size>>>(expmA, A, N, row, matsize);

    return output;
}
