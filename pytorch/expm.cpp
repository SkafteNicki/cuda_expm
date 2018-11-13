#include <torch/torch.h>
#include <cmath>
#include <math.h>
#include <algorithm>
#include "cblas.h"
#include <lapacke.h>
#include <iostream>
#include <iomanip>

using namespace std;

void printmat(const float* mat, const int row){
    cout << "[";
    for(int i = 0; i < row; i++){
        if(i != 0){cout << " ";}
        for(int j = 0; j < row; j++){
            cout << std::fixed << std::setprecision(3) << mat[i*row + j] << ",";
        }
        if(i == row-1){cout << "]";}
        cout << endl;
    }
}

float fronorm(const float* mat, const int matsize){
    float norm = 0;
    for(int i = 0; i < matsize; i++){
        norm += pow(abs(mat[i]), 2.0);
    }
    return sqrt(norm);
}

void pade13_forward(const float* mat, float* U, float* V, const int matsize, const int row){
    // Pade constants
    std::vector<float> beta;
    beta = {64764752532480000., 32382376266240000., 7771770303897600.,
            1187353796428800., 129060195264000., 10559470521600.,
            670442572800., 33522128640., 1323241920., 40840800.,
            960960., 16380., 182., 1.};

    float iden[matsize], mat2[matsize], mat4[matsize], mat6[matsize], temp[matsize];
    // Identity matrix
    for(int i = 0; i < row; i++){
        for(int j = 0; j < row; j++){
            if(i == j){
                iden[i*row + j] = 1.0;
            } else {
                iden[i*row + j] = 0.0;
            }
        }
    }

    // Calculate mat^2 = mat*mat
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat, row, mat, row, 0.0, mat2, row);
    // Calculate mat^4 = mat^2 * mat^2
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat2, row, mat2, row, 0.0, mat4, row);
    // Calculate mat^6 = mat^2 * mat^4
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat2, row, mat4, row, 0.0, mat6, row);
    
    // Calculate U
    for(int i = 0; i < matsize; i++){
        temp[i] = beta[13]*mat6[i]*beta[11]*mat4[i]+beta[9]*mat2[i];
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat6, row, temp, row, 0.0, temp, row);
    for(int i = 0; i < matsize; i++){
        temp[i] += beta[7]*mat6[i] + beta[5]*mat4[i] + beta[3]*mat2[i] + beta[1]*iden[i];
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat, row, temp, row, 0.0, U, row);
    
    // Calculate V
    for(int i = 0; i < matsize; i++){
        temp[i] = beta[12]*mat6[i] + beta[10]*mat4[i] + beta[8]*mat2[i];        
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat6, row, temp, row, 0.0, temp, row);
    for(int i = 0; i < matsize; i++){
        V[i] = temp[i] + beta[6]*mat6[i] + beta[4]*mat4[i] + beta[2]*mat2[i] + beta[0]*iden[i];
    }
    return;
}

void pade13_backward(const float* mat, float* dU, float* dV, const int matsize, const int row){
    // Pade constants
    std::vector<float> beta;
    beta = {64764752532480000. * 0, 32382376266240000. * 1, 7771770303897600. * 2,
            1187353796428800. * 3, 129060195264000. * 4, 10559470521600. * 5,
            670442572800. * 6, 33522128640. * 7, 1323241920. * 8, 40840800. * 9,
            960960. * 10, 16380. * 11, 182. * 12, 1. * 13};

    float iden[matsize], mat2[matsize], mat4[matsize], mat6[matsize], temp[matsize];
    // Identity matrix
    for(int i = 0; i < row; i++){
        for(int j = 0; j < row; j++){
            if(i == j){
                iden[i*row + j] = 1.0;
            } else {
                iden[i*row + j] = 0.0;
            }
        }
    }
    
    // Calculate mat^2 = mat*mat
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat, row, mat, row, 0.0, mat2, row);
    // Calculate mat^4 = mat^2 * mat^2
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat2, row, mat2, row, 0.0, mat4, row);
    // Calculate mat^6 = mat^2 * mat^4
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat2, row, mat4, row, 0.0, mat6, row);
    
    // Calculate dU
    for(int i = 0; i < matsize; i++){
        temp[i] = beta[13]*mat6[i] + beta[11]*mat4[i] + beta[9]*mat2[i] + beta[7]*iden[i];
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat6, row, temp, row, 0.0, temp, row);
    for(int i = 0; i < matsize; i++){
        dU[i] = temp[i] + beta[5]*mat4[i] + beta[3]*mat2[i] + beta[1]*iden[i];
    }
    
    // Calculate dV
    for(int i = 0; i < matsize; i++){
        temp[i] = beta[12]*mat4[i] + beta[10]*mat2[i];
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat6, row, temp, row, 0.0, temp, row);
    for(int i = 0; i < matsize; i++){
        temp[i] += beta[8]*mat6[i] + beta[6]*mat4[i] + beta[4]*mat2[i] + beta[2]*iden[i];
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat, row, temp, row, 0.0, dV, row);
    
    return;                    
}

at::Tensor expm_forward(at::Tensor input){
    // Problem size
    const auto N = input.size(0);
    const auto row = input.size(1);
    const auto col = input.size(2);
    const auto matsize = row*col;

    // Allocate output
    auto A = input.data<float>();
    auto output = at::zeros_like(input);
    auto expmA = output.data<float>();
    
    // For each matrix
    for(int i = 0; i < N; i++){
        // Copy to local matrix
        float mat[matsize]; 
        for(int j = 0; j < matsize; j++){
            mat[j] = A[matsize*i + j];
        }
        auto mat_norm = fronorm(mat, matsize);
        auto n_squaring = int(std::max(0.0, ceil(log2(mat_norm / 5.371920351148152))));
        
        // Do scaling
        for(int j = 0; j < matsize; j++){
            mat[j] /= pow(2,n_squaring);
        }

        // Calculate pade13 approximation matrices
        float U[matsize], V[matsize];
        pade13_forward(mat, U, V, matsize, row);
        
        // Calculate nominator and denominator
        float P[matsize], Q[matsize];
        for(int j = 0; j < matsize; j++){
            P[j] = U[j] + V[j];
            Q[j] = -U[j] + V[j];
        }
        
        // Solve the system Q*R = P (result is saved in P, LU factors stored in Q)
        int *ipiv; //  pivot indices that define the permutation matrix
        LAPACKE_sgesv(LAPACK_ROW_MAJOR, row, row, Q, row, ipiv, P, row);
        
        // Unsquare result
        float temp[matsize];
        for(int s = 0; s < n_squaring; s++){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                        P, row, P, row, 0.0, temp, row);
            for(int j = 0; j < matsize; j++){
                P[j] = temp[j];
            }
        }
        
        // Save result
        for(int j = 0; j < matsize; j++){
            expmA[matsize*i + j] = P[j];
        }
    }
    return output;
}

at::Tensor expm_backward(at::Tensor input){
    // Problem size
    const auto N = input.size(0);
    const auto row = input.size(1);
    const auto col = input.size(2);
    const auto matsize = row*col;
    
    // Allocate output
    auto A = input.data<float>();
    auto output = at::zeros_like(input);
    auto expmA = output.data<float>();
    
    // For each matrix
    for(int i = 0; i < N; i++){
        float mat[matsize];
        for(int j = 0; j < matsize; j++){
            mat[j] = A[matsize*i + j];
        }
        auto mat_norm = fronorm(mat, matsize);
        auto n_squaring = int(std::max(0.0, ceil(log2(mat_norm / 5.371920351148152))));
        
        // Do scaling
        for(int j = 0; j < matsize; j++){
            mat[j] /= pow(2,n_squaring);
        }
        
        // Calculate pade13 approximation matrices and their derivatives
        float U[matsize], V[matsize], dU[matsize], dV[matsize];
        pade13_forward(mat, U, V, matsize, row);
        pade13_backward(mat, dU, dV, matsize, row);
        
        // Calculate nominator and denominator
        float P[matsize], Q[matsize], dP[matsize], dQ[matsize];
        for(int j = 0; j < matsize; j++){
            P[j] = U[j] + V[j];
            Q[j] = -U[j] + V[j];
            dP[j] = dU[j] + dV[j];
            dQ[j] = -dU[j] + dV[j];
        }
        
        // Solve the system Q*R = P (result is saved in P, LU factors stored in Q)
        int *ipiv; //  pivot indices that define the permutation matrix
        LAPACKE_sgesv(LAPACK_ROW_MAJOR, row, row, Q, row, ipiv, P, row);
        
        // Calculate dP - dQ * R (result stored in dP)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, -1.0,
                    dQ, row, P, row, 1.0, dP, row);
        
        // Solve system Q*R = dP
        LAPACKE_sgesv(LAPACK_ROW_MAJOR, row, row, Q, row, ipiv, dP, row);
        
        // Unsquare result
        float temp[matsize];            
        for(int s = 0; s < n_squaring; s++){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                        dP, row, dP, row, 0.0, temp, row);
            for(int j = 0; j < matsize; j++){
                dP[j] = temp[j];
            }
        }
        
        // Save result
        for(int j = 0; j < matsize; j++){
            expmA[matsize*i + j] = dP[j];
        }
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &expm_forward, "expm forward");
    m.def("backward", &expm_backward, "expm backward");
}
