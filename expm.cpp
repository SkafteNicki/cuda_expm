#include <torch/torch.h>
#include <cmath>
#include <math.h>
#include <algorithm>
#include "cblas.h"
#include <lapacke.h>

using namespace std;

void printmat(const double* mat, const int row){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < row; j++){
            cout << mat[i*row + j] << ",";
        }
        cout << endl;
    }
}

float fronorm(const double* mat, const int matsize){
    float norm = 0;
    for(int i = 0; i < matsize; i++){
        norm += pow(abs(mat[i]), 2.0);
    }
    return sqrt(norm);
}

void pade13(const double* mat, double* U, double* V, const int matsize, const int row){
    // Pade constants
    std::vector<double> beta;
    beta = {64764752532480000., 32382376266240000., 7771770303897600.,
            1187353796428800., 129060195264000., 10559470521600.,
            670442572800., 33522128640., 1323241920., 40840800.,
            960960., 16380., 182., 1.};

    double iden[matsize], mat2[matsize], mat4[matsize], mat6[matsize];
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
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat, row, mat, row, 0.0, mat2, row);
    // Calculate mat^4 = mat^2 * mat^2
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat2, row, mat2, row, 0.0, mat4, row);
    // Calculate mat^6 = mat^2 * mat^4
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat2, row, mat4, row, 0.0, mat6, row);
    
    // Calculate U
    double temp[matsize];
    for(int i = 0; i < row; i++){
        for(int j = 0; j < row; j++){
            temp[i*row+j] = beta[13]*mat6[i*row+j]*beta[11]*mat4[i*row+j]+beta[9]*mat2[i*row+j];
        }
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat6, row, temp, row, 0.0, temp, row);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < row; j++){
            temp[i*row+j] += beta[7]*mat6[row*i+j] + beta[5]*mat4[row*i+j] + beta[3]*mat2[row*i+j] + beta[1]*iden[row*i+j];
        }
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat, row, temp, row, 0.0, U, row);
    
    // Calculate V
    for(int i = 0; i < row; i++){
        for(int j = 0; j < row; j++){
            temp[i*row+j] = beta[12]*mat6[row*i+j] + beta[10]*mat4[row*i+j] + beta[8]*mat2[row*i+j];
        }
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
                mat6, row, temp, row, 0.0, temp, row);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < row; j++){
            V[i*row+j] = temp[i*row+j] + beta[6]*mat6[i*row+j] + beta[4]*mat4[i*row+j] + beta[2]*mat2[i*row+j] + beta[0]*iden[i*row+j];
        }
    }
    return;
}

at::Tensor expm_forward(at::Tensor input){
    // Problem size
    const auto N = input.size(0);
    const auto row = input.size(1);
    const auto col = input.size(2);
    const auto matsize = row*col;

    // Allocate output
    auto A = input.data<double>();
    auto output = at::zeros_like(input);
    auto expmA = output.data<double>();
    
    // For each matrix
    for(int i = 0; i < N; i++){
        // Copy to local matrix
        double mat[matsize]; 
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
        double U[matsize], V[matsize];
        pade13(mat, U, V, matsize, row);
        
        // Calculate nominator and denominator
        double P[matsize], Q[matsize];
        for(int j = 0; j < matsize; j++){
            P[j] = U[j] + V[j];
            Q[j] = -U[j] + V[j];
        }
        
        // Solve the system Q*R = P (result is saved in P, LU factors stored in Q)
        int *ipiv; //  pivot indices that define the permutation matrix
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, row, row, Q, row, ipiv, P, row);
        
        // Unsquare result
        double temp[matsize];
        for(int s = 0; s < n_squaring; s++){
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, row, row, 1.0,
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &expm_forward, "expm forward");
}
