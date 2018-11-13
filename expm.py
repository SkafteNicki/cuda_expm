# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 06:41:30 2018

@author: nsde
"""
#%%
import torch

#%%
def torch_expm(A):
    """ """
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1,2), keepdim=True))
    
    # Scaling step
    maxnorm = torch.Tensor([5.371920351148152]).type(A.dtype).to(A.device)
    zero = torch.Tensor([0.0]).type(A.dtype).to(A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    Ascaled = A / 2.0**n_squarings    
    n_squarings = n_squarings.flatten().type(torch.int32)
    
    # Pade 13 approximation
    U, V = torch_pade13(Ascaled)
    P = U + V
    Q = -U + V
    R, _ = torch.gesv(P, Q) # solve P = Q*R
    
    # Unsquaring step
    expmA = [ ]
    for i in range(n_A):
        l = [R[i]]
        for _ in range(n_squarings[i]):
            l.append(l[-1].mm(l[-1]))
        expmA.append(l[-1])
    
    return torch.stack(expmA)

#%%
def torch_log2(x):
    return torch.log(x) / torch.log(torch.Tensor([2.0])).type(x.dtype).to(x.device)

#%%    
def torch_pade13(A):
    b = torch.Tensor([64764752532480000., 32382376266240000., 7771770303897600.,
                      1187353796428800., 129060195264000., 10559470521600.,
                      670442572800., 33522128640., 1323241920., 40840800.,
                      960960., 16380., 182., 1.]).type(A.dtype).to(A.device)
        
    ident = torch.eye(A.shape[1], dtype=A.dtype).to(A.device)
    A2 = torch.matmul(A,A)
    A4 = torch.matmul(A2,A2)
    A6 = torch.matmul(A4,A2)
    U = torch.matmul(A, torch.matmul(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = torch.matmul(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U, V

#%%
if __name__ == '__main__':
    from scipy.linalg import expm
    import numpy as np
    n = 10
    A = torch.randn(n,3,3)
    A[:,2,:] = 0
    
    expm_scipy = np.zeros_like(A)
    for i in range(n):
        expm_scipy[i] = expm(A[i].numpy())
    expm_torch = torch_expm(A)
    print('Difference: ', np.linalg.norm(expm_scipy - expm_torch))
