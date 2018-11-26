#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:21:57 2018

@author: nsde
"""

#%%
import torch
from torch.utils.cpp_extension import load
import warnings

#%%
with warnings.catch_warnings(record=True): # get rid of ABI-incompatible warning
    expm_cpu_module = load(name = 'expm_cpp',
                           sources = ['expm.cpp'],
                           extra_cflags = ['-llapacke', '-lblas', '-lm'],                    
                           verbose=True)
    
    expm_gpu_module = load(name = 'expm_cuda',
                           sources = ['expm_cuda.cpp', 'expm_cuda_kernel.cu', 'Utilities.cu'],
                           
                           extra_cuda_cflags = ['-lmagma', '-lmagmablas', '-lmagma'],
                           verbose=False)

#%%
class _ExpmOperatorAnalytic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        A = A.to(torch.float32)
        if A.is_cuda: res = expm_gpu_module.forward(A.contiguous())
        else: res = expm_cpu_module.forward(A.contiguous())
        ctx.save_for_backward(A)
        return res
    
    @staticmethod
    def backward(ctx, grad): # grad: [n_batch, n, n]
        # Get input from forward pass
        A = ctx.saved_tensors[0]
        if A.is_cuda: gradient = expm_gpu_module.backward(A.contiguous())
        else: gradient = expm_cpu_module.backward(A.contiguous())
        return gradient * grad
    
#%%
class _ExpmOperatorNumeric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        A = A.to(torch.float32)
        if A.is_cuda: res = expm_gpu_module.forward(A.contiguous())
        else: res = expm_cpu_module.forward(A.contiguous())
        ctx.save_for_backward(A, res)
        return res
    
    @staticmethod
    def backward(ctx, grad):
        # Get input from forward pass
        A, expmA = ctx.saved_tensors
        n = A.shape[1]
        
        # Finite difference constant
        h = 0.01
        
        gradient_outer = [ ]
        # Loop over all elements
        for i in range(n):
            gradient_inner = [ ]
            for j in range(n):
                # Permute matrix
                temp = A.clone()
                temp[:,i,j] += h
                
                # Calculate new matrix exponential
                if temp.is_cuda: res = expm_gpu_module.forward(temp.contiguous())
                else: res = expm_cpu_module.forward(temp.contiguous())
                
                # Finite difference
                diff = (res - expmA) / h
                gradient_inner.append((diff * grad).sum(dim=(1,2), keepdim=True)) # [n_batch, 1, 1]
            
            gradient_outer.append(torch.cat(gradient_inner, dim=2)) # [n_batch, 1, n]
                
        # Stack results
        return torch.cat(gradient_outer, dim=1) # [n_batch, n, n]

#%%
def expm(A, analytic=False):
    return _ExpmOperatorAnalytic.apply(A) if analytic else _ExpmOperatorNumeric.apply(A)

#%%
if __name__ == "__main__":
    A = torch.randn(1, 3, 3)
    print(expm(A))
    print(expm(A.cuda()))
#    
#    b = [1., .5000000000, .1200000000, 0.1833333333e-1, 0.1992753623e-2, 
#         0.1630434783e-3, 0.1035196687e-4, 5.175983437e-7, 
#         2.043151357e-8, 6.306022706e-10, 1.483770048e-11, 
#         2.529153492e-13, 2.810170546e-15, 1.544049751e-17]
#    
#    ident = torch.eye(A.shape[1], dtype=A.dtype).to(A.device)
#    A2 = torch.matmul(A,A)
#    A4 = torch.matmul(A2,A2)
#    A6 = torch.matmul(A4,A2)
#    U = torch.matmul(A, torch.matmul(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
#    V = torch.matmul(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
#    print(V)
#    
#    
#    