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
                           sources = ['expm_cuda.cpp', 'expm_cuda_kernel.cu'], 
                           verbose=False,
                           extra_include_paths = ['/usr/local/cuda-8.0/lib64/'])

#%%
class _ExpmOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        if A.is_cuda: res = expm_gpu_module.forward(A.contiguous())
        else: res = expm_cpu_module.forward(A.contiguous())
        ctx.save_for_backward(A, res)
        return res
    
    @staticmethod
    def backward(ctx, grad): # grad: [n_batch, n, n]
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
                temp = A
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
def expm(A):
    return _ExpmOperator.apply(A)

#%%
if __name__ == "__main__":
    A = 10*torch.randn(10, 3, 3).to(torch.double)
    A.requires_grad = True
    expmA_pytorch = expm(A)
    grad = torch.autograd.grad(expmA_pytorch.sum(), A)