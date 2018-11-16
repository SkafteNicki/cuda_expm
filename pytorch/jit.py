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
    A = torch.randn(2, 3, 3)
    print(A)
    print(expm(A.cuda()))
    