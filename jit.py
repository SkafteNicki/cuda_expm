#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:21:57 2018

@author: nsde
"""

#%%
import torch
from torch.utils.cpp_extension import load

#%%
expm_cpu_module = load(name = 'expm_cpp',
                       sources = ['expm.cpp'], 
                       verbose=False)

expm_gpu_module = load(name = 'expm_cuda',
                       sources = ['expm_cuda.cpp', 'expm_cuda_kernel.cu'], 
                       verbose=False,
                       extra_include_paths = ['/usr/local/cuda-8.0/lib64/'])

#%%
class _ExpmOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        if A.is_cuda: return expm_gpu_module.forward(A.contiguous())
        else: return expm_cpu_module.forward(A.contiguous())
    @staticmethod
    def backward(ctx, grad):
        pass

#%%
def expm(A):
    return _ExpmOperator.apply(A)

#%%
if __name__ == "__main__":
    A = torch.randn(10, 3, 3)
    expmA = expm(A)
    
    B = torch.randn(10, 3, 3)
    expmB = expm(B)