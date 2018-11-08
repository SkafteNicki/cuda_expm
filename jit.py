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
square_cuda = load(name = 'expm_cpp',
                   sources = ['expm.cpp'], 
                   verbose=False)

square_cuda = load(name = 'expm_cuda',
                   sources = ['expm_cuda.cpp', 'expm_cuda_kernel.cu'], 
                   verbose=False,
                   extra_include_paths = ['/usr/local/cuda-8.0/lib64/'])

