#include <torch/torch.h>

// Cuda forward declaration
at::Tensor expm_cuda_forward(at::Tensor input);

// Shortcuts for checking
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Function declaration
at::Tensor expm_forward(at::Tensor input){
    CHECK_INPUT(input);
    return expm_cuda_forward(input);
}

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &expm_forward, "expm forward (CUDA)");
}

