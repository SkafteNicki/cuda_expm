#include <torch/torch.h>

at::Tensor expm_forward(at::Tensor input){
    return input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &expm_forward, "expm forward");
}
