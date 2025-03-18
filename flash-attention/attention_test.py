import torch
from torch.utils.cpp_extension import load

# Compile and load the CUDA extension at runtime
vec_add_cuda = load(
    name="flash_attn_cuda",
    sources=["flash-attn.cu"],
    extra_cuda_cflags=["-O2"],
    verbose=True
)
