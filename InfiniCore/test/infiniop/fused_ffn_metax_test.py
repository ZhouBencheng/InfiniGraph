"""
Standalone fused_ffn correctness test for MetaX platform.

Uses PyTorch CPU for reference computation and infinirt C API for GPU memory
management, avoiding the need for a MACA-compatible PyTorch build.

Usage:
    export INFINI_ROOT=/root/InfiniCore/install
    export LD_LIBRARY_PATH=$INFINI_ROOT/lib:/opt/maca-2.32.0.6/lib:$LD_LIBRARY_PATH
    python3 fused_ffn_metax_test.py
"""

import os
import sys
import ctypes
from ctypes import c_int, c_size_t, c_ssize_t, c_float, c_uint64, c_void_p, POINTER, byref
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

# ==============================================================================
#  Library loading
# ==============================================================================

INFINI_ROOT = os.getenv("INFINI_ROOT") or str(Path.home() / ".infini")

def load_lib():
    libop_path = os.path.join(INFINI_ROOT, "lib", "libinfiniop.so")
    librt_path = os.path.join(INFINI_ROOT, "lib", "libinfinirt.so")
    assert os.path.exists(libop_path), f"Cannot find {libop_path}"
    assert os.path.exists(librt_path), f"Cannot find {librt_path}"
    librt = ctypes.CDLL(librt_path)
    libop = ctypes.CDLL(libop_path)
    return libop, librt

LIB, LIBRT = load_lib()

# ==============================================================================
#  Constants
# ==============================================================================

INFINI_DEVICE_METAX = 4
INFINI_DTYPE_F16 = 12
INFINI_DTYPE_F32 = 13
INFINI_DTYPE_BF16 = 19
INFINIRT_MEMCPY_H2D = 1
INFINIRT_MEMCPY_D2H = 2

DTYPE_SIZES = {INFINI_DTYPE_F16: 2, INFINI_DTYPE_BF16: 2, INFINI_DTYPE_F32: 4}
DTYPE_NAMES = {INFINI_DTYPE_F16: "F16", INFINI_DTYPE_BF16: "BF16", INFINI_DTYPE_F32: "F32"}
TORCH_DTYPES = {INFINI_DTYPE_F16: torch.float16, INFINI_DTYPE_BF16: torch.bfloat16, INFINI_DTYPE_F32: torch.float32}

# ==============================================================================
#  C API helpers
# ==============================================================================

infiniopHandle_t = c_void_p
infiniopTensorDescriptor_t = c_void_p
infiniopFusedFFNDescriptor_t = c_void_p
infinirtStream_t = c_void_p


def check(status, msg=""):
    if status != 0:
        raise RuntimeError(f"InfiniCore error {status}: {msg}")


def set_device(device_type, device_id=0):
    LIB.infinirtSetDevice.argtypes = [c_int, c_int]
    LIB.infinirtSetDevice.restype = c_int
    check(LIB.infinirtSetDevice(device_type, device_id), "infinirtSetDevice")


def create_handle():
    handle = infiniopHandle_t()
    check(LIB.infiniopCreateHandle(byref(handle)), "infiniopCreateHandle")
    return handle


def destroy_handle(handle):
    check(LIB.infiniopDestroyHandle(handle), "infiniopDestroyHandle")


def create_stream():
    stream = infinirtStream_t()
    check(LIB.infinirtStreamCreate(byref(stream)), "infinirtStreamCreate")
    return stream


def destroy_stream(stream):
    check(LIB.infinirtStreamDestroy(stream), "infinirtStreamDestroy")


def sync_stream(stream):
    check(LIB.infinirtStreamSynchronize(stream), "infinirtStreamSynchronize")


def gpu_malloc(size):
    ptr = c_void_p()
    check(LIB.infinirtMalloc(byref(ptr), c_size_t(size)), "infinirtMalloc")
    return ptr


def gpu_free(ptr):
    check(LIB.infinirtFree(ptr), "infinirtFree")


def memcpy_h2d(dst, src, size):
    check(LIB.infinirtMemcpy(dst, src, c_size_t(size), c_int(INFINIRT_MEMCPY_H2D)), "memcpy H2D")


def memcpy_d2h(dst, src, size):
    check(LIB.infinirtMemcpy(dst, src, c_size_t(size), c_int(INFINIRT_MEMCPY_D2H)), "memcpy D2H")


def create_tensor_desc(shape, strides, dtype):
    desc = infiniopTensorDescriptor_t()
    ndim = len(shape)
    c_shape = (c_size_t * ndim)(*shape)
    c_strides = (c_ssize_t * ndim)(*strides)
    check(LIB.infiniopCreateTensorDescriptor(
        byref(desc), c_size_t(ndim), c_shape, c_strides, c_int(dtype)
    ), "infiniopCreateTensorDescriptor")
    return desc


def destroy_tensor_desc(desc):
    check(LIB.infiniopDestroyTensorDescriptor(desc), "infiniopDestroyTensorDescriptor")


# ==============================================================================
#  GPU tensor helper
# ==============================================================================

class GpuTensor:
    """Manages a tensor with CPU data and GPU copy."""

    def __init__(self, cpu_tensor, dtype_id):
        self.cpu_tensor = cpu_tensor.contiguous()
        self.dtype_id = dtype_id
        self.nbytes = self.cpu_tensor.nelement() * self.cpu_tensor.element_size()
        self.gpu_ptr = gpu_malloc(self.nbytes)
        # Copy to GPU
        src_ptr = ctypes.c_void_p(self.cpu_tensor.data_ptr())
        memcpy_h2d(self.gpu_ptr, src_ptr, self.nbytes)
        # Create descriptor
        shape = list(self.cpu_tensor.shape)
        strides = list(self.cpu_tensor.stride())
        self.desc = create_tensor_desc(shape, strides, dtype_id)

    def read_back(self):
        """Copy GPU data back to a new CPU tensor."""
        result = torch.empty_like(self.cpu_tensor)
        dst_ptr = ctypes.c_void_p(result.data_ptr())
        memcpy_d2h(dst_ptr, self.gpu_ptr, self.nbytes)
        return result

    def free(self):
        destroy_tensor_desc(self.desc)
        gpu_free(self.gpu_ptr)


# ==============================================================================
#  Reference implementation
# ==============================================================================

def reference_fused_ffn(x, residual, norm_w, gate_up_w, down_w, eps):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + eps)
    normalized = (normalized * norm_w.float()).to(x.dtype)

    gate_up = F.linear(normalized, gate_up_w)
    di = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :di], gate_up[..., di:]

    hidden = F.silu(gate) * up
    out = F.linear(hidden, down_w)

    if residual is not None:
        out = out + residual

    return out


# ==============================================================================
#  Test function
# ==============================================================================

def test_fused_ffn(handle, stream, batch, hidden_dim, intermediate_dim, has_residual, dtype_id):
    torch_dtype = TORCH_DTYPES[dtype_id]
    dtype_name = DTYPE_NAMES[dtype_id]

    print(f"  Testing batch={batch} hidden={hidden_dim} inter={intermediate_dim} "
          f"residual={has_residual} dtype={dtype_name} ... ", end="", flush=True)

    weight_scale = 1.0 / (hidden_dim ** 0.5)
    epsilon = 1e-6

    # Generate random data on CPU
    x_cpu = torch.rand(batch, hidden_dim, dtype=torch_dtype, device="cpu")
    norm_w_cpu = torch.rand(hidden_dim, dtype=torch_dtype, device="cpu")
    gate_up_w_cpu = (torch.rand(2 * intermediate_dim, hidden_dim, dtype=torch_dtype, device="cpu") * weight_scale)
    down_w_cpu = (torch.rand(hidden_dim, intermediate_dim, dtype=torch_dtype, device="cpu") * weight_scale)
    residual_cpu = torch.rand(batch, hidden_dim, dtype=torch_dtype, device="cpu") if has_residual else None
    out_cpu = torch.zeros(batch, hidden_dim, dtype=torch_dtype, device="cpu")

    # Compute reference on CPU
    ref = reference_fused_ffn(x_cpu, residual_cpu, norm_w_cpu, gate_up_w_cpu, down_w_cpu, epsilon)

    # Upload to GPU
    x_gpu = GpuTensor(x_cpu, dtype_id)
    norm_w_gpu = GpuTensor(norm_w_cpu, dtype_id)
    gate_up_w_gpu = GpuTensor(gate_up_w_cpu, dtype_id)
    down_w_gpu = GpuTensor(down_w_cpu, dtype_id)
    residual_gpu = GpuTensor(residual_cpu, dtype_id) if has_residual else None
    out_gpu = GpuTensor(out_cpu, dtype_id)

    # Create fused_ffn descriptor
    descriptor = infiniopFusedFFNDescriptor_t()
    check(LIB.infiniopCreateFusedFFNDescriptor(
        handle,
        byref(descriptor),
        out_gpu.desc,
        x_gpu.desc,
        residual_gpu.desc if residual_gpu else None,
        norm_w_gpu.desc,
        gate_up_w_gpu.desc,
        down_w_gpu.desc,
        c_float(epsilon),
    ), "infiniopCreateFusedFFNDescriptor")

    # Destroy descriptors (no longer needed after create)
    x_gpu_desc_save = x_gpu.desc
    x_gpu.desc = None  # prevent double-free
    destroy_tensor_desc(x_gpu_desc_save)
    norm_w_desc_save = norm_w_gpu.desc
    norm_w_gpu.desc = None
    destroy_tensor_desc(norm_w_desc_save)
    gate_up_desc_save = gate_up_w_gpu.desc
    gate_up_w_gpu.desc = None
    destroy_tensor_desc(gate_up_desc_save)
    down_w_desc_save = down_w_gpu.desc
    down_w_gpu.desc = None
    destroy_tensor_desc(down_w_desc_save)
    out_desc_save = out_gpu.desc
    out_gpu.desc = None
    destroy_tensor_desc(out_desc_save)
    if residual_gpu:
        res_desc_save = residual_gpu.desc
        residual_gpu.desc = None
        destroy_tensor_desc(res_desc_save)

    # Get workspace size
    workspace_size = c_uint64(0)
    check(LIB.infiniopGetFusedFFNWorkspaceSize(descriptor, byref(workspace_size)),
          "infiniopGetFusedFFNWorkspaceSize")

    ws_ptr = None
    if workspace_size.value > 0:
        ws_ptr = gpu_malloc(workspace_size.value)

    # Execute
    check(LIB.infiniopFusedFFN(
        descriptor,
        ws_ptr,
        workspace_size,
        out_gpu.gpu_ptr,
        x_gpu.gpu_ptr,
        residual_gpu.gpu_ptr if residual_gpu else None,
        norm_w_gpu.gpu_ptr,
        gate_up_w_gpu.gpu_ptr,
        down_w_gpu.gpu_ptr,
        stream,
    ), "infiniopFusedFFN")

    sync_stream(stream)

    # Read back result
    result = out_gpu.read_back()

    # Compare
    atol = 1e-2 if dtype_id == INFINI_DTYPE_F16 else 5e-2
    rtol = 1e-2 if dtype_id == INFINI_DTYPE_F16 else 5e-2

    if torch.allclose(result, ref, atol=atol, rtol=rtol):
        print("\033[92mPASS\033[0m")
    else:
        max_diff = (result - ref).abs().max().item()
        mean_diff = (result - ref).abs().mean().item()
        print(f"\033[91mFAIL\033[0m (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
        # Show details for debugging
        print(f"    ref  stats: min={ref.min().item():.4f} max={ref.max().item():.4f} mean={ref.mean().item():.4f}")
        print(f"    result stats: min={result.min().item():.4f} max={result.max().item():.4f} mean={result.mean().item():.4f}")
        raise AssertionError(f"Correctness check failed for batch={batch} dtype={dtype_name}")

    # Cleanup
    check(LIB.infiniopDestroyFusedFFNDescriptor(descriptor), "destroy descriptor")
    if ws_ptr:
        gpu_free(ws_ptr)
    gpu_free(x_gpu.gpu_ptr)
    gpu_free(norm_w_gpu.gpu_ptr)
    gpu_free(gate_up_w_gpu.gpu_ptr)
    gpu_free(down_w_gpu.gpu_ptr)
    gpu_free(out_gpu.gpu_ptr)
    if residual_gpu:
        gpu_free(residual_gpu.gpu_ptr)


# ==============================================================================
#  Test cases
# ==============================================================================

TEST_CASES = [
    # (batch, hidden_dim, intermediate_dim, has_residual)
    (1, 2048, 5632, True),
    (2, 2048, 5632, True),
    (4, 2048, 5632, True),
    (8, 2048, 5632, True),
    (16, 2048, 5632, True),
    (32, 2048, 5632, True),
    (64, 2048, 5632, True),
    (128, 2048, 5632, True),
    # Different architectures
    (16, 4096, 11008, True),    # LLaMA-7B
    (16, 5120, 13824, True),    # LLaMA-13B
    (16, 3584, 18944, True),    # Qwen
    # Without residual
    (16, 2048, 5632, False),
    (32, 4096, 11008, False),
]

DTYPES = [INFINI_DTYPE_F16, INFINI_DTYPE_BF16]


def main():
    print("=" * 60)
    print("FusedFFN MetaX Correctness Test")
    print("=" * 60)

    # Set MetaX device
    set_device(INFINI_DEVICE_METAX, 0)
    handle = create_handle()
    stream = create_stream()

    try:
        for dtype_id in DTYPES:
            print(f"\n--- Testing dtype: {DTYPE_NAMES[dtype_id]} ---")
            for batch, hidden, inter, has_res in TEST_CASES:
                test_fused_ffn(handle, stream, batch, hidden, inter, has_res, dtype_id)
    finally:
        destroy_stream(stream)
        destroy_handle(handle)

    print("\n\033[92mAll tests passed!\033[0m")


if __name__ == "__main__":
    main()
