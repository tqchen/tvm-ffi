"""Tests for TVM-FFI builder with nop_body function using execution engine."""

import ctypes

import numpy as np
import pytest
import tvm_ffi

try:
    import torch
except ImportError:
    torch = None

from mlir_tvm_ffi import (  # type: ignore[import-not-found]
    ExecutionEngine,
    NopCallProvider,
    attach_ffi_func,
    spec,
)
from mlir_tvm_ffi._mlir import ir  # type: ignore[import-not-found]


def test_int_float_arguments() -> None:
    """Test nop_body with correct integer argument - should succeed."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        params = [spec.Var("x", "int32"), spec.Var("y", "float32"), spec.Var("z", "bool")]
        attach_ffi_func(module, "test_int_arguments", params, NopCallProvider())

        # Verify module and execute
        module.operation.verify()
        engine = ExecutionEngine(module, opt_level=2, shared_libs=[])
        func = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            engine.raw_lookup("__tvm_ffi_test_int_arguments")
        )
        # This should succeed with correct int argument
        func(42, 42.1, True)
        # This also works since int can be converted to float, bool can take int
        func(42, 0, 1)
        # we allow implicit conversion from bool to int/float
        func(True, False, 1)

        with pytest.raises(
            TypeError,
            match=r"Mismatched type on argument #0 when calling: "
            r"`test_int_arguments\(x: int32, y: float32, z: bool\)`, expected int",
        ):
            func(42.1, 42.1, True)

        with pytest.raises(
            TypeError,
            match=r"Expects 3 parameters when calling: "
            r"`test_int_arguments\(x: int32, y: float32, z: bool\)`",
        ):
            func(1, 2, 3, 4)


def test_signature_in_error_messages() -> None:
    """Test that function signatures are included in error messages."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        params = [spec.Var("x", "int32"), spec.Var("y", "float32")]
        attach_ffi_func(module, "test_signature", params, NopCallProvider())

        # Verify module and execute
        module.operation.verify()
        engine = ExecutionEngine(module, opt_level=2, shared_libs=[])
        func = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            engine.raw_lookup("__tvm_ffi_test_signature")
        )

        # Test type mismatch error includes signature
        with pytest.raises(
            TypeError,
            match=r"Mismatched type on argument #0 when calling: "
            r"`test_signature\(x: int32, y: float32\)`, expected int",
        ):
            func(42.1, 42.1)  # float instead of int

        # Test parameter count error includes signature
        with pytest.raises(
            TypeError,
            match=r"Expects 2 parameters when calling: "
            r"`test_signature\(x: int32, y: float32\)`",
        ):
            func(1, 2, 3)  # too many parameters

        # Test float type mismatch error includes signature
        with pytest.raises(
            TypeError,
            match=r"Mismatched type on argument #1 when calling: "
            r"`test_signature\(x: int32, y: float32\)`, expected float",
        ):
            func(42, "invalid")  # string instead of float


def test_opaque_handle_arguments() -> None:
    """Test nop_body with opaque handle argument - should succeed."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        params = [spec.Var("handle", "handle")]
        attach_ffi_func(module, "test_opaque_handle", params, NopCallProvider())

        # Verify module and execute
        module.operation.verify()
        engine = ExecutionEngine(module, opt_level=2, shared_libs=[])
        func = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            engine.raw_lookup("__tvm_ffi_test_opaque_handle")
        )

        test_data = ctypes.c_int(42)
        opaque_handle = ctypes.cast(ctypes.pointer(test_data), ctypes.c_void_p)

        # This should succeed with opaque handle
        func(opaque_handle)

        # Test type mismatch error for opaque handle
        with pytest.raises(
            TypeError,
            match=r"Mismatched type on argument #0 when calling: "
            r"`test_opaque_handle\(handle: handle\)`, expected handle",
        ):
            func(42)  # int instead of opaque handle


def test_tensor_arguments() -> None:
    """Test nop_body with tensor argument - should succeed."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with spec.DefaultConfig(device_type="cpu"):
            params = [spec.Tensor("a0", [2, 4], "int32")]
        attach_ffi_func(module, "test_tensor", params, NopCallProvider())

        # Verify module and execute
        module.operation.verify()
        engine = ExecutionEngine(module, opt_level=2, shared_libs=[])
        func = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            engine.raw_lookup("__tvm_ffi_test_tensor")
        )

        # Test valid stride continuity
        A = tvm_ffi.from_dlpack(np.zeros((2, 4), dtype=np.int32))
        func(A)

        # Test invalid stride continuity
        with pytest.raises(
            ValueError,
            match=r"Mismatched Tensor on argument #0 when calling: "
            r"`test_tensor\(a0: Tensor\(\[2, 4\], int32\)\)`, "
            r"expected contiguous",
        ):
            # Create a tensor with invalid stride continuity
            A = tvm_ffi.from_dlpack(np.zeros((4, 2), dtype=np.int32).T)
            assert A.strides == (1, 2)
            func(A)

        # Test dtype mismatch
        with pytest.raises(
            ValueError,
            match=r"Mismatched Tensor on argument #0 when calling: "
            r"`test_tensor\(a0: Tensor\(\[2, 4\], int32\)\)`, "
            r"expected dtype=int32",
        ):
            # Create a tensor with wrong dtype (float32 instead of int32)
            A_wrong_dtype = tvm_ffi.from_dlpack(np.zeros((2, 4), dtype=np.float32))
            func(A_wrong_dtype)

        # Test ndim mismatch
        with pytest.raises(
            ValueError,
            match=r"Mismatched Tensor on argument #0 when calling: "
            r"`test_tensor\(a0: Tensor\(\[2, 4\], int32\)\)`, "
            r"expected ndim=2",
        ):
            # Create a tensor with wrong number of dimensions (3D instead of 2D)
            A_wrong_ndim = tvm_ffi.from_dlpack(np.zeros((2, 4, 3), dtype=np.int32))
            func(A_wrong_ndim)

        # Test shape mismatch
        with pytest.raises(
            ValueError,
            match=r"Mismatched a0\.shape\[0\] on argument #0 when calling: "
            r"`test_tensor\(a0: Tensor\(\[2, 4\], int32\)\)`, "
            r"expected to be 2",
        ):
            # Create a tensor with wrong shape
            A_wrong_shape = tvm_ffi.from_dlpack(np.zeros((3, 5), dtype=np.int32))
            func(A_wrong_shape)

        # Test torch GPU tensor if available and GPU is present
        if torch is not None and torch.cuda.is_available():
            # Test device mismatch (GPU tensor when CPU expected)
            with pytest.raises(
                ValueError,
                match=r"Mismatched Tensor on argument #0 when calling: "
                r"`test_tensor\(a0: Tensor\(\[2, 4\], int32\)\)`, "
                r"expected device_type=cpu",
            ):
                # Create a GPU tensor
                torch_tensor = torch.zeros((2, 4), dtype=torch.int32, device="cuda")
                A_gpu = tvm_ffi.from_dlpack(torch_tensor)
                func(A_gpu)


def test_shape_arguments() -> None:
    """Test nop_body with shape argument - should succeed."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        params = [spec.Shape("s0", [1, spec.Var("n", "int32"), 3])]
        attach_ffi_func(module, "test_shape", params, NopCallProvider())

        # Verify module and execute
        module.operation.verify()
        engine = ExecutionEngine(module, opt_level=2, shared_libs=[])
        func = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            engine.raw_lookup("__tvm_ffi_test_shape")
        )

        # match can pass
        func(tvm_ffi.Shape((1, 2, 3)))
        func(tvm_ffi.Shape((1, 5, 3)))

        with pytest.raises(
            TypeError,
            match=r"Mismatched type on argument #0 when calling: "
            r"`test_shape\(s0: Shape\(\[1, n, 3\]\)\)`, expected ffi.Shape",
        ):
            func(1)

        # Test ndim mismatch
        with pytest.raises(
            ValueError,
            match=r"Mismatched Shape on argument #0 when calling: "
            r"`test_shape\(s0: Shape\(\[1, n, 3\]\)\)`, expected shape size=3",
        ):
            # Create a shape with wrong number of dimensions (2 instead of 3)
            wrong_shape = tvm_ffi.Shape((1, 2))  # Only 2 dimensions
            func(wrong_shape)

        # Test constant constraint mismatch
        with pytest.raises(
            ValueError,
            match=r"Mismatched s0\[0\] on argument #0 when calling: "
            r"`test_shape\(s0: Shape\(\[1, n, 3\]\)\)`, expected to be 1",
        ):
            # Create a shape with wrong first dimension (2 instead of 1)
            wrong_shape = tvm_ffi.Shape((2, 5, 3))  # First dimension is 2, not 1
            func(wrong_shape)


def test_symbolic_shape_constraints_matmul() -> None:
    """Test matrix multiplication with symbolic shape constraints."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()

        with spec.DefaultConfig(device_type="cpu"):
            n = spec.Var("n", "int32")
            m = spec.Var("m", "int32")
            k = spec.Var("k", "int32")
            params = [
                spec.Tensor("A", [n, k], "float32"),
                spec.Tensor("B", [k, m], "float32"),
                spec.Tensor("C", [n, m], "float32"),
            ]
        attach_ffi_func(module, "test_matmul", params, NopCallProvider())
        # Verify module and execute
        module.operation.verify()
        engine = ExecutionEngine(module, opt_level=2, shared_libs=[])
        func = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            engine.raw_lookup("__tvm_ffi_test_matmul")
        )

        # Test valid matrix multiplication shapes
        # A: [2, 3], B: [3, 4], C: [2, 4] - should succeed
        A = tvm_ffi.from_dlpack(np.zeros((2, 3), dtype=np.float32))
        B = tvm_ffi.from_dlpack(np.zeros((3, 4), dtype=np.float32))
        C = tvm_ffi.from_dlpack(np.zeros((2, 4), dtype=np.float32))
        func(A, B, C)

        # Test with different valid shapes
        # A: [5, 2], B: [2, 3], C: [5, 3] - should succeed
        A2 = tvm_ffi.from_dlpack(np.zeros((5, 2), dtype=np.float32))
        B2 = tvm_ffi.from_dlpack(np.zeros((2, 3), dtype=np.float32))
        C2 = tvm_ffi.from_dlpack(np.zeros((5, 3), dtype=np.float32))
        func(A2, B2, C2)

        # Test dimension mismatch in A
        with pytest.raises(
            ValueError,
            match=r"Mismatched B\.shape\[0\] on argument #1 when calling: "
            r"`test_matmul\(A: Tensor\(\[n, k\], float32\), B: Tensor\(\[k, m\], float32\), "
            r"C: Tensor\(\[n, m\], float32\)\)`, "
            r"symbolic constraint violated",
        ):
            # A: [2, 4], B: [3, 4], C: [2, 4] - A's second dim (4) != B's first dim (3)
            A_wrong = tvm_ffi.from_dlpack(np.zeros((2, 4), dtype=np.float32))
            B_wrong = tvm_ffi.from_dlpack(np.zeros((3, 4), dtype=np.float32))
            C_wrong = tvm_ffi.from_dlpack(np.zeros((2, 4), dtype=np.float32))
            func(A_wrong, B_wrong, C_wrong)

        # Test dimension mismatch in C
        with pytest.raises(
            ValueError,
            match=r"Mismatched C\.shape\[0\] on argument #2 when calling: "
            r"`test_matmul\(A: Tensor\(\[n, k\], float32\), B: Tensor\(\[k, m\], float32\), "
            r"C: Tensor\(\[n, m\], float32\)\)`, "
            r"symbolic constraint violated",
        ):
            # A: [2, 3], B: [3, 4], C: [3, 4] - C's first dim (3) != A's first dim (2)
            A_wrong2 = tvm_ffi.from_dlpack(np.zeros((2, 3), dtype=np.float32))
            B_wrong2 = tvm_ffi.from_dlpack(np.zeros((3, 4), dtype=np.float32))
            C_wrong2 = tvm_ffi.from_dlpack(np.zeros((3, 4), dtype=np.float32))
            func(A_wrong2, B_wrong2, C_wrong2)
