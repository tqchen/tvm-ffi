import jax
import jax.numpy as jnp
import tvm_ffi
import jax_tvm_ffi


@jax.jit
def add_one_jax(x):
    """JAX function that calls the 'add_one' C++ implementation."""
    call = jax.ffi.ffi_call(
        "add_one_cpu",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )
    y = call(x)
    z = call(y)
    return z


def main():
    mod: tvm_ffi.Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=r"""
            void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x->ndim == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x->dtype == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y->ndim == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y->dtype == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x->shape[0] == y->shape[0]) << "x and y must have the same shape";
              for (int i = 0; i < x->shape[0]; ++i) {
                static_cast<float*>(y->data)[i] = static_cast<float*>(x->data)[i] + 1;
              }
            }
        """,
        functions=["add_one_cpu"],
    )
    print("registered counter", jax_tvm_ffi.get_registered_counter())
    jax_tvm_ffi.regiister_ffi_target("add_one_cpu", mod.add_one_cpu, platform="cpu")

    # Run the JIT-compiled functions
    print("\n--- Testing add_one ---")
    cpu = jax.devices("cpu")[0]
    input_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32, device=cpu)
    # Although our dummy function doesn't compute anything, we can see the
    # print statements from the C++ side when this line is executed.
    output_array = add_one_jax(x)
    print(f"JAX call to 'add_one' completed. Input was: {input_array}")
