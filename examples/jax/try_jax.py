import jax
import numpy as np
import jax.numpy as jnp
import jax.ffi
import jax_tvm_ffi


@jax.jit
def add_one_jax(x):
    """JAX function that calls the 'add_one' C++ implementation."""
    call = jax.ffi.ffi_call(
        "custom_dispatch",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )
    return call(x)


@jax.jit
def mul_jax(x, y):
    """JAX function that calls the 'mul' C++ implementation."""
    call = jax.ffi.ffi_call(
        "custom_dispatch",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )
    return call(x, y)

# Run the JIT-compiled functions
print("\n--- Testing add_one ---")
input_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32, device = jax.devices("cpu")[0])
# Although our dummy function doesn't compute anything, we can see the
# print statements from the C++ side when this line is executed.
output_array = add_one_jax(x)
print(f"JAX call to 'add_one' completed. Input was: {input_array}")

