import jax
import numpy as np
import jax.numpy as jnp
import jax.ffi


def rms_norm_ref(x, eps=1e-5):
  scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
  return x / scale

def rms_norm_ffi(x, eps=1e-5):
  if x.dtype != jnp.float32:
    raise ValueError("Only the float32 dtype is implemented by rms_norm")

  call = jax.ffi.ffi_call(
    "rms_norm_ffi",
    jax.ShapeDtypeStruct(x.shape, x.dtype),
    vmap_method="broadcast_all",
  )
  return call(x, eps=np.float32(eps))

gpu = jax.devices("gpu")[0]

x = jnp.arange(32, dtype=jnp.float32, device=gpu)
print(jax.ffi.include_dir())
