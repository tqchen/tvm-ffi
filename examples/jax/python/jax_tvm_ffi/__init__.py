"""JAX TVM FFI Python package."""

import sys
from pathlib import Path

import jax.ffi
import tvm_ffi


def _load_lib() -> tvm_ffi.Module:
    # first look at the directory of the current file
    file_dir = Path(__file__).resolve().parent

    if sys.platform.startswith("win32"):
        lib_dll_name = "jax_tvm_ffi.dll"
    elif sys.platform.startswith("darwin"):
        lib_dll_name = "jax_tvm_ffi.dylib"
    else:
        lib_dll_name = "jax_tvm_ffi.so"

    dirs = [file_dir, file_dir / ".." / ".." / "build"]
    for dir in dirs:
        print(dir / lib_dll_name)
        if (dir / lib_dll_name).exists():
            lib_path = dir / lib_dll_name
            return tvm_ffi.load_module(str(lib_path))

    raise RuntimeError(f"Cannot find library: {lib_dll_name}")


_LIB = _load_lib()


def regiister_ffi_target(
    name: str, function: tvm_ffi.Function, platform: str = "cpu", *, allow_cuda_graph: bool = False
):
    """Function to register a ffi target for jax with tvm_ffi.Function

    Parameters
    ----------
    name: str
        The name of the ffi target

    function: tvm_ffi.Function
        The function to register

    platform: str
        The platform of the ffi target

    allow_cuda_graph: bool
        Whether the function can be used in cuda graph capture
    """
    traits = 1 if allow_cuda_graph else 0
    fn = jax.ffi.pycapsule(_LIB.Register(function, traits))
    jax.ffi.register_ffi_target(name, fn, platform=platform)
    return fn

def registered_count() -> int:
    return _LIB.RegisteredCount()
