"""Main module for TVM-FFI function generation and testing."""

from mlir_tvm_ffi import (  # type: ignore[import-not-found]
    DynamicParamPackCallProvider,
    attach_ffi_func,
    spec,
)
from mlir_tvm_ffi._mlir import ir  # type: ignore[import-not-found]


def main() -> None:
    """Demonstrate TVM-FFI function generation."""
    ctx = ir.Context()

    with ctx, ir.Location.unknown():
        module = ir.Module.create()

        n = spec.Var("n", "int32")
        t0 = spec.Tensor("t0", [n], "float32")
        t1 = spec.Tensor("t1", [n], "float32")
        attach_ffi_func(module, "add_one", [t0, t1], DynamicParamPackCallProvider("_mlir_add_one"))
        # Verify the module
        # module.operation.verify()
        print(module)


if __name__ == "__main__":
    main()
