"""Tests for MLIR type builder functionality."""

from mlir_tvm_ffi._mlir import ir  # type: ignore[import-not-found]
from mlir_tvm_ffi.mlir_builder import MLIRTypeBuilder  # type: ignore[import-not-found]


def test_mlir_type_builder() -> None:
    """Test MLIR type builder functionality."""
    with ir.Context(), ir.Location.unknown():
        builder = MLIRTypeBuilder()
        assert str(builder.i32_type) == "i32"
        assert str(builder.ui32_type) == "ui32", str(builder.ui32_type)
        assert str(builder.i64_type) == "i64"
        assert str(builder.i16_type) == "i16"
        assert str(builder.i8_type) == "i8"
        assert str(builder.f32_type) == "f32"
        assert str(builder.f64_type) == "f64"
        assert str(builder.ptr_type) == "!llvm.ptr"
        assert (
            str(builder.struct_type(fields=[builder.i32_type, builder.ui32_type]))
            == "!llvm.struct<(i32, ui32)>"
        )
        assert (
            str(builder.struct_type(fields=[builder.i32_type, builder.ui32_type], packed=True))
            == "!llvm.struct<packed (i32, ui32)>"
        )
        assert (
            str(builder.struct_type(name="MyStruct", fields=[builder.i32_type, builder.ui32_type]))
            == '!llvm.struct<"MyStruct", (i32, ui32)>'
        )
        assert (
            str(builder.identified_struct_type("MyStruct"))
            == '!llvm.struct<"MyStruct", (i32, ui32)>'
        )
