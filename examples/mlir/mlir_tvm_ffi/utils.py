"""Utilities for MLIR module compilation and object file generation."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from ._mlir import ExecutionEngine, ir


def dump_to_object_file(
    module: ir.Module,
    function_names: list[str],
    output_path: str,
    context: ir.Context,
    shared_libs: list[str] = [],
) -> None:
    """Dump the MLIR module to an object file using ExecutionEngine."""
    with context:
        # Verify the module
        module.operation.verify()

        # Create execution engine with object file dumping
        engine = ExecutionEngine(module, opt_level=2, shared_libs=shared_libs)
        for function_name in function_names:
            # need to do raw lookup for each functions
            # to trigger generation of the function
            engine.raw_lookup(function_name)

        # Dump the compiled object file
        engine.dump_to_object_file(output_path)
        print(f"Successfully created object file: {output_path}")


def compile_to_shared_library(object_file: str, output_path: str, compiler: str = "gcc") -> None:
    """Compile an object file to a shared library."""
    # Check if compiler is available
    if not shutil.which(compiler):
        raise RuntimeError(
            f"Compiler '{compiler}' not found. Please install it or specify a different compiler."
        )

    print(f"Compiling {object_file} to shared library: {output_path}")

    # Compile object file to shared library
    cmd = [compiler, "-shared", "-fPIC", object_file, "-o", output_path]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error compiling to shared library: {e.stderr}")
        raise


def dump_to_shared_library(
    module: ir.Module,
    function_names: list[str],
    output_path: str,
    context: ir.Context,
    compiler: str = "gcc",
    shared_libs: list[str] = [],
) -> None:
    """Dump the MLIR module directly to a shared library."""
    # Create a temporary object file
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as tmp_obj:
        tmp_obj_path = tmp_obj.name

    try:
        # First dump to object file
        dump_to_object_file(module, function_names, tmp_obj_path, context, shared_libs)

        # Then compile to shared library
        compile_to_shared_library(tmp_obj_path, output_path, compiler)
    finally:
        # Clean up temporary object file
        tmp_path = Path(tmp_obj_path)
        if tmp_path.exists():
            tmp_path.unlink()
