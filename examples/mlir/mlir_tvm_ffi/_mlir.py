"""Local MLIR module that re-exports cutlass MLIR functionality."""

from cutlass._mlir import ir  # type: ignore[import-not-found]
from cutlass._mlir.dialects import llvm  # type: ignore[import-not-found]
from cutlass._mlir.execution_engine import ExecutionEngine  # type: ignore[import-not-found]

__all__ = ["ExecutionEngine", "ir", "llvm"]
