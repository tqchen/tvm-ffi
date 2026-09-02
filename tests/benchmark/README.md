# Structural map cost benchmark

This opt-in C++ target measures the five-rung structural-map cost ladder on a
small expression model with the same two non-traversed base fields as TVM's
`ExprNode`. It has no TVM dependency.

```bash
cmake -S . -B build-benchmark -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG' \
  -DTVM_FFI_BUILD_BENCHMARKS=ON
cmake --build build-benchmark --target tvm_ffi_benchmark_structural_map_cost
taskset -c 22 build-benchmark/bin/tvm_ffi_benchmark_structural_map_cost
```

`TVM_FFI_BENCH_REPEATS` overrides the default iteration count. Run on an
otherwise idle, performance-governor core and retain the repository SHA with
the output. The reported unit is median nanoseconds per processed expression
node over eleven in-process samples.

The target detects whether the checked-out tvm-ffi exposes the historical
`StructuralWalk<order, deduplicate>` API used by the calibration pin. It enables
that pin's deduplicating walk when available and otherwise builds against the
current non-deduplicating walk API.

TVM calibration is intentionally separate from this target. Build
`tests/benchmark/structural_map_tvm_calibration.cc` against the matching TVM
checkout and its Release libraries; it has a TVM dependency and is not part of
the tvm-ffi CMake graph. For example:

```bash
c++ -std=c++20 -O3 -DNDEBUG \
  -I$TVM_SOURCE/include -I$TVM_SOURCE/3rdparty/tvm-ffi/include \
  -I$TVM_SOURCE/3rdparty/tvm-ffi/3rdparty/dlpack/include \
  tests/benchmark/structural_map_tvm_calibration.cc \
  -L$TVM_BUILD/lib -Wl,-rpath,$TVM_BUILD/lib \
  -ltvm_compiler -ltvm_runtime -ltvm_ffi \
  -o build-benchmark/bin/tvm_ffi_benchmark_tvm_calibration
taskset -c 22 build-benchmark/bin/tvm_ffi_benchmark_tvm_calibration
```
