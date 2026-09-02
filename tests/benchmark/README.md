<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Structural map cost benchmark

Two opt-in C++ targets measure the same five-rung structural-map cost ladder,
differing only in the expression model they run it on. Neither has a TVM
dependency.

- `tvm_ffi_benchmark_structural_map_tvm_ast_cost` runs the ladder on the TVM
  primitive expression node set copied verbatim into
  `src/ffi/testing/structural_map_tvm_ast.h`: the real `Var` / `IntImm` /
  `Add` / `Mul` / `FloorDiv` / `FloorMod` classes, their `span` and `ty`
  fields, their `_type_s_eq_hash_kind` values, and their registered
  `__s_visit__` / `__s_mutate__` / `__s_maybe_inplace_mutate__` bodies. It
  aborts before measuring if any fixture node type is missing a compiled hook,
  because a reflection fallback would measure a different code path. This is
  the target to use as a TVM proxy.
- `tvm_ffi_benchmark_structural_map_cost` runs the ladder on the smaller
  synthetic model that shares only the two non-traversed base fields. It is
  measurably cheaper than the copied AST and is kept for contrast, not as a
  proxy.

```bash
cmake -S . -B build-benchmark -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG' \
  -DTVM_FFI_BUILD_BENCHMARKS=ON
cmake --build build-benchmark
taskset -c 22 build-benchmark/bin/tvm_ffi_benchmark_structural_map_tvm_ast_cost
taskset -c 22 build-benchmark/bin/tvm_ffi_benchmark_structural_map_cost
```

`TVM_FFI_BENCH_REPEATS` overrides the default iteration count. Run on an
otherwise idle, performance-governor core and retain the repository SHA with
the output. The reported unit is median nanoseconds per processed expression
node over eleven in-process samples.

Both targets detect whether the checked-out tvm-ffi exposes the historical
`StructuralWalk<order, deduplicate>` API used by the calibration pin. They
enable that pin's deduplicating walk when available and otherwise build against
the current non-deduplicating walk API. The walk denominator differs between
the two APIs -- deduplicating walks report 15 and 155 nodes on the split/fuse
base and nested-4 fixtures, the non-deduplicating walk reports 17 and 185 -- so
a walk rung is only comparable against a run using the same API.

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
