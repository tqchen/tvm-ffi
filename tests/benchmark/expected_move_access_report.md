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

# Move-correct Expected payload access

## Scope and method

Measures the engine change in this branch and nothing else: the rvalue
`ExpectedUnsafe::GetData` overload, `ExpectedUnsafe::MoveDataAutoCast`, and the
four built-in container mutate hooks that now move the mapped value instead of
copying it. The benchmark hook bodies are unchanged apart from the mandatory
macro rename, so the hook-style rewrite is not part of these numbers.

Host: x86_64 AMD Ryzen 9 7950X, GCC 14.3.0 (conda-forge), `-O3 -DNDEBUG`,
`taskset -c 8`, schedutil governor. Before is `0cfb9a9`; after is that commit
plus the three implementation commits on this branch. The two binaries were run
interleaved for seven rounds and each cell is the median of those rounds, where
each round is itself the harness median over eleven in-process samples. Only
deltas transfer to another host.

Rung names map to the harness fields as `walk` = `rung1`,
`2a default mutate` = `rung2a`, `no-op map` = `rung3`, `changed map` = `rung4`.
Walk is the control: the visit hooks are untouched.

## Minimal AST (`tvm_ffi_benchmark_structural_map_cost`)

Nanoseconds per processed node.

| Fixture | Op | before | after | delta |
|---|---|---:|---:|---:|
| base distinct | walk | 4.11 | 4.16 | +0.05 |
| base distinct | 2a default mutate | 15.18 | 13.24 | -1.94 |
| base distinct | no-op map | 40.32 | 37.12 | -3.20 |
| base distinct | changed map | 45.51 | 42.95 | -2.55 |
| nested-4 distinct | walk | 3.99 | 4.11 | +0.12 |
| nested-4 distinct | 2a default mutate | 15.07 | 12.74 | -2.33 |
| nested-4 distinct | no-op map | 35.77 | 32.26 | -3.51 |
| nested-4 distinct | changed map | 45.21 | 41.92 | -3.29 |

The walk arms overlap round to round -- base distinct spans 4.08-4.14 before
and 4.10-4.21 after -- so the control is unchanged within noise. The mutate
arms do not overlap: base distinct rung 2a spans 14.98-16.37 before and
13.19-13.73 after.

## TVM primitive expression AST (`tvm_ffi_benchmark_structural_map_tvm_ast_cost`)

The proxy target for TVM. Nanoseconds per processed node.

| Fixture | Op | before | after | delta |
|---|---|---:|---:|---:|
| base distinct | walk | 6.65 | 6.87 | +0.22 |
| base distinct | 2a default mutate | 25.42 | 22.54 | -2.88 |
| base distinct | no-op map | 50.28 | 44.49 | -5.79 |
| base distinct | changed map | 62.76 | 56.07 | -6.69 |
| nested-4 distinct | walk | 7.24 | 7.18 | -0.06 |
| nested-4 distinct | 2a default mutate | 25.51 | 21.72 | -3.79 |
| nested-4 distinct | no-op map | 47.36 | 40.42 | -6.94 |
| nested-4 distinct | changed map | 64.08 | 55.88 | -8.20 |

Every mutate path is 11-15% faster on the TVM node set. Its hooks use
`std::move(ExpectedUnsafe::GetData(...)).cast<T>()`, which is the pattern the
rvalue overload repairs without touching any call site.
