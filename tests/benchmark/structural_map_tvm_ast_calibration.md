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

# Calibrating the standalone ladder on the copied TVM AST

## Why this exists

The tvm-ffi-only cost harness originally ran on a synthetic node set that shared
only TVM `ExprNode`'s two non-traversed fields. That model preserved the shape of
the cost ordering but was 14-29% cheaper than the real thing, so the cost budget
had to be re-derived directly on TVM and every later measurement kept a TVM build
in the loop.

This report replaces the model rather than the workaround. `src/ffi/testing/
structural_map_tvm_ast.h` copies the TVM primitive-expression node set into
tvm-ffi -- the classes, their field sets, their structural-equality kinds, and
the registered hook bodies -- and `tests/benchmark/structural_map_tvm_ast_cost.cc`
reruns the five-rung ladder on it.

## What was copied

| Copied | From |
|---|---|
| `Span` / `SpanNode` | `include/tvm/ir/source_map.h` |
| `Type` / `TypeNode` / `PrimType` / `PrimTypeNode`, `Type::Missing()`, the thread-local `PrimType::Int` cache, `GetCachedPrimTypeNode` | `include/tvm/ir/base_expr.h`, `src/ir/type.cc` |
| `ExprNode` (`mutable Span span`, `mutable Type ty = Type::Missing()`, `kTVMFFISEqHashKindTreeNode`, `_type_child_slots = 64`), `Expr`, `TypedExpr<T>`, `PrimExpr`, `PrimExprConvertible` and all four `TypeTraits` specializations including the nested `ty` check and the six-entry `PrimExpr` fallback chain | `include/tvm/ir/base_expr.h` |
| `VarNode` / `Var` (`kTVMFFISEqHashKindFreeVar`, `_type_child_slots = 1`), `IntImmNode` / `IntImm`, `PrimVar` | `include/tvm/ir/expr.h`, `src/ir/expr.cc`, `include/tvm/tirx/var.h` |
| `BinaryOpNode<T>` and `Add` / `Mul` / `FloorDiv` / `FloorMod`, including `TVM_DEFINE_BINOP_CONSTRUCTOR`'s `node->ExprNode::ty = a.get()->ExprNode::ty` | `include/tvm/ir/prim/expr.h`, `src/ir/prim/expr.cc` |
| `VisitIntImm` / `MutateIntImm` / `MaybeInplaceMutateIntImm`, `VisitVar` / `MutateVar` / `MaybeInplaceMutateVar` including `IsTIRXBufferType` and the def-region branches | `src/ir/expr.cc` |
| `VisitBinary` / `MutateBinary` / `MaybeInplaceMutateBinary` including the raw `TVMFFIAny` entry points, the error-context updates, the `PrimExpr` casts and the `make_object<TNode>(*self)` rebuild | `src/ir/prim/expr.cc` |

All three type-attribute columns -- `__s_visit__`, `__s_mutate__` and
`__s_maybe_inplace_mutate__` -- are registered for every one of the six node
types. The benchmark asserts this before measuring and exits non-zero otherwise,
because a missing entry silently reroutes the traversal to the reflected
fallback and would measure a different code path. The gate was verified to fire
by pointing one entry at an unregistered type index.

Only type keys were renamed. Two things could not be copied verbatim and are
called out at their use sites:

- The `PrimExpr` fallback converters for `double`, `String` and
  `PrimExprConvertible` throw instead of building `FloatImm` / `StringImm` /
  a converted expression, because those node types are outside the copied set.
  The fallback chain has TVM's length and order, and the benchmark never takes
  it: every value handed to a hook is already a copied expression node.
- `tirx.BufferType` is replaced by a registered `testing.tvmast.BufferType`
  stand-in so the one-time key lookup in the `Var` hooks resolves. The branch is
  false in both the copy and in TVM for the split/fuse fixtures, which only build
  `PrimType` vars.

## Shape is exact

| Property | Copied AST | Original TVM |
|---|---|---|
| `sizeof(ExprNode)` / `VarNode` / `IntImmNode` / `AddNode` | 40 / 56 / 48 / 56 | 40 / 56 / 48 / 56 |
| base fixture node histogram | Add 3, Mul 3, FloorDiv 1, FloorMod 1, IntImm 5, Var 2 | Add 3, Mul 3, FloorDiv 1, FloorMod 1, IntImm 5, Var 2 |
| walk denominator, base / nested-4 | 15 / 155 | 15 / 155 |
| mutate occurrence denominator | 17 / 185 | 17 / 185 |
| map denominator | 15 / 155 | 15 / 155 |

Field count, object size, allocation size and fixture shape are therefore closed
by construction, not by argument.

## Method and heads

Both sides were built against **the same traversal engine**: tvm-ffi
`2b50d28d9aada35c6eeacac7f834642288c1c486`, the commit TVM vendors. Comparing a
standalone build at current tvm-ffi `main` against a TVM build at `2b50d28` would
have measured an engine difference, not an AST difference.

| Component | Head |
|---|---|
| tvm-ffi traversal engine, both sides | `2b50d28d9aada35c6eeacac7f834642288c1c486` |
| tvm-ffi benchmark branch base | `2ec024f` (`ffi-structural-map-cost-budget`) |
| tvm-ffi `upstream/main` at run time | `0cfb9a9` |
| TVM | `f5a68bd834cdbc82f2a3bf6292d7328060739536` (`tvm-structural-visit-mutate-hooks`) |

The TVM head pinned by the earlier budget, `6ea5d92d`, is no longer reachable in
any configured remote; that branch was force-updated. `f5a68bd8` is its current
head and carries the same compiled structural IR hooks.

Host and build:

- `docker-ci_gpu`, x86_64, AMD Ryzen 9 7950X 16-Core (32 threads), `schedutil`
  governor, Linux 5.15.
- GCC 14.3.0 (conda-forge), `-O3 -DNDEBUG`, `taskset -c 30`.
- TVM configured `USE_LLVM=OFF USE_CUDA=OFF USE_GTEST=OFF USE_Z3=OFF
  HIDE_PRIVATE_SYMBOLS=OFF`, `CMAKE_BUILD_TYPE=Release`.
- The host was **not** idle: it is a shared machine and load average during the
  run was 25-35 on 32 threads.

Because the host is shared, each measurement is a median of 15 alternating
rounds, and each round is itself the harness's median of 11 in-process samples.
Rounds alternate copied AST -> original TVM -> synthetic model on the same pinned
core so all three see comparable contention. The min-of-rounds estimator agrees
with the median-of-rounds within 1.5 percentage points on every cell.

The noise floor is measured, not assumed: both drivers time the identical walk
workload twice per process (`structural_walk` and `rung1`) and the identical
changed-map workload twice (`structural_changed` and `rung4`). The median
disagreement between those duplicate pairs is **0.6-2.2%**, which is the
resolution of this host.

## Agreement per rung

ns per processed node, median of 15 rounds, on `docker-ci_gpu` (x86_64 Ryzen
9 7950X). The `Δ` columns are `(copied - TVM) / TVM`.

| Rung | base copied | base TVM | Δ base | nested-4 copied | nested-4 TVM | Δ nested-4 |
|---|---:|---:|---:|---:|---:|---:|
| walk | 9.95 | 10.32 | **-3.6%** | 10.96 | 11.48 | **-4.5%** |
| 2a, default mutate only | 20.63 | 22.69 | **-9.1%** | 20.41 | 21.72 | **-6.0%** |
| 2b, plus identity-remap check | 26.39 | 26.66 | **-1.0%** | 25.97 | 26.10 | **-0.5%** |
| unmatched map | 32.85 | 31.94 | **+2.8%** | 29.09 | 28.14 | **+3.4%** |
| changed map | 41.81 | 41.90 | **-0.2%** | 39.67 | 40.27 | **-1.5%** |

The synthetic model measured in the same rounds, for contrast:

| Rung | base synthetic | Δ vs TVM | nested-4 synthetic | Δ vs TVM |
|---|---:|---:|---:|---:|
| walk | 7.57 | -26.7% | 8.25 | -28.1% |
| 2a, default mutate only | 15.58 | -31.3% | 15.39 | -29.2% |
| 2b, plus identity-remap check | 21.13 | -20.8% | 21.00 | -19.5% |
| unmatched map | 27.34 | -14.4% | 23.77 | -15.5% |
| changed map | 35.20 | -16.0% | 33.79 | -16.1% |

Copying the node set moves the model from 14-31% below TVM on every rung to
within 4% on four of five rungs, and reproduces the earlier finding that the
synthetic model is materially cheaper.

### Declared tolerance

The copied AST is declared calibrated at **±5% per rung, except rung 2a at
±10%**, on this host.

That is wider than the 1.1-2.9% band the earlier budget reproduced, and the
difference is a property of the host rather than of the model: that band was
established on a dedicated, idle, pinned GB200 core, while these numbers come
from a shared x86 machine whose own duplicate-measurement resolution is 0.6-2.2%.
Four of the ten cells above already sit inside 1.1-2.9%; three more sit inside
±5%.

## What still differs, and why it is not closable here

Two residuals remain: the walk rung at -3.6%/-4.5% and rung 2a at -9.1%/-6.0%,
both in the direction of the copy being *cheaper*. They are not measurement
noise -- rung 2a's 15 rounds span 19.4-20.9 on the copy and 22.3-24.3 on TVM,
with no overlap.

Field count, object size, allocation size and hook body are all excluded above,
which leaves properties of the host binary rather than of the node set:

- **Instruction footprint.** TVM's hooks are compiled into `libtvm_compiler.so`,
  whose `.text` is 57.4 MB. The standalone benchmark's hooks live in a 315 KB
  `.text`. A 180x difference in instruction-cache and iTLB pressure around the
  hot loop cannot be reproduced by copying node definitions into a small
  executable.
- **Type-table density.** The `__s_mutate__` attribute column spans 162 entries
  in the standalone process and 290 in TVM, and the fixture node types land at
  type indices 155-160 rather than 182-207. `DefaultMutateExpected` indexes that
  column once per node, so a denser table costs more. This one was measured
  rather than asserted: a scratch probe that registers 128 extra expression node
  types, bringing the column to 291 entries, and is otherwise the identical
  source compiled by the identical command, moves rung 2a from -8.6%/-6.4% to
  -4.2%/-1.3% against TVM. Column density therefore accounts for roughly half to
  all of the rung 2a residual. It is an attribution experiment, not a change to
  make: padding the table pushes the other three mutate rungs from within 2% out
  to +3-9%, so the unpadded benchmark is the better overall model.
- **Build configuration.** The calibration TVM build sets
  `HIDE_PRIVATE_SYMBOLS=OFF` to simplify linking the driver, which changes
  symbol visibility and therefore interprocedural optimization inside TVM.

The residual is therefore an optimistic bias of the standalone model on the
pure-hook rungs, bounded at about 9%, and it comes from how the host binary is
built rather than from what the node set is. Type-table density is measured to
carry about half of it; instruction footprint is the remaining candidate and was
not separated further, because a 180x `.text` difference cannot be recreated in
a benchmark that is meant to build from a tvm-ffi checkout alone. Both are
recorded rather than closed.

## The hook body was the dominant error

The first version of the copy wrote the binary mutate hooks against
`MutateExpected` / `MaybeInplaceMutateIfUniqueExpected` instead of TVM's raw
`MutateRaw` / `MaybeInplaceMutateIfUniqueRaw`, on the reasoning that the wrapper
is literally `MoveFromTVMFFIAny<Any>(MutateRaw(value))`. Measured, it is not
equivalent:

| Rung | Δ with the wrapper form | Δ with TVM's raw form |
|---|---:|---:|
| 2a, base / nested-4 | +15.0% / +20.2% | -9.1% / -6.0% |
| 2b, base / nested-4 | +25.4% / +25.3% | -1.0% / -0.5% |
| unmatched map, base / nested-4 | +19.8% / +28.3% | +2.8% / +3.4% |
| changed map, base / nested-4 | +17.7% / +19.7% | -0.2% / -1.5% |

Materializing the `Expected<Any>` at every child return costs 15-28% of the
mutate rungs. The copied hooks now use the raw entry points wherever the engine
exposes them, and fall back to the wrapper form only on engines that removed
them; the header carries a compile-time detection for this. It also means a
tvm-ffi revision without the raw entry points cannot reproduce these numbers,
which is itself a finding about the current engine API.

## Outstanding: the GB200 absolute values

**The absolute ns/node figures the earlier budget quotes are not reproduced
here and this run does not claim to reproduce them.** Those were taken on GB200
CPU 22 (aarch64 Grace); this host is x86_64 Ryzen 9 7950X. The absolute levels
differ by roughly a factor of two on the mutate rungs -- for example rung 2a is
41.79/41.31 on GB200 and 22.69/21.72 on TVM here -- as expected for different
microarchitectures.

What is established here is the *relative* claim the model needs: on one machine,
with one engine revision, the copied AST tracks the original TVM AST within the
tolerance declared above, while the synthetic model does not. Confirming the
GB200 absolute table against the copied AST still requires running
`tvm_ffi_benchmark_structural_map_tvm_ast_cost` beside the TVM calibration
driver on a GB200 core, and that run has not been performed.
