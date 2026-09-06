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

# Structural traversal benchmark

One harness for the structural traversal APIs, on a long-lived branch that measurement
tasks **extend** rather than replace. Three earlier attempts each built their own harness,
fixtures, arm definitions and table format; the results were not comparable and two errors
reached written reports. Everything those reports had to restate — arms, fixtures, method,
provenance, and what makes a run invalid — is code here.

## Why this is a branch and not `main`

`real_tvm_bench.cc` links apache/tvm, which tvm-ffi cannot depend on, and it registers its own
structural hooks over the ones TVM installed from static init — which tvm-ffi's write-once
type-attribute guard would refuse. Neither belongs on `main`. The branch is rebased onto
`main` when the engine moves, and that rebase is when the harness is re-validated.

The guard is disabled by an off-by-default CMake option added on this branch,
`TVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE`, and the hook override is the only thing that needs
it.

## Layout

| file | what it owns |
| --- | --- |
| `timer.h` | the two timing loops and their constants, and nothing else |
| `bench_common.h` | the two floor arms, build-time provenance, printing |
| `mini_tir.h` | mini-TIR's node types, their hooks, and the traversal machinery the baseline arms measure |
| `tvm_hook_override.h` | every structural hook the benchmark dispatches into on real TVM types, and the installation that puts them over TVM's |
| `mini_tir_bench.cc` | mini-TIR fixtures, arms, `main` |
| `real_tvm_bench.cc` | real-TVM fixtures, arms, `main` |
| `build.sh` | builds each harness twice, with the `SeqStmt` in-place hook original and repaired, stamping provenance in |
| `report.py` | runs each binary N pinned processes, medians the process medians, renders the report |

The two hook headers are deliberately **not** factored against each other. Their bodies read
almost identically and are written out twice on purpose: a reader can follow either without
mentally instantiating a template, and a change to one cannot silently reshape the other.

The binary builds fixtures, runs arms, times them and prints timings. It collects no metadata
about itself. Node counts, occurrence counts and working sets are arithmetic the builder
already knows — it built the thing — so it declares them as constants next to the loop that
produces them. Rebuild counts are read off the hooks, which are local code here rather than
something buried in a TVM build. Nothing counts, records or asserts inside a timed loop.

## Why the hooks live here

A hook experiment is an edit to `tvm_hook_override.h` and a rebuild of one translation unit,
never a TVM branch and a full TVM build. That is the difference between answering a question
in a minute and in an afternoon, and it is what the `SeqStmt` in-place work was done on.

They are always installed — there is no dual mode. `walk_old` and `map_old` run on them too,
because `PostOrderVisit` and `Substitute` are themselves built on the structural engine, so
the old-versus-new comparison then differs only in the traversal API and not in the hooks
underneath it. **The consequence: no number this harness produces describes TVM's own
registered hook implementations.** The hooks here are a port of what TVM registers today, with
two deliberate divergences named at the top of `tvm_hook_override.h`.

## Arms

| arm | quantity | what it is |
| --- | --- | --- |
| `walk_floor` | walk | minimal hand-written vtable: type-attr lookup and hook dispatch, no callbacks |
| `walk_var` | walk | `StructuralWalk<kPostOrder>` with a `Var` link and an `Expr` catch-all link |
| `walk_never` | walk | the same shape whose first link can never match: prices link testing alone |
| `walk_functor` | walk | `IRApplyVisit` over `StmtExprVisitor`: the functor machinery, which does not go through structural hooks at all |
| `walk_old` | walk | `PostOrderVisit`, called directly |
| `map_floor` | map | minimal hand-written vtable mutator: the engine's lower bound |
| `map_never` | map | `StructuralMap<kPostOrder>` whose callback can never match |
| `map_identity` | map | `StructuralMap<Var>` returning the same `Var`: matches, rebuilds nothing |
| `map_replace` | map | `StructuralMap` performing the fixture's replacement: all `Var`s on split/fuse, a two-`Evaluate` swap on seq |
| `map_functor` | map | `IRSubstitute`'s shape over `StmtExprMutator`, minus the dtype check and buffer/attr handling TVM's own carries |
| `map_old` | map | `Substitute`, called directly. One fixture only. |

`*_functor` and `*_old` are both baselines, and they are different ones. `*_old` is the
pinned TVM's current implementation, which is engine-based; `*_functor` is the pre-structural
functor machinery that `main` still ships. Reading `*_old` as "before the engine" would be
wrong — it is already after.

**Every map arm runs in both ownership variants.** `retained` keeps the caller's handle alive,
so the root's refcount is at least two and the engine takes the copy-on-write path. `moved`
hands the sole reference over with `std::move`, so the engine takes the in-place path. These
are different workloads and a per-node optimization is worth different amounts in each. They
share a table, adjacent rows, because the comparison between them is a result in its own
right.

`map_replace` swaps its variables rather than replacing one-way, so repeating it in place is
stationary and the timed loop never drifts into a different workload. A fixture with a
pointer-shared subtree cannot be traversed in place repeatedly at all — the first in-place
rebuild un-shares the DAG — so those cases run over a pool of independent copies rebuilt
untimed between passes.

## Fixtures

| fixture | shape | `map_replace` replaces |
| --- | --- | --- |
| `split-fuse-shared` | `floordiv(o*16+i, 32)*32 + floormod(o*16+i, 32)` with one pointer-shared intermediate | every `Var` |
| `split-fuse-distinct` | the same with two structurally equal, pointer-distinct intermediates | every `Var` |
| `seq-L` | `SeqStmt` of `L` statements `Evaluate(o*(i+2) + inner)`, `L` in 16, 256, 16384 | a swap of two `Evaluate` nodes |

The `seq` fixtures are a **sparse update**, and they match `Evaluate` rather than `Var`. That is
deliberate: `Evaluate` is not a free variable, so no remap table is involved, and each element is
exactly one `Evaluate`, so "swap k pairs" is a controlled input. A `Var` callback would have
propagated through the remap to every element and the sparse update would silently have been a
dense one. The swap is an involution, so the fixture is stationary under repetition and needs no
pool. **The `map_replace` column therefore means different things on `seq` and on split/fuse; do
not read across the two.**

`build.sh` also emits `*_seqorig` binaries with the `SeqStmt` in-place hook in #20275's shape
rather than the repaired one, and `real_tvm_bench` runs a change-density sweep on `seq-256`,
swapping 1 to 64 pairs.

The three `seq` lengths are one per cache level: 16 inside a 32 KiB L1d, 256 inside a 1 MiB
L2, 16384 inside a 32 MiB L3. `report.py` names the level from that stated geometry.

Both harnesses build the same shapes, so their rows are structural counterparts: the same
node counts, the same occurrence counts, the same rebuild counts. Only the node sizes differ.

## Checks

Untimed, before any timing — nothing the harness verifies runs inside a timed loop.

- **In-place actually fires.** Every `moved` number rests on it, so the harness captures the
  input's raw node pointer (never an `ObjectRef`, which would itself be a reference and suppress
  what is being tested), runs the arm, and requires the moved root to be the same object and the
  retained root not to be.
- **The swap does what it claims.** Exactly the targeted elements differ, every other is
  pointer-identical to the input, and a second application restores the original pointers. This
  caught a density point whose swap pairs collided.
- **Hook coverage.** Every type a fixture dispatches on must have a harness hook, so a new
  fixture that reaches a new type names it instead of silently falling through to TVM's.
- **Port fidelity.** `./port_check.sh /path/to/tvm [sha]` re-extracts all fifteen ported hooks
  from the recorded revision and diffs them against `tvm_hook_override.h`.

## Method

Fixed in `timer.h`: one untimed warm-up pass, nine timed samples per arm per process with a
fixed per-fixture repeat count, the process value is the median of its nine samples, and the
reported value is the median of five pinned process medians. Repeat counts are plain constants
chosen so a sample lands in the milliseconds, not a calibration loop — #367 ran a fixed 20,000
traversals and this harness reproduces its numbers.

Nothing type-erased sits in the timed path. That is a correctness point, not a style one: the
arms are inlined into the loop through templates, and removing the `std::function` indirection
an earlier version had moved `walk` on split/fuse from 13.2 to 7.8 ns/node.

`ns/node` divides by the fixture's unique-node count, **the same divisor for every arm in a
row**. For the sparse-update fixtures `ns/node` amortizes one useful change over the whole
traversal, so those also get an absolute-cost table, which is the number a caller feels.

## Building and running

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE=ON
cmake --build build -j
cmake -S "$TVM" -B "$TVM/build" -DCMAKE_BUILD_TYPE=Release \
      -DTVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE=ON
cmake --build "$TVM/build" -j --target tvm_compiler
benchmarks/cpp/structural/build.sh --tvm "$TVM"
benchmarks/cpp/structural/report.py \
    --binary build_bench/real_tvm_bench --binary build_bench/mini_tir_bench --cpu 0
```

Point the TVM checkout's `3rdparty/tvm-ffi` submodule at this branch; TVM builds tvm-ffi
through `add_subdirectory`, so the option reaches it. Nothing in the TVM checkout is modified
— the harness lives here and links TVM from outside.

`build.sh` also emits `*_seqorig` binaries, the same sources with the `SeqStmt` in-place hook
in TVM's current shape rather than the repaired one.

## Extending it

Adding an arm means writing one function and adding a line to `kWalkArms` or `kMapArms` in the
harness's `.cc`. Adding a fixture means a builder plus the constants it declares about what it
built. Adding a hook experiment means editing `tvm_hook_override.h`. None of it requires
touching the other harness, the timer, or the report format.
