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

`real_tvm_bench.cc` links apache/tvm, which tvm-ffi cannot depend on, and the harness wraps
hooks that TVM already registered from static init, which needs tvm-ffi's write-once
type-attribute guard disabled. Neither belongs on `main`. The branch is rebased onto `main`
when the engine moves, and that rebase is when the harness is re-validated against current
code.

The guard is disabled by an off-by-default CMake option added on this branch,
`TVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE`. A run's provenance block states whether it was on.

## Layout

| file | what it owns |
| --- | --- |
| `bench_common.h` | everything except node types: the arm vocabulary, the method, working-set and cache accounting, hook-dispatch instrumentation, the assertions, the per-fixture driver, and the emission format |
| `mini_tir_bench.cc` | a mirror of TVM's TIR node shape built from tvm-ffi types alone, with its own hooks; no TVM checkout needed |
| `real_tvm_bench.cc` | the same fixtures on TVM's real node types, measuring the hooks TVM ships |
| `build.sh` | builds both and stamps provenance (harness commit, engine sha, TVM sha, flags) into the binaries |
| `report.py` | runs each binary N pinned processes, medians the process medians, renders the report |

The two harnesses answer different questions and must not be read as contradicting each
other. **Real TVM** asks whether migrating costs anything against what ships today.
**mini-TIR** asks how far the engine sits from a lean hand-written implementation; its
`map_old` model does no variable remapping and does not memoize, which is why its map column
can read as a regression where the real-TVM column reads as an improvement.

## Arms

Defined once, in `bench_common.h`. A measurement selects from this list; it does not
redefine it.

| arm | quantity | what it is |
| --- | --- | --- |
| `walk_floor` | walk | minimal hand-written vtable: type-attr lookup and hook dispatch, no callbacks |
| `walk` | walk | `StructuralWalk<kPostOrder>` with a `Var` link and an `Expr` catch-all link |
| `walk_never` | walk | the same shape whose first link can never match: prices link testing alone |
| `walk_old` | walk | the shipping post-order traversal (`PostOrderVisit`; a model in mini-TIR) |
| `map_floor` | map | minimal hand-written vtable mutator: the engine's lower bound |
| `map_never` | map | `StructuralMap<kPostOrder>` whose callback can never match |
| `map_identity` | map | `StructuralMap<Var>` returning the same `Var`: matches, rebuilds nothing |
| `map_replace` | map | `StructuralMap<Var>` replacing one variable |
| `map_old` | map | the shipping substitution (`Substitute`; a model in mini-TIR) |

**Every map arm runs in both ownership variants.** `retained` keeps the caller's handle
alive, so the root's refcount is at least two and the engine takes the copy-on-write path.
`moved` hands the sole reference over with `std::move`, so the engine takes the in-place
path. These are different workloads and a per-node optimization is worth different amounts
in each; reporting one hides how much of a result depends on ownership, which is the number
a caller deciding whether to `std::move` actually needs. Walk arms have no ownership axis.

`map_replace` swaps its two variables rather than replacing one-way, so repeating it in
place is stationary and the timed loop never drifts into a different workload. A fixture
with a pointer-shared subtree cannot be traversed in place repeatedly at all — the first
in-place rebuild un-shares the DAG — so those cases use a pool of independent copies rebuilt
untimed between timed batches.

## Fixtures

| fixture | shape |
| --- | --- |
| `split-fuse-shared` | `floordiv(o*16+i, 32)*32 + floormod(o*16+i, 32)` with one pointer-shared intermediate |
| `split-fuse-distinct` | the same expression with two structurally equal, pointer-distinct intermediates |
| `seq-L` | `SeqStmt` of `L` `Evaluate(o*(i+2) + i)` statements, `L` in 16, 64, 256, 1024, 4096, 16384 |

Both harnesses build the same shapes, so their rows are structural counterparts: the same
unique-node counts, the same occurrence counts, the same declared rebuild counts. Only the
node sizes differ, and the harness measures those instead of assuming them.

The `seq-L` sweep exists to make cache residency measured rather than asserted. Earlier
reports carried "this fixture is L1-resident, so absolute `ns/node` are best-case rates" as
prose, and then attached it to a fixture it no longer described. The harness computes each
fixture's working set — unique nodes times node size — and names the cache level it fits in
on the machine that ran it. The sweep deliberately spans inside L1d, past L1d, and past L2.

## Assertions that fail the run

A wrong measurement stops rather than reaching a table. Any failure exits the process
non-zero before anything is timed, and `report.py` refuses to render.

- **Fixture composition.** Unique node count and total working set against the fixture's
  declared expectations; every node's size must be known to the harness.
- **Rebuild and identity counts,** per ownership variant: what `map_replace` rebuilds
  retained, what it rebuilds moved, that `map_identity` and `map_never` return the input
  graph unchanged, and that `map_old` rebuilds exactly what `map_replace` does.
- **Dispatch-count equality between arms that claim the same traversal.** A "dispatch" is
  one entry into a registered structural hook, counted by wrapping every hook in the
  type-attribute columns for an untimed pass and restoring them before anything is timed.
  Every walk arm and every never-matching map arm must enter one hook per node occurrence; a
  matching map arm trades a hook for a callback on the first occurrence of each matched
  identity and is served from the identity remap afterwards, so its count must equal
  `occurrences - remap hits`, a declared fixture property.
- **Walk faster than map on the same fixture.** A walk returns a status word per node; a map
  returns an owned value, touches a refcount and has its parent run `same_as`, so a map can
  never be cheaper than the walk it dispatches identically to. A prior report published a
  table where it was, and the contradiction was caught in review rather than by the tool.

## Method

Fixed in `bench_common.h`, not restated per report: one untimed warm-up batch, nine timed
samples per arm per process, the batch sized so its two clock reads are noise against it and
the repeat count calibrated to about 30 ms per sample, the process value is the median of
its nine samples, and the reported value is the median of five pinned process medians.

`ns/node` divides by the fixture's unique-node count, **the same divisor for every arm in a
row** — never by an arm's own visit count.

## Building and running

The tvm-ffi build must be configured with the guard option, because the dispatch-count
assertions wrap registered hooks:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE=ON
cmake --build build -j
benchmarks/cpp/structural/build.sh
taskset -c 0 build_bench/mini_tir_bench
```

For the real-TVM harness, point a TVM checkout's `3rdparty/tvm-ffi` submodule at this branch
and configure TVM with the same option — TVM builds tvm-ffi through
`add_subdirectory(3rdparty/tvm-ffi)`, so the option reaches it. Nothing in the TVM checkout
is modified; the harness lives here and links TVM from outside.

```sh
cmake -S "$TVM" -B "$TVM/build" -DCMAKE_BUILD_TYPE=Release \
      -DTVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE=ON
cmake --build "$TVM/build" -j --target tvm_compiler
benchmarks/cpp/structural/build.sh --tvm "$TVM"
benchmarks/cpp/structural/report.py \
    --binary build_bench/mini_tir_bench --binary build_bench/real_tvm_bench --cpu 0
```

## Extending it

A new measurement adds a fixture or an arm here and reports the tables `report.py` prints.
It does not restate arms, format, method or provenance in its own task record. Adding an
arm means adding an `ArmId` and its `ArmSpec` in `bench_common.h` and binding it in both
harnesses' `RunWalkArm` / `RunMapArm`; adding a fixture means a `build` function and its
declared expectations, in both harnesses so the rows stay counterparts.
