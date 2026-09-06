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

One harness for the structural traversal APIs, on a long-lived branch that measurement tasks
**extend** rather than replace. Three earlier attempts each built their own harness, fixtures,
arm definitions and table format; the results were not comparable and two errors reached
written reports. Everything those reports had to restate — arms, fixtures, method, provenance,
and what makes a run invalid — is code here.

This file is the whole instruction set. If you are opening a benchmark task and were not part
of the earlier work, read it top to bottom once; it is shorter than re-deriving any of it.

## Contents

1. [Why this is a branch](#why-this-is-a-branch-and-not-main)
2. [Running it](#running-it)
3. [The arm vocabulary](#the-arm-vocabulary-which-arm-answers-which-question)
4. [The fixtures](#the-fixtures)
5. [Method, and what it protects against](#method-and-what-it-protects-against)
6. [Reading the tables](#reading-the-tables)
7. [The port](#the-port)
8. [Adding an arm or a fixture](#adding-an-arm-or-a-fixture)
9. [Layout](#layout)

## Why this is a branch and not `main`

`real_tvm_bench.cc` links apache/tvm, which tvm-ffi cannot depend on, and it registers its own
structural hooks over the ones TVM installed from static init — which tvm-ffi's write-once
type-attribute guard would refuse. Neither belongs on `main`. The branch is rebased onto `main`
when the engine moves, and that rebase is when the harness is re-validated against current code.

The guard is disabled by an off-by-default CMake option added on this branch,
`TVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE`, and the hook override is the only thing that needs it.

**Two harnesses, everything else shared.** `mini-TIR` builds its fixtures from tvm-ffi's own
types and registers its own hooks: no override, no patch, runs in tvm-ffi's tree alone. `real
TVM` links TVM and overrides the real hooks from `main()`, which runs after every
`TVM_FFI_STATIC_INIT_BLOCK`, and needs the guard patch. They differ only in the types they build
from and how hooks are installed; fixtures, arms, method, checks and report format are common.
Their rows are structural counterparts — same node counts, same occurrence counts, same rebuild
counts — so disagreement between them is a signal rather than noise.

## Running it

```sh
# tvm-ffi, with the guard disabled
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE=ON
cmake --build build -j

# TVM, with its 3rdparty/tvm-ffi submodule checked out to this branch.  TVM builds tvm-ffi
# through add_subdirectory, so the option reaches it from here.
cmake -S "$TVM" -B "$TVM/build" -DCMAKE_BUILD_TYPE=Release \
      -DTVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE=ON
cmake --build "$TVM/build" -j --target tvm_compiler

benchmarks/cpp/structural/build.sh --tvm "$TVM"
benchmarks/cpp/structural/report.py \
    --binary build_bench/real_tvm_bench --binary build_bench/mini_tir_bench \
    --cpu 0 --runs 5 --out tables.md
```

Nothing in the TVM checkout is modified. The harness lives here and links TVM from outside.

`build.sh` emits four binaries: `mini_tir_bench`, `real_tvm_bench`, and a `*_seqorig` of each
with the `SeqStmt` in-place hook in apache/tvm#20275's shipped shape rather than the harness's
variant. `report.py` takes any number of `--binary` arguments and renders one set of tables per
binary.

**Wall time.** A single `real_tvm_bench` process is roughly four minutes, `mini_tir_bench`
roughly three; `--runs 5` on both is about 35 minutes. The `seq-16384` fixture and the density
sweep are most of it. Pin with `--cpu`; leave the machine otherwise idle.

**Two-state runs** — building two tvm-ffi refs and comparing them — are not implemented yet.
See [Reading the tables](#reading-the-tables) for why absolutes from two separately compiled
binaries may not be compared directly, which is the problem that work exists to solve.

## The arm vocabulary: which arm answers which question

An arm is one traversal implementation run over one fixture. The set is fixed here so a
measurement task **selects** arms rather than defining them, which is what makes two reports
comparable.

### Walk arms

| arm | what it is | the question it answers |
| --- | --- | --- |
| `walk_floor` | hand-written minimal visitor: type-attr column lookup and hook dispatch, no callbacks | **the control.** The engine's lower bound. Nothing a callback-level change does should move it. |
| `walk_var` | `StructuralWalk<kPostOrder>` with a `Var` link and an `Expr` catch-all | the arm under study: what a real walk costs |
| `walk_never` | the same shape whose first link can never match | prices **link testing alone** — the gap to `floor` is the callback machinery with no callback firing |
| `walk_functor` | `IRApplyVisit` over `StmtExprVisitor`; does not go through structural hooks at all | baseline: **the pre-structural functor machinery** `main` still ships |
| `walk_old` | `PostOrderVisit`, called directly | baseline: **what the pinned TVM ships today** |

### Map arms

Map arms are split by the **operation** they perform, because two operations on the same graph
are not each other's baselines. This was a real error in an earlier report: the seq `replace`
column swapped `Evaluate` nodes while the `functor` and `old` columns beside it substituted
`Var`s, with nothing saying so.

**Expr-level — `Var` substitution.** Runs on every fixture. `Substitute` and `FunctorSubstitute`
both hook `VisitExpr_(const VarNode*)`, so these are the arms they are baselines for.

| arm | what it is | the question it answers |
| --- | --- | --- |
| `map_floor` | hand-written minimal mutator over the two mutate columns | **the control.** If it moves between two builds, the builds differ in something beyond what is under study and nothing else in the table can be trusted. |
| `map_never` | `StructuralMap<kPostOrder>` whose callback can never match | link testing alone, nothing rebuilt |
| `map_identity_var` | `StructuralMap` matching `Var`, returning the same `Var` | **matches but rebuilds nothing.** This is where a no-change optimization shows. |
| `map_replace_var` | `StructuralMap` substituting every `Var` through the remap | **the rebuild is real.** This is where rebuild cost shows. |
| `map_functor` | `IRSubstitute`'s shape over `StmtExprMutator` | baseline: the functor-era mutator |
| `map_old` | `Substitute`, called directly | baseline: the shipping API |

**Stmt-level — element swap.** Seq fixtures only; `split-fuse` is an `Expr` tree with no `Stmt`
nodes in it, so these arms have nothing to match there.

| arm | what it is | the question it answers |
| --- | --- | --- |
| `map_identity_stmt` | `StructuralMap` matching `Stmt`, returning the same `Stmt` | the Stmt-level ladder's no-rebuild rung |
| `map_replace_stmt` | swaps two whole `Evaluate` nodes | exercises the `SeqStmt` hook's **element in-place** path and the sparse-update cost |
| `map_replace_field` | swaps an `IntImm` *inside* two elements | the only arm that exercises **element-level in-place mutation**: replacing whole elements leaves nothing within an element to mutate |
| `map_splice` | maps an element to a nested `SeqStmt` | the only arm that reaches the hook's **splice** path |

**There is no functor baseline for the Stmt-level arms**, so no delta against `functor` or `old`
is printed for them. A cell that cannot be filled says `n/a` and carries a footnote; a blank
cell would read as missing data.

`map_functor` is a **floor for a functor-era mutator, not a reproduction of `Substitute`**. TVM's
`IRSubstitute` carries a dtype `ICHECK` per substitution plus buffer and attribute handling the
harness's `FunctorSubstitute` omits, which is why `map_functor` and `map_old` differ by around
20% while `walk_functor` and `walk_old` agree within 3%.

`*_functor` and `*_old` are **different baselines**, and a reader should not average them.
`*_old` is the pinned TVM's current implementation, which is already engine-based; `*_functor`
is the pre-structural functor machinery. Reading `*_old` as "before the engine" would be wrong.
Against `old` asks what migrating costs a caller of the shipping API; against `functor` asks
what the migration to the structural engine cost.

### The ownership axis

**Every map arm runs in both ownership variants.** `retained` keeps the caller's handle alive,
so the root's refcount is at least two and the engine takes the copy-on-write path. `moved`
hands the sole reference over with `std::move`, so the engine takes the in-place path. These are
different workloads and a per-node optimization is worth different amounts in each; reporting
one hides how much of a result depends on ownership, which is the number a caller deciding
whether to `std::move` actually needs. They share a table, adjacent rows, because the comparison
between them is a result in its own right. Walk arms have no ownership axis and appear once.

Replacement arms **swap** rather than replace one-way, so repeating one in place is stationary
and the timed loop never drifts into a different workload. A fixture with a pointer-shared
subtree cannot be traversed in place repeatedly at all — the first in-place rebuild un-shares
the DAG — so those cases run over a pool of independent copies rebuilt untimed between passes.

## The fixtures

| fixture | shape | scales? |
| --- | --- | --- |
| `split-fuse-shared` | `floordiv(o*16+i, 32)*32 + floormod(o*16+i, 32)`, one pointer-shared intermediate | no, N=12 |
| `split-fuse-distinct` | the same with two structurally equal, pointer-distinct intermediates | no, N=15 |
| `seq-L` | `SeqStmt` of `L` statements `Evaluate(o*(i+2) + inner)`, `L` in 16, 256, 16384 | **yes** |

**split/fuse** is a small `Expr` tree. It carries the Expr-level comparison against both
baselines, and the shared variant is the only fixture that exercises DAG un-sharing. It does not
scale, so it says nothing about working set.

**seq** is the one that scales, and the three lengths are one per cache level: 16 inside a
32 KiB L1d, 256 inside a 1 MiB L2, 16384 inside a 32 MiB L3. `report.py` names the level from
that stated geometry. Any question about how a result behaves as the tree stops fitting in cache
is a seq question.

**The seq fixtures carry two operations and they are not interchangeable.** The Stmt-level arm
swaps two `Evaluate` nodes: `Evaluate` is not a free variable, so no remap is involved, each
element is exactly one `Evaluate`, and "swap k pairs" is a controlled sparse input. The
Expr-level arm substitutes `Var`s, which propagates through the remap to every element and is a
dense rebuild. Both are legitimate; they are separate columns in separate tables because a delta
between them would be meaningless.

`N` and `occurrences` include the `Array` inside each `SeqStmt`: apache/tvm#20275's
`SeqStmtVisit` visits `self->seq` as a value rather than iterating its elements, so the
container is itself a visited node.

Fixture counts — unique nodes, occurrences, working set, rebuild counts — are **declared by the
builder**, not measured back by the binary. The builder built the thing and knows them; making
the binary re-derive them would put counting inside the measurement.

## Method, and what it protects against

Fixed in `timer.h` so no report restates it: one untimed warm-up pass, nine timed samples per
arm per process with a fixed per-fixture repeat count, the process value is the median of its
nine samples, and the reported value is the **median of five pinned process medians**. Repeat
counts are plain constants chosen so a sample lands in the milliseconds, not a calibration loop.

**The timed boundary is explicit.** Each measurement is a named `setup` / `batch` / `teardown`
triple, and `batch` is the only code between the two clock reads.

- *Outside, before:* the fixture is built and the result buffer is reserved for the whole batch,
  so no growth reallocation can land inside the clock.
- *Inside:* only the arm, `repeats` times, and whatever running it entails — **including the
  freeing caused by consuming the input under `moved`**. In-place mutation is partly about
  reusing storage the input owned, and hoisting that out would measure something that does not
  exist.
- *Outside, after:* the buffer is cleared, which is where every retained output is destroyed.
  That teardown is the caller's, not the mutation's.

This is not a detail. When destructor time was inside the clock, every arm paid to walk and free
what it had just built, which inflated precisely the arms that allocate most — `identity →
replace` was construction *plus* teardown. Moving it out took the rebuilding arms down 5-9% and
left `floor` and `never`, which build nothing, unmoved. **Do not regress it.**

**Nothing type-erased or instrumented sits in the timed path.** Arms are inlined into the loop
through templates. Removing a `std::function` indirection an earlier version had moved `walk` on
split/fuse from 13.2 to 7.8 ns/node — 40% of a quantity the report treats as near-floor.
Counting, recording and checking all happen in separate untimed passes.

**Absolutes from two separately compiled binaries are not comparable; deltas within one are.**
Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under
study. Within one binary, arm-to-arm deltas are sound. Across builds, only an interleaved
two-state run — A, B, A, B — makes a delta trustworthy, because thermal state and drift then hit
both equally. A number quoted against a previous report on a different day is the weakest form
of comparison there is, and the two-state work exists to replace it.

### Checks that fail the run

Untimed, before any timing. The point is that a wrong measurement stops rather than reaching a
table.

- **In-place actually fires.** Every `moved` number rests on it, so the harness captures the
  input's raw node pointer — never an `ObjectRef`, which would itself be a reference and suppress
  what is being tested — runs the arm, and requires the moved root to be the same object and the
  retained root not to be.
- **The swap does what it claims.** Exactly the targeted elements differ, every other is
  pointer-identical to the input, and a second application restores the original pointers. This
  caught a density point whose swap pairs collided.
- **Splice matches the reference.** The in-place hook must produce exactly what the
  non-in-place `MutateSeqStmtRaw` produces — it is an optimization of the same semantics. The
  two are differential-tested against each other across twelve cases: grow at first/last/middle,
  several in one pass, two adjacent, length-preserving, shrink at first/last/several, a splice
  that empties the sequence, and a wide splice that overflows capacity. Each case also asserts
  the reference's own output length, so agreement cannot be vacuous.
- **Hook coverage.** Every type a fixture dispatches on must have a harness hook, so a new
  fixture that reaches a new type names it instead of silently falling through to TVM's.
- **Port fidelity.** See [The port](#the-port).

**Splicing can only ever grow, through the public API.** `SeqStmt`'s constructor rejects both
length 0 and length 1, so a nested `SeqStmt` always carries at least two statements and the
shrink and length-unchanged directions are unreachable from outside. They are still tested, built
by bypassing the constructor, so the hook's behaviour on them is pinned rather than argued.

## Reading the tables

`ns/node` divides by the fixture's unique-node count, **the same divisor for every arm in a
row**, so a row is internally comparable and a column is comparable down the fixture list. For
the sparse-update fixtures `ns/node` amortizes one useful change over the whole traversal, so
those also get an absolute-cost table, which is the number a caller feels.

**The three arms that read as a set:**

| arm | expected | what a violation means |
| --- | --- | --- |
| `floor` | **flat** across any callback-level or protocol-level change | the two builds differ in something beyond what is under study; nothing else in the table can be trusted |
| `identity` | **improves** when a no-change optimization lands | it did not reach the path it was meant to |
| `replace` | **roughly neutral** — the rebuild is real work either way | a change aimed at unchanged nodes moved rebuilding cost, which needs explaining |

Any other pattern is a finding, not a footnote.

**Two standing caveats a reader needs to interpret any table here:**

1. **`rebuilt moved` counts are a lower bound.** Rebuild counts are pointer-identity set
   differences between input and output. Under `retained` the input is held alive, so a freed
   address cannot be reused and the counts are exact. Under `moved` the input graph is consumed,
   so a replacement can be allocated at the address a just-freed node occupied and the
   comparison undercounts. Timings are unaffected.
2. **`map_functor` is a floor, not a reproduction.** See [the arm
   table](#the-arm-vocabulary-which-arm-answers-which-question): `IRSubstitute` carries a dtype
   `ICHECK` per substitution and buffer/attr handling `FunctorSubstitute` omits.

**No number this harness produces describes TVM's own registered hook bodies.** The hooks are a
port, always installed over TVM's, with the deviations named at the top of
`tvm_hook_override.h`. `walk_old` and `map_old` run on them too, because `PostOrderVisit` and
`Substitute` are themselves built on the structural engine, so the old-versus-new comparison
differs only in the traversal API and not in the hooks underneath it.

## The port

`tvm_hook_override.h` holds fifteen structural hooks **copied verbatim** from apache/tvm#20275
— same names, same signatures, same order, same internal structure, grouped by the TVM file each
came from. Nothing is reordered, renamed or tidied, because a change prototyped here has to lift
back into apache/tvm as a patch. The intended differences are marked `HARNESS DEVIATION` in the
header.

A hook experiment is then an edit to one header and a rebuild of one translation unit, never a
TVM branch and a full TVM build. That is the difference between answering a question in a minute
and in an afternoon.

```sh
./port_check.sh /path/to/tvm            # against the sha recorded in the header
./port_check.sh /path/to/tvm <sha>      # against another revision, e.g. a rebased PR head
```

It re-extracts each function from the TVM revision and diffs it against the copy in the header.
**Run it before trusting any measurement**, and after every rebase of the PR.

**Re-porting when the PR moves:**

1. Fetch the new head and run `port_check.sh` against it. Every function it reports as `DRIFT`
   needs re-porting; `MISSING IN TVM` means the PR renamed or removed it and the `FUNCS` list at
   the top of `port_check.sh` needs updating.
2. Replace the drifted bodies verbatim. Do not hand-merge — copy.
3. Update the recorded sha in the header's `PORTED FROM` block. `port_check.sh` reads it from
   there, so that one line is the provenance.
4. If the PR bumped `3rdparty/tvm-ffi`, rebase this branch onto that ref so both harnesses build
   against the same engine.
5. Rebuild both harnesses and re-run. Re-porting is not complete until the numbers are refreshed:
   a hook change that alters a hot path invalidates the previous tables.

## Adding an arm or a fixture

The bar is that a new measurement stays comparable with existing reports.

**A new arm** is one function plus a line in `kWalkArms`, `kMapArms` or `kStmtMapArms`. Before
adding one, answer in a sentence *which question it answers that no existing arm does* — and put
that sentence in the arm table above. An arm whose question is already covered makes the tables
wider without making them say more. If it performs a different operation from the arms beside
it, it belongs in a different table, not a new column.

**A new fixture** is a builder plus the constants it declares about what it built: unique node
count, occurrence count, working-set bytes, and rebuild counts per ownership variant. Add it to
both harnesses with the same shape, so the rows stay structural counterparts. If it reaches a
node type no existing fixture does, the hook-coverage check will tell you.

**A hook experiment** is an edit to `tvm_hook_override.h` behind a build flag, so before and
after are measured on the same binary otherwise unchanged.

None of this requires touching the other harness, the timer, or the report format. If a change
would require touching the report format, that is a signal the arm set or the fixture set has
drifted from what the format was built to say — fix that first.

## Layout

| file | what it owns |
| --- | --- |
| `timer.h` | the two timing loops and their constants, and nothing else |
| `bench_common.h` | the two floor arms, build-time provenance, printing |
| `mini_tir.h` | mini-TIR's node types, their hooks, and the traversal machinery the baseline arms measure |
| `tvm_hook_override.h` | every structural hook the benchmark dispatches into on real TVM types, and the installation that puts them over TVM's |
| `mini_tir_bench.cc` | mini-TIR fixtures, arms, `main` |
| `real_tvm_bench.cc` | real-TVM fixtures, arms, checks, `main` |
| `build.sh` | builds each harness twice, `SeqStmt` in-place hook shipped and harness-variant, stamping provenance in |
| `port_check.sh` | diffs the ported hooks against the TVM revision they came from |
| `report.py` | runs each binary N pinned processes, medians the process medians, renders the tables |

The two hook headers are deliberately **not** factored against each other. Their bodies read
almost identically and are written out twice on purpose: a reader can follow either without
mentally instantiating a template, and a change to one cannot silently reshape the other.

The binary builds fixtures, runs arms, times them and prints timings. It collects no metadata
about itself and renders no tables — `report.py` owns the format. Provenance is stamped in at
build time by `build.sh` as compile-time constants, so **a report cannot omit what the tool
prints**.
