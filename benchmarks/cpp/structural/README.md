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
3. [Two-state runs](#two-state-runs)
4. [The arm vocabulary](#the-arm-vocabulary-which-arm-answers-which-question)
5. [The fixtures](#the-fixtures)
6. [Method, and what it protects against](#method-and-what-it-protects-against)
7. [Reading the tables](#reading-the-tables)
8. [The port](#the-port)
9. [Adding an arm or a fixture](#adding-an-arm-or-a-fixture)
10. [Layout](#layout)

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
from and how hooks are installed; arms, method, checks and report format are common. Where a
fixture exists in both, the rows are structural counterparts — same node counts, same
occurrence counts, same rebuild counts — so disagreement between them is a signal rather than
noise. `call-split-fuse` is real-TVM only: mini-TIR has no `Call` node, and the guards that
fixture exists to reach are `Call`'s.

**Two-state runs are real-TVM only**, for the same reason in reverse: mini-TIR's hooks are the
harness's own code rather than a port, so there is no per-state hook file for them.

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

`build.sh` emits `mini_tir_bench` and `real_tvm_bench`. `report.py` takes any number of
`--binary` arguments and renders one set of tables per binary.

**Wall time.** One `real_tvm_bench` process is about twenty seconds and `--runs 5` a couple of
minutes. Pin with `--cpu`; leave the machine otherwise idle.

## Two-state runs

Comparing two tvm-ffi refs. This is the only sound way to say that an engine change moved a
number: absolutes from two separately compiled binaries are not comparable, and this harness
has a worked example of a `map_floor` difference of -18% to -30% between two builds of
identical source.

```sh
benchmarks/cpp/structural/build.sh --tvm "$TVM" \
    --state 'A=62df2f5,hooks=tvm_hook_override_pre753.h,shim=state_shims/pre753_visit_return_none.h' \
    --state 'B=897ece6'

benchmarks/cpp/structural/report.py --two-state --cpu 0 --runs 5 \
    --state 'A=62df2f5:build_bench/real_tvm_bench_A' \
    --state 'B=897ece6:build_bench/real_tvm_bench_B' \
    --differs 'tvm-ffi #747 #749 #750 #751 #753 #756' --out compare.md
```

Four things make the comparison mean something, and each is a rule rather than a nicety.

**One hook file per state, each written against that state's own tvm-ffi API.** The harness
hooks call into the engine, and the engine's surface differs between states — entry points,
return types, macros available. `hooks=` selects a whole file. There is no conditional
compilation, no template parameter selecting an API, and no shared file with `#if`s: a hook
file that compiled against both states by accommodating both would measure a hybrid nobody
ships, which is what made an earlier bespoke benchmark useless. Duplication between the two
files is correct and expected — the same principle the harness already applies between
`tvm_hook_override.h` and `mini_tir.h`. A state file declares an `API ADAPTATION` block and
`port_check.sh --header` checks it; see [The port](#the-port).

**Interleaved processes, A/B/A/B.** Not all of A and then all of B. Thermal state, allocator
luck and drift then hit both states equally instead of landing on whichever ran second.
`report.py --two-state` does this; nothing else is a two-state run.

**apache/tvm held fixed.** Both states build the same TVM revision — the one `--tvm` points
at — so the only difference is the engine. `build.sh` gives each state its own TVM worktree
with `3rdparty/tvm-ffi` pointed at that state's engine, which is what makes one TVM revision
buildable against two engines at once.

**Build shims, where a state's engine is missing something TVM's own sources use.** `shim=`
force-includes a header into apache/tvm's translation units only, never the benchmark's. It
exists because the harness ports twenty-three of TVM's hooks and TVM has many more; the rest
still have to compile. A shim must be semantically identical to what the state's engine
already does, must be off every arm's dispatch path, and must say in its own header why both
are true. The empty shim `state_shims/none.h` is force-included for a state that needs
nothing, so both states are compiled with the same flag shape: a flag present on one side and
absent on the other is itself a build difference.

**What the report may claim.** Both absolutes and the delta appear in each cell, as
`30.81 —> 24.16 (-21.6%)`, and only the delta is claimed. `floor` is the control: it dispatches
no callbacks, so an engine or protocol change should not touch it, and if it moves the two
builds differ in something beyond what is under study and nothing else in that row can be
trusted.

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

**Each fixture family carries the one operation it is shaped for, and its arms are the arms
that operation has baselines for.** Two operations on the same graph are not each other's
baselines, and an earlier report printed them side by side without saying so: a column that
swapped `Evaluate` nodes sat next to `functor` and `old` columns that substituted `Var`s.

Every map arm is the same engine, `StructuralMap`, reached with a different callback — which
is what makes `subst` against `old` and against `functor` interpretable, and why they are not
three different engines:

| arm | mechanism |
| --- | --- |
| `subst` | `StructuralMap` with a `Var` callback |
| `old` | `Substitute` — the same engine at this pin, reached through a different API |
| `functor` | `FunctorSubstitute : StmtExprMutator` — pre-migration vtable path, no hooks |

So `subst vs old` prices the API and `subst vs functor` prices the migration.

**split/fuse — Expr-level `Var` substitution.** `Substitute` and `FunctorSubstitute` both hook
`VisitExpr_(const VarNode*)`, so these are the arms they are baselines for.

| arm | what it is | the question it answers |
| --- | --- | --- |
| `map_floor` | hand-written minimal mutator over the two mutate columns | **the control.** If it moves between two builds, the builds differ in something beyond what is under study and nothing else in the row can be trusted. |
| `map_never` | `StructuralMap<kPostOrder>` whose callback can never match | link testing alone, nothing rebuilt |
| `map_identity_var` | `StructuralMap` matching `Var`, returning the same `Var` | **matches but rebuilds nothing.** This is where a no-change optimization shows. |
| `map_subst` | `StructuralMap` substituting every `Var` through the remap | **the rebuild is real.** This is where rebuild cost shows. |
| `map_functor` | `IRSubstitute`'s shape over `StmtExprMutator` | baseline: the functor-era mutator |
| `map_old` | `Substitute`, called directly | baseline: the shipping API |

**seq — Stmt-level `Evaluate` swap.** `StmtExprMutator` hooks `VisitExpr_(const VarNode*)` and
does different work on the same graph, so it is not a baseline for this operation. `functor` and
`old` are therefore **not columns on the seq tables at all** rather than a column of `n/a`: a
baseline that cannot perform the operation is a column that belongs on another table.

| arm | what it is | the question it answers |
| --- | --- | --- |
| `map_floor` | as above | the control |
| `map_never` | as above | link testing alone |
| `map_identity_stmt` | `StructuralMap` matching `Stmt`, returning it unchanged | the no-rebuild rung |
| `map_swap` | matches `Stmt`, swaps two whole `Evaluate` nodes | exercises the `SeqStmt` hook's **element in-place** path and the sparse-update cost |

**Both seq arms bind the callback to `Stmt` and narrow to `Evaluate` inside it**, rather than
binding to `Evaluate` and letting the engine's link test do the narrowing. That is how a real
Stmt-level pass is written — passes match a base type and dispatch inside — and it is a
choice with a measurable consequence: matching narrowly moves work out of the callback and into
the engine's link test, and the two are not the same cost. It also gives `identity` and `swap`
the same link, so the difference between them is the rebuild and not how many nodes the link
accepted. Measuring the narrow/broad difference directly would be a narrow-match variant of
this same arm; there is no such arm today.

`map_splice` maps an element to a nested `SeqStmt` and is the only thing that reaches the
hook's splice path. It is not timed: it exists for the differential check, which is where the
question it answers is a correctness question rather than a cost one.

`map_functor` is a **floor for a functor-era mutator, not a reproduction of `Substitute`**. TVM's
`IRSubstitute` carries a dtype `ICHECK` per substitution plus buffer and attribute handling the
harness's `FunctorSubstitute` omits, which is why `map_functor` and `map_old` differ by around
20% while `walk_functor` and `walk_old` agree within 3%.

`*_functor` and `*_old` are **different baselines**, and a reader should not average them.
`*_old` is the pinned TVM's current implementation, which is already engine-based; `*_functor`
is the pre-structural functor machinery. Reading `*_old` as "before the engine" would be wrong.

### The ownership axis

**Every map arm runs in both ownership variants.** `retained` keeps the caller's handle alive,
so the root's refcount is at least two and the engine takes the copy-on-write path. `moved`
hands the sole reference over with `std::move`, so the engine takes the in-place path. These are
different workloads and a per-node optimization is worth different amounts in each; reporting
one hides how much of a result depends on ownership, which is the number a caller deciding
whether to `std::move` actually needs. They share a table, adjacent rows, because the comparison
between them is a result in its own right. Walk arms have no ownership axis and appear once.

Mutating arms **swap** rather than replace one-way, so repeating one in place is stationary
and the timed loop never drifts into a different workload. A fixture with a pointer-shared
subtree cannot be traversed in place repeatedly at all — the first in-place rebuild un-shares
the DAG — so those cases run over a pool of independent copies rebuilt untimed between passes.

## The fixtures

| fixture | shape | operation | scales? |
| --- | --- | --- | --- |
| `split-fuse-shared` | `floordiv(o*16+i, 32)*32 + floormod(o*16+i, 32)`, one pointer-shared intermediate | `subst` | no, N=12 |
| `split-fuse-distinct` | the same with two structurally equal, pointer-distinct intermediates | `subst` | no, N=15 |
| `call-split-fuse-shared` | the shared shape with every arithmetic node expressed as a `Call` | `subst` | no, N=18 |
| `call-split-fuse-distinct` | the distinct shape, likewise | `subst` | no, N=23 |
| `seq-L` | `SeqStmt` of `L` statements `Evaluate(o*(i+2) + inner)`, `L` in 16, 256, 16384 | `swap` | **yes** |

**split/fuse** is a small `Expr` tree. It carries the Expr-level comparison against both
baselines, and the shared variant is the only fixture that exercises DAG un-sharing. It does not
scale, so it says nothing about working set.

**call-split-fuse** is its row-for-row counterpart: identical topology, identical `Var`s, the
same six binary operations in the same order, with each expressed as a `Call` to an interned
builtin instead of a direct `Add`/`Mul`/`FloorDiv`/`FloorMod` node. The only variable between
the two is the node representation, so the delta between their rows is what a `Call` costs.

It is also the only fixture that reaches three hook decisions apache/tvm#20275 made and
justified on performance grounds:

- **descent into an `Array` inside a node**, `Call.args`. Every other fixture's children are
  direct typed fields, so the container path in `CallVisit`/`CallMutate` is otherwise dead.
- **the empty `ty_args` skip.** Nothing here has type arguments, so the guard is taken on every
  `Call`, on every arm, `floor` included.
- **the interned `Op` operator skip.**

Since the harness hooks are a verbatim port of 20275's, this fixture is the only place those
guards are exercised outside TVM's own tests. The operator's identity does not matter — the
traversal never evaluates the call, and the hooks skip an `OpNode` operator without looking at
which one — so four distinct builtins stand in for the four node types, which keeps the `op`
field varying exactly as the direct form's node type does.

**seq** is the one that scales, and the three lengths are one per cache level: 16 inside a
32 KiB L1d, 256 inside a 1 MiB L2, 16384 inside a 32 MiB L3. `report.py` names the level from
that stated geometry. Any question about how a result behaves as the tree stops fitting in cache
is a seq question. It carries the Stmt-level swap: `Evaluate` is not a free variable, so no
remap is involved, each element is exactly one `Evaluate`, and "swap k pairs" is a controlled
sparse input.

`N` and `occurrences` include the `Array` inside each `SeqStmt` and inside each `Call`:
apache/tvm#20275's hooks visit those containers as values rather than iterating their elements,
so a container is itself a visited node.

Fixture counts — unique nodes, occurrences, working set, rebuild counts — are **declared by the
builder**, not measured back by the binary: the builder built the thing and knows them, and
making the binary re-derive them would put counting inside the measurement. What the builder
cannot do is notice when a fixture is edited and its constants are not, which would be a silent
error in every `ns/node` in the row. So `CheckDeclaredCounts` walks the graph once, untimed,
before anything is measured, and fails the run with the numbers to paste in. It found two
stale rebuild counts the first time it ran.

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
  two are differential-tested against each other across thirteen cases: grow at
  first/last/middle, several in one pass, two adjacent, length-preserving, shrink at
  first/last/several, a wide splice, a sequence that ends at one element and one that ends at
  zero. Each case asserts the reference's own output shape, so agreement cannot be vacuous.
- **The no-op asymmetry is pinned.** From a1031a2177 the hooks normalize their result: an
  `Evaluate(0)` element is dropped, a sequence that ends at one element returns that element
  unwrapped, and one that ends at zero returns `Evaluate(0)`. Dropping happens only from the
  first changed element onward — the lead loop returns `self` untouched when nothing changed,
  and the rebuild path copies the prefix before the change with `InitRange` rather than
  replaying it. So the same `Evaluate(0)` survives or is dropped depending on where it sits
  relative to a change elsewhere in the sequence. That is deliberate: dropping no-ops on an
  unchanged pass would rewrite every sequence containing one and destroy the `same_as` fast
  path. It reads as a bug, it is untested in TVM's own tests, and `CheckNoOpAsymmetry` pins it.
- **The declared fixture counts are the real ones.** `CheckDeclaredCounts` walks each fixture
  once and fails the run if unique nodes, occurrences or rebuild counts differ from what the
  builder declared.
- **Hook coverage.** Every type a fixture dispatches on must have a harness hook, so a new
  fixture that reaches a new type names it instead of silently falling through to TVM's.
- **Port fidelity.** See [The port](#the-port).

**Splicing can only ever grow, through the public API.** `SeqStmt`'s constructor rejects both
length 0 and length 1, so a nested `SeqStmt` always carries at least two statements and the
shrink and length-unchanged directions are unreachable from outside. They are still tested, built
by bypassing the constructor, so the hook's behaviour on them is pinned rather than argued.

## Reading the tables

`ns/node` divides by the fixture's unique-node count, **the same divisor for every arm in a
row**, so a row is internally comparable and a column is comparable down the fixture list.

**`N` is not a results column.** It is fixture metadata, constant per fixture, and it lives in
the fixtures table rather than being repeated on every row of every results table. Nor is
`subst vs functor`: `old` is the shipping API, so `subst vs old` is the delta a caller's
decision turns on, and the historical question the functor comparison answers is settled and
belongs in a finding with its numbers quoted there.

**The three arms that read as a set** — the shape a two-state run is read in:

| arm | expected | what a violation means |
| --- | --- | --- |
| `floor` | **flat** across any callback-level or protocol-level change | the two builds differ in something beyond what is under study; nothing else in that row can be trusted |
| `identity` | **improves** when a no-change optimization lands | it did not reach the path it was meant to |
| `subst` / `swap` | **roughly neutral** — the rebuild is real work either way | a change aimed at unchanged nodes moved rebuilding cost, which needs explaining |

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

`tvm_hook_override.h` holds twenty-three structural hooks and helpers **copied verbatim** from
apache/tvm#20275 — same names, same signatures, same order, same internal structure, grouped by
the TVM file each came from. Nothing is reordered, renamed or tidied, because a change
prototyped here has to lift back into apache/tvm as a patch. The intended differences are
marked `HARNESS DEVIATION` in the header; there is one, and it is in the installation rather
than in any hook body.

**A per-state hook file** — `tvm_hook_override_pre753.h` today — is the same bodies written
against an older tvm-ffi API, for the two-state runs described above. It declares its single
`API ADAPTATION` and the apache/tvm revision that used that spelling, and `port_check.sh
--header FILE` checks it twice over: it undoes the declared substitution before diffing, so
DRIFT still means real drift, and it separately confirms the replacement text is what the PR
itself used at the revision cited. Both endpoints are checked against apache/tvm; nothing about
the adaptation is asserted by the harness alone. Delete the file when no two-state run reaches
back that far.

A hook experiment is then an edit to one header and a rebuild of one translation unit, never a
TVM branch and a full TVM build. That is the difference between answering a question in a minute
and in an afternoon.

```sh
./port_check.sh /path/to/tvm            # against the sha recorded in the header
./port_check.sh /path/to/tvm <sha>      # against another revision, e.g. a rebased PR head
./port_check.sh --header tvm_hook_override_pre753.h /path/to/tvm   # a state file
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
| `tvm_hook_override_pre753.h` | the same hooks written against a pre-#753 tvm-ffi, for two-state runs that reach back that far |
| `state_shims/` | headers force-included into apache/tvm's own translation units so TVM compiles against an older engine; never on the benchmark's include path |
| `mini_tir_bench.cc` | mini-TIR fixtures, arms, `main` |
| `real_tvm_bench.cc` | real-TVM fixtures, arms, checks, `main` |
| `build.sh` | builds each harness, single-state or one binary per engine state, stamping provenance in |
| `port_check.sh` | diffs the ported hooks against the TVM revision they came from |
| `report.py` | runs each binary N pinned processes, medians the process medians, renders the tables; `--two-state` interleaves A/B/A/B and renders the comparison |

The two hook headers are deliberately **not** factored against each other. Their bodies read
almost identically and are written out twice on purpose: a reader can follow either without
mentally instantiating a template, and a change to one cannot silently reshape the other.

The binary builds fixtures, runs arms, times them and prints timings. It collects no metadata
about itself and renders no tables — `report.py` owns the format. Provenance is stamped in at
build time by `build.sh` as compile-time constants, so **a report cannot omit what the tool
prints**.
