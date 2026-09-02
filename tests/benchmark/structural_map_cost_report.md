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

# StructuralMap per-node cost budget (GB200)

## Scope and method

This is measurement and attribution only; it makes no traversal-engine change.
The tvm-ffi-only model is pinned to
`2b50d28d9aada35c6eeacac7f834642288c1c486`. The original-TVM calibration is
pinned to `6ea5d92d8a8a2a0fcd31a8cfdabc77911cca4f08`, whose tvm-ffi submodule is the
same `2b50d28` commit.

Both drivers were compiled with GCC 13.3.0, `-O3 -DNDEBUG`, and run in the same
process shape pinned to GB200 CPU 22. Each value is the median of eleven
in-process samples. Walk values divide by deduplicated identities. Old mutate
and rungs 2a/2b divide by recursive occurrences (17 and 185); map values divide
by actual processed nodes after FreeVar identity caching (15 and 155).

The minimal nodes deliberately carry the same two non-traversed, ObjectRef-sized
`span` and `ty` fields as TVM `ExprNode`; binaries carry exactly `a` and `b`.
The four fixture shapes are base/nested-4 crossed with pointer-shared/distinct
subexpressions. The benchmark target itself depends only on tvm-ffi.

## Calibration gate

The original-TVM driver reproduced the prior audit closely:

| Fixture | Op | this run | #343 | difference |
|---|---|---:|---:|---:|
| base distinct | walk | 16.50 | 16.32 | +1.1% |
| base distinct | changed replace | 83.90 | 86.43 | -2.9% |
| base distinct | no-op replace | 58.52 | 60.22 | -2.8% |
| nested-4 distinct | walk | 14.27 | 13.93 | +2.5% |
| nested-4 distinct | changed replace | 82.84 | 84.71 | -2.2% |
| nested-4 distinct | no-op replace | 55.75 | 57.44 | -2.9% |

The identical old/structural comparison on both ASTs is:

| AST | Fixture | Op | old-O3 | structural-O3 |
|---|---|---|---:|---:|
| minimal ffi | base distinct | walk | 27.32 | 12.66 |
| minimal ffi | base distinct | changed replace | 23.14 | 65.44 |
| minimal ffi | base distinct | no-op replace | 13.82 | 43.99 |
| minimal ffi | nested-4 distinct | walk | 41.90 | 12.25 |
| minimal ffi | nested-4 distinct | changed replace | 23.34 | 64.15 |
| minimal ffi | nested-4 distinct | no-op replace | 13.05 | 39.72 |
| original TVM | base distinct | walk | 34.89 | 16.50 |
| original TVM | base distinct | changed replace | 69.94 | 83.90 |
| original TVM | base distinct | no-op replace | 42.59 | 58.52 |
| original TVM | nested-4 distinct | walk | 39.70 | 14.27 |
| original TVM | nested-4 distinct | changed replace | 70.99 | 82.84 |
| original TVM | nested-4 distinct | no-op replace | 41.31 | 55.75 |

Minimal/TVM agreement on the structural rows is -23.3%, -22.0%, and -24.8%
for base walk/changed/no-op, and -14.2%, -22.6%, and -28.8% for nested-4.
That is material disagreement, despite matching object field layout. Therefore
the minimal AST is **not** frozen as an absolute TVM performance proxy. The
remaining difference is attributable to real IR hook bodies/type hierarchy and
code placement, not arity or object size. To avoid projecting that residual,
the same ladder was also measured directly on the original TVM AST. Later
tvm-ffi-only optimization work may use the minimal target for before/after
deltas, but must retain a TVM calibration gate before claiming TVM absolute
costs.

## Five-rung ladder

Minimal tvm-ffi AST, ns per processed node:

| Fixture | 1 walk | 2a ownership | 2b remap check | 3 unmatched map | 4 changed map |
|---|---:|---:|---:|---:|---:|
| base shared | 13.77 | 28.31 | 32.88 | 43.00 | 65.27 |
| base distinct | 12.64 | 28.35 | 32.88 | 42.99 | 65.33 |
| nested-4 shared | 14.24 | 27.96 | 32.73 | 40.17 | 64.14 |
| nested-4 distinct | 12.23 | 27.96 | 32.74 | 40.28 | 64.16 |

Original TVM AST calibration, ns per processed node:

| Fixture | 1 walk | 2a ownership | 2b remap check | 3 unmatched map | 4 changed map |
|---|---:|---:|---:|---:|---:|
| base distinct | 16.50 | 41.79 | 46.32 | 56.35 | 83.89 |
| nested-4 distinct | 14.38 | 41.31 | 46.10 | 54.66 | 82.91 |

Rung 2a calls `DefaultMutateExpected` directly. Rung 2b wraps that exact call
in `MutateWithIdentityRemapExpected`; its remap get/set vtable slots are empty
stubs, so 2b-2a isolates `TVMFFIGetTypeInfo`, metadata loads, and enum tests.
Rung 3 is real `StructuralMap` with a callback whose type cannot match this AST.
Rung 4 matches `TestVar`/`VarNode` and rebuilds the target path. All roots have
a second owner, so no measurement takes the maybe-in-place path.

## Reconciled budget

The directly measured original-TVM components reconcile exactly (rounding to
0.01 ns):

| Fixture | ownership (2a-1) | remap check (2b-2a) | callback dispatch (3-2b) | reconstruction (4-3) | no-op gap (3-1) | changed gap (4-1) |
|---|---:|---:|---:|---:|---:|---:|
| base distinct | 25.29 | 4.53 | 10.03 | 27.54 | 39.85 | 67.39 |
| nested-4 distinct | 26.92 | 4.79 | 8.56 | 28.25 | 40.28 | 68.52 |

For example, nested-4 no-op is `26.92 + 4.79 + 8.56 = 40.27` ns/node,
equal to `54.66 - 14.38 = 40.28` after rounding. Adding reconstruction gives
`68.52`, equal to `82.91 - 14.38`. The audit-style no-op callback actually
matches each FreeVar and returns it unchanged; that costs another 2.17 ns/node
on base and 1.09 on nested-4, explaining the matching no-op values 58.52 and
55.75 without assigning callback work to ownership.

The minimal ladder has the same ordering and a stable remap-check cost
(4.54-4.78 ns/node), but understates ownership by about 11-13 ns/node and
reconstruction by about 4-5 ns/node. This is why its absolute budget is not
substituted for the directly measured TVM budget.

## Bounds and verdict

Rung 2a, 41.3-41.8 ns/node on original TVM, is the lower bound for the map path
as currently designed: a no-op structural mutation cannot do less than owning
and returning every child result. Rung 1, 14.4-16.5 ns/node, is the absolute
floor and the aspirational no-op target.

The result lies closer to the task's "2a near 50" branch than its "2a near 25"
branch. Ownership is the dominant avoidable cost (25-27 ns/node, 63-67% of the
unmatched no-op gap). Callback dispatch is second (9-10 ns/node); the identity
remap test is only 4.5-4.8 ns/node. The evidence-based follow-up is therefore
unchanged-path ownership first. Optimizing the remap metadata test alone cannot
recover most of the gap and should not outrank ownership work.
