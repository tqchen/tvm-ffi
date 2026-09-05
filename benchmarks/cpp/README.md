# Structural mutate benchmarks

Standalone harnesses for the `StructuralMap` engine. Neither depends on TVM, so the engine can
be measured and iterated on without a TVM rebuild. They are kept on this branch rather than in
the engine PR.

## `tir_shaped_mutate_benchmark.cc`

Mirrors TVM's TIR node hierarchy closely enough to reproduce its numbers:

- `HExprObj` — non-final base, 64 child slots, a `ty` field, `TreeNode` kind
- `HVarObj` — derives from it and is **`FreeVar` kind**, so every matched Var takes an identity
  remap get *and* set, the way `tvm::Var` does
- `HPrimExpr` — a *view* over any `HExprObj` whose `ty` holds an `HPrimType`, with the same
  `TypeTraits` shape as `tvm::TypedExpr`: checking a field costs an `IsObjectInstance` range
  test, a dereference to reach `ty`, and a second type check on that field
- binary ops — final, two `HPrimExpr` operands, and `MutateBinary`'s guard that skips result-type
  inference when neither operand's type changed

The fixture is the split/fuse shape, `floordiv(o*16+i, 32)*32 + floormod(o*16+i, 32)`.

Why the shape matters: measured against real TVM on the same expression, engine cost *above each
harness's own minimal-vtable floor* agrees to within 1%.

| | real TVM | this harness |
|---|---:|---:|
| minimal-vtable floor | 13.25 | 8.81 |
| hooked structural (replace) | 28.58 | 24.32 |
| **over the floor** | **+15.33** | **+15.51** |

The absolute floors differ because TVM's real hooks are heavier (`MutateBinary<AddNode>` is 595
instructions to this mirror's ~400, and TVM nodes carry spans and richer fields). The delta over
the floor is what transfers.

It is a model of the shape, not a clone: no `Span`, `ResultType` compares a single `bits` field
rather than full dtype logic, and only four binary ops are mirrored.

## `structural_map_floor_benchmark.cc`

The original floor benchmark: plain recursive descent, a minimal-vtable mutator as the engine's
lower bound, and hooked `StructuralMap` at 0% and ~50% match, plus a `PrimExpr`-shaped typed
hierarchy for the field-check cost.

**It under-represents the engine by roughly 6x** and should not be used alone to judge it: its
`match-leaf` callback matches a `TreeNode`-kind leaf through a plain `ObjectRef` field, so it
never touches the identity remap and never pays the view-type check. It reports ~+2.5 ns/node
over the floor where the TIR-shaped harness reports ~+15.5. Use it for descent and codegen
questions, not for the matched-callback path.

## Building

```sh
g++ -std=c++17 -O3 -DNDEBUG -Iinclude -I3rdparty/dlpack/include \
    benchmarks/cpp/tir_shaped_mutate_benchmark.cc \
    -Lbuild/lib -ltvm_ffi -Wl,-rpath,build/lib -o /tmp/tir_bench && taskset -c 4 /tmp/tir_bench
```

Report medians of several process runs, pinned; single runs on this harness do not resolve
differences below roughly half a nanosecond per node.
