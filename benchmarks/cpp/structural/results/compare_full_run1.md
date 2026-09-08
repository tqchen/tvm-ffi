### Provenance

| | |
| --- | --- |
| state gold | tvm-ffi `fe046c8c55078c6071e1c28311c36f5e10650b5d (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_hook_override_gold.h sha256:20bd4475ef7e1096b54629015ce86d40d35dffd6f9f02a918947fb527d8360d4` |
| state uc | tvm-ffi `fe046c8c55078c6071e1c28311c36f5e10650b5d (uc)` |
| state uc engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| what differs | one tree (bench/tvm-hook fe046c8 on upstream main e74e58f), one apache/tvm 2c28965, one libtvm_ffi/libtvm_compiler build; each executable is compiled against one engine header and one hook file: gold = structural_mutate_gold.h (GOLD 730d6fc byte for byte) + tvm_hook_override_gold.h; uc = structural_mutate.h (PR 46 as resolved, 7a569c5) + tvm_hook_override_uc.h; hook files and engines as reviewed at fe046c8, sha256 in the rows above |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved gold/uc |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.15 at start, 1.13 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved gold/uc, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse -- `floor` -- ns/node, gold/uc against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc |
| --- | --- | ---: | ---: |
| split-fuse-shared | retained | 8.65 | 5.88 (-32.0%) |
| split-fuse-shared | moved | 8.39 | 5.69 (-32.2%) |
| split-fuse-distinct | retained | 7.18 | 4.73 (-34.0%) |
| split-fuse-distinct | moved | 6.49 | 4.70 (-27.5%) |
| call-split-fuse-shared | retained | 8.25 | 6.44 (-22.0%) |
| call-split-fuse-shared | moved | 7.75 | 6.02 (-22.3%) |
| call-split-fuse-distinct | retained | 6.63 | 5.07 (-23.4%) |
| call-split-fuse-distinct | moved | 5.83 | 4.56 (-21.9%) |

#### Expr-level -- split/fuse -- `identity` -- ns/node, gold/uc against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc |
| --- | --- | ---: | ---: |
| split-fuse-shared | retained | 28.83 | 25.79 (-10.5%) |
| split-fuse-shared | moved | 28.98 | 25.94 (-10.5%) |
| split-fuse-distinct | retained | 23.09 | 20.64 (-10.6%) |
| split-fuse-distinct | moved | 22.93 | 20.77 (-9.4%) |
| call-split-fuse-shared | retained | 22.87 | 21.01 (-8.2%) |
| call-split-fuse-shared | moved | 23.93 | 22.15 (-7.5%) |
| call-split-fuse-distinct | retained | 18.07 | 16.42 (-9.2%) |
| call-split-fuse-distinct | moved | 17.99 | 16.87 (-6.2%) |

#### Expr-level -- split/fuse -- `subst` -- ns/node, gold/uc against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc |
| --- | --- | ---: | ---: |
| split-fuse-shared | retained | 37.66 | 35.99 (-4.5%) |
| split-fuse-shared | moved | 33.34 | 30.40 (-8.8%) |
| split-fuse-distinct | retained | 30.24 | 28.82 (-4.7%) |
| split-fuse-distinct | moved | 24.08 | 22.25 (-7.6%) |
| call-split-fuse-shared | retained | 37.98 | 38.01 (+0.1%) |
| call-split-fuse-shared | moved | 28.24 | 27.83 (-1.4%) |
| call-split-fuse-distinct | retained | 30.67 | 30.85 (+0.6%) |
| call-split-fuse-distinct | moved | 18.73 | 18.43 (-1.6%) |

#### Stmt-level -- seq -- `floor` -- ns/node, gold/uc against gold

`swap` swaps two whole `Evaluate` nodes.

| Case | own | gold | uc |
| --- | --- | ---: | ---: |
| seq-16 | retained | 9.48 | 6.86 (-27.7%) |
| seq-16 | moved | 8.71 | 6.59 (-24.4%) |
| seq-256 | retained | 9.77 | 7.13 (-27.1%) |
| seq-256 | moved | 8.79 | 6.59 (-25.0%) |
| seq-16384 | retained | 9.89 | 7.04 (-28.8%) |
| seq-16384 | moved | 8.86 | 6.68 (-24.5%) |

#### Stmt-level -- seq -- `identity` -- ns/node, gold/uc against gold

`swap` swaps two whole `Evaluate` nodes.

| Case | own | gold | uc |
| --- | --- | ---: | ---: |
| seq-16 | retained | 19.81 | 16.98 (-14.3%) |
| seq-16 | moved | 19.28 | 16.40 (-14.9%) |
| seq-256 | retained | 19.88 | 16.83 (-15.3%) |
| seq-256 | moved | 18.92 | 15.69 (-17.0%) |
| seq-16384 | retained | 19.96 | 16.77 (-16.0%) |
| seq-16384 | moved | 18.92 | 15.69 (-17.1%) |

#### Stmt-level -- seq -- `swap` -- ns/node, gold/uc against gold

`swap` swaps two whole `Evaluate` nodes.

| Case | own | gold | uc |
| --- | --- | ---: | ---: |
| seq-16 | retained | 20.99 | 18.92 (-9.9%) |
| seq-16 | moved | 19.65 | 17.36 (-11.7%) |
| seq-256 | retained | 20.75 | 18.86 (-9.1%) |
| seq-256 | moved | 19.32 | 17.28 (-10.5%) |
| seq-16384 | retained | 20.73 | 18.81 (-9.3%) |
| seq-16384 | moved | 19.34 | 17.26 (-10.8%) |

### Supporting

#### `never` -- ns/node, gold/uc against gold

| Case | own | gold | uc |
| --- | --- | ---: | ---: |
| split-fuse-shared | retained | 21.28 | 18.04 (-15.2%) |
| split-fuse-shared | moved | 19.73 | 16.55 (-16.1%) |
| split-fuse-distinct | retained | 17.07 | 14.17 (-17.0%) |
| split-fuse-distinct | moved | 15.27 | 12.39 (-18.9%) |
| call-split-fuse-shared | retained | 16.83 | 14.84 (-11.8%) |
| call-split-fuse-shared | moved | 17.15 | 15.05 (-12.3%) |
| call-split-fuse-distinct | retained | 13.23 | 11.55 (-12.7%) |
| call-split-fuse-distinct | moved | 13.16 | 11.61 (-11.8%) |
| seq-16 | retained | 17.17 | 13.68 (-20.3%) |
| seq-16 | moved | 15.78 | 12.76 (-19.2%) |
| seq-256 | retained | 17.03 | 13.39 (-21.4%) |
| seq-256 | moved | 15.32 | 12.43 (-18.9%) |
| seq-16384 | retained | 17.05 | 13.32 (-21.9%) |
| seq-16384 | moved | 15.30 | 12.46 (-18.6%) |

#### `functor` -- ns/node, gold/uc against gold

| Case | own | gold | uc |
| --- | --- | ---: | ---: |
| split-fuse-shared | retained | 35.74 | 35.73 (-0.0%) |
| split-fuse-shared | moved | 40.77 | 40.83 (+0.2%) |
| split-fuse-distinct | retained | 28.43 | 28.82 (+1.4%) |
| split-fuse-distinct | moved | 33.27 | 33.27 (-0.0%) |
| call-split-fuse-shared | retained | 57.14 | 57.40 (+0.5%) |
| call-split-fuse-shared | moved | 59.57 | 60.02 (+0.7%) |
| call-split-fuse-distinct | retained | 46.07 | 46.16 (+0.2%) |
| call-split-fuse-distinct | moved | 46.78 | 47.18 (+0.9%) |
| seq-16 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] |

#### `old` -- ns/node, gold/uc against gold

| Case | own | gold | uc |
| --- | --- | ---: | ---: |
| split-fuse-shared | retained | 53.83 | 53.50 (-0.6%) |
| split-fuse-shared | moved | 58.08 | 58.21 (+0.2%) |
| split-fuse-distinct | retained | 43.53 | 43.36 (-0.4%) |
| split-fuse-distinct | moved | 47.69 | 47.73 (+0.1%) |
| call-split-fuse-shared | retained | 64.65 | 64.77 (+0.2%) |
| call-split-fuse-shared | moved | 67.48 | 67.69 (+0.3%) |
| call-split-fuse-distinct | retained | 52.37 | 52.31 (-0.1%) |
| call-split-fuse-distinct | moved | 53.24 | 53.27 (+0.1%) |
| seq-16 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

