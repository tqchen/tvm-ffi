### Provenance

| | |
| --- | --- |
| state old | tvm-ffi `3666d8f8f7ee64d495a3c3442a2c51a79e893cd4 (old)` |
| state old engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state gold | tvm-ffi `fe046c8c55078c6071e1c28311c36f5e10650b5d (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_hook_override_gold.h sha256:20bd4475ef7e1096b54629015ce86d40d35dffd6f9f02a918947fb527d8360d4` |
| state uc | tvm-ffi `fe046c8c55078c6071e1c28311c36f5e10650b5d (uc)` |
| state uc engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| what differs | one tree (bench/tvm-hook 3666d8f on upstream main e74e58f), one apache/tvm 2c28965, one libtvm_ffi/libtvm_compiler build; each executable is compiled against one engine header and one hook file: old = structural_mutate_old.h (upstream main e74e58f's engine verbatim, the pre-UnchangedOr engine with #760/#761 in its base) + tvm_hook_override.h (the OLD hook file, untouched) with e74e58f's container hooks compiled in; gold = structural_mutate_gold.h (GOLD 730d6fc byte for byte) + tvm_hook_override_gold.h; uc = structural_mutate.h (PR 46 as resolved, 7a569c5) + tvm_hook_override_uc.h. The gold and uc binaries are the run-1 binaries, sha256-identical; sha256 of every engine and hook file in the rows above |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved old/gold/uc |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.98 at start, 1.14 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved old/gold/uc, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse -- `floor` -- ns/node, old/gold/uc against old

`subst` substitutes `Var`s.

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 11.51 | 8.42 (-26.9%) | 5.91 (-48.7%) |
| split-fuse-shared | moved | 11.48 | 7.98 (-30.5%) | 5.70 (-50.3%) |
| split-fuse-distinct | retained | 9.19 | 6.97 (-24.1%) | 4.66 (-49.3%) |
| split-fuse-distinct | moved | 9.42 | 6.37 (-32.4%) | 4.61 (-51.0%) |
| call-split-fuse-shared | retained | 13.80 | 8.35 (-39.5%) | 6.41 (-53.5%) |
| call-split-fuse-shared | moved | 13.90 | 7.72 (-44.5%) | 5.95 (-57.2%) |
| call-split-fuse-distinct | retained | 10.83 | 6.56 (-39.4%) | 5.04 (-53.4%) |
| call-split-fuse-distinct | moved | 10.81 | 5.87 (-45.7%) | 4.49 (-58.5%) |

#### Expr-level -- split/fuse -- `identity` -- ns/node, old/gold/uc against old

`subst` substitutes `Var`s.

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 31.71 | 28.84 (-9.0%) | 25.77 (-18.7%) |
| split-fuse-shared | moved | 31.43 | 29.01 (-7.7%) | 25.80 (-17.9%) |
| split-fuse-distinct | retained | 25.42 | 23.11 (-9.1%) | 20.60 (-19.0%) |
| split-fuse-distinct | moved | 24.88 | 22.96 (-7.7%) | 20.77 (-16.5%) |
| call-split-fuse-shared | retained | 29.57 | 22.98 (-22.3%) | 21.03 (-28.9%) |
| call-split-fuse-shared | moved | 29.31 | 24.09 (-17.8%) | 22.20 (-24.3%) |
| call-split-fuse-distinct | retained | 23.06 | 18.01 (-21.9%) | 16.46 (-28.6%) |
| call-split-fuse-distinct | moved | 22.36 | 18.10 (-19.1%) | 16.82 (-24.8%) |

#### Expr-level -- split/fuse -- `subst` -- ns/node, old/gold/uc against old

`subst` substitutes `Var`s.

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 38.69 | 37.83 (-2.2%) | 36.01 (-6.9%) |
| split-fuse-shared | moved | 34.70 | 33.43 (-3.6%) | 30.13 (-13.2%) |
| split-fuse-distinct | retained | 30.58 | 30.31 (-0.9%) | 28.77 (-5.9%) |
| split-fuse-distinct | moved | 25.81 | 24.10 (-6.6%) | 22.27 (-13.7%) |
| call-split-fuse-shared | retained | 39.21 | 38.11 (-2.8%) | 37.71 (-3.8%) |
| call-split-fuse-shared | moved | 31.88 | 28.26 (-11.4%) | 27.57 (-13.5%) |
| call-split-fuse-distinct | retained | 31.30 | 30.78 (-1.7%) | 31.03 (-0.9%) |
| call-split-fuse-distinct | moved | 23.16 | 18.79 (-18.9%) | 18.27 (-21.1%) |

#### Stmt-level -- seq -- `floor` -- ns/node, old/gold/uc against old

`swap` swaps two whole `Evaluate` nodes.

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| seq-16 | retained | 11.84 | 9.18 (-22.4%) | 6.96 (-41.2%) |
| seq-16 | moved | 10.68 | 8.56 (-19.9%) | 6.55 (-38.7%) |
| seq-256 | retained | 13.05 | 9.50 (-27.2%) | 7.06 (-45.9%) |
| seq-256 | moved | 11.78 | 8.77 (-25.6%) | 6.59 (-44.1%) |
| seq-16384 | retained | 13.17 | 9.50 (-27.9%) | 7.10 (-46.1%) |
| seq-16384 | moved | 11.89 | 8.68 (-27.0%) | 6.74 (-43.3%) |

#### Stmt-level -- seq -- `identity` -- ns/node, old/gold/uc against old

`swap` swaps two whole `Evaluate` nodes.

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| seq-16 | retained | 21.20 | 19.84 (-6.4%) | 16.98 (-19.9%) |
| seq-16 | moved | 19.65 | 19.31 (-1.7%) | 16.30 (-17.1%) |
| seq-256 | retained | 21.45 | 19.94 (-7.0%) | 16.82 (-21.6%) |
| seq-256 | moved | 19.29 | 18.98 (-1.6%) | 15.62 (-19.0%) |
| seq-16384 | retained | 21.48 | 19.97 (-7.0%) | 16.74 (-22.1%) |
| seq-16384 | moved | 19.22 | 18.93 (-1.5%) | 15.61 (-18.7%) |

#### Stmt-level -- seq -- `swap` -- ns/node, old/gold/uc against old

`swap` swaps two whole `Evaluate` nodes.

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| seq-16 | retained | 22.95 | 21.01 (-8.5%) | 18.90 (-17.7%) |
| seq-16 | moved | 21.14 | 19.72 (-6.7%) | 17.33 (-18.1%) |
| seq-256 | retained | 22.85 | 20.78 (-9.1%) | 18.86 (-17.5%) |
| seq-256 | moved | 20.70 | 19.40 (-6.3%) | 17.25 (-16.7%) |
| seq-16384 | retained | 22.79 | 20.68 (-9.2%) | 18.79 (-17.5%) |
| seq-16384 | moved | 20.64 | 19.36 (-6.2%) | 17.26 (-16.4%) |

### Supporting

#### `never` -- ns/node, old/gold/uc against old

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 23.08 | 21.36 (-7.5%) | 17.94 (-22.3%) |
| split-fuse-shared | moved | 21.82 | 19.66 (-9.9%) | 16.39 (-24.9%) |
| split-fuse-distinct | retained | 18.13 | 17.02 (-6.1%) | 14.19 (-21.7%) |
| split-fuse-distinct | moved | 17.31 | 15.16 (-12.4%) | 12.32 (-28.8%) |
| call-split-fuse-shared | retained | 21.80 | 16.94 (-22.3%) | 14.76 (-32.3%) |
| call-split-fuse-shared | moved | 22.08 | 17.05 (-22.7%) | 15.03 (-31.9%) |
| call-split-fuse-distinct | retained | 17.08 | 13.16 (-23.0%) | 11.57 (-32.3%) |
| call-split-fuse-distinct | moved | 17.30 | 13.11 (-24.2%) | 11.57 (-33.2%) |
| seq-16 | retained | 18.39 | 17.23 (-6.3%) | 13.64 (-25.8%) |
| seq-16 | moved | 16.82 | 15.72 (-6.5%) | 12.66 (-24.7%) |
| seq-256 | retained | 18.48 | 17.06 (-7.7%) | 13.35 (-27.8%) |
| seq-256 | moved | 16.13 | 15.30 (-5.1%) | 12.45 (-22.8%) |
| seq-16384 | retained | 18.35 | 17.07 (-7.0%) | 13.27 (-27.7%) |
| seq-16384 | moved | 16.02 | 15.20 (-5.1%) | 12.49 (-22.0%) |

#### `functor` -- ns/node, old/gold/uc against old

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.82 | 35.91 (+0.3%) | 35.58 (-0.7%) |
| split-fuse-shared | moved | 40.65 | 40.88 (+0.6%) | 40.35 (-0.7%) |
| split-fuse-distinct | retained | 28.32 | 28.47 (+0.5%) | 28.43 (+0.4%) |
| split-fuse-distinct | moved | 33.29 | 33.23 (-0.2%) | 33.14 (-0.5%) |
| call-split-fuse-shared | retained | 56.53 | 56.70 (+0.3%) | 57.21 (+1.2%) |
| call-split-fuse-shared | moved | 59.98 | 59.29 (-1.2%) | 59.91 (-0.1%) |
| call-split-fuse-distinct | retained | 46.11 | 45.67 (-0.9%) | 46.55 (+1.0%) |
| call-split-fuse-distinct | moved | 47.16 | 46.62 (-1.1%) | 47.22 (+0.1%) |
| seq-16 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |

#### `old` -- ns/node, old/gold/uc against old

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 53.87 | 54.06 (+0.4%) | 53.91 (+0.1%) |
| split-fuse-shared | moved | 58.17 | 58.46 (+0.5%) | 58.29 (+0.2%) |
| split-fuse-distinct | retained | 43.19 | 43.50 (+0.7%) | 43.39 (+0.5%) |
| split-fuse-distinct | moved | 47.36 | 47.79 (+0.9%) | 47.74 (+0.8%) |
| call-split-fuse-shared | retained | 65.14 | 64.88 (-0.4%) | 64.56 (-0.9%) |
| call-split-fuse-shared | moved | 68.34 | 67.75 (-0.9%) | 67.37 (-1.4%) |
| call-split-fuse-distinct | retained | 53.23 | 52.62 (-1.1%) | 52.67 (-1.0%) |
| call-split-fuse-distinct | moved | 53.82 | 53.25 (-1.1%) | 53.45 (-0.7%) |
| seq-16 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

