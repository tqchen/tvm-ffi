### Provenance

| | |
| --- | --- |
| state old | tvm-ffi `719b61793638322308cea242d9208af9dffecede (old)` |
| state old engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state gold | tvm-ffi `719b61793638322308cea242d9208af9dffecede (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_hook_override_gold.h sha256:20bd4475ef7e1096b54629015ce86d40d35dffd6f9f02a918947fb527d8360d4` |
| state uc | tvm-ffi `719b61793638322308cea242d9208af9dffecede (uc)` |
| state uc engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| what differs | one tree (bench/tvm-hook 719b617 on upstream main e74e58f), one apache/tvm 2c28965, one libtvm_ffi/libtvm_compiler build; each executable is compiled against one engine header and one hook file: old = structural_mutate_old.h (upstream main e74e58f's engine verbatim) + tvm_hook_override.h with e74e58f's container hooks compiled in; gold = structural_mutate_gold.h (GOLD 730d6fc byte for byte) + tvm_hook_override_gold.h; uc = structural_mutate.h (PR 46 as resolved, 7a569c5) + tvm_hook_override_uc.h. Hook files and engines byte-identical to fe046c8 / 3666d8f; only the driver changed (four add-tree fixtures added, seq elements redefined as Evaluate(IntImm)) |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved old/gold/uc |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 1.00 at start, 1.05 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved old/gold/uc, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse and add-tree -- `floor` -- ns/node, old/gold/uc against old

`subst` substitutes `Var`s.

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 11.58 | 8.32 (-28.1%) | 6.14 (-47.0%) |
| split-fuse-shared | moved | 11.53 | 7.98 (-30.8%) | 5.66 (-50.9%) |
| split-fuse-distinct | retained | 9.22 | 6.95 (-24.6%) | 4.93 (-46.5%) |
| split-fuse-distinct | moved | 9.44 | 6.34 (-32.9%) | 4.39 (-53.5%) |
| call-split-fuse-shared | retained | 13.77 | 8.21 (-40.4%) | 6.44 (-53.3%) |
| call-split-fuse-shared | moved | 13.94 | 7.60 (-45.4%) | 6.04 (-56.6%) |
| call-split-fuse-distinct | retained | 10.82 | 6.50 (-40.0%) | 5.06 (-53.2%) |
| call-split-fuse-distinct | moved | 10.85 | 5.73 (-47.2%) | 4.60 (-57.6%) |
| add-tree-shared | retained | 11.47 | 8.30 (-27.7%) | 6.12 (-46.7%) |
| add-tree-shared | moved | 11.59 | 7.97 (-31.3%) | 5.58 (-51.8%) |
| add-tree-distinct | retained | 9.19 | 6.88 (-25.1%) | 4.75 (-48.3%) |
| add-tree-distinct | moved | 9.21 | 6.11 (-33.7%) | 4.42 (-52.0%) |
| add-tree-intimm-shared | retained | 10.65 | 7.52 (-29.4%) | 4.75 (-55.4%) |
| add-tree-intimm-shared | moved | 9.58 | 7.04 (-26.5%) | 4.46 (-53.5%) |
| add-tree-intimm-distinct | retained | 8.56 | 6.02 (-29.7%) | 3.79 (-55.8%) |
| add-tree-intimm-distinct | moved | 7.70 | 5.31 (-31.0%) | 3.50 (-54.5%) |

#### Expr-level -- split/fuse and add-tree -- `identity` -- ns/node, old/gold/uc against old

`subst` substitutes `Var`s.

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 32.23 | 28.98 (-10.1%) | 25.92 (-19.6%) |
| split-fuse-shared | moved | 32.13 | 28.95 (-9.9%) | 25.87 (-19.5%) |
| split-fuse-distinct | retained | 25.80 | 23.16 (-10.2%) | 20.84 (-19.2%) |
| split-fuse-distinct | moved | 24.99 | 23.28 (-6.8%) | 20.78 (-16.9%) |
| call-split-fuse-shared | retained | 29.32 | 22.67 (-22.7%) | 21.18 (-27.8%) |
| call-split-fuse-shared | moved | 29.25 | 24.04 (-17.8%) | 22.24 (-23.9%) |
| call-split-fuse-distinct | retained | 23.18 | 17.75 (-23.4%) | 16.67 (-28.1%) |
| call-split-fuse-distinct | moved | 22.47 | 18.03 (-19.7%) | 16.73 (-25.5%) |
| add-tree-shared | retained | 31.73 | 29.29 (-7.7%) | 26.07 (-17.8%) |
| add-tree-shared | moved | 31.46 | 29.17 (-7.3%) | 25.91 (-17.7%) |
| add-tree-distinct | retained | 25.42 | 23.39 (-8.0%) | 20.93 (-17.7%) |
| add-tree-distinct | moved | 24.87 | 23.16 (-6.9%) | 20.88 (-16.1%) |
| add-tree-intimm-shared | retained | 12.32 | 10.31 (-16.3%) | 7.74 (-37.2%) |
| add-tree-intimm-shared | moved | 12.18 | 10.26 (-15.8%) | 7.75 (-36.4%) |
| add-tree-intimm-distinct | retained | 9.86 | 8.18 (-17.0%) | 6.21 (-37.0%) |
| add-tree-intimm-distinct | moved | 9.45 | 8.21 (-13.1%) | 6.14 (-35.0%) |

#### Expr-level -- split/fuse and add-tree -- `subst` -- ns/node, old/gold/uc against old

`subst` substitutes `Var`s.

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 38.51 | 37.94 (-1.5%) | 35.87 (-6.8%) |
| split-fuse-shared | moved | 34.65 | 33.36 (-3.7%) | 30.54 (-11.9%) |
| split-fuse-distinct | retained | 30.68 | 30.35 (-1.1%) | 28.49 (-7.1%) |
| split-fuse-distinct | moved | 25.90 | 24.36 (-5.9%) | 22.22 (-14.2%) |
| call-split-fuse-shared | retained | 38.54 | 38.04 (-1.3%) | 37.99 (-1.4%) |
| call-split-fuse-shared | moved | 31.80 | 28.34 (-10.9%) | 27.56 (-13.3%) |
| call-split-fuse-distinct | retained | 30.83 | 30.58 (-0.8%) | 31.17 (+1.1%) |
| call-split-fuse-distinct | moved | 23.22 | 19.02 (-18.1%) | 18.15 (-21.8%) |
| add-tree-shared | retained | 37.79 | 37.82 (+0.1%) | 35.61 (-5.8%) |
| add-tree-shared | moved | 34.87 | 33.19 (-4.8%) | 30.71 (-11.9%) |
| add-tree-distinct | retained | 30.21 | 30.57 (+1.2%) | 28.73 (-4.9%) |
| add-tree-distinct | moved | 25.98 | 24.21 (-6.8%) | 22.37 (-13.9%) |
| add-tree-intimm-shared | retained | 12.29 | 10.74 (-12.6%) | 7.98 (-35.1%) |
| add-tree-intimm-shared | moved | 12.58 | 10.25 (-18.5%) | 7.41 (-41.1%) |
| add-tree-intimm-distinct | retained | 10.02 | 8.66 (-13.6%) | 6.27 (-37.4%) |
| add-tree-intimm-distinct | moved | 9.39 | 7.93 (-15.5%) | 6.11 (-34.9%) |

#### Stmt-level -- seq -- `floor` -- ns/node, old/gold/uc against old

`swap` swaps two whole `Evaluate` nodes.

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| seq-16 | retained | 6.01 | 5.61 (-6.5%) | 4.33 (-27.9%) |
| seq-16 | moved | 4.31 | 4.61 (+7.1%) | 3.18 (-26.1%) |
| seq-256 | retained | 6.96 | 5.79 (-16.8%) | 4.22 (-39.3%) |
| seq-256 | moved | 4.50 | 4.61 (+2.4%) | 3.12 (-30.6%) |
| seq-16384 | retained | 7.06 | 5.74 (-18.7%) | 4.02 (-43.0%) |
| seq-16384 | moved | 4.39 | 4.43 (+0.9%) | 3.02 (-31.1%) |

#### Stmt-level -- seq -- `identity` -- ns/node, old/gold/uc against old

`swap` swaps two whole `Evaluate` nodes.

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| seq-16 | retained | 12.18 | 10.61 (-12.9%) | 9.76 (-19.8%) |
| seq-16 | moved | 10.00 | 11.54 (+15.4%) | 9.17 (-8.3%) |
| seq-256 | retained | 12.35 | 10.18 (-17.6%) | 9.47 (-23.3%) |
| seq-256 | moved | 9.44 | 10.98 (+16.4%) | 8.69 (-7.9%) |
| seq-16384 | retained | 12.53 | 10.25 (-18.2%) | 9.27 (-26.0%) |
| seq-16384 | moved | 9.36 | 10.98 (+17.4%) | 8.76 (-6.4%) |

#### Stmt-level -- seq -- `swap` -- ns/node, old/gold/uc against old

`swap` swaps two whole `Evaluate` nodes.

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| seq-16 | retained | 16.57 | 14.22 (-14.2%) | 14.67 (-11.5%) |
| seq-16 | moved | 12.50 | 12.96 (+3.7%) | 11.82 (-5.5%) |
| seq-256 | retained | 16.09 | 13.41 (-16.6%) | 13.98 (-13.1%) |
| seq-256 | moved | 11.35 | 12.16 (+7.1%) | 11.09 (-2.3%) |
| seq-16384 | retained | 15.96 | 13.40 (-16.1%) | 13.75 (-13.9%) |
| seq-16384 | moved | 11.22 | 12.13 (+8.1%) | 10.97 (-2.3%) |

### Supporting

#### `never` -- ns/node, old/gold/uc against old

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 23.12 | 21.08 (-8.8%) | 17.95 (-22.4%) |
| split-fuse-shared | moved | 21.94 | 19.63 (-10.5%) | 16.07 (-26.8%) |
| split-fuse-distinct | retained | 17.33 | 16.83 (-2.9%) | 13.36 (-22.9%) |
| split-fuse-distinct | moved | 17.26 | 15.34 (-11.1%) | 12.69 (-26.5%) |
| call-split-fuse-shared | retained | 21.84 | 16.67 (-23.7%) | 14.72 (-32.6%) |
| call-split-fuse-shared | moved | 22.29 | 16.88 (-24.3%) | 15.15 (-32.0%) |
| call-split-fuse-distinct | retained | 17.22 | 13.04 (-24.3%) | 11.52 (-33.1%) |
| call-split-fuse-distinct | moved | 17.40 | 12.96 (-25.5%) | 12.02 (-30.9%) |
| add-tree-shared | retained | 21.28 | 21.07 (-1.0%) | 16.67 (-21.7%) |
| add-tree-shared | moved | 21.71 | 19.53 (-10.0%) | 16.11 (-25.8%) |
| add-tree-distinct | retained | 16.99 | 16.80 (-1.1%) | 13.28 (-21.8%) |
| add-tree-distinct | moved | 17.13 | 14.95 (-12.7%) | 12.81 (-25.2%) |
| add-tree-intimm-shared | retained | 11.72 | 11.11 (-5.3%) | 7.67 (-34.6%) |
| add-tree-intimm-shared | moved | 11.76 | 10.28 (-12.6%) | 7.35 (-37.5%) |
| add-tree-intimm-distinct | retained | 9.43 | 8.86 (-6.0%) | 6.15 (-34.8%) |
| add-tree-intimm-distinct | moved | 9.34 | 8.02 (-14.1%) | 5.82 (-37.7%) |
| seq-16 | retained | 6.78 | 6.81 (+0.5%) | 5.97 (-11.9%) |
| seq-16 | moved | 5.81 | 6.14 (+5.8%) | 4.74 (-18.3%) |
| seq-256 | retained | 6.93 | 6.36 (-8.1%) | 5.54 (-20.0%) |
| seq-256 | moved | 5.51 | 5.69 (+3.2%) | 4.27 (-22.4%) |
| seq-16384 | retained | 6.93 | 6.39 (-7.7%) | 5.34 (-23.0%) |
| seq-16384 | moved | 5.42 | 5.59 (+3.2%) | 4.25 (-21.6%) |

#### `functor` -- ns/node, old/gold/uc against old

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.78 | 35.62 (-0.5%) | 35.52 (-0.7%) |
| split-fuse-shared | moved | 40.73 | 40.73 (+0.0%) | 41.00 (+0.7%) |
| split-fuse-distinct | retained | 28.60 | 28.43 (-0.6%) | 28.54 (-0.2%) |
| split-fuse-distinct | moved | 33.45 | 33.37 (-0.2%) | 33.43 (-0.1%) |
| call-split-fuse-shared | retained | 56.82 | 57.88 (+1.9%) | 58.14 (+2.3%) |
| call-split-fuse-shared | moved | 59.95 | 60.52 (+1.0%) | 61.34 (+2.3%) |
| call-split-fuse-distinct | retained | 45.76 | 46.47 (+1.5%) | 46.88 (+2.4%) |
| call-split-fuse-distinct | moved | 47.22 | 47.54 (+0.7%) | 48.35 (+2.4%) |
| add-tree-shared | retained | 35.83 | 35.96 (+0.4%) | 35.99 (+0.5%) |
| add-tree-shared | moved | 41.81 | 41.97 (+0.4%) | 42.21 (+1.0%) |
| add-tree-distinct | retained | 28.37 | 28.79 (+1.5%) | 28.68 (+1.1%) |
| add-tree-distinct | moved | 33.30 | 33.80 (+1.5%) | 33.67 (+1.1%) |
| add-tree-intimm-shared | retained | 19.10 | 19.32 (+1.1%) | 19.21 (+0.5%) |
| add-tree-intimm-shared | moved | 20.04 | 20.08 (+0.2%) | 20.01 (-0.1%) |
| add-tree-intimm-distinct | retained | 15.19 | 15.42 (+1.5%) | 15.33 (+0.9%) |
| add-tree-intimm-distinct | moved | 15.18 | 15.45 (+1.8%) | 15.35 (+1.1%) |
| seq-16 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |

#### `old` -- ns/node, old/gold/uc against old

| Case | own | old | gold | uc |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 53.92 | 54.24 (+0.6%) | 54.55 (+1.2%) |
| split-fuse-shared | moved | 58.22 | 58.95 (+1.3%) | 58.96 (+1.3%) |
| split-fuse-distinct | retained | 43.53 | 43.75 (+0.5%) | 43.82 (+0.7%) |
| split-fuse-distinct | moved | 47.35 | 47.91 (+1.2%) | 48.28 (+2.0%) |
| call-split-fuse-shared | retained | 64.99 | 66.24 (+1.9%) | 66.47 (+2.3%) |
| call-split-fuse-shared | moved | 68.21 | 68.61 (+0.6%) | 70.14 (+2.8%) |
| call-split-fuse-distinct | retained | 53.19 | 53.53 (+0.6%) | 54.30 (+2.1%) |
| call-split-fuse-distinct | moved | 54.01 | 53.91 (-0.2%) | 55.10 (+2.0%) |
| add-tree-shared | retained | 54.15 | 54.25 (+0.2%) | 54.11 (-0.1%) |
| add-tree-shared | moved | 59.13 | 59.20 (+0.1%) | 59.39 (+0.4%) |
| add-tree-distinct | retained | 43.33 | 43.51 (+0.4%) | 43.57 (+0.6%) |
| add-tree-distinct | moved | 47.66 | 47.90 (+0.5%) | 48.23 (+1.2%) |
| add-tree-intimm-shared | retained | 19.49 | 19.56 (+0.3%) | 19.53 (+0.2%) |
| add-tree-intimm-shared | moved | 20.51 | 20.55 (+0.2%) | 20.59 (+0.4%) |
| add-tree-intimm-distinct | retained | 15.52 | 15.50 (-0.1%) | 15.45 (-0.4%) |
| add-tree-intimm-distinct | moved | 15.57 | 15.47 (-0.7%) | 15.55 (-0.1%) |
| seq-16 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

