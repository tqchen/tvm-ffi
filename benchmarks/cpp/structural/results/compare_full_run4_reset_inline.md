### Provenance

| | |
| --- | --- |
| state old | tvm-ffi `719b61793638322308cea242d9208af9dffecede (old)` |
| state old engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state old-ri | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (old)` |
| state old-ri engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old-ri hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state gold | tvm-ffi `719b61793638322308cea242d9208af9dffecede (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_hook_override_gold.h sha256:20bd4475ef7e1096b54629015ce86d40d35dffd6f9f02a918947fb527d8360d4` |
| state gold-ri | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (gold)` |
| state gold-ri engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold-ri hook header | `tvm_hook_override_gold.h sha256:20bd4475ef7e1096b54629015ce86d40d35dffd6f9f02a918947fb527d8360d4` |
| state uc | tvm-ffi `719b61793638322308cea242d9208af9dffecede (uc)` |
| state uc engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| state uc-ri | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (uc)` |
| state uc-ri engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc-ri hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| what differs | unstated |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved old/old-ri/gold/gold-ri/uc/uc-ri |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.93 at start, 1.07 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved old/old-ri/gold/gold-ri/uc/uc-ri, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse and add-tree -- `floor` -- ns/node, old/old-ri/gold/gold-ri/uc/uc-ri against old

`subst` substitutes `Var`s.

| Case | own | old | old-ri | gold | gold-ri | uc | uc-ri |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 11.51 | 9.89 (-14.1%) | 8.79 (-23.6%) | 8.88 (-22.8%) | 6.17 (-46.4%) | 6.24 (-45.8%) |
| split-fuse-shared | moved | 11.52 | 9.94 (-13.7%) | 8.27 (-28.2%) | 8.75 (-24.0%) | 5.72 (-50.3%) | 5.99 (-48.0%) |
| split-fuse-distinct | retained | 9.20 | 7.91 (-14.0%) | 7.12 (-22.6%) | 7.53 (-18.1%) | 4.93 (-46.4%) | 5.09 (-44.7%) |
| split-fuse-distinct | moved | 9.40 | 7.96 (-15.3%) | 6.51 (-30.7%) | 6.92 (-26.3%) | 4.55 (-51.6%) | 4.92 (-47.6%) |
| call-split-fuse-shared | retained | 13.75 | 12.77 (-7.1%) | 8.34 (-39.4%) | 8.36 (-39.2%) | 6.44 (-53.2%) | 6.52 (-52.6%) |
| call-split-fuse-shared | moved | 13.93 | 12.88 (-7.6%) | 7.59 (-45.5%) | 7.85 (-43.7%) | 6.13 (-56.0%) | 6.14 (-55.9%) |
| call-split-fuse-distinct | retained | 10.80 | 10.00 (-7.4%) | 6.53 (-39.6%) | 6.64 (-38.5%) | 5.08 (-52.9%) | 5.08 (-53.0%) |
| call-split-fuse-distinct | moved | 10.82 | 10.11 (-6.6%) | 5.74 (-46.9%) | 6.10 (-43.6%) | 4.78 (-55.9%) | 4.72 (-56.4%) |
| add-tree-shared | retained | 11.50 | 9.97 (-13.3%) | 8.29 (-27.9%) | 9.13 (-20.6%) | 5.89 (-48.8%) | 6.26 (-45.5%) |
| add-tree-shared | moved | 11.55 | 9.92 (-14.1%) | 8.11 (-29.8%) | 9.09 (-21.3%) | 5.73 (-50.4%) | 6.24 (-45.9%) |
| add-tree-distinct | retained | 9.19 | 7.99 (-13.1%) | 6.82 (-25.8%) | 7.58 (-17.5%) | 4.72 (-48.7%) | 4.96 (-46.0%) |
| add-tree-distinct | moved | 9.21 | 7.95 (-13.7%) | 6.19 (-32.8%) | 7.23 (-21.5%) | 4.51 (-51.1%) | 5.19 (-43.6%) |
| add-tree-intimm-shared | retained | 10.76 | 10.23 (-4.9%) | 7.78 (-27.7%) | 7.99 (-25.8%) | 4.62 (-57.1%) | 5.08 (-52.8%) |
| add-tree-intimm-shared | moved | 9.58 | 9.56 (-0.3%) | 7.19 (-24.9%) | 7.78 (-18.8%) | 4.47 (-53.4%) | 5.31 (-44.6%) |
| add-tree-intimm-distinct | retained | 8.73 | 8.15 (-6.7%) | 6.04 (-30.9%) | 6.39 (-26.9%) | 3.76 (-57.0%) | 4.07 (-53.5%) |
| add-tree-intimm-distinct | moved | 7.68 | 7.63 (-0.7%) | 5.49 (-28.5%) | 6.09 (-20.6%) | 3.66 (-52.4%) | 4.23 (-44.9%) |

#### Expr-level -- split/fuse and add-tree -- `identity` -- ns/node, old/old-ri/gold/gold-ri/uc/uc-ri against old

`subst` substitutes `Var`s.

| Case | own | old | old-ri | gold | gold-ri | uc | uc-ri |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 32.21 | 29.72 (-7.7%) | 28.93 (-10.2%) | 26.84 (-16.7%) | 25.85 (-19.7%) | 22.85 (-29.1%) |
| split-fuse-shared | moved | 32.15 | 29.32 (-8.8%) | 28.90 (-10.1%) | 27.03 (-15.9%) | 25.79 (-19.8%) | 23.10 (-28.1%) |
| split-fuse-distinct | retained | 25.75 | 23.73 (-7.8%) | 23.19 (-10.0%) | 21.50 (-16.5%) | 20.72 (-19.5%) | 18.27 (-29.1%) |
| split-fuse-distinct | moved | 24.92 | 23.04 (-7.5%) | 23.29 (-6.5%) | 21.74 (-12.8%) | 20.68 (-17.0%) | 18.89 (-24.2%) |
| call-split-fuse-shared | retained | 29.43 | 27.87 (-5.3%) | 22.71 (-22.8%) | 21.15 (-28.1%) | 21.20 (-28.0%) | 20.58 (-30.1%) |
| call-split-fuse-shared | moved | 29.25 | 27.93 (-4.5%) | 23.90 (-18.3%) | 21.49 (-26.5%) | 22.24 (-24.0%) | 20.41 (-30.2%) |
| call-split-fuse-distinct | retained | 23.21 | 21.76 (-6.3%) | 17.76 (-23.5%) | 16.70 (-28.1%) | 16.62 (-28.4%) | 16.05 (-30.8%) |
| call-split-fuse-distinct | moved | 22.50 | 21.80 (-3.1%) | 18.02 (-19.9%) | 17.15 (-23.8%) | 16.71 (-25.7%) | 15.53 (-31.0%) |
| add-tree-shared | retained | 31.86 | 29.54 (-7.3%) | 29.29 (-8.0%) | 26.91 (-15.5%) | 26.08 (-18.1%) | 22.89 (-28.2%) |
| add-tree-shared | moved | 31.81 | 29.40 (-7.6%) | 29.18 (-8.3%) | 27.12 (-14.8%) | 25.87 (-18.7%) | 23.23 (-27.0%) |
| add-tree-distinct | retained | 25.40 | 23.60 (-7.1%) | 23.46 (-7.6%) | 21.54 (-15.2%) | 20.88 (-17.8%) | 18.32 (-27.9%) |
| add-tree-distinct | moved | 24.87 | 23.07 (-7.3%) | 23.23 (-6.6%) | 22.09 (-11.2%) | 20.76 (-16.5%) | 18.92 (-23.9%) |
| add-tree-intimm-shared | retained | 12.34 | 12.72 (+3.1%) | 10.16 (-17.7%) | 11.02 (-10.7%) | 8.42 (-31.7%) | 7.53 (-39.0%) |
| add-tree-intimm-shared | moved | 12.15 | 12.09 (-0.4%) | 10.28 (-15.3%) | 10.81 (-11.0%) | 7.96 (-34.5%) | 8.08 (-33.5%) |
| add-tree-intimm-distinct | retained | 9.91 | 10.16 (+2.5%) | 8.16 (-17.7%) | 8.46 (-14.6%) | 6.76 (-31.8%) | 6.12 (-38.2%) |
| add-tree-intimm-distinct | moved | 9.44 | 9.50 (+0.6%) | 8.22 (-12.9%) | 8.68 (-8.1%) | 6.20 (-34.3%) | 6.96 (-26.3%) |

#### Expr-level -- split/fuse and add-tree -- `subst` -- ns/node, old/old-ri/gold/gold-ri/uc/uc-ri against old

`subst` substitutes `Var`s.

| Case | own | old | old-ri | gold | gold-ri | uc | uc-ri |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 38.46 | 34.30 (-10.8%) | 37.95 (-1.3%) | 35.12 (-8.7%) | 35.83 (-6.8%) | 32.10 (-16.5%) |
| split-fuse-shared | moved | 34.49 | 31.86 (-7.6%) | 33.38 (-3.2%) | 29.83 (-13.5%) | 30.55 (-11.4%) | 27.47 (-20.4%) |
| split-fuse-distinct | retained | 30.67 | 27.35 (-10.8%) | 30.45 (-0.7%) | 27.91 (-9.0%) | 28.45 (-7.3%) | 25.58 (-16.6%) |
| split-fuse-distinct | moved | 25.99 | 24.24 (-6.7%) | 24.35 (-6.3%) | 21.92 (-15.7%) | 22.17 (-14.7%) | 20.02 (-23.0%) |
| call-split-fuse-shared | retained | 38.62 | 36.17 (-6.3%) | 38.15 (-1.2%) | 36.71 (-4.9%) | 37.48 (-2.9%) | 36.63 (-5.1%) |
| call-split-fuse-shared | moved | 31.73 | 29.46 (-7.2%) | 28.35 (-10.6%) | 27.09 (-14.6%) | 27.40 (-13.6%) | 25.36 (-20.1%) |
| call-split-fuse-distinct | retained | 31.19 | 28.45 (-8.8%) | 30.55 (-2.1%) | 28.71 (-8.0%) | 30.75 (-1.4%) | 28.85 (-7.5%) |
| call-split-fuse-distinct | moved | 23.20 | 21.57 (-7.0%) | 19.05 (-17.9%) | 17.74 (-23.5%) | 18.15 (-21.8%) | 16.20 (-30.2%) |
| add-tree-shared | retained | 38.26 | 34.93 (-8.7%) | 37.92 (-0.9%) | 34.67 (-9.4%) | 35.40 (-7.5%) | 31.94 (-16.5%) |
| add-tree-shared | moved | 34.74 | 32.16 (-7.4%) | 33.15 (-4.6%) | 30.00 (-13.6%) | 30.81 (-11.3%) | 26.94 (-22.5%) |
| add-tree-distinct | retained | 30.30 | 27.25 (-10.1%) | 30.70 (+1.3%) | 27.73 (-8.5%) | 28.49 (-6.0%) | 25.57 (-15.6%) |
| add-tree-distinct | moved | 25.95 | 24.28 (-6.4%) | 24.15 (-6.9%) | 21.91 (-15.5%) | 22.36 (-13.8%) | 20.11 (-22.5%) |
| add-tree-intimm-shared | retained | 13.49 | 12.70 (-5.8%) | 10.93 (-19.0%) | 10.88 (-19.3%) | 8.09 (-40.1%) | 7.14 (-47.0%) |
| add-tree-intimm-shared | moved | 12.56 | 12.33 (-1.9%) | 10.55 (-16.0%) | 10.70 (-14.9%) | 7.32 (-41.7%) | 7.45 (-40.7%) |
| add-tree-intimm-distinct | retained | 10.69 | 10.25 (-4.1%) | 8.79 (-17.8%) | 8.70 (-18.6%) | 6.53 (-38.9%) | 5.72 (-46.5%) |
| add-tree-intimm-distinct | moved | 9.36 | 9.29 (-0.7%) | 8.11 (-13.3%) | 8.58 (-8.3%) | 5.94 (-36.5%) | 6.41 (-31.5%) |

#### Stmt-level -- seq -- `floor` -- ns/node, old/old-ri/gold/gold-ri/uc/uc-ri against old

`swap` swaps two whole `Evaluate` nodes.

| Case | own | old | old-ri | gold | gold-ri | uc | uc-ri |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seq-16 | retained | 6.00 | 6.15 (+2.5%) | 5.60 (-6.6%) | 5.74 (-4.2%) | 4.31 (-28.2%) | 4.59 (-23.5%) |
| seq-16 | moved | 4.39 | 4.68 (+6.7%) | 4.55 (+3.7%) | 4.66 (+6.2%) | 3.29 (-24.9%) | 3.21 (-26.8%) |
| seq-256 | retained | 6.94 | 6.87 (-1.0%) | 5.79 (-16.6%) | 5.97 (-14.0%) | 4.24 (-38.9%) | 4.32 (-37.7%) |
| seq-256 | moved | 4.61 | 4.72 (+2.6%) | 4.61 (+0.1%) | 4.69 (+1.9%) | 3.28 (-28.7%) | 3.20 (-30.5%) |
| seq-16384 | retained | 7.06 | 7.02 (-0.5%) | 5.86 (-17.0%) | 6.01 (-14.9%) | 4.26 (-39.7%) | 4.35 (-38.4%) |
| seq-16384 | moved | 4.59 | 4.66 (+1.5%) | 4.60 (+0.4%) | 4.69 (+2.3%) | 3.21 (-30.1%) | 3.19 (-30.5%) |

#### Stmt-level -- seq -- `identity` -- ns/node, old/old-ri/gold/gold-ri/uc/uc-ri against old

`swap` swaps two whole `Evaluate` nodes.

| Case | own | old | old-ri | gold | gold-ri | uc | uc-ri |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seq-16 | retained | 12.16 | 11.64 (-4.2%) | 10.70 (-12.0%) | 10.26 (-15.6%) | 9.74 (-19.9%) | 8.86 (-27.2%) |
| seq-16 | moved | 10.02 | 9.66 (-3.6%) | 11.50 (+14.8%) | 10.98 (+9.5%) | 9.14 (-8.8%) | 8.36 (-16.6%) |
| seq-256 | retained | 12.32 | 11.85 (-3.8%) | 10.21 (-17.2%) | 9.77 (-20.7%) | 9.45 (-23.3%) | 8.57 (-30.5%) |
| seq-256 | moved | 9.48 | 9.47 (-0.1%) | 10.98 (+15.9%) | 10.43 (+10.0%) | 8.68 (-8.4%) | 7.86 (-17.1%) |
| seq-16384 | retained | 12.50 | 11.94 (-4.5%) | 10.43 (-16.6%) | 9.95 (-20.4%) | 9.25 (-26.0%) | 8.49 (-32.1%) |
| seq-16384 | moved | 9.41 | 9.45 (+0.4%) | 10.97 (+16.5%) | 10.29 (+9.4%) | 8.69 (-7.7%) | 7.91 (-16.0%) |

#### Stmt-level -- seq -- `swap` -- ns/node, old/old-ri/gold/gold-ri/uc/uc-ri against old

`swap` swaps two whole `Evaluate` nodes.

| Case | own | old | old-ri | gold | gold-ri | uc | uc-ri |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seq-16 | retained | 16.55 | 14.77 (-10.7%) | 14.23 (-14.0%) | 13.96 (-15.7%) | 14.62 (-11.6%) | 13.27 (-19.8%) |
| seq-16 | moved | 12.50 | 11.96 (-4.4%) | 12.99 (+3.9%) | 12.46 (-0.4%) | 11.81 (-5.5%) | 12.31 (-1.6%) |
| seq-256 | retained | 16.01 | 13.65 (-14.7%) | 13.40 (-16.3%) | 12.41 (-22.5%) | 13.95 (-12.9%) | 11.88 (-25.8%) |
| seq-256 | moved | 11.34 | 10.38 (-8.4%) | 12.19 (+7.5%) | 11.45 (+1.0%) | 11.08 (-2.3%) | 11.28 (-0.5%) |
| seq-16384 | retained | 15.98 | 13.41 (-16.0%) | 13.39 (-16.2%) | 12.18 (-23.8%) | 13.75 (-13.9%) | 11.62 (-27.3%) |
| seq-16384 | moved | 11.23 | 10.08 (-10.3%) | 12.11 (+7.8%) | 11.36 (+1.2%) | 10.95 (-2.5%) | 11.01 (-1.9%) |

### Supporting

#### `never` -- ns/node, old/old-ri/gold/gold-ri/uc/uc-ri against old

| Case | own | old | old-ri | gold | gold-ri | uc | uc-ri |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 23.17 | 20.74 (-10.5%) | 21.07 (-9.1%) | 18.84 (-18.7%) | 17.94 (-22.6%) | 16.27 (-29.8%) |
| split-fuse-shared | moved | 21.94 | 19.34 (-11.8%) | 19.65 (-10.4%) | 17.28 (-21.2%) | 16.19 (-26.2%) | 14.33 (-34.7%) |
| split-fuse-distinct | retained | 17.33 | 15.61 (-9.9%) | 16.99 (-1.9%) | 14.99 (-13.5%) | 13.30 (-23.3%) | 12.79 (-26.2%) |
| split-fuse-distinct | moved | 17.26 | 15.08 (-12.7%) | 15.28 (-11.5%) | 13.57 (-21.4%) | 12.64 (-26.8%) | 11.13 (-35.5%) |
| call-split-fuse-shared | retained | 21.74 | 20.10 (-7.5%) | 16.82 (-22.6%) | 15.74 (-27.6%) | 14.67 (-32.5%) | 13.24 (-39.1%) |
| call-split-fuse-shared | moved | 22.19 | 20.39 (-8.1%) | 16.81 (-24.2%) | 15.32 (-31.0%) | 15.07 (-32.1%) | 13.42 (-39.5%) |
| call-split-fuse-distinct | retained | 17.21 | 15.87 (-7.8%) | 13.03 (-24.3%) | 11.71 (-31.9%) | 11.44 (-33.5%) | 10.39 (-39.6%) |
| call-split-fuse-distinct | moved | 17.36 | 15.95 (-8.1%) | 12.97 (-25.3%) | 11.70 (-32.6%) | 11.90 (-31.5%) | 9.62 (-44.6%) |
| add-tree-shared | retained | 21.27 | 19.26 (-9.4%) | 21.35 (+0.4%) | 18.38 (-13.6%) | 16.74 (-21.3%) | 14.71 (-30.8%) |
| add-tree-shared | moved | 21.81 | 19.34 (-11.3%) | 19.63 (-10.0%) | 17.48 (-19.8%) | 16.05 (-26.4%) | 14.26 (-34.6%) |
| add-tree-distinct | retained | 16.95 | 15.38 (-9.3%) | 17.05 (+0.6%) | 14.66 (-13.5%) | 13.24 (-21.9%) | 11.72 (-30.9%) |
| add-tree-distinct | moved | 17.07 | 15.31 (-10.3%) | 15.02 (-12.0%) | 13.66 (-20.0%) | 12.75 (-25.3%) | 11.23 (-34.2%) |
| add-tree-intimm-shared | retained | 12.57 | 12.47 (-0.8%) | 11.07 (-11.9%) | 11.09 (-11.8%) | 7.63 (-39.3%) | 7.56 (-39.8%) |
| add-tree-intimm-shared | moved | 11.96 | 12.18 (+1.8%) | 10.36 (-13.4%) | 10.71 (-10.5%) | 7.40 (-38.2%) | 7.74 (-35.3%) |
| add-tree-intimm-distinct | retained | 10.14 | 9.94 (-2.0%) | 8.92 (-12.0%) | 8.87 (-12.5%) | 6.13 (-39.5%) | 6.02 (-40.6%) |
| add-tree-intimm-distinct | moved | 9.32 | 10.55 (+13.2%) | 8.02 (-13.9%) | 8.49 (-8.9%) | 5.93 (-36.3%) | 6.29 (-32.5%) |
| seq-16 | retained | 6.76 | 7.50 (+11.0%) | 6.83 (+1.1%) | 7.25 (+7.3%) | 5.93 (-12.2%) | 5.53 (-18.1%) |
| seq-16 | moved | 5.84 | 5.80 (-0.7%) | 6.05 (+3.6%) | 6.24 (+6.9%) | 4.78 (-18.1%) | 4.43 (-24.2%) |
| seq-256 | retained | 6.90 | 7.50 (+8.6%) | 6.33 (-8.3%) | 6.97 (+1.0%) | 5.52 (-20.0%) | 5.25 (-23.9%) |
| seq-256 | moved | 5.51 | 5.69 (+3.3%) | 5.58 (+1.3%) | 5.84 (+6.1%) | 4.27 (-22.4%) | 4.00 (-27.4%) |
| seq-16384 | retained | 6.92 | 7.30 (+5.5%) | 6.36 (-8.0%) | 6.94 (+0.3%) | 5.32 (-23.0%) | 5.02 (-27.5%) |
| seq-16384 | moved | 5.59 | 5.76 (+3.0%) | 5.49 (-1.8%) | 5.84 (+4.6%) | 4.26 (-23.8%) | 3.88 (-30.6%) |

#### `functor` -- ns/node, old/old-ri/gold/gold-ri/uc/uc-ri against old

| Case | own | old | old-ri | gold | gold-ri | uc | uc-ri |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.65 | 31.54 (-11.5%) | 35.45 (-0.6%) | 31.52 (-11.6%) | 35.37 (-0.8%) | 31.30 (-12.2%) |
| split-fuse-shared | moved | 40.64 | 35.00 (-13.9%) | 40.61 (-0.1%) | 36.00 (-11.4%) | 40.74 (+0.3%) | 34.90 (-14.1%) |
| split-fuse-distinct | retained | 28.54 | 25.25 (-11.5%) | 28.37 (-0.6%) | 25.33 (-11.2%) | 28.63 (+0.3%) | 25.23 (-11.6%) |
| split-fuse-distinct | moved | 33.42 | 28.66 (-14.2%) | 33.25 (-0.5%) | 28.58 (-14.5%) | 33.44 (+0.1%) | 28.62 (-14.4%) |
| call-split-fuse-shared | retained | 56.96 | 45.80 (-19.6%) | 57.67 (+1.3%) | 45.27 (-20.5%) | 57.94 (+1.7%) | 45.46 (-20.2%) |
| call-split-fuse-shared | moved | 60.18 | 48.71 (-19.1%) | 60.60 (+0.7%) | 48.61 (-19.2%) | 61.12 (+1.6%) | 48.99 (-18.6%) |
| call-split-fuse-distinct | retained | 46.09 | 37.26 (-19.2%) | 46.52 (+0.9%) | 36.62 (-20.5%) | 46.81 (+1.6%) | 36.86 (-20.0%) |
| call-split-fuse-distinct | moved | 47.40 | 38.66 (-18.4%) | 47.52 (+0.3%) | 38.48 (-18.8%) | 48.19 (+1.7%) | 38.31 (-19.2%) |
| add-tree-shared | retained | 36.01 | 31.87 (-11.5%) | 35.99 (-0.1%) | 31.98 (-11.2%) | 35.95 (-0.2%) | 31.83 (-11.6%) |
| add-tree-shared | moved | 42.16 | 35.34 (-16.2%) | 42.19 (+0.1%) | 36.00 (-14.6%) | 41.54 (-1.5%) | 35.47 (-15.9%) |
| add-tree-distinct | retained | 28.96 | 25.11 (-13.3%) | 28.92 (-0.2%) | 25.11 (-13.3%) | 28.83 (-0.5%) | 24.71 (-14.7%) |
| add-tree-distinct | moved | 33.79 | 28.54 (-15.5%) | 33.58 (-0.6%) | 29.21 (-13.5%) | 33.92 (+0.4%) | 28.54 (-15.5%) |
| add-tree-intimm-shared | retained | 19.15 | 17.28 (-9.8%) | 19.18 (+0.2%) | 17.59 (-8.2%) | 19.17 (+0.1%) | 17.32 (-9.6%) |
| add-tree-intimm-shared | moved | 20.02 | 17.69 (-11.7%) | 20.06 (+0.2%) | 17.91 (-10.5%) | 19.91 (-0.6%) | 17.88 (-10.7%) |
| add-tree-intimm-distinct | retained | 15.30 | 13.82 (-9.7%) | 15.28 (-0.2%) | 13.98 (-8.7%) | 15.30 (+0.0%) | 13.85 (-9.5%) |
| add-tree-intimm-distinct | moved | 15.33 | 13.80 (-10.0%) | 15.15 (-1.2%) | 14.10 (-8.1%) | 15.34 (+0.1%) | 13.87 (-9.6%) |
| seq-16 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |

#### `old` -- ns/node, old/old-ri/gold/gold-ri/uc/uc-ri against old

| Case | own | old | old-ri | gold | gold-ri | uc | uc-ri |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 53.78 | 47.79 (-11.1%) | 54.61 (+1.5%) | 47.78 (-11.2%) | 53.99 (+0.4%) | 48.00 (-10.8%) |
| split-fuse-shared | moved | 58.31 | 51.82 (-11.1%) | 58.89 (+1.0%) | 51.92 (-11.0%) | 58.85 (+0.9%) | 51.49 (-11.7%) |
| split-fuse-distinct | retained | 43.34 | 38.41 (-11.4%) | 43.95 (+1.4%) | 38.43 (-11.3%) | 43.57 (+0.5%) | 38.31 (-11.6%) |
| split-fuse-distinct | moved | 47.35 | 41.56 (-12.2%) | 48.03 (+1.4%) | 41.83 (-11.7%) | 47.74 (+0.8%) | 41.66 (-12.0%) |
| call-split-fuse-shared | retained | 65.86 | 53.59 (-18.6%) | 66.13 (+0.4%) | 54.36 (-17.5%) | 66.28 (+0.6%) | 53.43 (-18.9%) |
| call-split-fuse-shared | moved | 68.30 | 56.99 (-16.6%) | 68.75 (+0.7%) | 57.98 (-15.1%) | 69.64 (+2.0%) | 57.02 (-16.5%) |
| call-split-fuse-distinct | retained | 53.20 | 44.32 (-16.7%) | 53.44 (+0.4%) | 44.86 (-15.7%) | 54.10 (+1.7%) | 44.24 (-16.9%) |
| call-split-fuse-distinct | moved | 53.75 | 45.09 (-16.1%) | 54.33 (+1.1%) | 45.69 (-15.0%) | 54.90 (+2.2%) | 45.04 (-16.2%) |
| add-tree-shared | retained | 54.34 | 47.55 (-12.5%) | 54.40 (+0.1%) | 48.03 (-11.6%) | 54.35 (+0.0%) | 47.92 (-11.8%) |
| add-tree-shared | moved | 59.06 | 51.78 (-12.3%) | 59.26 (+0.4%) | 52.70 (-10.8%) | 59.20 (+0.2%) | 52.04 (-11.9%) |
| add-tree-distinct | retained | 43.18 | 37.75 (-12.6%) | 43.19 (+0.0%) | 38.58 (-10.6%) | 43.40 (+0.5%) | 38.16 (-11.6%) |
| add-tree-distinct | moved | 47.94 | 41.72 (-13.0%) | 47.68 (-0.5%) | 42.28 (-11.8%) | 48.43 (+1.0%) | 42.03 (-12.3%) |
| add-tree-intimm-shared | retained | 19.51 | 17.90 (-8.2%) | 19.58 (+0.4%) | 18.31 (-6.1%) | 19.56 (+0.3%) | 18.01 (-7.7%) |
| add-tree-intimm-shared | moved | 20.57 | 18.39 (-10.6%) | 20.62 (+0.3%) | 18.63 (-9.5%) | 20.51 (-0.3%) | 18.46 (-10.3%) |
| add-tree-intimm-distinct | retained | 15.66 | 14.25 (-9.0%) | 15.58 (-0.5%) | 14.32 (-8.6%) | 15.63 (-0.2%) | 14.38 (-8.2%) |
| add-tree-intimm-distinct | moved | 15.69 | 14.29 (-8.9%) | 15.61 (-0.5%) | 14.62 (-6.8%) | 15.72 (+0.2%) | 14.35 (-8.5%) |
| seq-16 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-256 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | retained | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |
| seq-16384 | moved | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] | n/a[^cmp-seq] |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

