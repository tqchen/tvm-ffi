### Provenance

| | |
| --- | --- |
| state old | tvm-ffi `719b61793638322308cea242d9208af9dffecede (old)` |
| state old engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state old-ri | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (old)` |
| state old-ri engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old-ri hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state old-rim | tvm-ffi `c9bb47d+rim (old)` |
| state old-rim engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old-rim hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state old-rimd | tvm-ffi `c9bb47d+rimd (old)` |
| state old-rimd engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old-rimd hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state gold | tvm-ffi `719b61793638322308cea242d9208af9dffecede (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_hook_override_gold.h sha256:20bd4475ef7e1096b54629015ce86d40d35dffd6f9f02a918947fb527d8360d4` |
| state gold-ri | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (gold)` |
| state gold-ri engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold-ri hook header | `tvm_hook_override_gold.h sha256:20bd4475ef7e1096b54629015ce86d40d35dffd6f9f02a918947fb527d8360d4` |
| state gold-rim | tvm-ffi `c9bb47d+rim (gold)` |
| state gold-rim engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold-rim hook header | `tvm_hook_override_gold.h sha256:20bd4475ef7e1096b54629015ce86d40d35dffd6f9f02a918947fb527d8360d4` |
| state gold-rimd | tvm-ffi `c9bb47d+rimd (gold)` |
| state gold-rimd engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold-rimd hook header | `tvm_hook_override_gold.h sha256:20bd4475ef7e1096b54629015ce86d40d35dffd6f9f02a918947fb527d8360d4` |
| state uc | tvm-ffi `719b61793638322308cea242d9208af9dffecede (uc)` |
| state uc engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| state uc-ri | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (uc)` |
| state uc-ri engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc-ri hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| state uc-rim | tvm-ffi `c9bb47d+rim (uc)` |
| state uc-rim engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc-rim hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| state uc-rimd | tvm-ffi `c9bb47d+rimd (uc)` |
| state uc-rimd engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc-rimd hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| what differs | unstated |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved old/old-ri/old-rim/old-rimd/gold/gold-ri/gold-rim/gold-rimd/uc/uc-ri/uc-rim/uc-rimd |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 1.44 at start, 1.14 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved old/old-ri/old-rim/old-rimd/gold/gold-ri/gold-rim/gold-rimd/uc/uc-ri/uc-rim/uc-rimd, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse and add-tree -- `floor` -- ns/node, old/old-ri/old-rim/old-rimd/gold/gold-ri/gold-rim/gold-rimd/uc/uc-ri/uc-rim/uc-rimd against old

`subst` substitutes `Var`s.

| Case | own | old | old-ri | old-rim | old-rimd | gold | gold-ri | gold-rim | gold-rimd | uc | uc-ri | uc-rim | uc-rimd |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 11.40 | 9.87 (-13.4%) | 10.05 (-11.8%) | 11.21 (-1.7%) | 8.79 (-22.9%) | 8.85 (-22.3%) | 8.62 (-24.4%) | 8.33 (-26.9%) | 5.58 (-51.0%) | 5.93 (-48.0%) | 5.95 (-47.8%) | 5.93 (-48.0%) |
| split-fuse-shared | moved | 11.48 | 9.91 (-13.7%) | 9.94 (-13.4%) | 11.35 (-1.2%) | 8.64 (-24.7%) | 9.00 (-21.6%) | 9.08 (-20.9%) | 8.47 (-26.2%) | 5.59 (-51.3%) | 6.09 (-47.0%) | 5.96 (-48.1%) | 5.77 (-49.8%) |
| split-fuse-distinct | retained | 9.16 | 7.92 (-13.5%) | 8.02 (-12.4%) | 8.94 (-2.3%) | 7.52 (-17.8%) | 7.70 (-15.9%) | 7.67 (-16.2%) | 7.10 (-22.5%) | 4.53 (-50.5%) | 4.91 (-46.4%) | 5.00 (-45.4%) | 4.97 (-45.7%) |
| split-fuse-distinct | moved | 9.43 | 7.98 (-15.4%) | 7.97 (-15.5%) | 9.16 (-2.8%) | 6.88 (-27.0%) | 7.17 (-24.0%) | 7.19 (-23.7%) | 6.76 (-28.3%) | 4.63 (-50.9%) | 4.89 (-48.2%) | 4.69 (-50.2%) | 4.64 (-50.8%) |

#### Expr-level -- split/fuse and add-tree -- `identity` -- ns/node, old/old-ri/old-rim/old-rimd/gold/gold-ri/gold-rim/gold-rimd/uc/uc-ri/uc-rim/uc-rimd against old

`subst` substitutes `Var`s.

| Case | own | old | old-ri | old-rim | old-rimd | gold | gold-ri | gold-rim | gold-rimd | uc | uc-ri | uc-rim | uc-rimd |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 31.97 | 29.29 (-8.4%) | 29.90 (-6.5%) | 31.26 (-2.2%) | 29.05 (-9.1%) | 26.55 (-16.9%) | 28.43 (-11.1%) | 27.77 (-13.1%) | 25.89 (-19.0%) | 22.95 (-28.2%) | 25.99 (-18.7%) | 25.07 (-21.6%) |
| split-fuse-shared | moved | 32.58 | 29.11 (-10.7%) | 29.78 (-8.6%) | 30.99 (-4.9%) | 29.08 (-10.7%) | 26.42 (-18.9%) | 28.30 (-13.1%) | 27.46 (-15.7%) | 26.02 (-20.1%) | 25.53 (-21.6%) | 25.92 (-20.5%) | 24.33 (-25.3%) |
| split-fuse-distinct | retained | 25.60 | 23.42 (-8.5%) | 23.97 (-6.3%) | 25.00 (-2.3%) | 23.33 (-8.9%) | 21.28 (-16.9%) | 22.77 (-11.0%) | 22.52 (-12.0%) | 20.74 (-19.0%) | 18.37 (-28.2%) | 20.64 (-19.4%) | 20.08 (-21.5%) |
| split-fuse-distinct | moved | 25.49 | 22.96 (-9.9%) | 23.71 (-7.0%) | 24.40 (-4.3%) | 23.14 (-9.2%) | 21.51 (-15.6%) | 22.89 (-10.2%) | 22.43 (-12.0%) | 20.97 (-17.7%) | 20.67 (-18.9%) | 20.66 (-19.0%) | 19.44 (-23.7%) |

#### Expr-level -- split/fuse and add-tree -- `subst` -- ns/node, old/old-ri/old-rim/old-rimd/gold/gold-ri/gold-rim/gold-rimd/uc/uc-ri/uc-rim/uc-rimd against old

`subst` substitutes `Var`s.

| Case | own | old | old-ri | old-rim | old-rimd | gold | gold-ri | gold-rim | gold-rimd | uc | uc-ri | uc-rim | uc-rimd |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 38.33 | 33.54 (-12.5%) | 33.48 (-12.6%) | 35.52 (-7.3%) | 38.47 (+0.4%) | 34.11 (-11.0%) | 34.72 (-9.4%) | 33.88 (-11.6%) | 35.37 (-7.7%) | 33.26 (-13.2%) | 32.08 (-16.3%) | 31.90 (-16.8%) |
| split-fuse-shared | moved | 34.95 | 31.96 (-8.5%) | 30.93 (-11.5%) | 33.71 (-3.5%) | 33.19 (-5.0%) | 29.49 (-15.6%) | 30.70 (-12.2%) | 30.42 (-13.0%) | 30.43 (-12.9%) | 27.98 (-19.9%) | 27.45 (-21.5%) | 27.89 (-20.2%) |
| split-fuse-distinct | retained | 30.76 | 26.89 (-12.6%) | 26.85 (-12.7%) | 28.52 (-7.3%) | 30.75 (-0.0%) | 27.14 (-11.8%) | 27.90 (-9.3%) | 27.31 (-11.2%) | 28.39 (-7.7%) | 26.39 (-14.2%) | 25.78 (-16.2%) | 25.33 (-17.6%) |
| split-fuse-distinct | moved | 25.75 | 24.21 (-6.0%) | 23.48 (-8.8%) | 25.68 (-0.3%) | 24.11 (-6.4%) | 21.91 (-14.9%) | 23.02 (-10.6%) | 22.19 (-13.8%) | 22.24 (-13.6%) | 20.28 (-21.3%) | 20.59 (-20.1%) | 20.65 (-19.8%) |

### Supporting

#### `never` -- ns/node, old/old-ri/old-rim/old-rimd/gold/gold-ri/gold-rim/gold-rimd/uc/uc-ri/uc-rim/uc-rimd against old

| Case | own | old | old-ri | old-rim | old-rimd | gold | gold-ri | gold-rim | gold-rimd | uc | uc-ri | uc-rim | uc-rimd |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 23.12 | 20.33 (-12.1%) | 20.72 (-10.4%) | 21.00 (-9.2%) | 21.44 (-7.3%) | 19.47 (-15.8%) | 19.30 (-16.5%) | 19.52 (-15.6%) | 18.42 (-20.4%) | 16.05 (-30.6%) | 15.91 (-31.2%) | 15.92 (-31.1%) |
| split-fuse-shared | moved | 21.88 | 19.41 (-11.3%) | 19.24 (-12.1%) | 19.39 (-11.4%) | 19.49 (-10.9%) | 17.84 (-18.5%) | 17.32 (-20.9%) | 17.50 (-20.0%) | 16.16 (-26.2%) | 14.13 (-35.4%) | 14.35 (-34.4%) | 13.49 (-38.3%) |
| split-fuse-distinct | retained | 17.52 | 16.28 (-7.1%) | 16.47 (-6.0%) | 16.70 (-4.7%) | 17.15 (-2.1%) | 15.57 (-11.1%) | 15.63 (-10.8%) | 15.53 (-11.4%) | 13.48 (-23.1%) | 12.76 (-27.2%) | 13.13 (-25.0%) | 12.91 (-26.3%) |
| split-fuse-distinct | moved | 17.33 | 15.37 (-11.3%) | 15.13 (-12.7%) | 15.43 (-11.0%) | 15.01 (-13.4%) | 13.76 (-20.6%) | 13.66 (-21.2%) | 13.82 (-20.2%) | 12.36 (-28.7%) | 11.06 (-36.2%) | 11.24 (-35.1%) | 10.53 (-39.2%) |

#### `functor` -- ns/node, old/old-ri/old-rim/old-rimd/gold/gold-ri/gold-rim/gold-rimd/uc/uc-ri/uc-rim/uc-rimd against old

| Case | own | old | old-ri | old-rim | old-rimd | gold | gold-ri | gold-rim | gold-rimd | uc | uc-ri | uc-rim | uc-rimd |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.38 | 31.61 (-10.6%) | 31.27 (-11.6%) | 30.26 (-14.5%) | 35.30 (-0.2%) | 31.79 (-10.1%) | 32.11 (-9.2%) | 30.44 (-13.9%) | 35.79 (+1.2%) | 31.76 (-10.2%) | 31.37 (-11.3%) | 30.45 (-13.9%) |
| split-fuse-shared | moved | 40.67 | 34.47 (-15.3%) | 34.63 (-14.9%) | 33.24 (-18.3%) | 40.45 (-0.5%) | 35.35 (-13.1%) | 35.42 (-12.9%) | 33.51 (-17.6%) | 40.28 (-1.0%) | 35.55 (-12.6%) | 34.54 (-15.1%) | 33.61 (-17.4%) |
| split-fuse-distinct | retained | 28.70 | 25.13 (-12.5%) | 25.06 (-12.7%) | 23.81 (-17.0%) | 28.62 (-0.3%) | 25.27 (-12.0%) | 25.33 (-11.8%) | 24.27 (-15.4%) | 28.62 (-0.3%) | 25.31 (-11.8%) | 24.86 (-13.4%) | 24.35 (-15.2%) |
| split-fuse-distinct | moved | 33.96 | 28.50 (-16.1%) | 28.13 (-17.2%) | 27.41 (-19.3%) | 33.75 (-0.6%) | 28.87 (-15.0%) | 28.93 (-14.8%) | 27.46 (-19.1%) | 33.08 (-2.6%) | 28.84 (-15.1%) | 28.49 (-16.1%) | 27.72 (-18.4%) |

#### `old` -- ns/node, old/old-ri/old-rim/old-rimd/gold/gold-ri/gold-rim/gold-rimd/uc/uc-ri/uc-rim/uc-rimd against old

| Case | own | old | old-ri | old-rim | old-rimd | gold | gold-ri | gold-rim | gold-rimd | uc | uc-ri | uc-rim | uc-rimd |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 53.77 | 47.75 (-11.2%) | 48.97 (-8.9%) | 46.81 (-12.9%) | 53.47 (-0.6%) | 48.41 (-10.0%) | 48.44 (-9.9%) | 46.97 (-12.7%) | 53.71 (-0.1%) | 47.86 (-11.0%) | 49.04 (-8.8%) | 47.38 (-11.9%) |
| split-fuse-shared | moved | 58.62 | 51.07 (-12.9%) | 52.48 (-10.5%) | 50.43 (-14.0%) | 58.13 (-0.8%) | 52.14 (-11.1%) | 52.31 (-10.8%) | 50.59 (-13.7%) | 58.42 (-0.4%) | 51.86 (-11.5%) | 52.12 (-11.1%) | 50.67 (-13.6%) |
| split-fuse-distinct | retained | 43.78 | 38.58 (-11.9%) | 39.65 (-9.4%) | 38.13 (-12.9%) | 43.43 (-0.8%) | 38.76 (-11.5%) | 39.09 (-10.7%) | 38.26 (-12.6%) | 44.00 (+0.5%) | 38.51 (-12.0%) | 39.34 (-10.1%) | 38.05 (-13.1%) |
| split-fuse-distinct | moved | 47.62 | 41.91 (-12.0%) | 42.42 (-10.9%) | 41.21 (-13.5%) | 47.50 (-0.3%) | 42.23 (-11.3%) | 42.63 (-10.5%) | 41.26 (-13.4%) | 48.16 (+1.1%) | 41.64 (-12.6%) | 42.52 (-10.7%) | 40.98 (-14.0%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

