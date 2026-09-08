### Provenance

| | |
| --- | --- |
| state old-ri | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (old)` |
| state old-ri engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old-ri hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state old-unchecked-ri | tvm-ffi `c9bb47d+764+unchecked (old)` |
| state old-unchecked-ri engine header | `structural_mutate_olduc.h sha256:3b5856c30cc152460026f8251d7965bcd352ac12c829114f30ba98403b4a3d7c` |
| state old-unchecked-ri hook header | `tvm_hook_override.h sha256:9d7c48991d79947d581f9203bb1303f96e5a5dcd63226a4dbc295af492c3b8a8` |
| state uc-chk-ri | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (uc)` |
| state uc-chk-ri engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc-chk-ri hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| state uc-unchecked-ri | tvm-ffi `c9bb47d+764+unchecked (uc)` |
| state uc-unchecked-ri engine header | `structural_mutate_ucuc.h sha256:110663684f4a5965a048b49b264ae26edc9b8e1fe5802b89db65a81aa7a6f7f3` |
| state uc-unchecked-ri hook header | `tvm_hook_override_ucuc.h sha256:f3564797e74d01f48438c84150b1b1b8053c9bb5c7fd4d5e9fe9c7c787c48bc3` |
| what differs | unstated |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved old-ri/old-unchecked-ri/uc-chk-ri/uc-unchecked-ri |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.41 at start, 1.08 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved old-ri/old-unchecked-ri/uc-chk-ri/uc-unchecked-ri, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse and add-tree -- `floor` -- ns/node, old-ri/old-unchecked-ri/uc-chk-ri/uc-unchecked-ri against old-ri

`subst` substitutes `Var`s.

| Case | own | old-ri | old-unchecked-ri | uc-chk-ri | uc-unchecked-ri |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 9.89 | 7.29 (-26.3%) | 5.86 (-40.7%) | 6.75 (-31.8%) |
| split-fuse-shared | moved | 9.93 | 7.04 (-29.1%) | 5.99 (-39.7%) | 5.78 (-41.8%) |
| split-fuse-distinct | retained | 7.92 | 5.73 (-27.7%) | 4.74 (-40.1%) | 5.04 (-36.3%) |
| split-fuse-distinct | moved | 7.94 | 5.59 (-29.7%) | 4.58 (-42.4%) | 4.48 (-43.6%) |

#### Expr-level -- split/fuse and add-tree -- `identity` -- ns/node, old-ri/old-unchecked-ri/uc-chk-ri/uc-unchecked-ri against old-ri

`subst` substitutes `Var`s.

| Case | own | old-ri | old-unchecked-ri | uc-chk-ri | uc-unchecked-ri |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 29.40 | 26.12 (-11.1%) | 22.99 (-21.8%) | 23.14 (-21.3%) |
| split-fuse-shared | moved | 29.14 | 25.90 (-11.1%) | 25.79 (-11.5%) | 23.69 (-18.7%) |
| split-fuse-distinct | retained | 23.49 | 20.94 (-10.9%) | 18.43 (-21.6%) | 18.57 (-21.0%) |
| split-fuse-distinct | moved | 22.98 | 20.21 (-12.1%) | 20.88 (-9.1%) | 19.02 (-17.2%) |

#### Expr-level -- split/fuse and add-tree -- `subst` -- ns/node, old-ri/old-unchecked-ri/uc-chk-ri/uc-unchecked-ri against old-ri

`subst` substitutes `Var`s.

| Case | own | old-ri | old-unchecked-ri | uc-chk-ri | uc-unchecked-ri |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 33.54 | 32.30 (-3.7%) | 33.29 (-0.7%) | 30.34 (-9.6%) |
| split-fuse-shared | moved | 32.01 | 29.00 (-9.4%) | 28.07 (-12.3%) | 25.37 (-20.7%) |
| split-fuse-distinct | retained | 26.89 | 25.87 (-3.8%) | 26.43 (-1.7%) | 24.36 (-9.4%) |
| split-fuse-distinct | moved | 24.28 | 21.21 (-12.7%) | 20.28 (-16.5%) | 19.47 (-19.8%) |

### Supporting

#### `never` -- ns/node, old-ri/old-unchecked-ri/uc-chk-ri/uc-unchecked-ri against old-ri

| Case | own | old-ri | old-unchecked-ri | uc-chk-ri | uc-unchecked-ri |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 20.30 | 17.10 (-15.7%) | 16.10 (-20.7%) | 15.79 (-22.2%) |
| split-fuse-shared | moved | 19.39 | 15.66 (-19.2%) | 14.17 (-26.9%) | 14.40 (-25.7%) |
| split-fuse-distinct | retained | 16.19 | 13.54 (-16.4%) | 12.81 (-20.9%) | 12.75 (-21.2%) |
| split-fuse-distinct | moved | 15.40 | 12.48 (-19.0%) | 11.10 (-27.9%) | 11.15 (-27.6%) |

#### `functor` -- ns/node, old-ri/old-unchecked-ri/uc-chk-ri/uc-unchecked-ri against old-ri

| Case | own | old-ri | old-unchecked-ri | uc-chk-ri | uc-unchecked-ri |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 31.57 | 31.70 (+0.4%) | 31.96 (+1.2%) | 31.89 (+1.0%) |
| split-fuse-shared | moved | 34.86 | 34.60 (-0.8%) | 35.21 (+1.0%) | 35.25 (+1.1%) |
| split-fuse-distinct | retained | 25.14 | 25.35 (+0.8%) | 25.31 (+0.7%) | 25.49 (+1.4%) |
| split-fuse-distinct | moved | 28.45 | 28.33 (-0.4%) | 28.75 (+1.0%) | 28.91 (+1.6%) |

#### `old` -- ns/node, old-ri/old-unchecked-ri/uc-chk-ri/uc-unchecked-ri against old-ri

| Case | own | old-ri | old-unchecked-ri | uc-chk-ri | uc-unchecked-ri |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 47.80 | 48.31 (+1.1%) | 47.76 (-0.1%) | 49.39 (+3.3%) |
| split-fuse-shared | moved | 51.07 | 52.09 (+2.0%) | 51.67 (+1.2%) | 53.07 (+3.9%) |
| split-fuse-distinct | retained | 38.55 | 38.79 (+0.6%) | 38.46 (-0.2%) | 39.84 (+3.3%) |
| split-fuse-distinct | moved | 41.69 | 42.12 (+1.0%) | 41.32 (-0.9%) | 43.22 (+3.7%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

