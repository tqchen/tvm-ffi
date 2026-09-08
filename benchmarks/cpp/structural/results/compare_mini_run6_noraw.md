### Provenance

| | |
| --- | --- |
| state old-ri | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (old)` |
| state old-ri engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old-ri hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state uc-chk-ri | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (uc)` |
| state uc-chk-ri engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc-chk-ri hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| state uc-noraw-ri | tvm-ffi `c9bb47d+764+noraw (uc)` |
| state uc-noraw-ri engine header | `structural_mutate_ucnr.h sha256:efe92b8a905d8f744d7be8e1216bec0c7118928aca17b00c772708fa38b0f1ab` |
| state uc-noraw-ri hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| what differs | unstated |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved old-ri/uc-chk-ri/uc-noraw-ri |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.17 at start, 0.99 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved old-ri/uc-chk-ri/uc-noraw-ri, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse and add-tree -- `floor` -- ns/node, old-ri/uc-chk-ri/uc-noraw-ri against old-ri

`subst` substitutes `Var`s.

| Case | own | old-ri | uc-chk-ri | uc-noraw-ri |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 9.88 | 5.84 (-40.8%) | 5.77 (-41.5%) |
| split-fuse-shared | moved | 9.93 | 6.00 (-39.6%) | 5.49 (-44.7%) |
| split-fuse-distinct | retained | 7.91 | 4.91 (-38.0%) | 4.66 (-41.1%) |
| split-fuse-distinct | moved | 7.95 | 4.83 (-39.3%) | 4.46 (-43.9%) |

#### Expr-level -- split/fuse and add-tree -- `identity` -- ns/node, old-ri/uc-chk-ri/uc-noraw-ri against old-ri

`subst` substitutes `Var`s.

| Case | own | old-ri | uc-chk-ri | uc-noraw-ri |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 29.35 | 23.01 (-21.6%) | 25.75 (-12.3%) |
| split-fuse-shared | moved | 29.09 | 25.73 (-11.6%) | 25.85 (-11.1%) |
| split-fuse-distinct | retained | 23.45 | 18.36 (-21.7%) | 20.66 (-11.9%) |
| split-fuse-distinct | moved | 22.97 | 20.82 (-9.3%) | 20.86 (-9.2%) |

#### Expr-level -- split/fuse and add-tree -- `subst` -- ns/node, old-ri/uc-chk-ri/uc-noraw-ri against old-ri

`subst` substitutes `Var`s.

| Case | own | old-ri | uc-chk-ri | uc-noraw-ri |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 33.48 | 33.24 (-0.7%) | 36.30 (+8.4%) |
| split-fuse-shared | moved | 31.87 | 28.07 (-11.9%) | 31.20 (-2.1%) |
| split-fuse-distinct | retained | 26.85 | 26.47 (-1.4%) | 29.28 (+9.0%) |
| split-fuse-distinct | moved | 24.21 | 20.32 (-16.1%) | 22.40 (-7.5%) |

### Supporting

#### `never` -- ns/node, old-ri/uc-chk-ri/uc-noraw-ri against old-ri

| Case | own | old-ri | uc-chk-ri | uc-noraw-ri |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 20.31 | 16.03 (-21.1%) | 18.42 (-9.3%) |
| split-fuse-shared | moved | 19.41 | 14.19 (-26.9%) | 16.68 (-14.1%) |
| split-fuse-distinct | retained | 16.35 | 12.77 (-21.9%) | 13.29 (-18.7%) |
| split-fuse-distinct | moved | 15.39 | 11.06 (-28.1%) | 13.19 (-14.2%) |

#### `functor` -- ns/node, old-ri/uc-chk-ri/uc-noraw-ri against old-ri

| Case | own | old-ri | uc-chk-ri | uc-noraw-ri |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 31.38 | 31.96 (+1.9%) | 31.92 (+1.7%) |
| split-fuse-shared | moved | 34.56 | 35.37 (+2.3%) | 34.93 (+1.1%) |
| split-fuse-distinct | retained | 25.11 | 25.29 (+0.7%) | 25.57 (+1.8%) |
| split-fuse-distinct | moved | 28.40 | 28.87 (+1.7%) | 28.95 (+1.9%) |

#### `old` -- ns/node, old-ri/uc-chk-ri/uc-noraw-ri against old-ri

| Case | own | old-ri | uc-chk-ri | uc-noraw-ri |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 47.68 | 47.73 (+0.1%) | 48.57 (+1.9%) |
| split-fuse-shared | moved | 51.18 | 51.44 (+0.5%) | 52.17 (+1.9%) |
| split-fuse-distinct | retained | 38.48 | 38.42 (-0.2%) | 39.06 (+1.5%) |
| split-fuse-distinct | moved | 41.81 | 41.59 (-0.5%) | 42.25 (+1.1%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

