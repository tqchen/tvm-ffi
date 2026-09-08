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
| state uc-novrr1-ri | tvm-ffi `c9bb47d+764+novrr1 (uc)` |
| state uc-novrr1-ri engine header | `structural_mutate_ucnv.h sha256:9b3259f7ee65310811f565163e7aa12a6c1eac47bb2cd394e9ce1b9a1f175a7d` |
| state uc-novrr1-ri hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| what differs | unstated |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved old-ri/uc-chk-ri/uc-noraw-ri/uc-novrr1-ri |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.91 at start, 1.04 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved old-ri/uc-chk-ri/uc-noraw-ri/uc-novrr1-ri, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse and add-tree -- `floor` -- ns/node, old-ri/uc-chk-ri/uc-noraw-ri/uc-novrr1-ri against old-ri

`subst` substitutes `Var`s.

| Case | own | old-ri | uc-chk-ri | uc-noraw-ri | uc-novrr1-ri |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 9.91 | 5.85 (-40.9%) | 5.89 (-40.6%) | 6.11 (-38.4%) |
| split-fuse-shared | moved | 9.94 | 5.89 (-40.7%) | 5.93 (-40.3%) | 5.94 (-40.2%) |
| split-fuse-distinct | retained | 7.93 | 4.94 (-37.7%) | 4.89 (-38.3%) | 4.87 (-38.6%) |
| split-fuse-distinct | moved | 7.93 | 4.74 (-40.2%) | 4.75 (-40.1%) | 4.86 (-38.7%) |

#### Expr-level -- split/fuse and add-tree -- `identity` -- ns/node, old-ri/uc-chk-ri/uc-noraw-ri/uc-novrr1-ri against old-ri

`subst` substitutes `Var`s.

| Case | own | old-ri | uc-chk-ri | uc-noraw-ri | uc-novrr1-ri |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 29.43 | 22.99 (-21.9%) | 23.56 (-19.9%) | 23.72 (-19.4%) |
| split-fuse-shared | moved | 29.17 | 25.74 (-11.8%) | 24.66 (-15.5%) | 23.80 (-18.4%) |
| split-fuse-distinct | retained | 23.51 | 18.42 (-21.6%) | 18.88 (-19.7%) | 18.95 (-19.4%) |
| split-fuse-distinct | moved | 22.98 | 20.58 (-10.4%) | 20.40 (-11.2%) | 18.89 (-17.8%) |

#### Expr-level -- split/fuse and add-tree -- `subst` -- ns/node, old-ri/uc-chk-ri/uc-noraw-ri/uc-novrr1-ri against old-ri

`subst` substitutes `Var`s.

| Case | own | old-ri | uc-chk-ri | uc-noraw-ri | uc-novrr1-ri |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 33.49 | 33.25 (-0.7%) | 31.13 (-7.1%) | 32.83 (-2.0%) |
| split-fuse-shared | moved | 31.88 | 28.10 (-11.9%) | 26.67 (-16.4%) | 27.37 (-14.2%) |
| split-fuse-distinct | retained | 26.90 | 26.46 (-1.6%) | 24.97 (-7.2%) | 26.02 (-3.3%) |
| split-fuse-distinct | moved | 24.19 | 20.27 (-16.2%) | 19.39 (-19.8%) | 20.22 (-16.4%) |

### Supporting

#### `never` -- ns/node, old-ri/uc-chk-ri/uc-noraw-ri/uc-novrr1-ri against old-ri

| Case | own | old-ri | uc-chk-ri | uc-noraw-ri | uc-novrr1-ri |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 20.42 | 16.28 (-20.3%) | 16.09 (-21.2%) | 16.12 (-21.1%) |
| split-fuse-shared | moved | 19.46 | 14.19 (-27.1%) | 14.38 (-26.1%) | 14.26 (-26.7%) |
| split-fuse-distinct | retained | 16.22 | 12.93 (-20.3%) | 12.97 (-20.0%) | 12.81 (-21.0%) |
| split-fuse-distinct | moved | 15.39 | 11.08 (-28.0%) | 11.66 (-24.3%) | 11.09 (-27.9%) |

#### `functor` -- ns/node, old-ri/uc-chk-ri/uc-noraw-ri/uc-novrr1-ri against old-ri

| Case | own | old-ri | uc-chk-ri | uc-noraw-ri | uc-novrr1-ri |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 31.87 | 31.71 (-0.5%) | 32.05 (+0.6%) | 31.70 (-0.6%) |
| split-fuse-shared | moved | 34.53 | 35.17 (+1.9%) | 35.17 (+1.9%) | 34.78 (+0.7%) |
| split-fuse-distinct | retained | 25.27 | 25.30 (+0.1%) | 25.54 (+1.0%) | 25.52 (+1.0%) |
| split-fuse-distinct | moved | 28.58 | 28.87 (+1.0%) | 28.69 (+0.4%) | 28.60 (+0.1%) |

#### `old` -- ns/node, old-ri/uc-chk-ri/uc-noraw-ri/uc-novrr1-ri against old-ri

| Case | own | old-ri | uc-chk-ri | uc-noraw-ri | uc-novrr1-ri |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 48.08 | 47.57 (-1.1%) | 49.33 (+2.6%) | 48.05 (-0.1%) |
| split-fuse-shared | moved | 51.15 | 51.51 (+0.7%) | 52.73 (+3.1%) | 51.53 (+0.7%) |
| split-fuse-distinct | retained | 38.66 | 38.52 (-0.4%) | 39.78 (+2.9%) | 38.65 (-0.0%) |
| split-fuse-distinct | moved | 41.97 | 41.50 (-1.1%) | 42.68 (+1.7%) | 42.00 (+0.1%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

