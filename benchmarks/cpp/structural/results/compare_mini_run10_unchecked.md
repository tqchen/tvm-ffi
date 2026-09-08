### Provenance

| | |
| --- | --- |
| state old-ri | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (old)` |
| state old-ri engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old-ri hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state uc-chk-ri | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (uc)` |
| state uc-chk-ri engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc-chk-ri hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| state uc-unchecked-ri | tvm-ffi `c9bb47d+764+unchecked (uc)` |
| state uc-unchecked-ri engine header | `structural_mutate_ucuc.h sha256:110663684f4a5965a048b49b264ae26edc9b8e1fe5802b89db65a81aa7a6f7f3` |
| state uc-unchecked-ri hook header | `tvm_hook_override_ucuc.h sha256:f3564797e74d01f48438c84150b1b1b8053c9bb5c7fd4d5e9fe9c7c787c48bc3` |
| what differs | unstated |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved old-ri/uc-chk-ri/uc-unchecked-ri |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.38 at start, 1.04 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved old-ri/uc-chk-ri/uc-unchecked-ri, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse and add-tree -- `floor` -- ns/node, old-ri/uc-chk-ri/uc-unchecked-ri against old-ri

`subst` substitutes `Var`s.

| Case | own | old-ri | uc-chk-ri | uc-unchecked-ri |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 9.86 | 5.90 (-40.2%) | 6.40 (-35.2%) |
| split-fuse-shared | moved | 9.90 | 5.92 (-40.2%) | 5.88 (-40.6%) |
| split-fuse-distinct | retained | 7.90 | 4.94 (-37.5%) | 5.15 (-34.8%) |
| split-fuse-distinct | moved | 7.97 | 4.84 (-39.3%) | 4.65 (-41.7%) |

#### Expr-level -- split/fuse and add-tree -- `identity` -- ns/node, old-ri/uc-chk-ri/uc-unchecked-ri against old-ri

`subst` substitutes `Var`s.

| Case | own | old-ri | uc-chk-ri | uc-unchecked-ri |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 29.32 | 23.04 (-21.4%) | 23.15 (-21.0%) |
| split-fuse-shared | moved | 29.15 | 25.85 (-11.3%) | 23.70 (-18.7%) |
| split-fuse-distinct | retained | 23.44 | 18.39 (-21.5%) | 18.54 (-20.9%) |
| split-fuse-distinct | moved | 23.05 | 20.77 (-9.9%) | 19.07 (-17.2%) |

#### Expr-level -- split/fuse and add-tree -- `subst` -- ns/node, old-ri/uc-chk-ri/uc-unchecked-ri against old-ri

`subst` substitutes `Var`s.

| Case | own | old-ri | uc-chk-ri | uc-unchecked-ri |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 33.42 | 33.45 (+0.1%) | 30.43 (-8.9%) |
| split-fuse-shared | moved | 31.89 | 28.10 (-11.9%) | 25.45 (-20.2%) |
| split-fuse-distinct | retained | 26.87 | 26.61 (-1.0%) | 24.42 (-9.1%) |
| split-fuse-distinct | moved | 24.24 | 20.28 (-16.3%) | 19.44 (-19.8%) |

### Supporting

#### `never` -- ns/node, old-ri/uc-chk-ri/uc-unchecked-ri against old-ri

| Case | own | old-ri | uc-chk-ri | uc-unchecked-ri |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 20.29 | 16.20 (-20.2%) | 16.00 (-21.2%) |
| split-fuse-shared | moved | 19.43 | 14.20 (-26.9%) | 14.31 (-26.3%) |
| split-fuse-distinct | retained | 16.31 | 12.94 (-20.7%) | 12.63 (-22.6%) |
| split-fuse-distinct | moved | 15.36 | 11.06 (-28.0%) | 11.13 (-27.6%) |

#### `functor` -- ns/node, old-ri/uc-chk-ri/uc-unchecked-ri against old-ri

| Case | own | old-ri | uc-chk-ri | uc-unchecked-ri |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 31.76 | 31.92 (+0.5%) | 31.72 (-0.1%) |
| split-fuse-shared | moved | 34.53 | 35.27 (+2.2%) | 34.97 (+1.3%) |
| split-fuse-distinct | retained | 25.18 | 25.35 (+0.7%) | 25.40 (+0.9%) |
| split-fuse-distinct | moved | 28.50 | 28.84 (+1.2%) | 28.68 (+0.7%) |

#### `old` -- ns/node, old-ri/uc-chk-ri/uc-unchecked-ri against old-ri

| Case | own | old-ri | uc-chk-ri | uc-unchecked-ri |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 47.63 | 47.70 (+0.1%) | 49.76 (+4.5%) |
| split-fuse-shared | moved | 51.14 | 51.45 (+0.6%) | 52.90 (+3.4%) |
| split-fuse-distinct | retained | 38.55 | 38.54 (-0.0%) | 39.98 (+3.7%) |
| split-fuse-distinct | moved | 41.74 | 41.53 (-0.5%) | 42.79 (+2.5%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

