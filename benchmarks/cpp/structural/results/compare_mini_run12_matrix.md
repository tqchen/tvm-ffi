### Provenance

| | |
| --- | --- |
| state old-checked | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (old)` |
| state old-checked engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old-checked hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state old-unchecked | tvm-ffi `c9bb47d+764+unchecked (old)` |
| state old-unchecked engine header | `structural_mutate_olduc.h sha256:3b5856c30cc152460026f8251d7965bcd352ac12c829114f30ba98403b4a3d7c` |
| state old-unchecked hook header | `tvm_hook_override.h sha256:9d7c48991d79947d581f9203bb1303f96e5a5dcd63226a4dbc295af492c3b8a8` |
| state old-borrowed | tvm-ffi `c9bb47d+764+ty (old)` |
| state old-borrowed engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old-borrowed hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state uc-checked | tvm-ffi `c9bb47d1f44709361678202868b44fe480356116-dirty (uc)` |
| state uc-checked engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc-checked hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| state uc-unchecked | tvm-ffi `c9bb47d+764+unchecked (uc)` |
| state uc-unchecked engine header | `structural_mutate_ucuc.h sha256:110663684f4a5965a048b49b264ae26edc9b8e1fe5802b89db65a81aa7a6f7f3` |
| state uc-unchecked hook header | `tvm_hook_override_ucuc.h sha256:f3564797e74d01f48438c84150b1b1b8053c9bb5c7fd4d5e9fe9c7c787c48bc3` |
| state uc-borrowed | tvm-ffi `c9bb47d+764+ty (uc)` |
| state uc-borrowed engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc-borrowed hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| what differs | unstated |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved old-checked/old-unchecked/old-borrowed/uc-checked/uc-unchecked/uc-borrowed |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.47 at start, 1.13 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved old-checked/old-unchecked/old-borrowed/uc-checked/uc-unchecked/uc-borrowed, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse and add-tree -- `floor` -- ns/node, old-checked/old-unchecked/old-borrowed/uc-checked/uc-unchecked/uc-borrowed against old-checked

`subst` substitutes `Var`s.

| Case | own | old-checked | old-unchecked | old-borrowed | uc-checked | uc-unchecked | uc-borrowed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 9.90 | 7.28 (-26.5%) | 7.52 (-24.0%) | 5.91 (-40.3%) | 6.34 (-35.9%) | 5.90 (-40.4%) |
| split-fuse-shared | moved | 9.91 | 7.05 (-28.8%) | 7.36 (-25.7%) | 6.06 (-38.8%) | 5.85 (-41.0%) | 5.71 (-42.3%) |
| split-fuse-distinct | retained | 7.91 | 5.80 (-26.7%) | 6.01 (-24.0%) | 4.96 (-37.3%) | 5.11 (-35.3%) | 4.79 (-39.4%) |
| split-fuse-distinct | moved | 7.96 | 5.60 (-29.6%) | 5.90 (-25.9%) | 4.77 (-40.0%) | 4.68 (-41.2%) | 4.57 (-42.5%) |

#### Expr-level -- split/fuse and add-tree -- `identity` -- ns/node, old-checked/old-unchecked/old-borrowed/uc-checked/uc-unchecked/uc-borrowed against old-checked

`subst` substitutes `Var`s.

| Case | own | old-checked | old-unchecked | old-borrowed | uc-checked | uc-unchecked | uc-borrowed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 29.29 | 26.13 (-10.8%) | 26.18 (-10.6%) | 23.01 (-21.4%) | 23.16 (-20.9%) | 23.10 (-21.1%) |
| split-fuse-shared | moved | 29.16 | 25.84 (-11.4%) | 26.19 (-10.2%) | 25.47 (-12.6%) | 23.76 (-18.5%) | 23.83 (-18.3%) |
| split-fuse-distinct | retained | 23.52 | 20.91 (-11.1%) | 20.96 (-10.9%) | 18.35 (-22.0%) | 18.61 (-20.9%) | 18.49 (-21.4%) |
| split-fuse-distinct | moved | 22.98 | 20.15 (-12.3%) | 20.49 (-10.8%) | 20.70 (-9.9%) | 19.13 (-16.8%) | 19.46 (-15.3%) |

#### Expr-level -- split/fuse and add-tree -- `subst` -- ns/node, old-checked/old-unchecked/old-borrowed/uc-checked/uc-unchecked/uc-borrowed against old-checked

`subst` substitutes `Var`s.

| Case | own | old-checked | old-unchecked | old-borrowed | uc-checked | uc-unchecked | uc-borrowed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 33.45 | 32.47 (-2.9%) | 32.37 (-3.2%) | 33.23 (-0.6%) | 30.37 (-9.2%) | 30.82 (-7.9%) |
| split-fuse-shared | moved | 31.84 | 29.05 (-8.8%) | 29.10 (-8.6%) | 28.18 (-11.5%) | 25.49 (-20.0%) | 26.29 (-17.4%) |
| split-fuse-distinct | retained | 26.90 | 25.92 (-3.6%) | 26.00 (-3.4%) | 26.40 (-1.9%) | 24.45 (-9.1%) | 24.59 (-8.6%) |
| split-fuse-distinct | moved | 24.16 | 21.19 (-12.3%) | 21.70 (-10.2%) | 20.28 (-16.1%) | 19.42 (-19.6%) | 19.77 (-18.2%) |

### Supporting

#### `never` -- ns/node, old-checked/old-unchecked/old-borrowed/uc-checked/uc-unchecked/uc-borrowed against old-checked

| Case | own | old-checked | old-unchecked | old-borrowed | uc-checked | uc-unchecked | uc-borrowed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 20.36 | 17.07 (-16.2%) | 17.48 (-14.1%) | 16.06 (-21.1%) | 16.02 (-21.3%) | 16.20 (-20.4%) |
| split-fuse-shared | moved | 19.50 | 15.60 (-20.0%) | 16.04 (-17.7%) | 14.19 (-27.2%) | 14.29 (-26.7%) | 14.42 (-26.0%) |
| split-fuse-distinct | retained | 16.08 | 13.53 (-15.8%) | 13.91 (-13.4%) | 12.90 (-19.8%) | 12.77 (-20.6%) | 12.41 (-22.8%) |
| split-fuse-distinct | moved | 15.41 | 12.46 (-19.2%) | 12.33 (-20.0%) | 11.07 (-28.2%) | 11.14 (-27.7%) | 11.50 (-25.4%) |

#### `functor` -- ns/node, old-checked/old-unchecked/old-borrowed/uc-checked/uc-unchecked/uc-borrowed against old-checked

| Case | own | old-checked | old-unchecked | old-borrowed | uc-checked | uc-unchecked | uc-borrowed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 31.57 | 31.79 (+0.7%) | 31.82 (+0.8%) | 32.05 (+1.5%) | 31.74 (+0.6%) | 31.86 (+0.9%) |
| split-fuse-shared | moved | 35.05 | 34.70 (-1.0%) | 35.03 (-0.1%) | 35.21 (+0.4%) | 35.14 (+0.3%) | 34.83 (-0.6%) |
| split-fuse-distinct | retained | 25.13 | 25.29 (+0.6%) | 25.72 (+2.3%) | 25.30 (+0.7%) | 25.41 (+1.1%) | 25.35 (+0.9%) |
| split-fuse-distinct | moved | 28.49 | 28.29 (-0.7%) | 28.64 (+0.5%) | 28.78 (+1.0%) | 28.64 (+0.5%) | 28.65 (+0.6%) |

#### `old` -- ns/node, old-checked/old-unchecked/old-borrowed/uc-checked/uc-unchecked/uc-borrowed against old-checked

| Case | own | old-checked | old-unchecked | old-borrowed | uc-checked | uc-unchecked | uc-borrowed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 47.97 | 48.25 (+0.6%) | 48.29 (+0.7%) | 47.99 (+0.0%) | 49.52 (+3.2%) | 48.21 (+0.5%) |
| split-fuse-shared | moved | 51.50 | 51.78 (+0.5%) | 51.72 (+0.4%) | 51.59 (+0.2%) | 52.94 (+2.8%) | 52.10 (+1.2%) |
| split-fuse-distinct | retained | 38.61 | 38.69 (+0.2%) | 38.98 (+1.0%) | 38.66 (+0.1%) | 39.89 (+3.3%) | 38.48 (-0.3%) |
| split-fuse-distinct | moved | 41.91 | 42.23 (+0.8%) | 42.30 (+0.9%) | 41.58 (-0.8%) | 42.96 (+2.5%) | 41.49 (-1.0%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

