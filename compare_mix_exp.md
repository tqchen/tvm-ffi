### Provenance

| | |
| --- | --- |
| state gold | tvm-ffi `f9f52bfc6c06c9658390c32aabbf2b207af6e77e (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_override_gold.h sha256:47233c529e10a1ca3442d5b6fa1287f130a309a65d65441ea91ec7fd01a63b87` |
| state uc-proxy | tvm-ffi `f9f52bfc6c06c9658390c32aabbf2b207af6e77e (uc)` |
| state uc-proxy engine header | `structural_mutate.h sha256:ed40dd82d5e97edee66bb1c655cd012fbb06e77de4649331e2d626a98b112b90` |
| state uc-proxy hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| state uc-inlinedtor | tvm-ffi `5e29d1e516264b3a322ea6190738661665089cb8 (uc-inlinedtor)` |
| state uc-inlinedtor engine header | `structural_mutate.h sha256:cd3fe38bda1c8d72c2d8922a3a12f4ca7c46bc5235eb3c6fe90eaa10b8013471` |
| state uc-inlinedtor hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| what differs | experiment: gold and uc-proxy are the run-2 binaries (fe350f9); uc-inlinedtor is f9f52bf^ (original TVM_FFI_S_MUTATE_RETURN_UNCHANGED) plus TVM_FFI_INLINE special members on UnchangedOr<T> and nothing else (side branch bench/387-experiment-inline-dtor 5e29d1e) |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved gold/uc-proxy/uc-inlinedtor |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.23 at start, 1.06 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved gold/uc-proxy/uc-inlinedtor, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse -- `floor` -- ns/node, gold/uc-proxy/uc-inlinedtor against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-proxy | uc-inlinedtor |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 8.61 | 8.53 (-1.0%) | 7.51 (-12.8%) |
| split-fuse-shared | moved | 8.91 | 8.71 (-2.2%) | 7.87 (-11.6%) |
| split-fuse-distinct | retained | 7.40 | 7.08 (-4.2%) | 6.35 (-14.2%) |
| split-fuse-distinct | moved | 7.17 | 7.01 (-2.1%) | 6.21 (-13.3%) |

#### Expr-level -- split/fuse -- `identity` -- ns/node, gold/uc-proxy/uc-inlinedtor against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-proxy | uc-inlinedtor |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.42 | 39.94 (+12.8%) | 33.74 (-4.8%) |
| split-fuse-shared | moved | 34.91 | 39.58 (+13.4%) | 36.91 (+5.7%) |
| split-fuse-distinct | retained | 28.46 | 32.10 (+12.8%) | 26.93 (-5.4%) |
| split-fuse-distinct | moved | 28.39 | 34.54 (+21.7%) | 31.23 (+10.0%) |

#### Expr-level -- split/fuse -- `subst` -- ns/node, gold/uc-proxy/uc-inlinedtor against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-proxy | uc-inlinedtor |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 44.35 | 51.02 (+15.0%) | 45.55 (+2.7%) |
| split-fuse-shared | moved | 40.92 | 48.58 (+18.7%) | 43.35 (+5.9%) |
| split-fuse-distinct | retained | 35.96 | 40.35 (+12.2%) | 36.19 (+0.6%) |
| split-fuse-distinct | moved | 29.35 | 35.81 (+22.0%) | 32.23 (+9.8%) |

### Supporting

#### `never` -- ns/node, gold/uc-proxy/uc-inlinedtor against gold

| Case | own | gold | uc-proxy | uc-inlinedtor |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 21.49 | 22.61 (+5.2%) | 20.56 (-4.3%) |
| split-fuse-shared | moved | 21.31 | 22.54 (+5.8%) | 18.54 (-13.0%) |
| split-fuse-distinct | retained | 17.18 | 18.36 (+6.9%) | 16.63 (-3.2%) |
| split-fuse-distinct | moved | 17.25 | 18.37 (+6.5%) | 16.46 (-4.6%) |

#### `functor` -- ns/node, gold/uc-proxy/uc-inlinedtor against gold

| Case | own | gold | uc-proxy | uc-inlinedtor |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.54 | 35.50 (-0.1%) | 35.52 (-0.1%) |
| split-fuse-shared | moved | 40.90 | 40.45 (-1.1%) | 40.09 (-2.0%) |
| split-fuse-distinct | retained | 28.21 | 28.29 (+0.3%) | 28.41 (+0.7%) |
| split-fuse-distinct | moved | 33.88 | 33.52 (-1.1%) | 33.23 (-1.9%) |

#### `old` -- ns/node, gold/uc-proxy/uc-inlinedtor against gold

| Case | own | gold | uc-proxy | uc-inlinedtor |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 55.33 | 54.82 (-0.9%) | 54.97 (-0.6%) |
| split-fuse-shared | moved | 59.72 | 59.07 (-1.1%) | 59.06 (-1.1%) |
| split-fuse-distinct | retained | 44.81 | 44.02 (-1.8%) | 44.03 (-1.7%) |
| split-fuse-distinct | moved | 48.81 | 48.51 (-0.6%) | 48.07 (-1.5%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

