### Provenance

| | |
| --- | --- |
| state gold | tvm-ffi `f9f52bfc6c06c9658390c32aabbf2b207af6e77e (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_override_gold.h sha256:47233c529e10a1ca3442d5b6fa1287f130a309a65d65441ea91ec7fd01a63b87` |
| state gold-expected-marked | tvm-ffi `a2c2744f1813a5848cc2f77a395ff3f1911db530 (gold-expected-marked)` |
| state gold-expected-marked engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold-expected-marked hook header | `tvm_override_gold.h sha256:47233c529e10a1ca3442d5b6fa1287f130a309a65d65441ea91ec7fd01a63b87` |
| state uc-inline-dtor | tvm-ffi `5e29d1e516264b3a322ea6190738661665089cb8 (uc-inlinedtor)` |
| state uc-inline-dtor engine header | `structural_mutate.h sha256:cd3fe38bda1c8d72c2d8922a3a12f4ca7c46bc5235eb3c6fe90eaa10b8013471` |
| state uc-inline-dtor hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| state uc-inline-dtor-expected-marked | tvm-ffi `a2c2744f1813a5848cc2f77a395ff3f1911db530 (uc-inline-dtor-expected-marked)` |
| state uc-inline-dtor-expected-marked engine header | `structural_mutate.h sha256:cd3fe38bda1c8d72c2d8922a3a12f4ca7c46bc5235eb3c6fe90eaa10b8013471` |
| state uc-inline-dtor-expected-marked hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| what differs | experiment 2: gold and uc-inline-dtor are the existing binaries; the two -expected-marked binaries are rebuilt from a2c2744, which adds the five explicitly defaulted TVM_FFI_INLINE special members to Expected<T> in expected.h and nothing else |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved gold/gold-expected-marked/uc-inline-dtor/uc-inline-dtor-expected-marked |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.50 at start, 1.06 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved gold/gold-expected-marked/uc-inline-dtor/uc-inline-dtor-expected-marked, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse -- `floor` -- ns/node, gold/gold-expected-marked/uc-inline-dtor/uc-inline-dtor-expected-marked against gold

`subst` substitutes `Var`s.

| Case | own | gold | gold-expected-marked | uc-inline-dtor | uc-inline-dtor-expected-marked |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 8.62 | 8.33 (-3.4%) | 7.73 (-10.3%) | 5.90 (-31.6%) |
| split-fuse-shared | moved | 8.97 | 8.44 (-5.9%) | 7.92 (-11.6%) | 6.37 (-29.0%) |
| split-fuse-distinct | retained | 7.35 | 7.44 (+1.2%) | 6.33 (-14.0%) | 4.96 (-32.5%) |
| split-fuse-distinct | moved | 7.22 | 7.03 (-2.5%) | 6.23 (-13.7%) | 5.44 (-24.6%) |

#### Expr-level -- split/fuse -- `identity` -- ns/node, gold/gold-expected-marked/uc-inline-dtor/uc-inline-dtor-expected-marked against gold

`subst` substitutes `Var`s.

| Case | own | gold | gold-expected-marked | uc-inline-dtor | uc-inline-dtor-expected-marked |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.31 | 30.12 (-14.7%) | 33.64 (-4.7%) | 26.46 (-25.1%) |
| split-fuse-shared | moved | 34.70 | 29.77 (-14.2%) | 36.91 (+6.4%) | 27.03 (-22.1%) |
| split-fuse-distinct | retained | 28.24 | 24.26 (-14.1%) | 26.76 (-5.3%) | 21.07 (-25.4%) |
| split-fuse-distinct | moved | 28.33 | 23.73 (-16.2%) | 31.06 (+9.6%) | 23.11 (-18.4%) |

#### Expr-level -- split/fuse -- `subst` -- ns/node, gold/gold-expected-marked/uc-inline-dtor/uc-inline-dtor-expected-marked against gold

`subst` substitutes `Var`s.

| Case | own | gold | gold-expected-marked | uc-inline-dtor | uc-inline-dtor-expected-marked |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 43.81 | 38.98 (-11.0%) | 45.51 (+3.9%) | 34.74 (-20.7%) |
| split-fuse-shared | moved | 40.88 | 34.23 (-16.3%) | 43.13 (+5.5%) | 34.24 (-16.2%) |
| split-fuse-distinct | retained | 35.65 | 31.31 (-12.2%) | 36.00 (+1.0%) | 27.75 (-22.2%) |
| split-fuse-distinct | moved | 29.45 | 24.87 (-15.5%) | 32.11 (+9.0%) | 25.05 (-15.0%) |

### Supporting

#### `never` -- ns/node, gold/gold-expected-marked/uc-inline-dtor/uc-inline-dtor-expected-marked against gold

| Case | own | gold | gold-expected-marked | uc-inline-dtor | uc-inline-dtor-expected-marked |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 22.50 | 20.77 (-7.7%) | 20.42 (-9.2%) | 18.14 (-19.4%) |
| split-fuse-shared | moved | 21.18 | 19.24 (-9.1%) | 18.43 (-13.0%) | 15.99 (-24.5%) |
| split-fuse-distinct | retained | 16.97 | 15.64 (-7.8%) | 15.17 (-10.6%) | 14.30 (-15.7%) |
| split-fuse-distinct | moved | 17.10 | 15.60 (-8.8%) | 16.33 (-4.5%) | 13.63 (-20.3%) |

#### `functor` -- ns/node, gold/gold-expected-marked/uc-inline-dtor/uc-inline-dtor-expected-marked against gold

| Case | own | gold | gold-expected-marked | uc-inline-dtor | uc-inline-dtor-expected-marked |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.33 | 35.50 (+0.5%) | 35.26 (-0.2%) | 35.97 (+1.8%) |
| split-fuse-shared | moved | 40.35 | 40.04 (-0.8%) | 39.89 (-1.1%) | 40.78 (+1.1%) |
| split-fuse-distinct | retained | 28.47 | 28.59 (+0.4%) | 28.30 (-0.6%) | 28.52 (+0.2%) |
| split-fuse-distinct | moved | 33.62 | 33.43 (-0.6%) | 33.20 (-1.3%) | 33.47 (-0.5%) |

#### `old` -- ns/node, gold/gold-expected-marked/uc-inline-dtor/uc-inline-dtor-expected-marked against gold

| Case | own | gold | gold-expected-marked | uc-inline-dtor | uc-inline-dtor-expected-marked |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 54.78 | 54.55 (-0.4%) | 54.63 (-0.3%) | 53.90 (-1.6%) |
| split-fuse-shared | moved | 58.61 | 59.15 (+0.9%) | 58.96 (+0.6%) | 58.65 (+0.1%) |
| split-fuse-distinct | retained | 44.31 | 44.20 (-0.3%) | 43.73 (-1.3%) | 43.63 (-1.5%) |
| split-fuse-distinct | moved | 48.27 | 48.15 (-0.3%) | 47.92 (-0.7%) | 48.03 (-0.5%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

