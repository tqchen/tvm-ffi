### Provenance

| | |
| --- | --- |
| state gold | tvm-ffi `f9f52bfc6c06c9658390c32aabbf2b207af6e77e (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_override_gold.h sha256:47233c529e10a1ca3442d5b6fa1287f130a309a65d65441ea91ec7fd01a63b87` |
| state uc-prefix | tvm-ffi `7d94093053aa75bac9d5f82d8a8f0ff2f6f4922a (uc-prefix)` |
| state uc-prefix engine header | `structural_mutate.h sha256:7996b957aea127c313ccaf8cebc98759d86d8050ee416f9ef9d2e4adce9481a7` |
| state uc-prefix hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| state uc-proxy | tvm-ffi `f9f52bfc6c06c9658390c32aabbf2b207af6e77e (uc)` |
| state uc-proxy engine header | `structural_mutate.h sha256:ed40dd82d5e97edee66bb1c655cd012fbb06e77de4649331e2d626a98b112b90` |
| state uc-proxy hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| state uc-inline-dtor | tvm-ffi `5e29d1e516264b3a322ea6190738661665089cb8 (uc-inlinedtor)` |
| state uc-inline-dtor engine header | `structural_mutate.h sha256:cd3fe38bda1c8d72c2d8922a3a12f4ca7c46bc5235eb3c6fe90eaa10b8013471` |
| state uc-inline-dtor hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| state uc-both-marked | tvm-ffi `a2c2744f1813a5848cc2f77a395ff3f1911db530 (uc-inline-dtor-expected-marked)` |
| state uc-both-marked engine header | `structural_mutate.h sha256:cd3fe38bda1c8d72c2d8922a3a12f4ca7c46bc5235eb3c6fe90eaa10b8013471` |
| state uc-both-marked hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| what differs | consolidated: gold (structural_mutate_gold.h, GOLD 730d6fc); uc-prefix = UC 31f6948 exactly as shipped (original TVM_FFI_S_MUTATE_RETURN_UNCHANGED, UnchangedOr and Expected unmarked), rebuilt from f9f52bf^; uc-proxy = f9f52bf; uc-inline-dtor = cdf0687 (original macro, UnchangedOr special members TVM_FFI_INLINE); uc-both-marked = 0780862 (that plus Expected<T> special members TVM_FFI_INLINE). One tree, one apache/tvm 2c28965, one libtvm build, same hook file for every uc column |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved gold/uc-prefix/uc-proxy/uc-inline-dtor/uc-both-marked |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.25 at start, 1.18 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved gold/uc-prefix/uc-proxy/uc-inline-dtor/uc-both-marked, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse -- `floor` -- ns/node, gold/uc-prefix/uc-proxy/uc-inline-dtor/uc-both-marked against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-prefix | uc-proxy | uc-inline-dtor | uc-both-marked |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 9.59 | 10.96 (+14.4%) | 8.52 (-11.1%) | 7.55 (-21.3%) | 6.94 (-27.6%) |
| split-fuse-shared | moved | 8.98 | 10.59 (+17.9%) | 8.73 (-2.8%) | 7.84 (-12.7%) | 6.14 (-31.6%) |
| split-fuse-distinct | retained | 8.02 | 9.12 (+13.7%) | 7.14 (-10.9%) | 6.27 (-21.8%) | 5.73 (-28.5%) |
| split-fuse-distinct | moved | 7.18 | 7.93 (+10.4%) | 7.04 (-1.9%) | 6.22 (-13.4%) | 4.94 (-31.2%) |

#### Expr-level -- split/fuse -- `identity` -- ns/node, gold/uc-prefix/uc-proxy/uc-inline-dtor/uc-both-marked against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-prefix | uc-proxy | uc-inline-dtor | uc-both-marked |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.17 | 43.57 (+23.9%) | 39.75 (+13.0%) | 33.52 (-4.7%) | 26.46 (-24.8%) |
| split-fuse-shared | moved | 34.56 | 42.22 (+22.2%) | 39.26 (+13.6%) | 36.69 (+6.2%) | 27.04 (-21.8%) |
| split-fuse-distinct | retained | 28.14 | 34.87 (+23.9%) | 31.87 (+13.2%) | 26.78 (-4.8%) | 21.01 (-25.4%) |
| split-fuse-distinct | moved | 28.08 | 37.53 (+33.6%) | 34.29 (+22.1%) | 31.06 (+10.6%) | 23.06 (-17.9%) |

#### Expr-level -- split/fuse -- `subst` -- ns/node, gold/uc-prefix/uc-proxy/uc-inline-dtor/uc-both-marked against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-prefix | uc-proxy | uc-inline-dtor | uc-both-marked |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 44.13 | 48.91 (+10.8%) | 50.61 (+14.7%) | 45.24 (+2.5%) | 34.60 (-21.6%) |
| split-fuse-shared | moved | 40.53 | 51.20 (+26.3%) | 48.16 (+18.8%) | 43.13 (+6.4%) | 34.11 (-15.8%) |
| split-fuse-distinct | retained | 35.72 | 39.00 (+9.2%) | 40.08 (+12.2%) | 35.97 (+0.7%) | 27.62 (-22.7%) |
| split-fuse-distinct | moved | 29.18 | 35.83 (+22.8%) | 35.56 (+21.8%) | 31.95 (+9.5%) | 24.99 (-14.4%) |

### Supporting

#### `never` -- ns/node, gold/uc-prefix/uc-proxy/uc-inline-dtor/uc-both-marked against gold

| Case | own | gold | uc-prefix | uc-proxy | uc-inline-dtor | uc-both-marked |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 22.27 | 24.28 (+9.0%) | 22.45 (+0.8%) | 20.41 (-8.4%) | 18.08 (-18.8%) |
| split-fuse-shared | moved | 21.21 | 24.15 (+13.9%) | 22.30 (+5.2%) | 18.46 (-12.9%) | 15.96 (-24.8%) |
| split-fuse-distinct | retained | 17.04 | 19.44 (+14.0%) | 18.04 (+5.8%) | 16.39 (-3.8%) | 14.34 (-15.9%) |
| split-fuse-distinct | moved | 17.09 | 19.23 (+12.6%) | 18.22 (+6.6%) | 16.30 (-4.6%) | 13.62 (-20.3%) |

#### `functor` -- ns/node, gold/uc-prefix/uc-proxy/uc-inline-dtor/uc-both-marked against gold

| Case | own | gold | uc-prefix | uc-proxy | uc-inline-dtor | uc-both-marked |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.21 | 35.47 (+0.7%) | 35.31 (+0.3%) | 35.35 (+0.4%) | 35.69 (+1.4%) |
| split-fuse-shared | moved | 40.33 | 39.98 (-0.9%) | 40.17 (-0.4%) | 40.18 (-0.4%) | 40.53 (+0.5%) |
| split-fuse-distinct | retained | 28.08 | 28.32 (+0.8%) | 28.20 (+0.4%) | 28.41 (+1.2%) | 28.42 (+1.2%) |
| split-fuse-distinct | moved | 33.61 | 33.20 (-1.2%) | 33.54 (-0.2%) | 32.98 (-1.9%) | 33.40 (-0.6%) |

#### `old` -- ns/node, gold/uc-prefix/uc-proxy/uc-inline-dtor/uc-both-marked against gold

| Case | own | gold | uc-prefix | uc-proxy | uc-inline-dtor | uc-both-marked |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 54.32 | 55.09 (+1.4%) | 54.43 (+0.2%) | 54.67 (+0.6%) | 53.76 (-1.0%) |
| split-fuse-shared | moved | 58.73 | 58.90 (+0.3%) | 58.86 (+0.2%) | 58.83 (+0.2%) | 58.62 (-0.2%) |
| split-fuse-distinct | retained | 44.25 | 44.02 (-0.5%) | 44.06 (-0.4%) | 44.19 (-0.1%) | 43.36 (-2.0%) |
| split-fuse-distinct | moved | 48.04 | 47.95 (-0.2%) | 48.16 (+0.3%) | 47.66 (-0.8%) | 47.84 (-0.4%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

