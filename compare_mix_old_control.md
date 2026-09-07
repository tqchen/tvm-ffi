### Provenance

| | |
| --- | --- |
| state old | tvm-ffi `c81d04398d1b1682512418e8b6d29aeeacd82004 (old)` |
| state old engine header | `structural_mutate.h (expected.h sha256:d2c00d84378ff6c687076d62a684dc13c662b97fa371ba6219f3cba84dc0ae1f) sha256:6f68aaca05c6962de93fd80fb32daaa22d6f688404aad8ae93a7efdcdb9636d9` |
| state old hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state old-expected-marked | tvm-ffi `985f85b8ed93b87efeb8ade1bf1efa378dc72d64 (old-expected-marked)` |
| state old-expected-marked engine header | `structural_mutate.h (expected.h sha256:8f1c03afdbae588e318e1f46d18be16c6a1e81d15bfbdc6ded10fc43c03b5992) sha256:6f68aaca05c6962de93fd80fb32daaa22d6f688404aad8ae93a7efdcdb9636d9` |
| state old-expected-marked hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state gold | tvm-ffi `f9f52bfc6c06c9658390c32aabbf2b207af6e77e (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_override_gold.h sha256:47233c529e10a1ca3442d5b6fa1287f130a309a65d65441ea91ec7fd01a63b87` |
| state uc-both-marked | tvm-ffi `a2c2744f1813a5848cc2f77a395ff3f1911db530 (uc-inline-dtor-expected-marked)` |
| state uc-both-marked engine header | `structural_mutate.h sha256:cd3fe38bda1c8d72c2d8922a3a12f4ca7c46bc5235eb3c6fe90eaa10b8013471` |
| state uc-both-marked hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| what differs | control: old = base 0909bb4 as shipped (pre-unchanged engine, expected.h unmarked, hooks tvm_hook_override.h); old-expected-marked = the same plus the five Expected<T> special members TVM_FFI_INLINE and nothing else; gold and uc-both-marked are the existing binaries |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved old/old-expected-marked/gold/uc-both-marked |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.41 at start, 1.04 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved old/old-expected-marked/gold/uc-both-marked, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse -- `floor` -- ns/node, old/old-expected-marked/gold/uc-both-marked against old

`subst` substitutes `Var`s.

| Case | own | old | old-expected-marked | gold | uc-both-marked |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 11.02 | 10.19 (-7.6%) | 8.34 (-24.3%) | 5.78 (-47.6%) |
| split-fuse-shared | moved | 13.32 | 12.11 (-9.1%) | 8.85 (-33.6%) | 6.06 (-54.5%) |
| split-fuse-distinct | retained | 8.80 | 8.14 (-7.5%) | 7.27 (-17.4%) | 4.77 (-45.8%) |
| split-fuse-distinct | moved | 12.30 | 11.13 (-9.5%) | 7.21 (-41.4%) | 4.94 (-59.9%) |

#### Expr-level -- split/fuse -- `identity` -- ns/node, old/old-expected-marked/gold/uc-both-marked against old

`subst` substitutes `Var`s.

| Case | own | old | old-expected-marked | gold | uc-both-marked |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 39.90 | 31.76 (-20.4%) | 35.13 (-12.0%) | 26.39 (-33.9%) |
| split-fuse-shared | moved | 41.76 | 33.06 (-20.8%) | 34.61 (-17.1%) | 27.00 (-35.3%) |
| split-fuse-distinct | retained | 31.90 | 25.41 (-20.4%) | 28.10 (-11.9%) | 21.00 (-34.2%) |
| split-fuse-distinct | moved | 34.43 | 27.06 (-21.4%) | 27.94 (-18.9%) | 22.98 (-33.2%) |

#### Expr-level -- split/fuse -- `subst` -- ns/node, old/old-expected-marked/gold/uc-both-marked against old

`subst` substitutes `Var`s.

| Case | own | old | old-expected-marked | gold | uc-both-marked |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 43.95 | 38.76 (-11.8%) | 43.72 (-0.5%) | 34.66 (-21.1%) |
| split-fuse-shared | moved | 44.47 | 37.33 (-16.0%) | 40.52 (-8.9%) | 34.05 (-23.4%) |
| split-fuse-distinct | retained | 35.21 | 30.48 (-13.5%) | 35.63 (+1.2%) | 27.62 (-21.6%) |
| split-fuse-distinct | moved | 34.63 | 28.34 (-18.2%) | 29.00 (-16.3%) | 24.91 (-28.1%) |

### Supporting

#### `never` -- ns/node, old/old-expected-marked/gold/uc-both-marked against old

| Case | own | old | old-expected-marked | gold | uc-both-marked |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 24.95 | 22.34 (-10.5%) | 22.33 (-10.5%) | 18.00 (-27.9%) |
| split-fuse-shared | moved | 25.13 | 22.90 (-8.9%) | 21.06 (-16.2%) | 15.94 (-36.6%) |
| split-fuse-distinct | retained | 19.69 | 17.09 (-13.2%) | 17.13 (-13.0%) | 14.26 (-27.6%) |
| split-fuse-distinct | moved | 20.86 | 18.81 (-9.8%) | 16.98 (-18.6%) | 13.62 (-34.7%) |

#### `functor` -- ns/node, old/old-expected-marked/gold/uc-both-marked against old

| Case | own | old | old-expected-marked | gold | uc-both-marked |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.38 | 35.57 (+0.5%) | 35.19 (-0.5%) | 35.63 (+0.7%) |
| split-fuse-shared | moved | 40.01 | 40.39 (+0.9%) | 40.36 (+0.9%) | 40.54 (+1.3%) |
| split-fuse-distinct | retained | 27.99 | 28.37 (+1.3%) | 27.99 (-0.0%) | 28.76 (+2.8%) |
| split-fuse-distinct | moved | 33.08 | 33.32 (+0.7%) | 33.50 (+1.3%) | 33.49 (+1.2%) |

#### `old` -- ns/node, old/old-expected-marked/gold/uc-both-marked against old

| Case | own | old | old-expected-marked | gold | uc-both-marked |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 54.72 | 54.65 (-0.1%) | 54.82 (+0.2%) | 53.83 (-1.6%) |
| split-fuse-shared | moved | 59.71 | 59.12 (-1.0%) | 58.90 (-1.4%) | 58.77 (-1.6%) |
| split-fuse-distinct | retained | 44.13 | 44.16 (+0.1%) | 44.09 (-0.1%) | 43.77 (-0.8%) |
| split-fuse-distinct | moved | 48.59 | 48.13 (-1.0%) | 48.25 (-0.7%) | 47.79 (-1.6%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

