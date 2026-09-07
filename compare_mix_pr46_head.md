### Provenance

| | |
| --- | --- |
| state gold | tvm-ffi `f9f52bfc6c06c9658390c32aabbf2b207af6e77e (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_override_gold.h sha256:47233c529e10a1ca3442d5b6fa1287f130a309a65d65441ea91ec7fd01a63b87` |
| state uc-both-marked | tvm-ffi `a2c2744f1813a5848cc2f77a395ff3f1911db530 (uc-inline-dtor-expected-marked)` |
| state uc-both-marked engine header | `structural_mutate.h sha256:cd3fe38bda1c8d72c2d8922a3a12f4ca7c46bc5235eb3c6fe90eaa10b8013471` |
| state uc-both-marked hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| state uc-badf3a4 | tvm-ffi `badf3a477849c5a8ab9b8c9cfc30462b7731c947 (uc-badf3a4, PR 46 head)` |
| state uc-badf3a4 engine header | `structural_mutate.h (expected.h sha256:ad1270654f4500d43a8eaf6cb367dbbe2226d106e24494c6e22b0d4f574498b6) sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc-badf3a4 hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| what differs | gold and uc-both-marked are the existing binaries; uc-badf3a4 is PR 46 head badf3a477849c5a8ab9b8c9cfc30462b7731c947 (refactor/structural-mutate-unchanged-or on staging: GOLD engine shape, VarRemapGetRaw, recovery fixes, UnchangedReturnProxy, marked special members on UnchangedOr, Expected<T>, Expected<void>) built with the same UC hook file, gcc 14.3.0 here against its author validation on gcc 11.4 |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved gold/uc-both-marked/uc-badf3a4 |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.29 at start, 1.00 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved gold/uc-both-marked/uc-badf3a4, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse -- `floor` -- ns/node, gold/uc-both-marked/uc-badf3a4 against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-both-marked | uc-badf3a4 |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 8.59 | 5.86 (-31.8%) | 5.68 (-33.9%) |
| split-fuse-shared | moved | 8.97 | 6.25 (-30.4%) | 7.70 (-14.2%) |
| split-fuse-distinct | retained | 7.33 | 4.92 (-32.8%) | 4.70 (-35.9%) |
| split-fuse-distinct | moved | 7.17 | 4.92 (-31.4%) | 7.71 (+7.6%) |

#### Expr-level -- split/fuse -- `identity` -- ns/node, gold/uc-both-marked/uc-badf3a4 against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-both-marked | uc-badf3a4 |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.14 | 26.35 (-25.0%) | 26.57 (-24.4%) |
| split-fuse-shared | moved | 34.57 | 26.81 (-22.5%) | 28.60 (-17.3%) |
| split-fuse-distinct | retained | 28.15 | 20.94 (-25.6%) | 21.22 (-24.6%) |
| split-fuse-distinct | moved | 28.01 | 23.08 (-17.6%) | 24.70 (-11.8%) |

#### Expr-level -- split/fuse -- `subst` -- ns/node, gold/uc-both-marked/uc-badf3a4 against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-both-marked | uc-badf3a4 |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 43.67 | 34.69 (-20.6%) | 34.56 (-20.9%) |
| split-fuse-shared | moved | 40.41 | 34.13 (-15.6%) | 33.10 (-18.1%) |
| split-fuse-distinct | retained | 35.69 | 27.68 (-22.5%) | 27.65 (-22.5%) |
| split-fuse-distinct | moved | 29.06 | 24.92 (-14.3%) | 25.09 (-13.7%) |

### Supporting

#### `never` -- ns/node, gold/uc-both-marked/uc-badf3a4 against gold

| Case | own | gold | uc-both-marked | uc-badf3a4 |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 22.35 | 18.05 (-19.2%) | 18.60 (-16.7%) |
| split-fuse-shared | moved | 21.01 | 15.91 (-24.3%) | 17.80 (-15.3%) |
| split-fuse-distinct | retained | 16.94 | 14.23 (-16.0%) | 13.05 (-22.9%) |
| split-fuse-distinct | moved | 17.00 | 13.60 (-20.0%) | 15.86 (-6.7%) |

#### `functor` -- ns/node, gold/uc-both-marked/uc-badf3a4 against gold

| Case | own | gold | uc-both-marked | uc-badf3a4 |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.11 | 35.80 (+2.0%) | 35.40 (+0.8%) |
| split-fuse-shared | moved | 40.28 | 40.87 (+1.5%) | 40.14 (-0.4%) |
| split-fuse-distinct | retained | 27.93 | 28.87 (+3.4%) | 28.65 (+2.6%) |
| split-fuse-distinct | moved | 33.46 | 33.63 (+0.5%) | 33.29 (-0.5%) |

#### `old` -- ns/node, gold/uc-both-marked/uc-badf3a4 against gold

| Case | own | gold | uc-both-marked | uc-badf3a4 |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 54.36 | 53.88 (-0.9%) | 54.16 (-0.4%) |
| split-fuse-shared | moved | 58.69 | 58.77 (+0.1%) | 58.48 (-0.4%) |
| split-fuse-distinct | retained | 44.07 | 43.43 (-1.4%) | 44.11 (+0.1%) |
| split-fuse-distinct | moved | 48.06 | 47.54 (-1.1%) | 47.69 (-0.8%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

