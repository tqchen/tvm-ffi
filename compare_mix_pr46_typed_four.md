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
| state uc-badf3a4-typed | tvm-ffi `4e7ff399699cd9314847ef6053ffcbd331e9f743 (uc-badf3a4-typed)` |
| state uc-badf3a4-typed engine header | `structural_mutate.h (expected.h sha256:ad1270654f4500d43a8eaf6cb367dbbe2226d106e24494c6e22b0d4f574498b6) sha256:5ba81bb17bf7d97b9baafe638a7a778ab6f0e4d694a5dcf47f81de9f9993a466` |
| state uc-badf3a4-typed hook header | `tvm_override_uc.h sha256:37438e29f082440001af053b6f0458368490db5b391e088e314199d11ddc0de8` |
| what differs | one variable on PR 46 head badf3a4: uc-badf3a4-typed restores 31f6948 typed const T& overloads of MaybeInplaceMutateExpected and MaybeInplaceMutateIfUniqueExpected beside the AnyView forms; the hook names <Any> at its one unexecuted Var-ty site so that compiles |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved gold/uc-both-marked/uc-badf3a4/uc-badf3a4-typed |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.41 at start, 1.02 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved gold/uc-both-marked/uc-badf3a4/uc-badf3a4-typed, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse -- `floor` -- ns/node, gold/uc-both-marked/uc-badf3a4/uc-badf3a4-typed against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-both-marked | uc-badf3a4 | uc-badf3a4-typed |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 8.38 | 5.85 (-30.2%) | 5.58 (-33.4%) | 5.96 (-28.8%) |
| split-fuse-shared | moved | 8.83 | 6.14 (-30.4%) | 7.70 (-12.9%) | 5.77 (-34.7%) |
| split-fuse-distinct | retained | 7.23 | 4.98 (-31.2%) | 4.60 (-36.5%) | 4.58 (-36.7%) |
| split-fuse-distinct | moved | 7.13 | 5.13 (-28.0%) | 7.87 (+10.4%) | 4.65 (-34.8%) |

#### Expr-level -- split/fuse -- `identity` -- ns/node, gold/uc-both-marked/uc-badf3a4/uc-badf3a4-typed against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-both-marked | uc-badf3a4 | uc-badf3a4-typed |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.00 | 26.40 (-24.6%) | 26.55 (-24.1%) | 26.62 (-23.9%) |
| split-fuse-shared | moved | 34.45 | 27.06 (-21.4%) | 28.58 (-17.0%) | 26.41 (-23.3%) |
| split-fuse-distinct | retained | 28.08 | 21.00 (-25.2%) | 21.24 (-24.3%) | 21.31 (-24.1%) |
| split-fuse-distinct | moved | 28.03 | 22.98 (-18.0%) | 24.79 (-11.6%) | 23.19 (-17.2%) |

#### Expr-level -- split/fuse -- `subst` -- ns/node, gold/uc-both-marked/uc-badf3a4/uc-badf3a4-typed against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-both-marked | uc-badf3a4 | uc-badf3a4-typed |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 43.72 | 34.61 (-20.8%) | 34.59 (-20.9%) | 35.12 (-19.7%) |
| split-fuse-shared | moved | 40.67 | 34.03 (-16.3%) | 33.26 (-18.2%) | 32.08 (-21.1%) |
| split-fuse-distinct | retained | 35.57 | 27.60 (-22.4%) | 27.71 (-22.1%) | 28.16 (-20.8%) |
| split-fuse-distinct | moved | 29.01 | 24.85 (-14.4%) | 25.16 (-13.3%) | 24.38 (-16.0%) |

### Supporting

#### `never` -- ns/node, gold/uc-both-marked/uc-badf3a4/uc-badf3a4-typed against gold

| Case | own | gold | uc-both-marked | uc-badf3a4 | uc-badf3a4-typed |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 22.13 | 17.97 (-18.8%) | 18.52 (-16.3%) | 18.52 (-16.3%) |
| split-fuse-shared | moved | 21.06 | 15.88 (-24.6%) | 17.85 (-15.2%) | 16.15 (-23.3%) |
| split-fuse-distinct | retained | 16.82 | 14.23 (-15.4%) | 13.42 (-20.2%) | 13.32 (-20.8%) |
| split-fuse-distinct | moved | 16.98 | 13.52 (-20.4%) | 15.77 (-7.1%) | 13.44 (-20.8%) |

#### `functor` -- ns/node, gold/uc-both-marked/uc-badf3a4/uc-badf3a4-typed against gold

| Case | own | gold | uc-both-marked | uc-badf3a4 | uc-badf3a4-typed |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.38 | 35.78 (+1.1%) | 35.37 (-0.0%) | 35.46 (+0.2%) |
| split-fuse-shared | moved | 40.39 | 40.62 (+0.6%) | 40.14 (-0.6%) | 40.19 (-0.5%) |
| split-fuse-distinct | retained | 28.33 | 28.68 (+1.2%) | 28.43 (+0.3%) | 28.65 (+1.1%) |
| split-fuse-distinct | moved | 33.50 | 33.37 (-0.4%) | 33.19 (-0.9%) | 33.45 (-0.2%) |

#### `old` -- ns/node, gold/uc-both-marked/uc-badf3a4/uc-badf3a4-typed against gold

| Case | own | gold | uc-both-marked | uc-badf3a4 | uc-badf3a4-typed |
| --- | --- | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 54.74 | 53.57 (-2.1%) | 54.12 (-1.1%) | 54.07 (-1.2%) |
| split-fuse-shared | moved | 58.61 | 58.38 (-0.4%) | 58.38 (-0.4%) | 58.56 (-0.1%) |
| split-fuse-distinct | retained | 44.14 | 43.53 (-1.4%) | 43.69 (-1.0%) | 43.96 (-0.4%) |
| split-fuse-distinct | moved | 48.11 | 47.66 (-0.9%) | 47.64 (-1.0%) | 47.64 (-1.0%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

