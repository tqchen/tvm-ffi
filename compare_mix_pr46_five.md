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
| state uc-badf3a4-rebuild | tvm-ffi `badf3a477849c5a8ab9b8c9cfc30462b7731c947 (uc-badf3a4-rebuild)` |
| state uc-badf3a4-rebuild engine header | `structural_mutate.h (expected.h sha256:ad1270654f4500d43a8eaf6cb367dbbe2226d106e24494c6e22b0d4f574498b6) sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc-badf3a4-rebuild hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| state uc-badf3a4-typed | tvm-ffi `4e7ff399699cd9314847ef6053ffcbd331e9f743 (uc-badf3a4-typed)` |
| state uc-badf3a4-typed engine header | `structural_mutate.h (expected.h sha256:ad1270654f4500d43a8eaf6cb367dbbe2226d106e24494c6e22b0d4f574498b6) sha256:5ba81bb17bf7d97b9baafe638a7a778ab6f0e4d694a5dcf47f81de9f9993a466` |
| state uc-badf3a4-typed hook header | `tvm_override_uc.h sha256:37438e29f082440001af053b6f0458368490db5b391e088e314199d11ddc0de8` |
| what differs | five-way: gold and uc-both-marked existing; uc-badf3a4 = PR 46 head; uc-badf3a4-rebuild = the same head rebuilt from byte-identical source (only provenance strings differ); uc-badf3a4-typed = the head with 31f6948 typed const T& in-place overloads restored |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved gold/uc-both-marked/uc-badf3a4/uc-badf3a4-rebuild/uc-badf3a4-typed |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.24 at start, 1.07 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved gold/uc-both-marked/uc-badf3a4/uc-badf3a4-rebuild/uc-badf3a4-typed, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse -- `floor` -- ns/node, gold/uc-both-marked/uc-badf3a4/uc-badf3a4-rebuild/uc-badf3a4-typed against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-both-marked | uc-badf3a4 | uc-badf3a4-rebuild | uc-badf3a4-typed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 8.52 | 5.84 (-31.5%) | 5.54 (-35.0%) | 5.52 (-35.3%) | 6.64 (-22.1%) |
| split-fuse-shared | moved | 8.86 | 6.02 (-32.1%) | 7.57 (-14.6%) | 7.53 (-15.0%) | 5.80 (-34.6%) |
| split-fuse-distinct | retained | 7.13 | 4.79 (-32.9%) | 4.63 (-35.1%) | 4.53 (-36.5%) | 5.18 (-27.4%) |
| split-fuse-distinct | moved | 7.19 | 4.93 (-31.4%) | 7.73 (+7.5%) | 7.67 (+6.7%) | 4.64 (-35.4%) |

#### Expr-level -- split/fuse -- `identity` -- ns/node, gold/uc-both-marked/uc-badf3a4/uc-badf3a4-rebuild/uc-badf3a4-typed against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-both-marked | uc-badf3a4 | uc-badf3a4-rebuild | uc-badf3a4-typed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.12 | 26.38 (-24.9%) | 26.55 (-24.4%) | 26.71 (-23.9%) | 26.68 (-24.0%) |
| split-fuse-shared | moved | 34.58 | 26.94 (-22.1%) | 28.59 (-17.3%) | 27.84 (-19.5%) | 26.49 (-23.4%) |
| split-fuse-distinct | retained | 28.11 | 20.96 (-25.5%) | 21.20 (-24.6%) | 21.24 (-24.4%) | 21.37 (-24.0%) |
| split-fuse-distinct | moved | 28.12 | 23.05 (-18.0%) | 24.76 (-12.0%) | 24.00 (-14.6%) | 23.21 (-17.4%) |

#### Expr-level -- split/fuse -- `subst` -- ns/node, gold/uc-both-marked/uc-badf3a4/uc-badf3a4-rebuild/uc-badf3a4-typed against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc-both-marked | uc-badf3a4 | uc-badf3a4-rebuild | uc-badf3a4-typed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 43.70 | 34.65 (-20.7%) | 34.77 (-20.4%) | 35.09 (-19.7%) | 35.27 (-19.3%) |
| split-fuse-shared | moved | 40.54 | 34.03 (-16.1%) | 33.07 (-18.4%) | 32.86 (-19.0%) | 32.30 (-20.3%) |
| split-fuse-distinct | retained | 35.52 | 27.60 (-22.3%) | 27.76 (-21.9%) | 28.03 (-21.1%) | 28.21 (-20.6%) |
| split-fuse-distinct | moved | 28.98 | 24.98 (-13.8%) | 25.08 (-13.5%) | 25.05 (-13.6%) | 24.71 (-14.7%) |

### Supporting

#### `never` -- ns/node, gold/uc-both-marked/uc-badf3a4/uc-badf3a4-rebuild/uc-badf3a4-typed against gold

| Case | own | gold | uc-both-marked | uc-badf3a4 | uc-badf3a4-rebuild | uc-badf3a4-typed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 22.39 | 18.06 (-19.4%) | 18.61 (-16.9%) | 18.69 (-16.5%) | 18.65 (-16.7%) |
| split-fuse-shared | moved | 21.04 | 15.85 (-24.7%) | 17.75 (-15.6%) | 18.26 (-13.2%) | 16.02 (-23.9%) |
| split-fuse-distinct | retained | 16.96 | 14.24 (-16.1%) | 13.12 (-22.6%) | 13.61 (-19.7%) | 13.57 (-20.0%) |
| split-fuse-distinct | moved | 16.94 | 13.56 (-19.9%) | 15.80 (-6.7%) | 15.66 (-7.5%) | 13.48 (-20.4%) |

#### `functor` -- ns/node, gold/uc-both-marked/uc-badf3a4/uc-badf3a4-rebuild/uc-badf3a4-typed against gold

| Case | own | gold | uc-both-marked | uc-badf3a4 | uc-badf3a4-rebuild | uc-badf3a4-typed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.21 | 35.98 (+2.2%) | 35.40 (+0.5%) | 35.45 (+0.7%) | 35.51 (+0.9%) |
| split-fuse-shared | moved | 40.21 | 40.62 (+1.0%) | 40.06 (-0.4%) | 40.14 (-0.2%) | 40.31 (+0.3%) |
| split-fuse-distinct | retained | 27.80 | 28.46 (+2.4%) | 28.55 (+2.7%) | 28.42 (+2.2%) | 28.47 (+2.4%) |
| split-fuse-distinct | moved | 33.42 | 33.30 (-0.4%) | 33.48 (+0.2%) | 33.20 (-0.7%) | 33.44 (+0.1%) |

#### `old` -- ns/node, gold/uc-both-marked/uc-badf3a4/uc-badf3a4-rebuild/uc-badf3a4-typed against gold

| Case | own | gold | uc-both-marked | uc-badf3a4 | uc-badf3a4-rebuild | uc-badf3a4-typed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split-fuse-shared | retained | 54.55 | 53.74 (-1.5%) | 54.46 (-0.2%) | 54.51 (-0.1%) | 54.30 (-0.4%) |
| split-fuse-shared | moved | 58.69 | 58.78 (+0.2%) | 58.82 (+0.2%) | 58.27 (-0.7%) | 58.69 (+0.0%) |
| split-fuse-distinct | retained | 44.16 | 43.55 (-1.4%) | 44.20 (+0.1%) | 43.70 (-1.1%) | 43.93 (-0.5%) |
| split-fuse-distinct | moved | 48.20 | 47.86 (-0.7%) | 47.94 (-0.5%) | 47.72 (-1.0%) | 47.93 (-0.6%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

