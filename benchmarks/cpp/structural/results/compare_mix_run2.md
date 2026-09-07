### Provenance

| | |
| --- | --- |
| state gold | tvm-ffi `f9f52bfc6c06c9658390c32aabbf2b207af6e77e (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_override_gold.h sha256:47233c529e10a1ca3442d5b6fa1287f130a309a65d65441ea91ec7fd01a63b87` |
| state uc | tvm-ffi `f9f52bfc6c06c9658390c32aabbf2b207af6e77e (uc)` |
| state uc engine header | `structural_mutate.h sha256:ed40dd82d5e97edee66bb1c655cd012fbb06e77de4649331e2d626a98b112b90` |
| state uc hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| state mixed | tvm-ffi `f9f52bfc6c06c9658390c32aabbf2b207af6e77e (mixed)` |
| state mixed engine header | `structural_mutate_mixed.h sha256:82df66a519275ee5a078116330b8fe4d8fb37e388940296436172be25098dab1` |
| state mixed hook header | `tvm_override_mixed.h sha256:499da2ddf35daa7a14591a666e6b6b5dc2d820f82f31fb750f3aa80ff1a5a4ad` |
| what differs | run 2, after f9f52bf: as run 1, except that structural_mutate.h (uc) now returns the unchanged marker through details::UnchangedReturnProxy, with no carrier temporary and no destructor on the unchanged exit; gold and mixed are unchanged source rebuilt from the same tree |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved gold/uc/mixed |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 2.31 at start, 1.21 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved gold/uc/mixed, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse -- `floor` -- ns/node, gold/uc/mixed against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc | mixed |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 8.47 | 8.81 (+4.0%) | 11.43 (+34.9%) |
| split-fuse-shared | moved | 8.88 | 8.78 (-1.1%) | 10.62 (+19.6%) |
| split-fuse-distinct | retained | 7.30 | 7.17 (-1.8%) | 9.40 (+28.7%) |
| split-fuse-distinct | moved | 7.05 | 7.02 (-0.5%) | 7.95 (+12.8%) |

#### Expr-level -- split/fuse -- `identity` -- ns/node, gold/uc/mixed against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc | mixed |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.55 | 39.82 (+12.0%) | 40.37 (+13.6%) |
| split-fuse-shared | moved | 34.97 | 39.46 (+12.8%) | 38.14 (+9.1%) |
| split-fuse-distinct | retained | 28.43 | 31.86 (+12.1%) | 32.40 (+14.0%) |
| split-fuse-distinct | moved | 28.56 | 34.27 (+20.0%) | 32.57 (+14.0%) |

#### Expr-level -- split/fuse -- `subst` -- ns/node, gold/uc/mixed against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc | mixed |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 44.34 | 51.22 (+15.5%) | 44.46 (+0.3%) |
| split-fuse-shared | moved | 41.22 | 48.66 (+18.0%) | 45.06 (+9.3%) |
| split-fuse-distinct | retained | 36.17 | 40.38 (+11.6%) | 37.04 (+2.4%) |
| split-fuse-distinct | moved | 29.50 | 35.87 (+21.6%) | 33.88 (+14.8%) |

### Supporting

#### `never` -- ns/node, gold/uc/mixed against gold

| Case | own | gold | uc | mixed |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 22.56 | 22.55 (-0.0%) | 26.93 (+19.4%) |
| split-fuse-shared | moved | 21.46 | 22.47 (+4.7%) | 24.55 (+14.4%) |
| split-fuse-distinct | retained | 17.32 | 18.11 (+4.6%) | 21.38 (+23.5%) |
| split-fuse-distinct | moved | 17.21 | 18.32 (+6.5%) | 18.72 (+8.8%) |

#### `functor` -- ns/node, gold/uc/mixed against gold

| Case | own | gold | uc | mixed |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.63 | 35.44 (-0.5%) | 35.61 (-0.0%) |
| split-fuse-shared | moved | 40.71 | 40.56 (-0.4%) | 40.32 (-1.0%) |
| split-fuse-distinct | retained | 28.03 | 28.67 (+2.3%) | 28.64 (+2.2%) |
| split-fuse-distinct | moved | 33.57 | 33.75 (+0.5%) | 33.47 (-0.3%) |

#### `old` -- ns/node, gold/uc/mixed against gold

| Case | own | gold | uc | mixed |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 54.98 | 54.75 (-0.4%) | 54.68 (-0.6%) |
| split-fuse-shared | moved | 59.23 | 59.08 (-0.2%) | 58.96 (-0.4%) |
| split-fuse-distinct | retained | 44.36 | 44.18 (-0.4%) | 44.41 (+0.1%) |
| split-fuse-distinct | moved | 48.41 | 48.18 (-0.5%) | 48.43 (+0.0%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

