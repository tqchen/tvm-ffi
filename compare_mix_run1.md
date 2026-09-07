### Provenance

| | |
| --- | --- |
| state gold | tvm-ffi `e865d5ce0cd3cdbff62d115c4829b74a09733cc4 (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_override_gold.h sha256:47233c529e10a1ca3442d5b6fa1287f130a309a65d65441ea91ec7fd01a63b87` |
| state uc | tvm-ffi `e865d5ce0cd3cdbff62d115c4829b74a09733cc4 (uc)` |
| state uc engine header | `structural_mutate.h sha256:7996b957aea127c313ccaf8cebc98759d86d8050ee416f9ef9d2e4adce9481a7` |
| state uc hook header | `tvm_override_uc.h sha256:5197cae9a73927cfc717ec53a297731cbcd58f7e1be519ff149848f547b14882` |
| state mixed | tvm-ffi `e865d5ce0cd3cdbff62d115c4829b74a09733cc4 (mixed)` |
| state mixed engine header | `structural_mutate_mixed.h sha256:82df66a519275ee5a078116330b8fe4d8fb37e388940296436172be25098dab1` |
| state mixed hook header | `tvm_override_mixed.h sha256:499da2ddf35daa7a14591a666e6b6b5dc2d820f82f31fb750f3aa80ff1a5a4ad` |
| what differs | one tree (bench/386-gold-uc-mix e865d5c), one apache/tvm 2c28965, one libtvm_ffi/libtvm_compiler build; each executable is compiled against one engine header and one hook file: gold = structural_mutate_gold.h (GOLD 730d6fc byte for byte) + tvm_override_gold.h; uc = structural_mutate.h (UC 31f6948) + tvm_override_uc.h (UC*g*, the store kept off the unchanged path); mixed = structural_mutate_mixed.h (GOLD carrying the UC protocol alongside) + tvm_override_mixed.h (the UC hooks on the mixed entries) |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved gold/uc/mixed |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.91 at start, 1.10 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved gold/uc/mixed, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse -- `floor` -- ns/node, gold/uc/mixed against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc | mixed |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 8.55 | 10.99 (+28.6%) | 11.55 (+35.1%) |
| split-fuse-shared | moved | 8.82 | 10.60 (+20.2%) | 10.79 (+22.4%) |
| split-fuse-distinct | retained | 7.25 | 9.06 (+25.0%) | 9.78 (+35.0%) |
| split-fuse-distinct | moved | 7.19 | 7.89 (+9.7%) | 8.14 (+13.3%) |

#### Expr-level -- split/fuse -- `identity` -- ns/node, gold/uc/mixed against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc | mixed |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 38.40 | 44.30 (+15.4%) | 39.72 (+3.4%) |
| split-fuse-shared | moved | 37.63 | 43.85 (+16.5%) | 38.80 (+3.1%) |
| split-fuse-distinct | retained | 30.75 | 35.46 (+15.3%) | 31.88 (+3.7%) |
| split-fuse-distinct | moved | 31.42 | 36.23 (+15.3%) | 32.40 (+3.1%) |

#### Expr-level -- split/fuse -- `subst` -- ns/node, gold/uc/mixed against gold

`subst` substitutes `Var`s.

| Case | own | gold | uc | mixed |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 43.94 | 52.97 (+20.5%) | 46.15 (+5.0%) |
| split-fuse-shared | moved | 41.08 | 50.41 (+22.7%) | 45.67 (+11.2%) |
| split-fuse-distinct | retained | 35.43 | 42.23 (+19.2%) | 37.61 (+6.2%) |
| split-fuse-distinct | moved | 29.56 | 37.68 (+27.5%) | 34.18 (+15.6%) |

### Supporting

#### `never` -- ns/node, gold/uc/mixed against gold

| Case | own | gold | uc | mixed |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 22.48 | 24.70 (+9.9%) | 28.56 (+27.1%) |
| split-fuse-shared | moved | 21.41 | 24.35 (+13.8%) | 24.76 (+15.7%) |
| split-fuse-distinct | retained | 17.21 | 19.65 (+14.2%) | 22.40 (+30.1%) |
| split-fuse-distinct | moved | 17.38 | 18.86 (+8.5%) | 18.39 (+5.8%) |

#### `functor` -- ns/node, gold/uc/mixed against gold

| Case | own | gold | uc | mixed |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 35.75 | 35.59 (-0.5%) | 35.84 (+0.3%) |
| split-fuse-shared | moved | 40.79 | 40.05 (-1.8%) | 40.35 (-1.1%) |
| split-fuse-distinct | retained | 28.37 | 28.17 (-0.7%) | 28.46 (+0.3%) |
| split-fuse-distinct | moved | 33.94 | 33.19 (-2.2%) | 33.38 (-1.7%) |

#### `old` -- ns/node, gold/uc/mixed against gold

| Case | own | gold | uc | mixed |
| --- | --- | ---: | ---: | ---: |
| split-fuse-shared | retained | 55.08 | 55.72 (+1.2%) | 55.04 (-0.1%) |
| split-fuse-shared | moved | 59.68 | 59.88 (+0.3%) | 59.20 (-0.8%) |
| split-fuse-distinct | retained | 44.45 | 44.63 (+0.4%) | 44.46 (+0.0%) |
| split-fuse-distinct | moved | 48.62 | 48.51 (-0.2%) | 48.50 (-0.3%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

