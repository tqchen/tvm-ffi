### Provenance

| | |
| --- | --- |
| state old-expected-marked | tvm-ffi `985f85b8ed93b87efeb8ade1bf1efa378dc72d64 (old-expected-marked)` |
| state old-expected-marked engine header | `structural_mutate.h (expected.h sha256:8f1c03afdbae588e318e1f46d18be16c6a1e81d15bfbdc6ded10fc43c03b5992) sha256:6f68aaca05c6962de93fd80fb32daaa22d6f688404aad8ae93a7efdcdb9636d9` |
| state old-expected-marked hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state old-expected-marked-rawless | tvm-ffi `f8b41aa816c38c9f1a6fd532e4fd5610c2faea0a (old-expected-marked-rawless)` |
| state old-expected-marked-rawless engine header | `structural_mutate.h (expected.h sha256:8f1c03afdbae588e318e1f46d18be16c6a1e81d15bfbdc6ded10fc43c03b5992) sha256:c29fcb17993f85280355dc112731e640ffb15ef97726fa48b5e0f9b9685d8e03` |
| state old-expected-marked-rawless hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| what differs | one variable: DefaultMutateRaw returns Expected<Any> (DefaultMutateExpected body) and its raw callers convert with ExpectedUnsafe::MoveToTVMFFIAny at the vtable boundary; nothing else |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved old-expected-marked/old-expected-marked-rawless |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.33 at start, 0.97 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |

**Absolutes come from separately compiled binaries and are not comparable across states; only the delta in each cell is claimed.** Two builds differ in inlining, layout and allocator luck for reasons unrelated to what is under study. The processes are interleaved old-expected-marked/old-expected-marked-rawless, so drift and thermal state hit all of them equally.

### Headline

**The drift band is `functor` / `old`.** Those arms are TVM's own `Substitute` and `StmtExprMutator` and do not dispatch through the structural engine at all, so nothing under study can move them: whatever they move by is this run's noise, and a row inside that band is not a result. **`floor` is not a general control.** `MinimalMutatorObj` carries its own `Mutate`, which never reaches `DefaultMutateRaw`, so `floor` is blind by construction to an engine change below the vtable; it prices the callback layer and the hook file, and a hook rewrite moves it by construction. `identity` and `subst`/`swap` are read against whatever the states differ in, named above.

#### Expr-level -- split/fuse -- `floor` -- ns/node, old-expected-marked/old-expected-marked-rawless against old-expected-marked

`subst` substitutes `Var`s.

| Case | own | old-expected-marked | old-expected-marked-rawless |
| --- | --- | ---: | ---: |
| split-fuse-shared | retained | 10.19 | 10.29 (+1.0%) |
| split-fuse-shared | moved | 12.13 | 12.29 (+1.4%) |
| split-fuse-distinct | retained | 8.15 | 8.16 (+0.2%) |
| split-fuse-distinct | moved | 11.13 | 11.23 (+1.0%) |

#### Expr-level -- split/fuse -- `identity` -- ns/node, old-expected-marked/old-expected-marked-rawless against old-expected-marked

`subst` substitutes `Var`s.

| Case | own | old-expected-marked | old-expected-marked-rawless |
| --- | --- | ---: | ---: |
| split-fuse-shared | retained | 31.70 | 31.98 (+0.9%) |
| split-fuse-shared | moved | 32.99 | 33.22 (+0.7%) |
| split-fuse-distinct | retained | 25.43 | 25.67 (+1.0%) |
| split-fuse-distinct | moved | 27.10 | 27.33 (+0.9%) |

#### Expr-level -- split/fuse -- `subst` -- ns/node, old-expected-marked/old-expected-marked-rawless against old-expected-marked

`subst` substitutes `Var`s.

| Case | own | old-expected-marked | old-expected-marked-rawless |
| --- | --- | ---: | ---: |
| split-fuse-shared | retained | 38.69 | 38.24 (-1.2%) |
| split-fuse-shared | moved | 37.31 | 37.18 (-0.3%) |
| split-fuse-distinct | retained | 30.49 | 30.45 (-0.1%) |
| split-fuse-distinct | moved | 28.32 | 28.49 (+0.6%) |

### Supporting

#### `never` -- ns/node, old-expected-marked/old-expected-marked-rawless against old-expected-marked

| Case | own | old-expected-marked | old-expected-marked-rawless |
| --- | --- | ---: | ---: |
| split-fuse-shared | retained | 21.48 | 22.26 (+3.6%) |
| split-fuse-shared | moved | 22.89 | 22.60 (-1.3%) |
| split-fuse-distinct | retained | 17.12 | 16.86 (-1.6%) |
| split-fuse-distinct | moved | 18.87 | 19.01 (+0.8%) |

#### `functor` -- ns/node, old-expected-marked/old-expected-marked-rawless against old-expected-marked

| Case | own | old-expected-marked | old-expected-marked-rawless |
| --- | --- | ---: | ---: |
| split-fuse-shared | retained | 35.48 | 35.47 (-0.0%) |
| split-fuse-shared | moved | 40.23 | 40.24 (+0.0%) |
| split-fuse-distinct | retained | 28.28 | 28.45 (+0.6%) |
| split-fuse-distinct | moved | 33.15 | 33.40 (+0.8%) |

#### `old` -- ns/node, old-expected-marked/old-expected-marked-rawless against old-expected-marked

| Case | own | old-expected-marked | old-expected-marked-rawless |
| --- | --- | ---: | ---: |
| split-fuse-shared | retained | 54.53 | 55.18 (+1.2%) |
| split-fuse-shared | moved | 59.02 | 59.63 (+1.0%) |
| split-fuse-distinct | retained | 44.10 | 44.50 (+0.9%) |
| split-fuse-distinct | moved | 48.01 | 48.71 (+1.5%) |

[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for those rows rather than missing from them.

