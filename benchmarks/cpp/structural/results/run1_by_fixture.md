# bench/tvm-hook `fe046c8`: GOLD vs UC, all fixtures, one interleave on Ryzen 9 7950X

Harness commit `fe046c8c55078c6071e1c28311c36f5e10650b5d` (branch `bench/tvm-hook`) on upstream `main` `e74e58f9d8a6fb1ece5722d0715f68187a8e0385`; UC engine = PR 46 as resolved, `7a569c5a5b1be1318900ffe0b3723341a084b0af`; GOLD engine = `730d6fc` byte for byte; `expected.h` sha256 `1b62f4d408328408280b2e167ed5bdc7f4a34b6032842da166ca9516c4a8b877` (shared by both executables). Both hook files and both engines as reviewed at `fe046c8`; their sha256 are in the provenance rows below, stamped by the executables themselves. Raw `report.py` output (one table per arm) is `compare_full_run1.md` beside this file; this file is the same numbers arranged one table per fixture.

## Provenance

| | |
| --- | --- |
| state gold | tvm-ffi `fe046c8c55078c6071e1c28311c36f5e10650b5d (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_hook_override_gold.h sha256:20bd4475ef7e1096b54629015ce86d40d35dffd6f9f02a918947fb527d8360d4` |
| state uc | tvm-ffi `fe046c8c55078c6071e1c28311c36f5e10650b5d (uc)` |
| state uc engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| what differs | one tree (bench/tvm-hook fe046c8 on upstream main e74e58f), one apache/tvm 2c28965, one libtvm_ffi/libtvm_compiler build; each executable is compiled against one engine header and one hook file: gold = structural_mutate_gold.h (GOLD 730d6fc byte for byte) + tvm_hook_override_gold.h; uc = structural_mutate.h (PR 46 as resolved, 7a569c5) + tvm_hook_override_uc.h; hook files and engines as reviewed at fe046c8, sha256 in the rows above |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved gold/uc |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.15 at start, 1.13 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |


**Method.** `report.py --two-state`, 21 pinned (`taskset -c 20`) processes per state, strict interleave gold/uc/gold/uc, quiet gate on both signals at start and end (load 0.15 / 1.13, no competing benchmark process), both binaries sha256-checked before and after the run and unchanged (`real_tvm_bench_gold` `f669457e...`, `real_tvm_bench_uc` `be49f1db...`). Per process: one untimed warm-up, nine timed samples per arm, process value = median of nine; reported value = median of the 21 process medians. `ns/node` divides by each fixture's declared unique-node count `N` (fixtures table: split-fuse-shared 12, split-fuse-distinct 15, call-split-fuse-shared 18, call-split-fuse-distinct 23, seq-16 68, seq-256 1028, seq-16384 65540), the same divisor for every arm in a row. Absolutes come from two separately compiled binaries and are not comparable across states; only the interleaved delta is claimed.

**Drift band.** `functor` and `old` never enter the structural engine, so their movement is this run's noise: **-0.6% to +1.4%** over the 16 control cells. A delta inside that band is not a result.

**Checks that fail the run** (README, "Checks that fail the run"), executed untimed by every one of the 42 processes before its first timed sample; a failure exits nonzero and `report.py` refuses the run. All passed in every process: in-place actually fires (`CheckInplace`, moved root same object, retained root not); an identity arm rebuilds nothing (`CheckIdentityPointers`, both ownerships); the swap does what it claims (`CheckSeqSwap`, exactly the targeted elements differ, involution restores the pointers; all three lengths); splice matches the reference (`CheckSpliceAgainstReference`, thirteen cases); the no-op asymmetry is pinned (`CheckNoOpAsymmetry`); the declared fixture counts are the real ones (`CheckDeclaredCounts`, all seven fixtures); hook coverage (`CheckHookCoverage`, every type each fixture dispatches on has a harness hook in both hook files). Not exercised by this run: layout parity and port fidelity, which belong to the mini-TIR fidelity run and to `tvm_hook_override.h`'s port check respectively; the per-engine hook files are hand-written and were reviewed, not port-checked.


## Tables, one per fixture -- ns/node, UC delta against GOLD


### `split-fuse-shared`

| arm | own | gold | uc | uc vs gold |
| --- | --- | ---: | ---: | ---: |
| floor | retained | 8.65 | 5.88 | -32.0% |
| floor | moved | 8.39 | 5.69 | -32.2% |
| identity | retained | 28.83 | 25.79 | -10.5% |
| identity | moved | 28.98 | 25.94 | -10.5% |
| subst | retained | 37.66 | 35.99 | -4.4% |
| subst | moved | 33.34 | 30.40 | -8.8% |
| never | retained | 21.28 | 18.04 | -15.2% |
| never | moved | 19.73 | 16.55 | -16.1% |
| functor | retained | 35.74 | 35.73 | -0.0% |
| functor | moved | 40.77 | 40.83 | +0.1% |
| old | retained | 53.83 | 53.50 | -0.6% |
| old | moved | 58.08 | 58.21 | +0.2% |

### `split-fuse-distinct`

| arm | own | gold | uc | uc vs gold |
| --- | --- | ---: | ---: | ---: |
| floor | retained | 7.18 | 4.73 | -34.1% |
| floor | moved | 6.49 | 4.70 | -27.6% |
| identity | retained | 23.09 | 20.64 | -10.6% |
| identity | moved | 22.93 | 20.77 | -9.4% |
| subst | retained | 30.24 | 28.82 | -4.7% |
| subst | moved | 24.08 | 22.25 | -7.6% |
| never | retained | 17.07 | 14.17 | -17.0% |
| never | moved | 15.27 | 12.39 | -18.9% |
| functor | retained | 28.43 | 28.82 | +1.4% |
| functor | moved | 33.27 | 33.27 | +0.0% |
| old | retained | 43.53 | 43.36 | -0.4% |
| old | moved | 47.69 | 47.73 | +0.1% |

### `call-split-fuse-shared`

| arm | own | gold | uc | uc vs gold |
| --- | --- | ---: | ---: | ---: |
| floor | retained | 8.25 | 6.44 | -21.9% |
| floor | moved | 7.75 | 6.02 | -22.3% |
| identity | retained | 22.87 | 21.01 | -8.1% |
| identity | moved | 23.93 | 22.15 | -7.4% |
| subst | retained | 37.98 | 38.01 | +0.1% |
| subst | moved | 28.24 | 27.83 | -1.5% |
| never | retained | 16.83 | 14.84 | -11.8% |
| never | moved | 17.15 | 15.05 | -12.2% |
| functor | retained | 57.14 | 57.40 | +0.5% |
| functor | moved | 59.57 | 60.02 | +0.8% |
| old | retained | 64.65 | 64.77 | +0.2% |
| old | moved | 67.48 | 67.69 | +0.3% |

### `call-split-fuse-distinct`

| arm | own | gold | uc | uc vs gold |
| --- | --- | ---: | ---: | ---: |
| floor | retained | 6.63 | 5.07 | -23.5% |
| floor | moved | 5.83 | 4.56 | -21.8% |
| identity | retained | 18.07 | 16.42 | -9.1% |
| identity | moved | 17.99 | 16.87 | -6.2% |
| subst | retained | 30.67 | 30.85 | +0.6% |
| subst | moved | 18.73 | 18.43 | -1.6% |
| never | retained | 13.23 | 11.55 | -12.7% |
| never | moved | 13.16 | 11.61 | -11.8% |
| functor | retained | 46.07 | 46.16 | +0.2% |
| functor | moved | 46.78 | 47.18 | +0.9% |
| old | retained | 52.37 | 52.31 | -0.1% |
| old | moved | 53.24 | 53.27 | +0.1% |

### `seq-16`

| arm | own | gold | uc | uc vs gold |
| --- | --- | ---: | ---: | ---: |
| floor | retained | 9.48 | 6.86 | -27.6% |
| floor | moved | 8.71 | 6.59 | -24.3% |
| identity | retained | 19.81 | 16.98 | -14.3% |
| identity | moved | 19.28 | 16.40 | -14.9% |
| swap | retained | 20.99 | 18.92 | -9.9% |
| swap | moved | 19.65 | 17.36 | -11.7% |
| never | retained | 17.17 | 13.68 | -20.3% |
| never | moved | 15.78 | 12.76 | -19.1% |

### `seq-256`

| arm | own | gold | uc | uc vs gold |
| --- | --- | ---: | ---: | ---: |
| floor | retained | 9.77 | 7.13 | -27.0% |
| floor | moved | 8.79 | 6.59 | -25.0% |
| identity | retained | 19.88 | 16.83 | -15.3% |
| identity | moved | 18.92 | 15.69 | -17.1% |
| swap | retained | 20.75 | 18.86 | -9.1% |
| swap | moved | 19.32 | 17.28 | -10.6% |
| never | retained | 17.03 | 13.39 | -21.4% |
| never | moved | 15.32 | 12.43 | -18.9% |

### `seq-16384`

| arm | own | gold | uc | uc vs gold |
| --- | --- | ---: | ---: | ---: |
| floor | retained | 9.89 | 7.04 | -28.8% |
| floor | moved | 8.86 | 6.68 | -24.6% |
| identity | retained | 19.96 | 16.77 | -16.0% |
| identity | moved | 18.92 | 15.69 | -17.1% |
| swap | retained | 20.73 | 18.81 | -9.3% |
| swap | moved | 19.34 | 17.26 | -10.8% |
| never | retained | 17.05 | 13.32 | -21.9% |
| never | moved | 15.30 | 12.46 | -18.6% |

## Readings, one per fixture

Band is -0.6% to +1.4%; every delta quoted below is outside it unless it says otherwise. Absolutes are not compared across the two binaries.

- **`split-fuse-shared`.** UC below GOLD on every engine-touching arm, both ownerships: `floor` -32% (retained and moved alike, 5.9 against 8.7 ns/node), `identity` -10.5%, `subst` -4.5% retained / -8.8% moved, `never` -15..-16%. The round-1 floor gap is gone and inverted; the moved-slower-than-retained inversion the `badf3a4` head showed on `floor` is absent (uc 5.69 moved against 5.88 retained).
- **`split-fuse-distinct`.** Same picture: `floor` -34% retained / -27.5% moved, `identity` -10.6% / -9.4%, `subst` -4.7% / -7.6%, `never` -17% / -19%. The unshared tree moves `floor` and `never` a little more than the shared one.
- **`call-split-fuse-shared`.** The Call hook (container field, three skip guards) keeps the shape but narrows it: `floor` -22% both ownerships, `identity` -8.2% / -7.5%, `never` -12% / -12%. `subst` is **inside the band** (+0.1% / -1.4%): on the rebuilding arm with `Array<Expr>` descent, the two states are indistinguishable here.
- **`call-split-fuse-distinct`.** As shared: `floor` -23.5% / -21.8%, `identity` -9.1% / -6.2%, `never` -12.7% / -11.8%; `subst` inside the band (+0.6% / -1.6%). So on both Call fixtures the UC advantage is confined to the unchanged and unchanged-heavy arms, and the changed-path cost (`subst`) is at parity.
- **`seq-16`.** The Stmt-level hooks, first timed on this construction: `floor` -27.6% retained / -24.3% moved, `identity` -14.3% / -14.9%, `swap` -9.9% / -11.7%, `never` -20.3% / -19.1%. UC below GOLD on every row.
- **`seq-256`.** `floor` -27.0% / -25.0%, `identity` -15.3% / -17.1%, `swap` -9.1% / -10.6%, `never` -21.4% / -18.9%. Per-node values are within a few percent of `seq-16`'s in both states.
- **`seq-16384`.** `floor` -28.8% / -24.6%, `identity` -16.0% / -17.1%, `swap` -9.3% / -10.8%, `never` -21.9% / -18.6%. Again flat against the shorter lengths (3.5 MB working set, inside L3).

**What seq shows, explicitly.** The seq path -- `SeqStmtMutate` / `SeqStmtMaybeInplaceMutate` with the element loop, the `Evaluate` hooks, and the swap's two-element rebuild -- is where the H200 and GB200 ports disagreed (`ValueOrUnchanged(element)` lvalue against `std::move(element)`), and this is the first run with the standardized `std::move(element)` form on the copy-on-write sites and the explicit `IsUnchanged()` / `ValueUnchecked()` tests on the in-place sites. On it UC is below GOLD on all 24 seq cells, by a margin that is (a) the largest of any fixture on `never` (-19..-22%: the link test over 16..16384 elements, no callback, so almost pure per-element hook and carrier cost), (b) larger than the Expr fixtures' on `identity` (-14..-17% against -6..-11%), and (c) present on the rebuilding `swap` arm too (-9..-12%), where the Call fixtures' rebuilding arm sat inside the band. It is independent of sequence length across three orders of magnitude in both ownerships, so it is a per-element cost, not a cache or allocation effect. The `retained`/`moved` split is small on every seq row (1-4 points), i.e. the in-place element loop and the copy-on-write loop price the same in both states.

**Overall.** At `fe046c8` -- PR 46 as `7a569c5` on `e74e58f`, the reviewed hook pair -- UC is at or below GOLD on every one of the 78 engine-touching cells; the only cells at parity are the four Call `subst` cells, and none is above the band. The round-1 question (UC floor +10..+29% above GOLD) does not reproduce on this base with these hooks; the engine-side fixes that landed in PR 46 and #761, plus the early return in the in-place hooks, account for the sign change, and this run does not separate their shares.
