# bench/tvm-hook: OLD vs GOLD vs UC, all fixtures, one three-state interleave on Ryzen 9 7950X

Harness commit `3666d8f8f7ee64d495a3c3442a2c51a79e893cd4` (branch `bench/tvm-hook`) on upstream `main` `e74e58f9d8a6fb1ece5722d0715f68187a8e0385`. OLD engine = `e74e58f`'s `include/tvm/ffi/extra/structural_mutate.h` verbatim (sha256 `6f68aaca05c6962de93fd80fb32daaa22d6f688404aad8ae93a7efdcdb9636d9`), carried as `structural_mutate_old.h` with a seven-line banner (stamped sha256 `25e5c571...` below) under the untouched OLD hook file `tvm_hook_override.h`; the old executables also compile `e74e58f`'s container hooks so the OLD engine's Array descent is OLD's own. UC engine = PR 46 as resolved, `7a569c5a5b1be1318900ffe0b3723341a084b0af`; GOLD engine = `730d6fc` byte for byte; `expected.h` sha256 `1b62f4d408328408280b2e167ed5bdc7f4a34b6032842da166ca9516c4a8b877` (shared by all three executables). **The gold and uc executables are the run-1 binaries, byte-identical** (`real_tvm_bench_gold` sha256 `f669457ec8fd317fa29fda5521b7ec10f903345bb5c822a053fa1d5e7a3ca026`, `real_tvm_bench_uc` `be49f1dbff1b590107833d60df2f5a2fa0641a5f0e2b7f0aef6e14a66738d5ad`, built from `fe046c8`; their hook files and engines were not rebuilt); `real_tvm_bench_old` is `cb4792a57b5165b7f5d8ba0eec584ab2a103474d58c00c9c7f9cdfb407a10f0e`, built from `3666d8f`. All three were sha256-checked before and after the run and unchanged. Raw `report.py` output (one table per arm) is `compare_full_run2_old.md`; this file is the same numbers arranged one table per fixture.

## Provenance

| | |
| --- | --- |
| state old | tvm-ffi `3666d8f8f7ee64d495a3c3442a2c51a79e893cd4 (old)` |
| state old engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state gold | tvm-ffi `fe046c8c55078c6071e1c28311c36f5e10650b5d (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_hook_override_gold.h sha256:20bd4475ef7e1096b54629015ce86d40d35dffd6f9f02a918947fb527d8360d4` |
| state uc | tvm-ffi `fe046c8c55078c6071e1c28311c36f5e10650b5d (uc)` |
| state uc engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| what differs | one tree (bench/tvm-hook 3666d8f on upstream main e74e58f), one apache/tvm 2c28965, one libtvm_ffi/libtvm_compiler build; each executable is compiled against one engine header and one hook file: old = structural_mutate_old.h (upstream main e74e58f's engine verbatim, the pre-UnchangedOr engine with #760/#761 in its base) + tvm_hook_override.h (the OLD hook file, untouched) with e74e58f's container hooks compiled in; gold = structural_mutate_gold.h (GOLD 730d6fc byte for byte) + tvm_hook_override_gold.h; uc = structural_mutate.h (PR 46 as resolved, 7a569c5) + tvm_hook_override_uc.h. The gold and uc binaries are the run-1 binaries, sha256-identical; sha256 of every engine and hook file in the rows above |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved old/gold/uc |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 0.98 at start, 1.14 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |


**Method.** `report.py --two-state` with three states, 21 pinned (`taskset -c 20`) processes per state, strict rotation old/gold/uc, quiet gate on both signals at start and end (load 0.98 / 1.14, no competing benchmark process). Per process: one untimed warm-up, nine timed samples per arm, process value = median of nine; reported value = median of the 21 process medians. `ns/node` divides by each fixture's declared unique-node count `N` (split-fuse-shared 12, split-fuse-distinct 15, call-split-fuse-shared 18, call-split-fuse-distinct 23, seq-16 68, seq-256 1028, seq-16384 65540), the same divisor for every arm in a row. Absolutes come from three separately compiled binaries and are not comparable across states; only the interleaved deltas are claimed.

**Drift band.** `functor` and `old` never enter the structural engine: over the 32 control cells (16 fixtures-by-ownership x two arms, gold and uc against old) they move by **-1.4% to +1.2%**. A delta inside that band is not a result.

**Checks that fail the run** (README), executed untimed by every one of the 63 processes before its first timed sample; all passed: in-place fires, identity rebuilds nothing, swap involution, splice vs reference (13 cases), no-op asymmetry, declared counts (all seven fixtures), hook coverage (in all three hook files). Layout parity and port fidelity are not part of a gold/uc/old run.


## Tables, one per fixture -- ns/node; deltas gold vs old, uc vs old, uc vs gold


### `split-fuse-shared`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 11.51 | 8.42 | 5.91 | -26.8% | -48.7% | -29.8% |
| floor | moved | 11.48 | 7.98 | 5.70 | -30.5% | -50.3% | -28.6% |
| identity | retained | 31.71 | 28.84 | 25.77 | -9.1% | -18.7% | -10.6% |
| identity | moved | 31.43 | 29.01 | 25.80 | -7.7% | -17.9% | -11.1% |
| subst | retained | 38.69 | 37.83 | 36.01 | -2.2% | -6.9% | -4.8% |
| subst | moved | 34.70 | 33.43 | 30.13 | -3.7% | -13.2% | -9.9% |
| never | retained | 23.08 | 21.36 | 17.94 | -7.5% | -22.3% | -16.0% |
| never | moved | 21.82 | 19.66 | 16.39 | -9.9% | -24.9% | -16.6% |
| functor | retained | 35.82 | 35.91 | 35.58 | +0.3% | -0.7% | -0.9% |
| functor | moved | 40.65 | 40.88 | 40.35 | +0.6% | -0.7% | -1.3% |
| old | retained | 53.87 | 54.06 | 53.91 | +0.4% | +0.1% | -0.3% |
| old | moved | 58.17 | 58.46 | 58.29 | +0.5% | +0.2% | -0.3% |

### `split-fuse-distinct`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 9.19 | 6.97 | 4.66 | -24.2% | -49.3% | -33.1% |
| floor | moved | 9.42 | 6.37 | 4.61 | -32.4% | -51.1% | -27.6% |
| identity | retained | 25.42 | 23.11 | 20.60 | -9.1% | -19.0% | -10.9% |
| identity | moved | 24.88 | 22.96 | 20.77 | -7.7% | -16.5% | -9.5% |
| subst | retained | 30.58 | 30.31 | 28.77 | -0.9% | -5.9% | -5.1% |
| subst | moved | 25.81 | 24.10 | 22.27 | -6.6% | -13.7% | -7.6% |
| never | retained | 18.13 | 17.02 | 14.19 | -6.1% | -21.7% | -16.6% |
| never | moved | 17.31 | 15.16 | 12.32 | -12.4% | -28.8% | -18.7% |
| functor | retained | 28.32 | 28.47 | 28.43 | +0.5% | +0.4% | -0.1% |
| functor | moved | 33.29 | 33.23 | 33.14 | -0.2% | -0.5% | -0.3% |
| old | retained | 43.19 | 43.50 | 43.39 | +0.7% | +0.5% | -0.3% |
| old | moved | 47.36 | 47.79 | 47.74 | +0.9% | +0.8% | -0.1% |

### `call-split-fuse-shared`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 13.80 | 8.35 | 6.41 | -39.5% | -53.6% | -23.2% |
| floor | moved | 13.90 | 7.72 | 5.95 | -44.5% | -57.2% | -22.9% |
| identity | retained | 29.57 | 22.98 | 21.03 | -22.3% | -28.9% | -8.5% |
| identity | moved | 29.31 | 24.09 | 22.20 | -17.8% | -24.3% | -7.8% |
| subst | retained | 39.21 | 38.11 | 37.71 | -2.8% | -3.8% | -1.0% |
| subst | moved | 31.88 | 28.26 | 27.57 | -11.4% | -13.5% | -2.4% |
| never | retained | 21.80 | 16.94 | 14.76 | -22.3% | -32.3% | -12.9% |
| never | moved | 22.08 | 17.05 | 15.03 | -22.8% | -31.9% | -11.8% |
| functor | retained | 56.53 | 56.70 | 57.21 | +0.3% | +1.2% | +0.9% |
| functor | moved | 59.98 | 59.29 | 59.91 | -1.2% | -0.1% | +1.0% |
| old | retained | 65.14 | 64.88 | 64.56 | -0.4% | -0.9% | -0.5% |
| old | moved | 68.34 | 67.75 | 67.37 | -0.9% | -1.4% | -0.6% |

### `call-split-fuse-distinct`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 10.83 | 6.56 | 5.04 | -39.4% | -53.5% | -23.2% |
| floor | moved | 10.81 | 5.87 | 4.49 | -45.7% | -58.5% | -23.5% |
| identity | retained | 23.06 | 18.01 | 16.46 | -21.9% | -28.6% | -8.6% |
| identity | moved | 22.36 | 18.10 | 16.82 | -19.1% | -24.8% | -7.1% |
| subst | retained | 31.30 | 30.78 | 31.03 | -1.7% | -0.9% | +0.8% |
| subst | moved | 23.16 | 18.79 | 18.27 | -18.9% | -21.1% | -2.8% |
| never | retained | 17.08 | 13.16 | 11.57 | -23.0% | -32.3% | -12.1% |
| never | moved | 17.30 | 13.11 | 11.57 | -24.2% | -33.1% | -11.7% |
| functor | retained | 46.11 | 45.67 | 46.55 | -1.0% | +1.0% | +1.9% |
| functor | moved | 47.16 | 46.62 | 47.22 | -1.1% | +0.1% | +1.3% |
| old | retained | 53.23 | 52.62 | 52.67 | -1.1% | -1.1% | +0.1% |
| old | moved | 53.82 | 53.25 | 53.45 | -1.1% | -0.7% | +0.4% |

### `seq-16`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 11.84 | 9.18 | 6.96 | -22.5% | -41.2% | -24.2% |
| floor | moved | 10.68 | 8.56 | 6.55 | -19.9% | -38.7% | -23.5% |
| identity | retained | 21.20 | 19.84 | 16.98 | -6.4% | -19.9% | -14.4% |
| identity | moved | 19.65 | 19.31 | 16.30 | -1.7% | -17.0% | -15.6% |
| swap | retained | 22.95 | 21.01 | 18.90 | -8.5% | -17.6% | -10.0% |
| swap | moved | 21.14 | 19.72 | 17.33 | -6.7% | -18.0% | -12.1% |
| never | retained | 18.39 | 17.23 | 13.64 | -6.3% | -25.8% | -20.8% |
| never | moved | 16.82 | 15.72 | 12.66 | -6.5% | -24.7% | -19.5% |

### `seq-256`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 13.05 | 9.50 | 7.06 | -27.2% | -45.9% | -25.7% |
| floor | moved | 11.78 | 8.77 | 6.59 | -25.6% | -44.1% | -24.9% |
| identity | retained | 21.45 | 19.94 | 16.82 | -7.0% | -21.6% | -15.6% |
| identity | moved | 19.29 | 18.98 | 15.62 | -1.6% | -19.0% | -17.7% |
| swap | retained | 22.85 | 20.78 | 18.86 | -9.1% | -17.5% | -9.2% |
| swap | moved | 20.70 | 19.40 | 17.25 | -6.3% | -16.7% | -11.1% |
| never | retained | 18.48 | 17.06 | 13.35 | -7.7% | -27.8% | -21.7% |
| never | moved | 16.13 | 15.30 | 12.45 | -5.1% | -22.8% | -18.6% |

### `seq-16384`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 13.17 | 9.50 | 7.10 | -27.9% | -46.1% | -25.3% |
| floor | moved | 11.89 | 8.68 | 6.74 | -27.0% | -43.3% | -22.4% |
| identity | retained | 21.48 | 19.97 | 16.74 | -7.0% | -22.1% | -16.2% |
| identity | moved | 19.22 | 18.93 | 15.61 | -1.5% | -18.8% | -17.5% |
| swap | retained | 22.79 | 20.68 | 18.79 | -9.3% | -17.6% | -9.1% |
| swap | moved | 20.64 | 19.36 | 17.26 | -6.2% | -16.4% | -10.8% |
| never | retained | 18.35 | 17.07 | 13.27 | -7.0% | -27.7% | -22.3% |
| never | moved | 16.02 | 15.20 | 12.49 | -5.1% | -22.0% | -17.8% |

## Readings, one per fixture

Band is -1.4% to +1.2% (gold and uc against old on `functor`/`old`); every delta quoted below is outside it unless it says otherwise. Absolutes are not compared across the three binaries. "OLD->GOLD" is the gold-vs-old column, "GOLD->UC" the uc-vs-gold column.

- **`split-fuse-shared`.** The two steps are of the same size on `floor`: OLD->GOLD -27% / -31% (retained / moved), GOLD->UC -30% / -29%, so UC lands at -49% / -50% of OLD. On `identity` and `never` the UC step is the larger one (identity -9% then -11%; never -8% / -10% then -16% / -17%, UC -22% / -25% of OLD). `subst`: OLD->GOLD -2% / -4%, GOLD->UC -5% / -10%; UC -7% / -13% of OLD.
- **`split-fuse-distinct`.** Same shape: `floor` -24% / -32% then -33% / -28% (UC -49% / -51% of OLD); `identity` -9% / -8% then -11% / -10% (UC -19% / -17%); `never` -6% / -12% then -17% / -19% (UC -22% / -29%); `subst` -1% / -7% then -5% / -8% (UC -6% / -14%; the retained OLD->GOLD step is inside the band).
- **`call-split-fuse-shared`.** Here the OLD->GOLD step is the large one on the Call path: `floor` -40% / -45%, `identity` -22% / -18%, `never` -22% / -23%; GOLD->UC adds -23% / -23%, -9% / -8%, -13% / -12%. UC against OLD: `floor` -54% / -57%, `identity` -29% / -24%, `never` -32% / -32%. `subst` retained is inside the band across all three states (-2.8%, -1.0%); `subst` moved is OLD->GOLD -11%, GOLD->UC -2.4% (inside the band), UC -14% of OLD.
- **`call-split-fuse-distinct`.** As shared: `floor` -39% / -46% then -23% / -24% (UC -54% / -59% of OLD); `identity` -22% / -19% then -9% / -7% (UC -29% / -25%); `never` -23% / -24% then -12% / -12% (UC -32% / -33%); `subst` retained inside the band for all three, `subst` moved -19% OLD->GOLD then -2.8% (inside the band), UC -21% of OLD.
- **`seq-16`.** `floor` OLD->GOLD -23% / -20%, GOLD->UC -24% / -24% (UC -41% / -39% of OLD). `identity` OLD->GOLD -6% retained and **-1.7% moved, inside the band**, GOLD->UC -14% / -16% (UC -20% / -17% of OLD). `swap` -9% / -7% then -10% / -12% (UC -18% / -18%). `never` -6% / -7% then -21% / -20% (UC -26% / -25%).
- **`seq-256`.** `floor` -27% / -26% then -26% / -25% (UC -46% / -44% of OLD); `identity` -7% / **-1.6% (band)** then -16% / -18% (UC -22% / -19%); `swap` -9% / -6% then -9% / -11% (UC -18% / -17%); `never` -8% / -5% then -22% / -19% (UC -28% / -23%).
- **`seq-16384`.** `floor` -28% / -27% then -25% / -22% (UC -46% / -43% of OLD); `identity` -7% / **-1.5% (band)** then -16% / -18% (UC -22% / -19%); `swap` -9% / -6% then -9% / -11% (UC -18% / -16%); `never` -7% / -5% then -22% / -18% (UC -28% / -22%). Per-node values are flat across 16/256/16384 in all three states.

**What OLD adds to the picture.** On the Expr fixtures the OLD->GOLD step and the GOLD->UC step are comparable on `floor` (split-fuse) or OLD->GOLD dominates (Call, where GOLD's typed container descent already takes -40% off `floor`); on `identity` and `never` the UC step is the larger of the two on split-fuse and the smaller on Call. On seq the ordering is unambiguous: GOLD's step is small on every engine-dispatched arm (`identity` moved is inside the band, `swap` and `never` -5..-9%) and UC's is the large one (`identity` -14..-18%, `swap` -9..-12%, `never` -18..-22%), so of UC's -17..-28% on the seq engine arms against OLD, most is the UC protocol's, not GOLD's. `floor` on seq splits evenly, -20..-28% per step. Every UC-vs-OLD cell on an engine-touching arm is below the band except the four Call `subst` retained cells, which sit inside it for all three states; no cell is above.
