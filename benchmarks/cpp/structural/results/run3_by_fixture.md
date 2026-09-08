# bench/tvm-hook: OLD vs GOLD vs UC, eleven fixtures, one three-state interleave on Ryzen 9 7950X (Round 3)

Harness commit `719b61793638322308cea242d9208af9dffecede` (branch `bench/tvm-hook`) on upstream `main` `e74e58f9d8a6fb1ece5722d0715f68187a8e0385`. This commit adds the four Add-tree fixtures (both leaf kinds) and redefines the seq element as `Evaluate(IntImm(i + 1))`; **hook files and engines are byte-identical to the previous runs** -- `tvm_hook_override_uc.h` `8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd`, `tvm_hook_override_gold.h` `20bd4475ef7e1096b54629015ce86d40d35dffd6f9f02a918947fb527d8360d4`, `tvm_hook_override.h` `ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff`, `structural_mutate.h` (UC, PR 46 as resolved `7a569c5`) `61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b`, `structural_mutate_gold.h` (GOLD `730d6fc`) `3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101`, `structural_mutate_old.h` (upstream `e74e58f`'s engine, original sha256 `6f68aaca...`) `25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8`, `expected.h` `1b62f4d408328408280b2e167ed5bdc7f4a34b6032842da166ca9516c4a8b877` -- only the driver `real_tvm_bench.cc` changed, so all three executables were rebuilt from `719b617` (`real_tvm_bench_old` `c2f4d821...`, `real_tvm_bench_gold` `752fb084...`, `real_tvm_bench_uc` `4e8655b2...`; sha256-checked before and after the run, unchanged). Raw `report.py` output (one table per arm) is `compare_full_run3.md`; this file is the same numbers arranged one table per fixture.

## Provenance

| | |
| --- | --- |
| state old | tvm-ffi `719b61793638322308cea242d9208af9dffecede (old)` |
| state old engine header | `structural_mutate_old.h sha256:25e5c57135e774d909dc6e141129faa255e15ceb9e56e1a194ad387e070aacf8` |
| state old hook header | `tvm_hook_override.h sha256:ccd0a39feeff2c7aa4cdf8f3ee688c399c10a5fabbcfc4032a8526ee1ae29eff` |
| state gold | tvm-ffi `719b61793638322308cea242d9208af9dffecede (gold)` |
| state gold engine header | `structural_mutate_gold.h sha256:3bda287c921f476db17e9692d31ae139ccd113ed20acc631808feff3a9101101` |
| state gold hook header | `tvm_hook_override_gold.h sha256:20bd4475ef7e1096b54629015ce86d40d35dffd6f9f02a918947fb527d8360d4` |
| state uc | tvm-ffi `719b61793638322308cea242d9208af9dffecede (uc)` |
| state uc engine header | `structural_mutate.h sha256:61ff237292e6048b67480c02b76326dff84cc09b3ccc1adb68884f5af20a670b` |
| state uc hook header | `tvm_hook_override_uc.h sha256:8d6a8eccb2e2e09101ea12a7a15bfd8ac897ea5eee4bc7b8ff3135f81faa65bd` |
| what differs | one tree (bench/tvm-hook 719b617 on upstream main e74e58f), one apache/tvm 2c28965, one libtvm_ffi/libtvm_compiler build; each executable is compiled against one engine header and one hook file: old = structural_mutate_old.h (upstream main e74e58f's engine verbatim) + tvm_hook_override.h with e74e58f's container hooks compiled in; gold = structural_mutate_gold.h (GOLD 730d6fc byte for byte) + tvm_hook_override_gold.h; uc = structural_mutate.h (PR 46 as resolved, 7a569c5) + tvm_hook_override_uc.h. Hook files and engines byte-identical to fe046c8 / 3666d8f; only the driver changed (four add-tree fixtures added, seq elements redefined as Evaluate(IntImm)) |
| apache/tvm | `2c289655fb5c3770463685d084032b1cc5b5aad5`, identical in every state |
| hooks | one file per state, each written against that state's own tvm-ffi API; `port_check.sh --header` checks each against apache/tvm |
| method | one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture repeat count, process value = median of its 9 samples, reported value = median of 5 pinned process medians, processes interleaved old/gold/uc |
| processes | 21 per state |
| machine | quiet -- no other benchmark process at start or end, load average 1.00 at start, 1.05 at end. Both are checked: a pinned single-threaded run is invisible to load average, and a parallel build is invisible to the process check |


**Method.** `report.py --two-state` with three states, 21 pinned (`taskset -c 20`) processes per state, strict rotation old/gold/uc, quiet gate on both signals at start and end (load 1.00 / 1.05, no competing benchmark process). Per process: one untimed warm-up, nine timed samples per arm, process value = median of nine; reported value = median of the 21 process medians. `ns/node` divides by each fixture's declared unique-node count `N` (split-fuse 12 / 15, call-split-fuse 18 / 23, add-tree 12 / 15, add-tree-intimm 12 / 15, seq-16 34, seq-256 514, seq-16384 32770 -- **the seq divisor changed with the element shape**, so seq `ns/node` here is not comparable with earlier runs). Absolutes come from three separately compiled binaries and are not comparable across states; only the interleaved deltas are claimed.

**Drift band.** `functor` and `old` never enter the structural engine: over the 64 control cells (gold and uc against old, 16 Expr fixture-by-ownership rows x two arms) they move by **-0.7% to +2.8%**; the +2..+2.8% cells are all on the two Call fixtures, where every control drifted up in the same direction this run. A delta inside that band is not a result; a delta under about 3% on the Call fixtures is read with that in mind.

**Checks that fail the run** (README), executed untimed by every one of the 63 processes before its first timed sample; all passed on all eleven fixtures: in-place fires, identity rebuilds nothing, swap involution (keyed on the constant, all three lengths), splice vs reference (13 cases), no-op asymmetry, declared counts (unique / occurrences / rebuilds, including the zero rebuild counts of the IntImm-leaf trees and `2L + 2` for seq), hook coverage in all three hook files. Layout parity and port fidelity are not part of a gold/uc/old run (and mini-TIR's seq fixture still has the old element shape, as the README now says).


## Tables, one per fixture -- ns/node; deltas gold vs old, uc vs old, uc vs gold


### `split-fuse-shared`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 11.58 | 8.32 | 6.14 | -28.2% | -47.0% | -26.2% |
| floor | moved | 11.53 | 7.98 | 5.66 | -30.8% | -50.9% | -29.1% |
| identity | retained | 32.23 | 28.98 | 25.92 | -10.1% | -19.6% | -10.6% |
| identity | moved | 32.13 | 28.95 | 25.87 | -9.9% | -19.5% | -10.6% |
| subst | retained | 38.51 | 37.94 | 35.87 | -1.5% | -6.9% | -5.5% |
| subst | moved | 34.65 | 33.36 | 30.54 | -3.7% | -11.9% | -8.5% |
| never | retained | 23.12 | 21.08 | 17.95 | -8.8% | -22.4% | -14.8% |
| never | moved | 21.94 | 19.63 | 16.07 | -10.5% | -26.8% | -18.1% |
| functor | retained | 35.78 | 35.62 | 35.52 | -0.4% | -0.7% | -0.3% |
| functor | moved | 40.73 | 40.73 | 41.00 | +0.0% | +0.7% | +0.7% |
| old | retained | 53.92 | 54.24 | 54.55 | +0.6% | +1.2% | +0.6% |
| old | moved | 58.22 | 58.95 | 58.96 | +1.3% | +1.3% | +0.0% |

### `split-fuse-distinct`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 9.22 | 6.95 | 4.93 | -24.6% | -46.5% | -29.1% |
| floor | moved | 9.44 | 6.34 | 4.39 | -32.8% | -53.5% | -30.8% |
| identity | retained | 25.80 | 23.16 | 20.84 | -10.2% | -19.2% | -10.0% |
| identity | moved | 24.99 | 23.28 | 20.78 | -6.8% | -16.8% | -10.7% |
| subst | retained | 30.68 | 30.35 | 28.49 | -1.1% | -7.1% | -6.1% |
| subst | moved | 25.90 | 24.36 | 22.22 | -5.9% | -14.2% | -8.8% |
| never | retained | 17.33 | 16.83 | 13.36 | -2.9% | -22.9% | -20.6% |
| never | moved | 17.26 | 15.34 | 12.69 | -11.1% | -26.5% | -17.3% |
| functor | retained | 28.60 | 28.43 | 28.54 | -0.6% | -0.2% | +0.4% |
| functor | moved | 33.45 | 33.37 | 33.43 | -0.2% | -0.1% | +0.2% |
| old | retained | 43.53 | 43.75 | 43.82 | +0.5% | +0.7% | +0.2% |
| old | moved | 47.35 | 47.91 | 48.28 | +1.2% | +2.0% | +0.8% |

### `call-split-fuse-shared`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 13.77 | 8.21 | 6.44 | -40.4% | -53.2% | -21.6% |
| floor | moved | 13.94 | 7.60 | 6.04 | -45.5% | -56.7% | -20.5% |
| identity | retained | 29.32 | 22.67 | 21.18 | -22.7% | -27.8% | -6.6% |
| identity | moved | 29.25 | 24.04 | 22.24 | -17.8% | -24.0% | -7.5% |
| subst | retained | 38.54 | 38.04 | 37.99 | -1.3% | -1.4% | -0.1% |
| subst | moved | 31.80 | 28.34 | 27.56 | -10.9% | -13.3% | -2.8% |
| never | retained | 21.84 | 16.67 | 14.72 | -23.7% | -32.6% | -11.7% |
| never | moved | 22.29 | 16.88 | 15.15 | -24.3% | -32.0% | -10.2% |
| functor | retained | 56.82 | 57.88 | 58.14 | +1.9% | +2.3% | +0.4% |
| functor | moved | 59.95 | 60.52 | 61.34 | +1.0% | +2.3% | +1.4% |
| old | retained | 64.99 | 66.24 | 66.47 | +1.9% | +2.3% | +0.3% |
| old | moved | 68.21 | 68.61 | 70.14 | +0.6% | +2.8% | +2.2% |

### `call-split-fuse-distinct`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 10.82 | 6.50 | 5.06 | -39.9% | -53.2% | -22.2% |
| floor | moved | 10.85 | 5.73 | 4.60 | -47.2% | -57.6% | -19.7% |
| identity | retained | 23.18 | 17.75 | 16.67 | -23.4% | -28.1% | -6.1% |
| identity | moved | 22.47 | 18.03 | 16.73 | -19.8% | -25.5% | -7.2% |
| subst | retained | 30.83 | 30.58 | 31.17 | -0.8% | +1.1% | +1.9% |
| subst | moved | 23.22 | 19.02 | 18.15 | -18.1% | -21.8% | -4.6% |
| never | retained | 17.22 | 13.04 | 11.52 | -24.3% | -33.1% | -11.7% |
| never | moved | 17.40 | 12.96 | 12.02 | -25.5% | -30.9% | -7.3% |
| functor | retained | 45.76 | 46.47 | 46.88 | +1.6% | +2.4% | +0.9% |
| functor | moved | 47.22 | 47.54 | 48.35 | +0.7% | +2.4% | +1.7% |
| old | retained | 53.19 | 53.53 | 54.30 | +0.6% | +2.1% | +1.4% |
| old | moved | 54.01 | 53.91 | 55.10 | -0.2% | +2.0% | +2.2% |

### `add-tree-shared`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 11.47 | 8.30 | 6.12 | -27.6% | -46.6% | -26.3% |
| floor | moved | 11.59 | 7.97 | 5.58 | -31.2% | -51.9% | -30.0% |
| identity | retained | 31.73 | 29.29 | 26.07 | -7.7% | -17.8% | -11.0% |
| identity | moved | 31.46 | 29.17 | 25.91 | -7.3% | -17.6% | -11.2% |
| subst | retained | 37.79 | 37.82 | 35.61 | +0.1% | -5.8% | -5.8% |
| subst | moved | 34.87 | 33.19 | 30.71 | -4.8% | -11.9% | -7.5% |
| never | retained | 21.28 | 21.07 | 16.67 | -1.0% | -21.7% | -20.9% |
| never | moved | 21.71 | 19.53 | 16.11 | -10.0% | -25.8% | -17.5% |
| functor | retained | 35.83 | 35.96 | 35.99 | +0.4% | +0.4% | +0.1% |
| functor | moved | 41.81 | 41.97 | 42.21 | +0.4% | +1.0% | +0.6% |
| old | retained | 54.15 | 54.25 | 54.11 | +0.2% | -0.1% | -0.3% |
| old | moved | 59.13 | 59.20 | 59.39 | +0.1% | +0.4% | +0.3% |

### `add-tree-distinct`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 9.19 | 6.88 | 4.75 | -25.1% | -48.3% | -31.0% |
| floor | moved | 9.21 | 6.11 | 4.42 | -33.7% | -52.0% | -27.7% |
| identity | retained | 25.42 | 23.39 | 20.93 | -8.0% | -17.7% | -10.5% |
| identity | moved | 24.87 | 23.16 | 20.88 | -6.9% | -16.0% | -9.8% |
| subst | retained | 30.21 | 30.57 | 28.73 | +1.2% | -4.9% | -6.0% |
| subst | moved | 25.98 | 24.21 | 22.37 | -6.8% | -13.9% | -7.6% |
| never | retained | 16.99 | 16.80 | 13.28 | -1.1% | -21.8% | -21.0% |
| never | moved | 17.13 | 14.95 | 12.81 | -12.7% | -25.2% | -14.3% |
| functor | retained | 28.37 | 28.79 | 28.68 | +1.5% | +1.1% | -0.4% |
| functor | moved | 33.30 | 33.80 | 33.67 | +1.5% | +1.1% | -0.4% |
| old | retained | 43.33 | 43.51 | 43.57 | +0.4% | +0.6% | +0.1% |
| old | moved | 47.66 | 47.90 | 48.23 | +0.5% | +1.2% | +0.7% |

### `add-tree-intimm-shared`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 10.65 | 7.52 | 4.75 | -29.4% | -55.4% | -36.8% |
| floor | moved | 9.58 | 7.04 | 4.46 | -26.5% | -53.4% | -36.6% |
| identity | retained | 12.32 | 10.31 | 7.74 | -16.3% | -37.2% | -24.9% |
| identity | moved | 12.18 | 10.26 | 7.75 | -15.8% | -36.4% | -24.5% |
| subst | retained | 12.29 | 10.74 | 7.98 | -12.6% | -35.1% | -25.7% |
| subst | moved | 12.58 | 10.25 | 7.41 | -18.5% | -41.1% | -27.7% |
| never | retained | 11.72 | 11.11 | 7.67 | -5.2% | -34.6% | -31.0% |
| never | moved | 11.76 | 10.28 | 7.35 | -12.6% | -37.5% | -28.5% |
| functor | retained | 19.10 | 19.32 | 19.21 | +1.2% | +0.6% | -0.6% |
| functor | moved | 20.04 | 20.08 | 20.01 | +0.2% | -0.1% | -0.3% |
| old | retained | 19.49 | 19.56 | 19.53 | +0.4% | +0.2% | -0.2% |
| old | moved | 20.51 | 20.55 | 20.59 | +0.2% | +0.4% | +0.2% |

### `add-tree-intimm-distinct`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 8.56 | 6.02 | 3.79 | -29.7% | -55.7% | -37.0% |
| floor | moved | 7.70 | 5.31 | 3.50 | -31.0% | -54.5% | -34.1% |
| identity | retained | 9.86 | 8.18 | 6.21 | -17.0% | -37.0% | -24.1% |
| identity | moved | 9.45 | 8.21 | 6.14 | -13.1% | -35.0% | -25.2% |
| subst | retained | 10.02 | 8.66 | 6.27 | -13.6% | -37.4% | -27.6% |
| subst | moved | 9.39 | 7.93 | 6.11 | -15.5% | -34.9% | -23.0% |
| never | retained | 9.43 | 8.86 | 6.15 | -6.0% | -34.8% | -30.6% |
| never | moved | 9.34 | 8.02 | 5.82 | -14.1% | -37.7% | -27.4% |
| functor | retained | 15.19 | 15.42 | 15.33 | +1.5% | +0.9% | -0.6% |
| functor | moved | 15.18 | 15.45 | 15.35 | +1.8% | +1.1% | -0.6% |
| old | retained | 15.52 | 15.50 | 15.45 | -0.1% | -0.5% | -0.3% |
| old | moved | 15.57 | 15.47 | 15.55 | -0.6% | -0.1% | +0.5% |

### `seq-16`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 6.01 | 5.61 | 4.33 | -6.7% | -28.0% | -22.8% |
| floor | moved | 4.31 | 4.61 | 3.18 | +7.0% | -26.2% | -31.0% |
| identity | retained | 12.18 | 10.61 | 9.76 | -12.9% | -19.9% | -8.0% |
| identity | moved | 10.00 | 11.54 | 9.17 | +15.4% | -8.3% | -20.5% |
| swap | retained | 16.57 | 14.22 | 14.67 | -14.2% | -11.5% | +3.2% |
| swap | moved | 12.50 | 12.96 | 11.82 | +3.7% | -5.4% | -8.8% |
| never | retained | 6.78 | 6.81 | 5.97 | +0.4% | -11.9% | -12.3% |
| never | moved | 5.81 | 6.14 | 4.74 | +5.7% | -18.4% | -22.8% |

### `seq-256`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 6.96 | 5.79 | 4.22 | -16.8% | -39.4% | -27.1% |
| floor | moved | 4.50 | 4.61 | 3.12 | +2.4% | -30.7% | -32.3% |
| identity | retained | 12.35 | 10.18 | 9.47 | -17.6% | -23.3% | -7.0% |
| identity | moved | 9.44 | 10.98 | 8.69 | +16.3% | -7.9% | -20.9% |
| swap | retained | 16.09 | 13.41 | 13.98 | -16.7% | -13.1% | +4.3% |
| swap | moved | 11.35 | 12.16 | 11.09 | +7.1% | -2.3% | -8.8% |
| never | retained | 6.93 | 6.36 | 5.54 | -8.2% | -20.1% | -12.9% |
| never | moved | 5.51 | 5.69 | 4.27 | +3.3% | -22.5% | -25.0% |

### `seq-16384`

| arm | own | old | gold | uc | gold vs old | uc vs old | uc vs gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | retained | 7.06 | 5.74 | 4.02 | -18.7% | -43.1% | -30.0% |
| floor | moved | 4.39 | 4.43 | 3.02 | +0.9% | -31.2% | -31.8% |
| identity | retained | 12.53 | 10.25 | 9.27 | -18.2% | -26.0% | -9.6% |
| identity | moved | 9.36 | 10.98 | 8.76 | +17.3% | -6.4% | -20.2% |
| swap | retained | 15.96 | 13.40 | 13.75 | -16.0% | -13.8% | +2.6% |
| swap | moved | 11.22 | 12.13 | 10.97 | +8.1% | -2.2% | -9.6% |
| never | retained | 6.93 | 6.39 | 5.34 | -7.8% | -22.9% | -16.4% |
| never | moved | 5.42 | 5.59 | 4.25 | +3.1% | -21.6% | -24.0% |

## Readings, one per fixture

Band is -0.7% to +2.8%; every delta quoted is outside it unless it says otherwise. Absolutes are not compared across the three binaries. "OLD->GOLD" is the gold-vs-old column, "GOLD->UC" the uc-vs-gold column.

- **`split-fuse-shared`.** As in the previous run, to the point: `floor` OLD->GOLD -28% / -31%, GOLD->UC -26% / -29% (UC -47% / -51% of OLD); `identity` -10% then -11% (UC -20%); `subst` -1.5% (band) / -4% then -6% / -9% (UC -7% / -12%); `never` -9% / -11% then -15% / -18% (UC -22% / -27%).
- **`split-fuse-distinct`.** Same: `floor` -25% / -33% then -29% / -31% (UC -47% / -54%); `identity` -10% / -7% then -10% / -11% (UC -19% / -17%); `subst` -1% (band) / -6% then -6% / -9%; `never` -3% (band) / -11% then -21% / -17% (UC -23% / -27%).
- **`call-split-fuse-shared`.** OLD->GOLD carries the Call path: `floor` -40% / -46%, `identity` -23% / -18%, `never` -24% / -24%; GOLD->UC adds -22% / -21%, -7% / -8%, -12% / -10% (UC -53% / -57%, -28% / -24%, -33% / -32% of OLD). `subst` retained is inside the band across all three; `subst` moved -11% then -3% (band), UC -13% of OLD.
- **`call-split-fuse-distinct`.** As shared: `floor` -40% / -47% then -22% / -20% (UC -53% / -58%); `identity` -23% / -20% then -6% / -7%; `never` -24% / -26% then -12% / -7%; `subst` retained inside the band, `subst` moved -18% then -5%, UC -22% of OLD.
- **`add-tree-shared`** (new). Row for row the split-fuse-shared numbers: `floor` 11.47 / 8.30 / 6.12 against 11.58 / 8.32 / 6.14, `identity` 31.7 / 29.3 / 26.1 against 32.2 / 29.0 / 25.9, and the same deltas (`floor` -28% / -31% then -26% / -30%; `identity` -8% / -7% then -11% / -11%; `never` -1% (band) / -10% then -21% / -18%; `subst` +0.1% (band) / -5% then -6% / -8%). Replacing four operator types by one `Add` changes nothing measurable in any state: node-type variety is not a cost here.
- **`add-tree-distinct`** (new). Likewise split-fuse-distinct's numbers within a few percent (`floor` 9.19 / 6.88 / 4.75 against 9.22 / 6.95 / 4.93) and the same deltas (`floor` -25% / -34% then -31% / -28%; `identity` -8% / -7% then -11% / -10%; `never` -1% (band) / -13% then -21% / -14%).
- **`add-tree-intimm-shared`** (new). Removing the two `Var` leaves removes two thirds of the traversal: `identity` falls from 31.7 / 29.3 / 26.1 to **12.3 / 10.3 / 7.7** ns/node, `never` from 21.3 / 21.1 / 16.7 to 11.7 / 11.1 / 7.7, and the functor/`old` baselines from 36 / 54 to 19 / 20. So the `Var` path -- remap lookup and bind, the `ty` guard, the name-carrying node -- is about 20 ns/node of the 26-32 ns/node identity cost on the Var-leaf trees, in every state. On what is left the engine deltas are the largest of any fixture: OLD->GOLD `floor` -29% / -27%, `identity` -16%, `subst` -13% / -19%, `never` -5% / -13%; GOLD->UC `floor` -37% / -37%, `identity` -25% / -25%, `subst` -26% / -28%, `never` -31% / -29%; UC against OLD `floor` -55% / -53%, `identity` -37% / -36%, `subst` -35% / -41%, `never` -35% / -38%. `subst` here is a substitution pass whose callback never fires (rebuild counts zero), so it reads as a second `identity` through the `Var`-bound link, and it does: 12.3 / 10.7 / 8.0 beside `identity`'s 12.3 / 10.3 / 7.7.
- **`add-tree-intimm-distinct`** (new). Same picture at N=15: `identity` 9.9 / 8.2 / 6.2, GOLD->UC -24% / -25%, UC -37% / -35% of OLD; `floor` -30% / -31% then -37% / -34% (UC -56% / -55%); `subst` -14% / -16% then -28% / -23%; `never` -6% / -14% then -31% / -27%.
- **`seq-16`** (redefined: `Evaluate(IntImm(i + 1))`, N = 34, not comparable with earlier seq rows). The container loop and two leaf hooks, nothing else, and the picture changes with ownership. Retained: OLD->GOLD `floor` -7%, `identity` -13%, `swap` -14%, `never` +0.4% (band); GOLD->UC `floor` -23%, `identity` -8%, `never` -12%, and `swap` **+3.2%** -- UC slightly above GOLD on the copy-on-write swap rebuild, just outside the band. Moved: **GOLD is above OLD** on every arm -- `floor` +7%, `identity` +15%, `swap` +4%, `never` +6% -- and UC takes it back: GOLD->UC `floor` -31%, `identity` -21%, `swap` -9%, `never` -23%, so UC against OLD is `floor` -26%, `identity` -8%, `never` -18%, and `swap` -5% (near the band).
- **`seq-256`.** Retained: OLD->GOLD `floor` -17%, `identity` -18%, `swap` -17%, `never` -8%; GOLD->UC `floor` -27%, `identity` -7%, `never` -13%, `swap` **+4.3%**. Moved: OLD->GOLD `floor` +2% (band), `identity` **+16%**, `swap` +7%, `never` +3% (band); GOLD->UC `floor` -32%, `identity` -21%, `swap` -9%, `never` -25%; UC against OLD `floor` -31%, `identity` -8%, `swap` -2% (band), `never` -23%.
- **`seq-16384`.** As seq-256 within a point or two: retained OLD->GOLD -19% / -18% / -16% / -8% (`floor` / `identity` / `swap` / `never`), GOLD->UC -30% / -10% / **+2.6%** (band edge) / -16%; moved OLD->GOLD +1% (band) / **+17%** / +8% / +3%, GOLD->UC -32% / -20% / -10% / -24%; UC against OLD -31% / -6% / -2% (band) / -22%. Flat across the three lengths in all three states.

**What Round 3 adds.** (1) The Add-only trees reproduce split/fuse's rows almost to the tenth in every state, so the engine comparison on the Expr fixtures does not depend on operator variety, and `add-tree-*` can stand in for `split-fuse-*` where a single hook body is wanted. (2) The IntImm-leaf trees put a number on the `Var` path: about 20 ns/node of the ~26-32 ns/node identity traversal on the Var-leaf trees is `Var` handling, the same in OLD, GOLD and UC -- the engines differ on the remaining ~8-12 ns/node, and there UC's advantage is at its largest (-24..-31% against GOLD, -35..-41% against OLD on identity/subst/never). (3) On the redefined seq -- the bare container loop -- the retained (copy-on-write) rows keep the familiar ordering except that UC's `swap` is slightly above GOLD's (+2.6..+4.3%, at the band's edge on one length); the moved (in-place) rows do not: GOLD's in-place element loop is slower than OLD's (`identity` +15..+17%, `swap` +4..+8%), and UC's is 20-32% below GOLD's and 6-31% below OLD's on every arm but `swap`, where UC and OLD are within the band. With the Var/Mul/Add subtree gone, the old seq rows' "GOLD small step, UC large step" was mostly the element subtrees; on the container loop itself GOLD's in-place path is a regression against OLD and UC's is not.
