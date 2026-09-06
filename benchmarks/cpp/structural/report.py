#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Run the structural harnesses and render the report.

The reporting format lives here, not in a task record: rows are cases, columns are arms, one
quantity per table, `ns/node` on the fixture's declared unique-node count with the same
divisor for every arm in a row, and every delta column says in its own header what it
compares.  `N` is not repeated on results rows -- it is fixture metadata and lives in the
fixtures table.

    # one state
    ./report.py --binary ../../../build_bench/real_tvm_bench --cpu 0

    # two states, interleaved A/B/A/B, rendering the `## Compare` tables
    ./report.py --two-state --state 'A=62df2f5:../../../build_bench/real_tvm_bench_A' \\
                --state 'B=897ece6:../../../build_bench/real_tvm_bench_B' \\
                --differs 'the five structural commits #747 #749 #750 #751 #753' --cpu 0

Two-state runs interleave the processes rather than running all of A and then all of B, so
thermal state, allocator luck and drift hit both states equally.  Absolutes from two
separately compiled binaries are not comparable; only the interleaved delta is claimed.
"""
import argparse
import os
import statistics
import subprocess
import sys

WALK_ARMS = ["walk_floor", "walk_var", "walk_never", "walk_functor", "walk_old"]
# split/fuse: Expr-level Var substitution, with the two baselines that perform it.
EXPR_MAP_ARMS = ["map_floor", "map_never", "map_identity_var", "map_subst",
                 "map_functor", "map_old"]
# seq: the Stmt-level Evaluate swap at scale. `StmtExprMutator` does different work on the
# same graph, so `functor` and `old` are not baselines for it and are not columns here.
SEQ_MAP_ARMS = ["map_floor", "map_never", "map_identity_stmt", "map_swap"]

# Which arm each fixture family's mutating column is, keyed by the `kind` the binary declares.
MUTATE_ARM = {"subst": "map_subst", "swap": "map_swap"}
IDENTITY_ARM = {"subst": "map_identity_var", "swap": "map_identity_stmt"}

# The benchmark machine's data-cache geometry, stated rather than probed.
CACHE = [("L1d", 32 * 1024), ("L2", 1024 * 1024), ("L3", 32 * 1024 * 1024)]


def fits(working_set):
    for name, size in CACHE:
        if working_set <= size:
            return name
    return "DRAM"


def run_once(binary, cpu):
    cmd = (["taskset", "-c", str(cpu)] if cpu is not None else []) + [binary]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit("%s exited %d, so this run is invalid" % (binary, proc.returncode))
    return parse(proc.stdout)


def parse(text):
    out = {"provenance": {}, "fixtures": {}, "order": [], "results": {}}
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        parts = line.split("\t")
        if parts[0] == "#provenance":
            out["provenance"][parts[1]] = parts[2]
        elif parts[0] == "#fixture":
            _, harness, name, unique, occ, ws, reb_r, reb_m, kind = parts
            out["fixtures"][name] = dict(unique=int(unique), occurrences=int(occ),
                                         working_set=int(ws), rebuilt_retained=int(reb_r),
                                         rebuilt_moved=int(reb_m), kind=kind)
            out["order"].append(name)
        elif parts[0] == "#result":
            _, harness, fixture, ownership, arm, ns = parts
            out["results"][(fixture, ownership, arm)] = float(ns)
    return out


def median_runs(runs):
    merged = dict(runs[0])
    merged["results"] = {k: statistics.median([r["results"][k] for r in runs])
                         for k in runs[0]["results"]}
    return merged


def human(n):
    for unit in ["B", "KiB", "MiB"]:
        if n < 1024 or unit == "MiB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0


HEADER = {"map_identity_var": "identity", "map_identity_stmt": "identity"}


def header(arm):
    """The arm without its `walk_`/`map_` prefix: the table title already says which."""
    return HEADER.get(arm, arm.partition("_")[2])


def pct(new, base):
    return "**%+.1f%%**" % (100.0 * (new - base) / base)


def ns_per_node(merged, fixture, ownership, arm):
    ns = merged["results"].get((fixture, ownership, arm))
    return None if ns is None else ns / merged["fixtures"][fixture]["unique"]


def render(merged, runs, out):
    prov = merged["provenance"]
    harness = prov["harness"]
    fixtures = merged["fixtures"]
    names = merged["order"]
    w = out.write

    w("### Provenance -- %s\n\n| | |\n| --- | --- |\n" % harness)
    for key, label in [("harness_branch_commit", "harness branch commit"),
                       ("tvm_ffi_engine_sha", "tvm-ffi engine sha"),
                       ("tvm_sha", "apache/tvm sha"),
                       ("compiler", "compiler"), ("flags", "flags"),
                       ("structural_hooks", "structural hooks")]:
        if key in prov:
            w("| %s | `%s` |\n" % (label, prov[key]))
    w("| method | %s |\n| processes | %d |\n\n" % (prov["method"], len(runs)))

    w("### Fixtures -- %s\n\n" % harness)
    w("| fixture | N | occurrences | working set | fits | operation | rebuilt retained "
      "| rebuilt moved |\n")
    w("| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |\n")
    for name in names:
        f = fixtures[name]
        w("| `%s` | %d | %d | %s | %s | `%s` | %d | %d |\n"
          % (name, f["unique"], f["occurrences"], human(f["working_set"]),
             fits(f["working_set"]), f["kind"], f["rebuilt_retained"], f["rebuilt_moved"]))
    w("\n")

    w("#### %s -- walk -- ns/node\n\n" % harness)
    w("| Case | " + " | ".join(header(a) for a in WALK_ARMS) + " | var<br>vs old |\n")
    w("| --- |" + " ---: |" * (len(WALK_ARMS) + 1) + "\n")
    for name in names:
        row = [ns_per_node(merged, name, "-", a) for a in WALK_ARMS]
        w("| %s | %s | %s |\n"
          % (name, " | ".join("%.2f" % v for v in row), pct(row[1], row[4])))
    w("\n")

    for kind, arms, title, note in [
        ("subst", EXPR_MAP_ARMS, "map, Expr-level `Var` substitution",
         "Every column substitutes `Var`s, which is the operation `Substitute` and\n"
         "`FunctorSubstitute` perform, so `functor` and `old` are baselines for `subst` here\n"
         "and the delta column is like-for-like.\n"),
        ("swap", SEQ_MAP_ARMS, "map, Stmt-level `Evaluate` swap",
         "A different operation: swap two whole `Evaluate` nodes rather than substitute\n"
         "`Var`s. `StmtExprMutator` hooks `VisitExpr_(const VarNode*)` and does different work\n"
         "on the same graph, so it is not a baseline for this operation and does not appear.\n"),
    ]:
        rows = [n for n in names if fixtures[n]["kind"] == kind]
        if not rows:
            continue
        w("#### %s -- %s -- ns/node, both ownership variants\n\n" % (harness, title))
        w(note + "\n")
        delta = kind == "subst"
        w("| Case | own | " + " | ".join(header(a) for a in arms) +
          (" | subst<br>vs old |\n" if delta else " |\n"))
        w("| --- | --- |" + " ---: |" * (len(arms) + (1 if delta else 0)) + "\n")
        for name in rows:
            for ownership in ["retained", "moved"]:
                vals = [ns_per_node(merged, name, ownership, a) for a in arms]
                cells = " | ".join("%.2f" % v for v in vals)
                if delta:
                    cells += " | " + pct(vals[arms.index("map_subst")],
                                         vals[arms.index("map_old")])
                w("| %s | %s | %s |\n" % (name, ownership, cells))
        w("\n")


# ---------------------------------------------------------------------------
# Two-state mode.
# ---------------------------------------------------------------------------

def interleave(states, cpu, runs):
    """Run A, B, A, B, ... so drift and thermal state hit both states equally.

    The alternative -- all of A and then all of B -- lets anything that changes over the run
    land entirely on one state and read as a result. This report already carries a worked
    example: a `map_floor` comparison across two separately compiled binaries showed -18% to
    -30% on every row, which an interleaved A/B proved was a build artifact and not a change.
    """
    collected = {label: [] for label, _, _ in states}
    for _ in range(runs):
        for label, _ref, binary in states:
            collected[label].append(run_once(binary, cpu))
    return {label: median_runs(rs) for label, rs in collected.items()}


def cell(a, b):
    """`30.81 -> 24.16 (-21.6%)`: both absolutes and the delta, in one cell."""
    if a is None or b is None:
        return None
    return "%.2f &rarr; %.2f (%+.1f%%)" % (a, b, 100.0 * (b - a) / a)


def render_compare(merged, states, runs, differs, out):
    w = out.write
    a_label, a_ref, _ = states[0]
    b_label, b_ref, _ = states[1]
    fixtures = merged[a_label]["fixtures"]
    names = merged[a_label]["order"]

    def row_cell(name, ownership, arm):
        return cell(ns_per_node(merged[a_label], name, ownership, arm),
                    ns_per_node(merged[b_label], name, ownership, arm))

    w("### Provenance\n\n| | |\n| --- | --- |\n")
    w("| state %s | tvm-ffi `%s` |\n" % (a_label, merged[a_label]["provenance"]["tvm_ffi_engine_sha"]))
    w("| state %s | tvm-ffi `%s` |\n" % (b_label, merged[b_label]["provenance"]["tvm_ffi_engine_sha"]))
    w("| what differs | %s |\n" % differs)
    w("| apache/tvm | `%s`, identical in both states |\n"
      % merged[a_label]["provenance"]["tvm_sha"])
    w("| hooks | one file per state, each written against that state's own tvm-ffi API; "
      "`port_check.sh --header` checks both against apache/tvm |\n")
    w("| method | %s, processes interleaved %s/%s/%s/%s |\n"
      % (merged[a_label]["provenance"]["method"], a_label, b_label, a_label, b_label))
    w("| processes | %d per state |\n\n" % runs)
    w("**Absolutes come from two separately compiled binaries and are not comparable across "
      "states; only the delta in each cell is claimed.** Two builds differ in inlining, layout "
      "and allocator luck for reasons unrelated to what is under study. The processes are "
      "interleaved %s/%s/%s/%s, so drift and thermal state hit both equally.\n\n"
      % (a_label, b_label, a_label, b_label))

    w("### Headline\n\n")
    w("`floor` is the control: it dispatches no callbacks, so a protocol or engine change "
      "should not touch it. **If `floor` moves, the two builds differ in something beyond "
      "what is being compared and nothing else in that row can be trusted.** `identity` and "
      "`subst`/`swap` are read against whatever the two states differ in, named above. Any "
      "other pattern is a finding.\n\n")

    for kind, title, note in [
        ("subst", "Expr-level -- split/fuse", "`subst` substitutes `Var`s."),
        ("swap", "Stmt-level -- seq", "`swap` swaps two whole `Evaluate` nodes."),
    ]:
        rows = [n for n in names if fixtures[n]["kind"] == kind]
        if not rows:
            continue
        arms = ["map_floor", IDENTITY_ARM[kind], MUTATE_ARM[kind]]
        w("#### %s -- ns/node, %s &rarr; %s\n\n%s\n\n" % (title, a_label, b_label, note))
        w("| Case | own | floor | identity | %s |\n" % header(arms[2]))
        w("| --- | --- | ---: | ---: | ---: |\n")
        for name in rows:
            for ownership in ["retained", "moved"]:
                cells = [row_cell(name, ownership, a) for a in arms]
                w("| %s | %s | %s |\n"
                  % (name, ownership, " | ".join(c if c else "n/a" for c in cells)))
        w("\n")

    w("### Supporting\n\n")
    w("| Case | own | never | functor | old |\n| --- | --- | ---: | ---: | ---: |\n")
    na = "n/a[^cmp-seq]"
    for name in names:
        kind = fixtures[name]["kind"]
        for ownership in ["retained", "moved"]:
            never = row_cell(name, ownership, "map_never")
            if kind == "swap":
                functor = old = na
            else:
                functor = row_cell(name, ownership, "map_functor") or na
                old = row_cell(name, ownership, "map_old") or na
            w("| %s | %s | %s | %s | %s |\n" % (name, ownership, never or na, functor, old))
    w("\n[^cmp-seq]: `functor` and `old` substitute `Var`s. The seq fixtures carry the "
      "Stmt-level `Evaluate` swap, which neither performs, so they are not baselines for "
      "those rows rather than missing from them.\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", action="append", default=[])
    ap.add_argument("--two-state", action="store_true")
    ap.add_argument("--state", action="append", default=[],
                    help="LABEL=REF:PATH, given twice for a two-state run")
    ap.add_argument("--differs", default="unstated",
                    help="what differs between the two states -- the only thing the delta "
                         "can be attributed to")
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    if args.two_state:
        if len(args.state) != 2:
            raise SystemExit("--two-state needs exactly two --state arguments")
        states = []
        for spec in args.state:
            label, _, rest = spec.partition("=")
            ref, _, path = rest.partition(":")
            states.append((label, ref, os.path.abspath(path)))
        merged = interleave(states, args.cpu, args.runs)
        render_compare(merged, states, args.runs, args.differs, out)
    else:
        for binary in args.binary:
            binary = os.path.abspath(binary)
            runs = [run_once(binary, args.cpu) for _ in range(args.runs)]
            render(median_runs(runs), runs, out)
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
