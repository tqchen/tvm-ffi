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

# Machine state at both ends of the run, stamped into every provenance table.
MACHINE = {"before": None, "after": None}

# The benchmark machine's data-cache geometry, stated rather than probed.
CACHE = [("L1d", 32 * 1024), ("L2", 1024 * 1024), ("L3", 32 * 1024 * 1024)]


def fits(working_set):
    for name, size in CACHE:
        if working_set <= size:
            return name
    return "DRAM"


# ---------------------------------------------------------------------------
# The quiet-machine gate.
#
# A run of this harness is one single-threaded process pinned to one core, so on a many-core
# host a run in progress and an empty machine give the same reading: load ~2, ~98% idle.
# Load average sees a parallel `cmake --build` and nothing else.  The check that sees another
# run is process presence, so that is the check, and it is recorded with the run rather than
# left to whoever remembers to look.
#
# Two corollaries the interleaving does not cover.  Interleaving protects a delta against slow
# drift; it protects neither absolutes nor anything at all against a competing pinned run,
# because that run contends unevenly across arms -- the allocating arms absorb most of it -- so
# it moves cells relative to each other and not together.  And `--cpu` is core isolation, not
# workload isolation: `taskset` grants a private core, never private L3 or memory bandwidth.
# ---------------------------------------------------------------------------

def competing_processes():
    """Benchmark processes on this host that are not part of this run."""
    try:
        listing = subprocess.run(["ps", "-eo", "pid=,ppid=,args="],
                                 capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return ["(ps unavailable -- the quiet-machine check could not run)"]
    rows = []
    for line in listing.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2]))

    me = os.getpid()
    parent = {pid: ppid for pid, ppid, _ in rows}
    children = {}
    for pid, ppid, _ in rows:
        children.setdefault(ppid, []).append(pid)
    related, cur = set(), me
    while cur and cur not in related:            # this process and everything that launched it
        related.add(cur)
        cur = parent.get(cur, 0)
    stack = [me]                                  # and everything it launched
    while stack:
        for kid in children.get(stack.pop(), []):
            if kid not in related:
                related.add(kid)
                stack.append(kid)

    # Matched on what is being executed -- the first two argv tokens' basenames -- and not on
    # anything the command line merely mentions.  A shell wrapper whose text contains
    # "report.py" (another agent's own `pgrep`, say) is not a benchmark.  Under-detection is
    # the dangerous direction and a wider match would seem the safer error, but a check that
    # trips on a `pgrep` is a check someone switches off, which is the same failure with
    # extra steps.
    def is_benchmark(args):
        for token in args.split()[:2]:
            base = os.path.basename(token)
            if base == "report.py" or "_bench" in base:
                return True
        return False

    return ["%d %s" % (pid, args) for pid, _ppid, args in rows
            if pid not in related and is_benchmark(args)]


def machine_state():
    try:
        load = os.getloadavg()[0]
    except (OSError, AttributeError):
        load = float("nan")
    return {"load": load, "competing": competing_processes()}


def machine_row(before, after):
    """One provenance row saying what the machine was doing, checked the way that works."""
    seen, competing = set(), []
    for entry in (before["competing"] if before else []) + (after["competing"] if after else []):
        pid = entry.split(None, 1)[0]
        if pid not in seen:
            seen.add(pid)
            competing.append(entry)
    loads = "load average %.2f at start, %.2f at end" % (before["load"], after["load"])
    if competing:
        return ("**contended** -- %d other benchmark process(es) live during this run (%s). "
                "Absolutes are not usable and the deltas are suspect; %s"
                % (len(competing), "; ".join(c.split(None, 1)[0] for c in competing), loads))
    return ("quiet -- no other benchmark process at start or end, %s. Checked by process "
            "presence rather than load, because a pinned single-threaded run reads as an idle "
            "machine" % loads)


def run_once(binary, cpu):
    cmd = (["taskset", "-c", str(cpu)] if cpu is not None else []) + [binary]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit("%s exited %d, so this run is invalid" % (binary, proc.returncode))
    return parse(proc.stdout)


def parse(text):
    out = {"provenance": {}, "fixtures": {}, "order": [], "results": {}, "nodesizes": {}}
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
        elif parts[0] == "#nodesize":
            _, harness, node, size = parts
            out["nodesizes"][node] = int(size)
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
    w("| method | %s |\n| processes | %d |\n" % (prov["method"], len(runs)))
    w("| machine | %s |\n\n" % machine_row(MACHINE["before"], MACHINE["after"]))

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
    w("| processes | %d per state |\n" % runs)
    w("| machine | %s |\n\n" % machine_row(MACHINE["before"], MACHINE["after"]))
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


# ---------------------------------------------------------------------------
# Fidelity mode: mini-TIR against real TVM, within one host.
#
# The requirement the harness is built to: where a fixture exists in both harnesses, the two
# must agree, on THIS host.  Cross-host comparison is not the goal and is not attempted --
# different compilers and architectures make nothing cross-host attributable to either side.
#
# Two preconditions are checked rather than assumed, and either one failing invalidates the
# comparison outright:
#
#   * ONE ENGINE.  Both binaries must be stamped with the same tvm-ffi engine sha.  Point the
#     TVM checkout's 3rdparty/tvm-ffi at this checkout and build.sh does that; otherwise mini
#     and real link different engines and the delta is an engine delta wearing a fidelity
#     delta's clothes.
#   * ONE LAYOUT.  Every counterpart node type must be the same size in both.  mini-TIR's node
#     set is a reduced set of apache/tvm's, and the reduction is in WHICH types exist, never in
#     what one of them contains.
# ---------------------------------------------------------------------------

# mini-TIR node types and their apache/tvm counterparts, by the logical name both emit.
COUNTERPART_NODES = ["Span", "Type", "PrimType", "Expr", "Var", "IntImm", "FloatImm", "Call",
                     "Add", "Mul", "FloorDiv", "FloorMod", "Stmt", "Evaluate", "SeqStmt"]

FIDELITY_WALK_ARMS = WALK_ARMS
FIDELITY_MAP_ARMS = {"subst": EXPR_MAP_ARMS, "swap": SEQ_MAP_ARMS}


def check_fidelity_preconditions(a, b, a_label, b_label):
    """Refuse to render a comparison that would be read as a fidelity result and is not."""
    a_engine = a["provenance"].get("tvm_ffi_engine_sha", "?")
    b_engine = b["provenance"].get("tvm_ffi_engine_sha", "?")
    if a_engine != b_engine:
        raise SystemExit(
            "fidelity run invalid: %s links tvm-ffi %s and %s links %s.\n"
            "  Build both against one engine -- point the TVM checkout's 3rdparty/tvm-ffi at\n"
            "  this checkout and rerun build.sh -- or the delta is an engine delta."
            % (a_label, a_engine, b_label, b_engine))
    mismatched = []
    for node in COUNTERPART_NODES:
        sa, sb = a["nodesizes"].get(node), b["nodesizes"].get(node)
        if sa is None or sb is None:
            mismatched.append((node, sa, sb, "not emitted"))
        elif sa != sb:
            mismatched.append((node, sa, sb, "different size"))
    if mismatched:
        lines = ["fidelity run invalid: counterpart node layouts disagree."]
        for node, sa, sb, why in mismatched:
            lines.append("  %-10s %s=%s  %s=%s  (%s)" % (node, a_label, sa, b_label, sb, why))
        lines.append("  mini-TIR's nodes are apache/tvm's node layouts; the only permitted")
        lines.append("  difference between the harnesses is which node types exist.")
        raise SystemExit("\n".join(lines))


def render_fidelity(merged, labels, runs, out):
    a_label, b_label = labels
    a, b = merged[a_label], merged[b_label]
    check_fidelity_preconditions(a, b, a_label, b_label)
    w = out.write

    def delta(x, y):
        if x is None or y is None:
            return None
        return "%.2f / %.2f (%+.1f%%)" % (x, y, 100.0 * (y - x) / x)

    w("### Fidelity -- %s against %s, one host\n\n" % (a_label, b_label))
    w("| | |\n| --- | --- |\n")
    w("| tvm-ffi engine | `%s`, identical in both |\n"
      % a["provenance"].get("tvm_ffi_engine_sha", "?"))
    w("| apache/tvm | `%s` |\n" % a["provenance"].get("tvm_sha", "?"))
    w("| compiler | `%s` |\n" % a["provenance"].get("compiler", "?"))
    w("| node layouts | all %d counterpart types identical in size |\n" % len(COUNTERPART_NODES))
    w("| method | %s, processes interleaved %s/%s/%s/%s |\n"
      % (a["provenance"]["method"], a_label, b_label, a_label, b_label))
    w("| processes | %d per harness |\n" % runs)
    w("| machine | %s |\n\n" % machine_row(MACHINE["before"], MACHINE["after"]))
    w("Cells are `%s / %s (delta)`. **This is a within-host comparison and the only kind the "
      "harness makes**: both binaries were built by one compiler on one machine against one "
      "engine, and the processes are interleaved so drift and thermal state hit both "
      "equally.\n\n" % (a_label, b_label))

    w("| node | bytes |\n| --- | ---: |\n")
    for node in COUNTERPART_NODES:
        w("| `%s` | %d |\n" % (node, a["nodesizes"][node]))
    w("\n")

    common = [n for n in a["order"] if n in b["fixtures"]]
    w("#### walk -- ns/node\n\n")
    w("| Case | " + " | ".join(header(arm) for arm in FIDELITY_WALK_ARMS) + " |\n")
    w("| --- |" + " ---: |" * len(FIDELITY_WALK_ARMS) + "\n")
    for name in common:
        cells = [delta(ns_per_node(a, name, "-", arm), ns_per_node(b, name, "-", arm))
                 for arm in FIDELITY_WALK_ARMS]
        w("| %s | %s |\n" % (name, " | ".join(c or "n/a" for c in cells)))
    w("\n")

    for kind, title in [("subst", "map, Expr-level `Var` substitution"),
                        ("swap", "map, Stmt-level `Evaluate` swap")]:
        rows = [n for n in common if a["fixtures"][n]["kind"] == kind]
        if not rows:
            continue
        arms = FIDELITY_MAP_ARMS[kind]
        w("#### %s -- ns/node\n\n" % title)
        w("| Case | own | " + " | ".join(header(arm) for arm in arms) + " |\n")
        w("| --- | --- |" + " ---: |" * len(arms) + "\n")
        for name in rows:
            for ownership in ["retained", "moved"]:
                cells = [delta(ns_per_node(a, name, ownership, arm),
                               ns_per_node(b, name, ownership, arm)) for arm in arms]
                w("| %s | %s | %s |\n" % (name, ownership, " | ".join(c or "n/a" for c in cells)))
        w("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", action="append", default=[])
    ap.add_argument("--two-state", action="store_true")
    ap.add_argument("--fidelity", action="store_true",
                    help="compare the two --binary harnesses within this host: check that "
                         "they share one engine and one node layout, then render their "
                         "interleaved agreement tables")
    ap.add_argument("--state", action="append", default=[],
                    help="LABEL=REF:PATH, given twice for a two-state run")
    ap.add_argument("--differs", default="unstated",
                    help="what differs between the two states -- the only thing the delta "
                         "can be attributed to")
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="-")
    ap.add_argument("--allow-contention", action="store_true",
                    help="run even though another benchmark process is live, and say so in "
                         "the provenance. For a deliberately contended run only")
    args = ap.parse_args()

    MACHINE["before"] = machine_state()
    if MACHINE["before"]["competing"] and not args.allow_contention:
        raise SystemExit(
            "another benchmark process is live, so this run would be contended:\n  %s\n\n"
            "Load average will not tell you this -- a pinned single-threaded run reads as an "
            "idle machine -- which is why the check is process presence. Wait for the other "
            "run to finish, or pass --allow-contention to record an explicitly contended run."
            % "\n  ".join(MACHINE["before"]["competing"]))

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
        MACHINE["after"] = machine_state()
        render_compare(merged, states, args.runs, args.differs, out)
    elif args.fidelity:
        if len(args.binary) != 2:
            raise SystemExit("--fidelity needs exactly two --binary arguments")
        paths = [os.path.abspath(p) for p in args.binary]
        # Labelled by path, then renamed from the provenance the run itself carries: naming
        # them up front would mean running each binary an extra time to read its name.
        states = [(paths[i], "", paths[i]) for i in (0, 1)]
        by_path = interleave(states, args.cpu, args.runs)
        labels = [by_path[p]["provenance"]["harness"] for p in paths]
        if labels[0] == labels[1]:
            raise SystemExit("--fidelity compares two different harnesses; both are %s"
                             % labels[0])
        merged = {labels[i]: by_path[paths[i]] for i in (0, 1)}
        MACHINE["after"] = machine_state()
        render_fidelity(merged, labels, args.runs, out)
    else:
        for binary in args.binary:
            binary = os.path.abspath(binary)
            runs = [run_once(binary, args.cpu) for _ in range(args.runs)]
            MACHINE["after"] = machine_state()
            render(median_runs(runs), runs, out)
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
