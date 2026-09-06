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
compares.  Each binary is run `--runs` times pinned to one CPU; the reported value is the
median of the process medians.

    ./report.py --binary ../../../build_bench/mini_tir_bench --cpu 0
"""
import argparse
import os
import statistics
import subprocess
import sys

WALK_ARMS = ["walk_floor", "walk_var", "walk_never", "walk_functor", "walk_old"]
# The Expr-level ladder and its two baselines: every arm here substitutes Vars, so every
# column is the same operation and every delta between them is a real comparison.
VAR_MAP_ARMS = ["map_floor", "map_never", "map_identity_var", "map_replace_var",
                "map_functor", "map_old"]
# Stmt-level, seq fixtures only. No functor baseline: StmtExprMutator hooks
# VisitExpr_(const VarNode*) and does different work on the same graph.
STMT_MAP_ARMS = ["map_floor", "map_never", "map_identity_stmt", "map_replace_stmt"]

# A cell is never blank. Where an arm cannot run on a row, it says so in a word and the
# footnote under the table gives the reason.
NA = "n/a[^1]"

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
    return proc.stdout


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


# The operation each column performs, printed in the header so a reader scanning only the
# tables can see which columns are comparable with which.
OPERATION = {
    "map_identity_var": "identity<br>Var",
    "map_replace_var": "replace<br>Var subst",
    "map_functor": "functor<br>Var subst",
    "map_old": "old<br>Var subst",
    "map_identity_stmt": "identity<br>Stmt",
    "map_replace_stmt": "replace<br>swap 2 Eval",
}


def header(arm):
    """`map_floor` renders as `map<br>floor`; arms with an operation name say which."""
    if arm in OPERATION:
        return OPERATION[arm]
    quantity, _, rest = arm.partition("_")
    return "%s<br>%s" % (quantity, rest)


def pct(new, base):
    return "**%+.1f%%**" % (100.0 * (new - base) / base)


def render(merged, runs, out):
    prov = merged["provenance"]
    harness = prov["harness"]
    fixtures = merged["fixtures"]
    results = [n for n in merged["order"] if not n.startswith("density-")]
    w = out.write

    w("### Provenance -- %s\n\n| | |\n| --- | --- |\n" % harness)
    for key, label in [("harness_branch_commit", "harness branch commit"),
                       ("tvm_ffi_engine_sha", "tvm-ffi engine sha"),
                       ("tvm_sha", "apache/tvm sha"),
                       ("compiler", "compiler"), ("flags", "flags"),
                       ("structural_hooks", "structural hooks"),
                       ("seqstmt_inplace_hook", "SeqStmt in-place hook")]:
        if key in prov:
            w("| %s | `%s` |\n" % (label, prov[key]))
    w("| method | %s |\n| processes | %d |\n\n" % (prov["method"], len(runs)))

    w("### Fixtures -- %s\n\n" % harness)
    w("| fixture | N | occurrences | working set | fits | replaces | rebuilt retained "
      "| rebuilt moved |\n")
    w("| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |\n")
    for name in results:
        f = fixtures[name]
        w("| `%s` | %d | %d | %s | %s | %s | %d | %d |\n"
          % (name, f["unique"], f["occurrences"], human(f["working_set"]),
             fits(f["working_set"]), f["kind"].replace("_", " "),
             f["rebuilt_retained"], f["rebuilt_moved"]))
    w("\n")

    def cell(fixture, ownership, arm):
        ns = merged["results"].get((fixture, ownership, arm))
        return None if ns is None else ns / fixtures[fixture]["unique"]

    w("#### %s -- walk -- ns/node\n\n" % harness)
    w("| Case | N | " + " | ".join(header(a) for a in WALK_ARMS) +
      " | var<br>vs old | var<br>vs functor |\n")
    w("| --- | ---: |" + " ---: |" * (len(WALK_ARMS) + 2) + "\n")
    for name in results:
        row = [cell(name, "-", a) for a in WALK_ARMS]
        w("| %s | %d | %s | %s | %s |\n"
          % (name, fixtures[name]["unique"],
             " | ".join("%.2f" % v if v is not None else "" for v in row),
             pct(row[1], row[4]), pct(row[1], row[3])))
    w("\n")

    def fmt(v):
        return NA if v is None else "%.2f" % v

    w("#### %s -- map, Var substitution -- ns/node, both ownership variants\n\n" % harness)
    w("Every column substitutes `Var`s, which is the operation `Substitute` and\n"
      "`FunctorSubstitute` perform, so `functor` and `old` are baselines for `replace` here\n"
      "and the two delta columns are like-for-like.\n\n")
    w("| Case | ownership | N | " + " | ".join(header(a) for a in VAR_MAP_ARMS) +
      " | replace<br>vs functor | replace<br>vs old |\n")
    w("| --- | --- | ---: |" + " ---: |" * (len(VAR_MAP_ARMS) + 2) + "\n")
    for name in results:
        for ownership in ["retained", "moved"]:
            row = [cell(name, ownership, a) for a in VAR_MAP_ARMS]
            w("| %s | %s | %d | %s | %s | %s |\n"
              % (name, ownership, fixtures[name]["unique"],
                 " | ".join(fmt(v) for v in row),
                 pct(row[3], row[4]) if row[3] and row[4] else NA,
                 pct(row[3], row[5]) if row[3] and row[5] else NA))
    w("\n")

    w("#### %s -- map, Stmt-level element swap -- ns/node\n\n" % harness)
    w("A different operation from the table above: swap two whole `Evaluate` nodes rather\n"
      "than substitute `Var`s. It is what exercises the `SeqStmt` hook's element in-place and\n"
      "splice paths, and it has no functor baseline, so no delta against `functor` or `old`\n"
      "appears here or in the table above.\n\n")
    w("| Case | ownership | N | " + " | ".join(header(a) for a in STMT_MAP_ARMS) + " |\n")
    w("| --- | --- | ---: |" + " ---: |" * len(STMT_MAP_ARMS) + "\n")
    for name in results:
        for ownership in ["retained", "moved"]:
            row = [cell(name, ownership, a) for a in STMT_MAP_ARMS]
            w("| %s | %s | %d | %s |\n"
              % (name, ownership, fixtures[name]["unique"], " | ".join(fmt(v) for v in row)))
    w("\n[^1]: `split-fuse` is an `Expr` tree with no `Stmt` nodes in it, so a Stmt-level arm\n"
      "has nothing to match and is not run on those rows.\n\n")

    # The sparse-update fixtures: ns/node amortizes one useful change over the whole
    # traversal, which is exactly the cost that should be visible rather than hidden.
    sparse = [n for n in results if fixtures[n]["kind"] == "single_var"]
    if sparse:
        w("#### %s -- sparse update: cost of changing one variable\n\n" % harness)
        # One variable occurrence changes, so ns/traversal is also ns per changed node:
        # the whole traversal buys one replacement. That is the cost ns/node would hide.
        w("| Case | ownership | N | ns per traversal | ns per changed element | rebuilt |\n")
        w("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for name in sparse:
            f = fixtures[name]
            for ownership in ["retained", "moved"]:
                ns = merged["results"].get((name, ownership, "map_replace_stmt"))
                if ns is None:
                    continue
                rebuilt = f["rebuilt_retained"] if ownership == "retained" else f["rebuilt_moved"]
                # Two elements change, so the traversal buys two swaps.
                w("| %s | %s | %d | %.0f | %.0f | %d |\n"
                  % (name, ownership, f["unique"], ns, ns / 2.0, rebuilt))
        w("\n")


def render_extras(merged, out):
    w = out.write
    density = sorted({k[0] for k in merged["results"] if k[0].startswith("density-")},
                     key=lambda n: int(n.split("-")[1].split("of")[0]))
    if density:
        w("#### change-density sweep -- `map_replace` on seq-256, ns per traversal\n\n")
        w("| changed of 256 | retained | moved | moved vs retained |\n")
        w("| --- | ---: | ---: | ---: |\n")
        for name in density:
            r = merged["results"].get((name, "retained", "map_replace_stmt"))
            m = merged["results"].get((name, "moved", "map_replace_stmt"))
            if r is None or m is None:
                continue
            w("| %s | %.0f | %.0f | %s |\n"
              % (name.split("-")[1].replace("of256", ""), r, m, pct(m, r)))
        w("\n")
    intra_seq = sorted({k[0] for k in merged["results"] if k[0].startswith("intra-seq-")},
                       key=lambda n: int(n.rsplit("-", 1)[1]))
    intra_den = sorted({k[0] for k in merged["results"] if k[0].startswith("intra-density-")},
                       key=lambda n: int(n.split("-")[2].split("of")[0]))
    if intra_seq or intra_den:
        w("#### intra-element workload -- `map_replace_field`, ns per traversal\n\n")
        w("Changes a leaf *inside* two elements rather than replacing the elements, so each\n"
          "changed `Evaluate` has its field updated and is a candidate for in-place mutation.\n\n")
        w("| case | retained | moved | moved vs retained |\n| --- | ---: | ---: | ---: |\n")
        for name in intra_seq + intra_den:
            r = merged["results"].get((name, "retained", "map_replace_field"))
            m = merged["results"].get((name, "moved", "map_replace_field"))
            if r is None or m is None:
                continue
            label = name.replace("intra-seq-", "L=").replace("intra-density-", "").replace(
                "of256", " of 256 changed, L=256")
            w("| %s | %.0f | %.0f | %s |\n" % (label, r, m, pct(m, r)))
        w("\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", action="append", required=True)
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    for binary in args.binary:
        binary = os.path.abspath(binary)
        runs = [parse(run_once(binary, args.cpu)) for _ in range(args.runs)]
        merged = median_runs(runs)
        render(merged, runs, out)
        render_extras(merged, out)
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
