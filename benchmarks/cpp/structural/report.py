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
MAP_ARMS = ["map_floor", "map_never", "map_identity", "map_replace", "map_functor", "map_old"]

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


def header(arm):
    """`map_floor` renders as `map<br>floor`: full arm name, no column width."""
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

    w("#### %s -- map -- ns/node, both ownership variants\n\n" % harness)
    w("| Case | ownership | N | " + " | ".join(header(a) for a in MAP_ARMS) +
      " | replace<br>vs functor |\n")
    w("| --- | --- | ---: |" + " ---: |" * (len(MAP_ARMS) + 1) + "\n")
    for name in results:
        for ownership in ["retained", "moved"]:
            row = [cell(name, ownership, a) for a in MAP_ARMS]
            comparable = fixtures[name]["kind"] == "all_vars"
            w("| %s | %s | %d | %s | %s |\n"
              % (name, ownership, fixtures[name]["unique"],
                 " | ".join("%.2f" % v if v is not None else "" for v in row),
                 pct(row[3], row[4]) if comparable and row[4] is not None else "--"))
    w("\n")

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
                ns = merged["results"].get((name, ownership, "map_replace"))
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
            r = merged["results"].get((name, "retained", "map_replace"))
            m = merged["results"].get((name, "moved", "map_replace"))
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
    noremap = [k for k in merged["results"] if k[2] == "map_replace_noremap"]
    if noremap:
        w("#### remap probe -- `map_replace_noremap`, ns/node\n\n")
        w("| Case | ownership | never | noremap | identity | replace |\n")
        w("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for fixture, ownership, _ in sorted(noremap):
            n = merged["fixtures"][fixture]["unique"]
            def g(arm):
                v = merged["results"].get((fixture, ownership, arm))
                return "" if v is None else "%.2f" % (v / n)
            w("| %s | %s | %s | %s | %s | %s |\n"
              % (fixture, ownership, g("map_never"), g("map_replace_noremap"),
                 g("map_identity"), g("map_replace")))
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
