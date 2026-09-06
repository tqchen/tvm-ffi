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
"""Run the structural-traversal harnesses and render the fixed report.

The reporting format lives here, not in a task record: rows are cases, columns are arms,
one quantity per table, `ns/node` on the fixture's unique-node count with the same divisor
for every arm in a row, and deltas always against the same baseline.

    ./report.py --binary ../../../build_bench/mini_tir_bench \
                --binary ../../../build_bench/real_tvm_bench --cpu 0

Each binary is run `--runs` times pinned to one CPU; the reported value is the median of the
process medians.  A process whose assertions fail exits non-zero and aborts the report --
that is the point of the assertions.
"""
import argparse
import os
import statistics
import subprocess
import sys

# The order arms appear in a table. Defined once here, matching bench_common.h's vocabulary.
WALK_ARMS = ["walk_floor", "walk", "walk_never", "walk_old"]
MAP_ARMS = ["map_floor", "map_never", "map_identity", "map_replace", "map_old"]


def run_once(binary, cpu):
    cmd = (["taskset", "-c", str(cpu)] if cpu is not None else []) + [binary]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            "%s exited %d: an assertion failed, so this run is invalid" % (binary, proc.returncode)
        )
    return proc.stdout


def parse(text):
    out = {"provenance": {}, "fixtures": {}, "results": {}, "counters": {}, "asserts": []}
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        parts = line.split("\t")
        tag = parts[0]
        if tag == "#provenance":
            out["provenance"][parts[1]] = parts[2]
        elif tag == "#fixture":
            _, harness, fixture, ownership, unique, occ, ws, fits, changed, unchanged = parts
            out["fixtures"][(harness, fixture, ownership)] = dict(
                unique=int(unique), occurrences=int(occ), working_set=int(ws), fits=fits,
                changed=int(changed), unchanged=int(unchanged))
        elif tag == "#result":
            _, harness, fixture, ownership, arm, unique, ns = parts
            out["results"][(harness, fixture, ownership, arm)] = (int(unique), float(ns))
        elif tag == "#counter":
            _, harness, fixture, ownership, arm, name, value = parts
            out["counters"][(harness, fixture, ownership, arm, name)] = int(value)
        elif tag == "#assert":
            out["asserts"].append((parts[1], parts[2], parts[3], parts[4]))
    return out


def median_runs(runs):
    merged = dict(runs[0])
    merged["results"] = {}
    for key in runs[0]["results"]:
        unique = runs[0]["results"][key][0]
        values = [r["results"][key][1] for r in runs if key in r["results"]]
        merged["results"][key] = (unique, statistics.median(values))
    return merged


def fmt(value):
    return "%.2f" % value


def render(merged, runs, out):
    prov = merged["provenance"]
    harness = prov["harness"]
    w = out.write

    w("### Provenance -- %s\n\n" % harness)
    w("| | |\n| --- | --- |\n")
    for key, label in [
        ("harness_branch_commit", "harness branch commit"),
        ("tvm_ffi_engine_sha", "tvm-ffi engine sha"),
        ("tvm_sha", "apache/tvm sha"),
        ("tvm_ffi_submodule_pin", "tvm-ffi submodule pin"),
        ("machine", "machine"),
        ("compiler", "compiler"),
        ("flags", "flags"),
        ("pinning", "pinning"),
        ("write_once_attr_guard", "write-once attr guard"),
    ]:
        w("| %s | `%s` |\n" % (label, prov.get(key, "")))
    w("| cache | L1d %s B, L2 %s B, L3 %s B |\n" % (
        prov.get("cache_l1d_bytes"), prov.get("cache_l2_bytes"), prov.get("cache_l3_bytes")))
    w("| method | %s |\n" % prov["method"])
    w("| processes | %d |\n\n" % len(runs))

    w("### Fixtures -- %s\n\n" % harness)
    w("| fixture | ownership | unique nodes | occurrences | working set | fits | changed | unchanged |\n")
    w("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |\n")
    seen = []
    for (h, fixture, ownership), info in merged["fixtures"].items():
        if h != harness:
            continue
        seen.append((fixture, ownership, info))
    for fixture, ownership, info in seen:
        w("| `%s` | %s | %d | %d | %s | %s | %d | %d |\n" % (
            fixture, ownership, info["unique"], info["occurrences"],
            human(info["working_set"]), info["fits"], info["changed"], info["unchanged"]))
    w("\n")

    fixtures = []
    for (h, fixture, ownership) in merged["fixtures"]:
        if h == harness and fixture not in fixtures:
            fixtures.append(fixture)

    w("### %s -- walk -- ns/node\n\n" % harness)
    w("| Case | N | " + " | ".join("`%s`" % a for a in WALK_ARMS) + " | delta vs `walk_old` |\n")
    w("| --- | ---: |" + " ---: |" * (len(WALK_ARMS) + 1) + "\n")
    for fixture in fixtures:
        row = []
        n = None
        for arm in WALK_ARMS:
            key = (harness, fixture, "-", arm)
            if key not in merged["results"]:
                row.append(None)
                continue
            n, ns = merged["results"][key]
            row.append(ns / n)
        if n is None:
            continue
        delta = "" if row[1] is None or row[3] is None else "**%+.1f%%**" % (
            100.0 * (row[1] - row[3]) / row[3])
        w("| %s | %d | %s | %s |\n" % (
            fixture, n, " | ".join(fmt(v) if v is not None else "" for v in row), delta))
    w("\n")

    for ownership in ["retained", "moved"]:
        w("### %s -- map, %s -- ns/node\n\n" % (harness, ownership))
        w("| Case | N | " + " | ".join("`%s`" % a for a in MAP_ARMS) + " | delta vs `map_old` |\n")
        w("| --- | ---: |" + " ---: |" * (len(MAP_ARMS) + 1) + "\n")
        for fixture in fixtures:
            row = []
            n = None
            for arm in MAP_ARMS:
                key = (harness, fixture, ownership, arm)
                if key not in merged["results"]:
                    row.append(None)
                    continue
                n, ns = merged["results"][key]
                row.append(ns / n)
            if n is None:
                continue
            delta = "" if row[3] is None or row[4] is None else "**%+.1f%%**" % (
                100.0 * (row[3] - row[4]) / row[4])
            w("| %s, %s | %d | %s | %s |\n" % (
                fixture, ownership, n,
                " | ".join(fmt(v) if v is not None else "" for v in row), delta))
        w("\n")

    w("### %s -- dispatch and rebuild counts\n\n" % harness)
    w("| fixture | ownership | arm | counter | value |\n| --- | --- | --- | --- | ---: |\n")
    for (h, fixture, ownership, arm, name), value in merged["counters"].items():
        if h != harness or name == "occurrences" or (name == "unique_nodes"):
            continue
        w("| `%s` | %s | `%s` | %s | %d |\n" % (fixture, ownership, arm, name, value))
    w("\n")

    names = [a[1] for a in merged["asserts"] if a[0] == harness]
    w("### %s -- assertions\n\n" % harness)
    w("%d assertions, all passing in every one of the %d processes.\n\n" % (len(names), len(runs)))


def human(n):
    for unit in ["B", "KiB", "MiB"]:
        if n < 1024 or unit == "MiB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0


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
        for run in runs:
            bad = [a for a in run["asserts"] if a[2] != "pass"]
            if bad:
                raise SystemExit("assertion failed: %s" % bad)
        render(median_runs(runs), runs, out)
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
