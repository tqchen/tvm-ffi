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

"""Matched JSON performance measurements; validation branch only."""

import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
HARNESS = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
STATES = ("baseline", "candidate")
REPEATS = 41


def capture(args: list[str]) -> dict[str, Any]:
    """Capture a diagnostic command without hiding its exit status."""
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return {
        "command": args,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def snapshot() -> dict[str, Any]:
    """Record visible process and CPU state around the measurement window."""
    data = {"time": time.time(), "load": os.getloadavg() if hasattr(os, "getloadavg") else None}
    data["processes"] = capture(
        ["tasklist", "/FO", "CSV"] if os.name == "nt" else ["ps", "-eo", "pid,ppid,comm"]
    )
    if Path("/proc/stat").exists():
        data["proc_stat"] = Path("/proc/stat").read_text()
    return data


def sha(path: Path) -> str:
    """Hash an exact executable or runtime library."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


result = {
    "platform": platform.platform(),
    "machine": platform.machine(),
    "runner_image": os.environ.get("ImageOS"),
    "runner_image_version": os.environ.get("ImageVersion"),
    "run_id": os.environ.get("GITHUB_RUN_ID"),
    "validation_head": os.environ.get("GITHUB_SHA"),
    "repeats": REPEATS,
    "samples": [],
    "summary": [],
    "builds": {},
}
paths = {}
for state in STATES:
    source = ROOT / state
    build = ROOT / ("build-" + state)
    config = [
        "cmake",
        "-S",
        str(HARNESS),
        "-B",
        str(build),
        "-DFFI_SOURCE=" + str(source),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DTVM_FFI_BUILD_TESTS=OFF",
    ]
    with (OUT / (state + "-build.log")).open("w") as log:
        subprocess.run(config, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build),
                "--config",
                "Release",
                "--target",
                "json_probe",
                "--parallel",
                "4",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    exe, lib = map(Path, (build / "paths-Release.txt").read_text().splitlines())
    paths[state] = exe
    shutil.copy2(build / "CMakeCache.txt", OUT / (state + "-CMakeCache.txt"))
    metadata = {
        "head": capture(["git", "-C", str(source), "rev-parse", "HEAD"])["stdout"].strip(),
        "configuration": config,
        "executable_sha256": sha(exe),
        "library_sha256": sha(lib),
        "library_bytes": lib.stat().st_size,
    }
    if platform.system() == "Linux":
        metadata["dependencies"] = capture(["readelf", "-d", str(lib)])
        metadata["symbol_versions"] = capture(["readelf", "--version-info", str(lib)])
        metadata["undefined_symbols"] = capture(["nm", "-D", "--undefined-only", str(lib)])
    elif platform.system() == "Darwin":
        metadata["dependencies"] = capture(["otool", "-L", str(lib)])
        metadata["load_commands"] = capture(["otool", "-l", str(lib)])
        metadata["undefined_symbols"] = capture(["nm", "-u", str(lib)])
    else:
        vswhere = (
            Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
        )
        installation = Path(
            capture([str(vswhere), "-latest", "-property", "installationPath"])["stdout"].strip()
        )
        dumpbin = sorted(installation.glob("VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe"))[-1]
        metadata["dependencies"] = capture([str(dumpbin), "/DEPENDENTS", str(lib)])
        metadata["imports"] = capture([str(dumpbin), "/IMPORTS", str(lib)])
    result["builds"][state] = metadata
    with (OUT / (state + "-numeric.txt")).open("w") as log:
        subprocess.run([str(exe), "validate"], stdout=log, check=True)

baseline_numeric = (OUT / "baseline-numeric.txt").read_bytes()
candidate_numeric = (OUT / "candidate-numeric.txt").read_bytes()
result["numeric_corpus_equal"] = baseline_numeric == candidate_numeric
# Keep evidence even if a platform has a numerical difference.
(OUT / "build-metadata.json").write_text(json.dumps(result, indent=2))
assert result["numeric_corpus_equal"], "Numeric corpus differs; inspect uploaded evidence"

prefix = []
if hasattr(os, "sched_getaffinity"):
    cpus = sorted(os.sched_getaffinity(0))
    result["available_cpus"] = cpus
    result["benchmark_cpu"] = cpus[0]
    prefix = ["taskset", "-c", str(cpus[0])]
    if len(cpus) > 1:
        os.sched_setaffinity(0, {cpus[-1]})
    result["driver_affinity"] = sorted(os.sched_getaffinity(0))
    sibling = Path(f"/sys/devices/system/cpu/cpu{cpus[0]}/topology/thread_siblings_list")
    result["benchmark_cpu_siblings"] = sibling.read_text() if sibling.exists() else None
else:
    result["affinity"] = "Not pinned: platform has no portable CPU-affinity API in this harness"


def measure(state: str, mode: str, case: str, iterations: int) -> tuple[float, int]:
    """Run one timed workload and return its duration and output checksum."""
    value = subprocess.check_output(
        [*prefix, str(paths[state]), mode, case, str(iterations)], text=True
    )
    ns, checksum = value.split()
    return float(ns), int(checksum)


result["before"] = snapshot()
time.sleep(2)
for mode in ["parse", "write"]:
    for case in ["int", "float", "integral_float", "scientific", "ints", "floats", "mixed"]:
        pilot, _ = measure("baseline", mode, case, 2000)
        iterations = max(1000, min(20000000, int(0.4e9 / pilot)))
        groups = {state: [] for state in STATES}
        ratios = []
        for repeat in range(REPEATS):
            values = {}
            checksums = []
            for state in STATES if repeat % 2 == 0 else STATES[::-1]:
                ns, checksum = measure(state, mode, case, iterations)
                values[state] = ns
                checksums.append(checksum)
                groups[state].append(ns)
                result["samples"].append(
                    dict(
                        state=state,
                        mode=mode,
                        case=case,
                        repeat=repeat,
                        iterations=iterations,
                        ns=ns,
                        checksum=checksum,
                    )
                )
            assert checksums[0] == checksums[1], "Benchmark outputs differ"
            ratios.append(values["candidate"] / values["baseline"])
        row = {
            "mode": mode,
            "case": case,
            "iterations": iterations,
            "paired_ratios": ratios,
            "paired_ratio_median": statistics.median(ratios),
        }
        for state in STATES:
            values = groups[state]
            row[state] = statistics.median(values)
            row[state + "_spread_pct"] = (
                100 * (max(values) - min(values)) / statistics.median(values)
            )
        result["summary"].append(row)
        print(json.dumps(row), flush=True)
result["after"] = snapshot()
# Descriptive bootstrap interval; shared-runner/systematic noise remains outside it.
rng = random.Random(955)
for row in result["summary"]:
    boot = sorted(
        statistics.median(rng.choices(row["paired_ratios"], k=REPEATS)) for _ in range(2000)
    )
    row["paired_ratio_bootstrap_95pct"] = [boot[50], boot[1949]]
(OUT / "performance.json").write_text(json.dumps(result, indent=2))
summary = [
    "## Matched JSON performance",
    "",
    f"Platform: {result['platform']}; 41 alternating pairs per workload.",
    "Shared hosted runner: results are measurements, not an isolation guarantee.",
    "",
    "| Workload | Baseline ns/op | Candidate ns/op | Median reduction | Paired ratio 95% interval |",
    "|---|---:|---:|---:|---:|",
]
for row in result["summary"]:
    lo, hi = row["paired_ratio_bootstrap_95pct"]
    summary.append(
        f"| {row['mode']}/{row['case']} | {row['baseline']:.2f} | {row['candidate']:.2f} | {100 * (1 - row['candidate'] / row['baseline']):.2f}% | {lo:.4f}-{hi:.4f} |"
    )
text = "\n".join(summary) + "\n"
(OUT / "summary.md").write_text(text)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
        stream.write(text)
