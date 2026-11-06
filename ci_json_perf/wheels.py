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

"""Build a matched pair of macOS wheels before the timing window."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

root = Path.cwd()
out = root / "results"
out.mkdir(exist_ok=True)
report = {
    "environment": {
        k: os.environ.get(k)
        for k in ["SETUPTOOLS_SCM_PRETEND_VERSION", "SOURCE_DATE_EPOCH", "MACOSX_DEPLOYMENT_TARGET"]
    },
    "python": sys.version,
    "tools": {
        name: importlib.metadata.version(name)
        for name in ["build", "scikit-build-core", "cython", "setuptools-scm", "ninja", "delocate"]
    },
    "states": {},
}
for state in ["baseline", "candidate"]:
    dest = out / (state + "-wheels")
    with (out / (state + "-wheel-build.log")).open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(dest),
                str(root / state),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        wheel = next(dest.glob("*.whl"))
        subprocess.run(
            [
                "delocate-wheel",
                "--require-archs",
                "arm64",
                "-w",
                str(out / (state + "-repaired")),
                str(wheel),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    wheel = next((out / (state + "-repaired")).glob("*.whl"))
    entry = {
        "wheel": wheel.name,
        "bytes": wheel.stat().st_size,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "native": {},
    }
    with zipfile.ZipFile(wheel) as z:
        for member in z.infolist():
            if member.filename.endswith((".so", ".dylib")):
                entry["native"][member.filename] = {
                    "bytes": member.file_size,
                    "compressed": member.compress_size,
                    "sha256": hashlib.sha256(z.read(member.filename)).hexdigest(),
                }
    report["states"][state] = entry
(out / "wheel-sizes.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
