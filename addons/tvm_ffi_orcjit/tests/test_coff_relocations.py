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
"""Native checks for ARM64 COFF instruction relocation addends."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from tvm_ffi_orcjit import ExecutionSession


@pytest.mark.skipif(
    sys.platform != "win32" or platform.machine().lower() not in ("arm64", "aarch64"),
    reason="requires native AArch64 COFF execution",
)
def test_coff_page_addends_and_scaled_loads(tmp_path: Path) -> None:
    """COFF ADRP addends are bytes; ADD and SIMD loads use distinct scaling."""
    compiler = shutil.which("clang")
    if compiler is None:
        pytest.skip("clang assembler is unavailable")
    source = Path(__file__).parent / "sources" / "arm64" / "coff_page_addends.s"
    obj = tmp_path / "page_addends.obj"
    subprocess.run(
        [compiler, "--target=aarch64-pc-windows-msvc", "-c", str(source), "-o", str(obj)],
        check=True,
    )
    assert int.from_bytes(obj.read_bytes()[:2], "little") == 0xAA64
    module = ExecutionSession().load_module(str(obj))
    assert module.probe() == 101
