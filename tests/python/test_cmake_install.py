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
"""Exercise standalone and embedded CMake install component boundaries."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("embedded,python_module", [(False, False), (True, False), (True, True)])
def test_install_component(tmp_path: Path, embedded: bool, python_module: bool) -> None:
    """A parent default component must never capture FFI's native or Python payload."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for install integration tests")
    assert cmake is not None
    source = Path(__file__).resolve().parents[2]
    project = source
    if embedded:
        project = tmp_path / "parent"
        project.mkdir()
        (project / "parent.txt").write_text("parent payload")
        (project / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.18)\n"
            "project(parent LANGUAGES C CXX)\n"
            "set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME parent_payload)\n"
            f'add_subdirectory("{source.as_posix()}" ffi)\n'
            'install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/parent.txt" DESTINATION .)\n'
        )
    build = tmp_path / "build"

    def run(*args: str) -> None:
        subprocess.run(args, check=True, cwd=tmp_path)

    run(
        cmake,
        "-S",
        str(project),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_DEFAULT_COMPONENT_NAME=parent_payload",
        "-DTVM_FFI_USE_LIBBACKTRACE=OFF",
        f"-DTVM_FFI_BUILD_PYTHON_MODULE={'ON' if python_module else 'OFF'}",
        f"-DPython_EXECUTABLE={sys.executable}",
    )
    targets = ["tvm_ffi_shared", "tvm_ffi_testing"]
    targets.append("tvm_ffi_cython" if python_module else "tvm_ffi_static")
    run(
        cmake, "--build", str(build), "--config", "Release", "--parallel", "2", "--target", *targets
    )

    def install(name: str, component: str | None = None) -> tuple[Path, set[str]]:
        prefix = tmp_path / name
        args = [cmake, "--install", str(build), "--config", "Release", "--prefix", str(prefix)]
        if component is not None:
            args.extend(["--component", component])
        run(*args)
        return prefix, {p.relative_to(prefix).as_posix() for p in prefix.rglob("*") if p.is_file()}

    ffi_prefix, ffi_files = install("ffi", "tvm_ffi")
    _, parent_files = install("parent-install", "parent_payload")
    _, all_files = install("all")
    assert "include/tvm/ffi/c_api.h" in ffi_files
    assert "include/dlpack/dlpack.h" in ffi_files
    assert any("tvm_ffi" in name and name.startswith("lib/") for name in ffi_files)
    assert parent_files == ({"parent.txt"} if embedded else set())
    assert all_files == ffi_files | parent_files
    assert not (ffi_files & parent_files)
    assert not install("unknown", "unrelated")[1]
    if python_module:
        assert "share/cmake/tvm_ffi/tvm_ffi-config.cmake" in ffi_files
        assert "src/ffi/object.cc" in ffi_files
        assert "include/tvm_ffi_python_helpers.h" in ffi_files
        assert any(name.startswith("core.") for name in ffi_files)
        assert not any("tvm_ffi_static" in name for name in ffi_files)
        # wheel.packages supplies these pure Python files; CMake supplies the
        # matching extension and native libraries being exercised here.
        package = tmp_path / "python" / "tvm_ffi"
        shutil.copytree(source / "python" / "tvm_ffi", package)
        shutil.copytree(ffi_prefix, package, dirs_exist_ok=True)
        subprocess.run(
            [sys.executable, "-c", "import tvm_ffi; assert tvm_ffi.get_global_func"],
            check=True,
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(package.parent)},
        )
    else:
        assert not any(name.startswith("core.") for name in ffi_files)
        assert any("tvm_ffi_static" in name for name in ffi_files)
