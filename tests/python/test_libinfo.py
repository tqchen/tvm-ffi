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
"""Tests for the tvm-ffi-config command line utility."""

import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest
from tvm_ffi import libinfo


def _stdout_for(*args: str) -> str:
    """Invoke tvm-ffi-config with the provided arguments and return stdout with trailing whitespace removed."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tvm_ffi.config",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""
    return result.stdout.strip()


@pytest.mark.parametrize(
    ("flag", "expected_fn", "is_dir"),
    [
        ("--includedir", libinfo.find_include_path, True),
        ("--dlpack-includedir", libinfo.find_dlpack_include_path, True),
        ("--cmakedir", libinfo.find_cmake_path, True),
        ("--sourcedir", libinfo.find_source_path, True),
        ("--cython-lib-path", libinfo.find_cython_lib, False),
    ],
)
def test_basic_path_flags(flag: str, expected_fn: Callable[[], str], is_dir: bool) -> None:
    output = _stdout_for(flag)
    assert output == expected_fn()
    path = Path(output)
    assert path.exists()
    assert path.is_dir() if is_dir else path.is_file()


def test_libdir_matches_library_parent() -> None:
    expected_dir = Path(libinfo.find_libtvm_ffi()).parent
    output = _stdout_for("--libdir")
    assert output == str(expected_dir)
    assert Path(output).is_dir()
    assert Path(libinfo.find_libtvm_ffi()).is_file()


def test_libfiles_reports_platform_library() -> None:
    output = _stdout_for("--libfiles")
    if sys.platform.startswith("win32"):
        expected = libinfo.find_windows_implib()
    else:
        expected = libinfo.find_libtvm_ffi()
    assert output == expected
    assert Path(output).is_file()


def test_libs_reports_link_target() -> None:
    output = _stdout_for("--libs")
    if sys.platform.startswith("win32"):
        assert output == libinfo.find_windows_implib()
    else:
        assert output == "-ltvm_ffi"


def test_cxxflags_include_paths_and_standard() -> None:
    include_dir = libinfo.find_include_path()
    dlpack_dir = libinfo.find_dlpack_include_path()
    assert _stdout_for("--cxxflags") == f"-I{include_dir} -I{dlpack_dir} -std=c++17"


def test_cflags_include_paths() -> None:
    include_dir = libinfo.find_include_path()
    dlpack_dir = libinfo.find_dlpack_include_path()
    assert _stdout_for("--cflags") == f"-I{include_dir} -I{dlpack_dir}"


def test_ldflags_only_on_unix() -> None:
    output = _stdout_for("--ldflags")
    if sys.platform.startswith("win32"):
        assert output == ""
    else:
        libdir = Path(libinfo.find_libtvm_ffi()).parent
        assert output == f"-L{libdir}"
        assert libdir.is_dir()


def test_cmakedir_contains_config_file() -> None:
    cmake_dir = Path(_stdout_for("--cmakedir"))
    assert (cmake_dir / "tvm_ffi-config.cmake").is_file()


def test_find_python_helper_include_path() -> None:
    path = libinfo.find_python_helper_include_path()
    assert Path(path).is_dir()
    assert (Path(path) / "tvm_ffi_python_helpers.h").is_file()


def _platform_lib_filename(target_name: str) -> str:
    """Return the platform-appropriate shared library filename for ``target_name``."""
    if sys.platform.startswith("win32"):
        return f"{target_name}.dll"
    if sys.platform.startswith("darwin"):
        return f"lib{target_name}.dylib"
    return f"lib{target_name}.so"


def test_find_library_by_basename_extra_lib_paths(tmp_path: Path) -> None:
    """A directory passed via ``extra_lib_paths`` is searched in the dev fallback."""
    lib_file = tmp_path / _platform_lib_filename("dummy")
    lib_file.write_bytes(b"")

    result = libinfo._find_library_by_basename(
        "nonexistent-dist-name", "dummy", extra_lib_paths=[tmp_path]
    )

    assert isinstance(result, Path)
    assert result.resolve() == lib_file.resolve()
    # Sanity: the result must live under the temp tree, not tvm_ffi's own tree.
    assert tmp_path.resolve() in result.resolve().parents


def test_find_library_by_basename_extra_lib_paths_type_error(tmp_path: Path) -> None:
    """Non-Path elements in ``extra_lib_paths`` raise a TypeError naming the offender."""
    with pytest.raises(TypeError, match=r"extra_lib_paths\[1\] must be a pathlib\.Path"):
        libinfo._find_library_by_basename(
            "apache-tvm-ffi",
            "tvm_ffi",
            extra_lib_paths=[tmp_path, str(tmp_path)],  # ty: ignore[invalid-argument-type]
        )


def test_find_library_by_basename_search_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fallback and missing policies leave metadata and explicit paths active."""
    library = tmp_path / "lib" / _platform_lib_filename("dummy")
    library.parent.mkdir()
    library.touch()
    monkeypatch.setattr(libinfo, "_rel_top_directory", lambda: tmp_path)
    monkeypatch.setattr(libinfo, "_dev_top_directory", lambda: tmp_path)
    monkeypatch.setattr(libinfo, "_split_env_var", lambda *_: [str(library.parent)])
    finder = libinfo._find_library_by_basename
    assert finder("nonexistent-dist-name", "dummy") == library.resolve()
    assert (
        finder("nonexistent-dist-name", "dummy", use_fallback_path=False, allow_missing=True)
        is None
    )
    with pytest.raises(RuntimeError, match="Cannot find library"):
        finder("nonexistent-dist-name", "dummy", use_fallback_path=False)
    assert (
        finder("nonexistent-dist-name", "dummy", [library.parent], use_fallback_path=False)
        == library.resolve()
    )
    with pytest.raises(TypeError, match=r"must be a pathlib\.Path"):
        finder(
            "nonexistent-dist-name",
            "dummy",
            [str(library.parent)],  # ty: ignore[invalid-argument-type]
            allow_missing=True,
        )
    with pytest.raises(TypeError):
        finder(
            "nonexistent-dist-name",
            "dummy",
            [],
            True,  # ty: ignore[too-many-positional-arguments]
            False,
        )
    # A metadata hit remains preferred over explicit paths even with fallback off.
    metadata = tmp_path / "dummy.dist-info"
    metadata.mkdir()
    (metadata / "RECORD").write_text(f"lib/{library.name},,\n")
    monkeypatch.setattr(libinfo.im, "distribution", lambda _: libinfo.im.PathDistribution(metadata))
    assert finder("dummy", "dummy", [tmp_path], use_fallback_path=False) == library.resolve()
