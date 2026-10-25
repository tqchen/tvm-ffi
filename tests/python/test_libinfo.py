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
from types import SimpleNamespace
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


def test_find_shared_library_path_extra_lib_paths(tmp_path: Path) -> None:
    """A directory passed via ``extra_lib_paths`` is searched in the dev fallback."""
    lib_file = tmp_path / _platform_lib_filename("dummy")
    lib_file.write_bytes(b"")

    result = libinfo.find_shared_library_path(
        "nonexistent-dist-name", "dummy", extra_lib_paths=[tmp_path / "missing", str(tmp_path)]
    )

    assert isinstance(result, str)
    assert result == str(lib_file.resolve())
    # Sanity: the result must live under the temp tree, not tvm_ffi's own tree.
    assert tmp_path.resolve() in Path(result).parents


def test_find_shared_library_path_extra_lib_paths_type_error(tmp_path: Path) -> None:
    """Unsupported elements in ``extra_lib_paths`` raise a TypeError naming the offender."""
    with pytest.raises(TypeError, match=r"extra_lib_paths\[1\] must be str or pathlib\.Path"):
        libinfo.find_shared_library_path(
            "apache-tvm-ffi",
            "tvm_ffi",
            extra_lib_paths=[tmp_path, 123],  # ty: ignore[invalid-argument-type]
        )


def test_resolve_first_valid_path(tmp_path: Path) -> None:
    """Mixed candidates are resolved before applying the predicate in order."""
    candidates = [str(tmp_path / "missing"), tmp_path]
    seen = []

    def is_directory(path: Path) -> bool:
        assert isinstance(path, Path)
        seen.append(path)
        return path.is_dir()

    assert libinfo.resolve_first_valid_path(candidates, is_directory) == str(tmp_path.resolve())
    assert seen == [tmp_path / "missing", tmp_path]
    assert libinfo.resolve_first_valid_path(candidates, lambda _: False) is None
    assert libinfo.resolve_first_valid_path(candidates, lambda _: Path("relative")) == "relative"

    def skip_missing(path: Path) -> bool:
        if not path.is_dir():
            raise OSError("missing")
        return True

    assert libinfo.resolve_first_valid_path(candidates, skip_missing) == str(tmp_path.resolve())


def test_private_libinfo_compatibility() -> None:
    """Private callers retain the finder Path return and resolver behavior."""
    result = libinfo._find_library_by_basename("apache-tvm-ffi", "tvm_ffi")
    assert isinstance(result, Path)
    assert result.parent.is_dir()
    assert str(result) == libinfo.find_shared_library_path("apache-tvm-ffi", "tvm_ffi")
    assert libinfo._resolve_and_validate([result], lambda _: True) == str(result)


def test_find_shared_library_path_without_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling defaults retains metadata precedence and explicit directories."""

    def unexpected_lookup(*args: object) -> None:
        pytest.fail("lookup consulted a disabled default search source")

    monkeypatch.setattr(libinfo, "_rel_top_directory", unexpected_lookup)
    monkeypatch.setattr(libinfo, "_dev_top_directory", unexpected_lookup)
    monkeypatch.setattr(libinfo, "_split_env_var", unexpected_lookup)
    filename = _platform_lib_filename("dummy")
    distribution = SimpleNamespace(
        _path=tmp_path / "dummy.dist-info", read_text=lambda _: f"metadata/{filename},,"
    )
    monkeypatch.setattr(libinfo.im, "distribution", lambda _: distribution)
    library = tmp_path / filename
    library.touch()
    assert libinfo.find_shared_library_path(
        "dummy-package", "dummy", [str(tmp_path)], use_default_paths=False
    ) == str((tmp_path / "metadata" / filename).resolve())

    def missing_distribution(package: str) -> None:
        raise libinfo.im.PackageNotFoundError(package)

    monkeypatch.setattr(libinfo.im, "distribution", missing_distribution)
    assert libinfo.find_shared_library_path(
        "dummy-package", "dummy", [str(tmp_path)], use_default_paths=False
    ) == str(library.resolve())
    assert (
        libinfo.find_shared_library_path(
            "dummy-package", "dummy", [], use_default_paths=False, allow_missing=True
        )
        is None
    )
    with pytest.raises(RuntimeError, match="Cannot find library"):
        libinfo.find_shared_library_path("dummy-package", "dummy", use_default_paths=False)


def test_find_shared_library_path_default_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Built-in directories remain enabled by default and can be disabled."""
    library = tmp_path / "lib" / _platform_lib_filename("dummy")
    library.parent.mkdir()
    library.touch()
    monkeypatch.setattr(libinfo, "_rel_top_directory", lambda: tmp_path)
    monkeypatch.setattr(libinfo, "_dev_top_directory", lambda: tmp_path)
    assert libinfo.find_shared_library_path("nonexistent-dist-name", "dummy") == str(
        library.resolve()
    )
    assert (
        libinfo.find_shared_library_path(
            "nonexistent-dist-name", "dummy", use_default_paths=False, allow_missing=True
        )
        is None
    )


def test_find_shared_library_path_allow_missing(tmp_path: Path) -> None:
    """Missing-result policy changes only lookup misses, not invalid arguments."""
    target_name = "nonexistent_library_for_libinfo_test"
    with pytest.raises(RuntimeError, match="Cannot find library"):
        libinfo.find_shared_library_path("nonexistent-dist-name", target_name)
    assert (
        libinfo.find_shared_library_path("nonexistent-dist-name", target_name, allow_missing=True)
        is None
    )
    with pytest.raises(TypeError, match=r"must be str or pathlib\.Path"):
        libinfo.find_shared_library_path(
            "unused-package",
            "dummy",
            [123],  # ty: ignore[invalid-argument-type]
            allow_missing=True,
            use_default_paths=False,
        )
    with pytest.raises(TypeError):
        libinfo.find_shared_library_path(
            "unused-package",
            "dummy",
            [tmp_path],
            True,  # ty: ignore[too-many-positional-arguments]
            True,
        )
