#!/usr/bin/env bash
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
#
# Build the structural benchmarks once per engine in this tree, from one tree, in one go.
#
#   ./build_mix.sh --tvm /path/to/tvm      # apache/tvm whose 3rdparty/tvm-ffi is THIS checkout,
#                                          # configured with -DTVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE=ON
#                                          # and built (--target tvm_compiler)
#
# Seven executables come out, each compiled against exactly one engine header and one hook file:
#
#   variant  engine header                                    hook file                  drivers
#   gold     include/tvm/ffi/extra/structural_mutate_gold.h   tvm_hook_override_gold.h   split_fuse_bench, real_tvm_bench
#   uc       include/tvm/ffi/extra/structural_mutate.h        tvm_hook_override_uc.h     split_fuse_bench, real_tvm_bench
#   mixed    include/tvm/ffi/extra/structural_mutate_mixed.h  tvm_hook_override_mixed.h  split_fuse_bench
#   old      include/tvm/ffi/extra/structural_mutate_old.h    tvm_hook_override.h        split_fuse_bench, real_tvm_bench
#
# old is upstream main e74e58f's engine verbatim (structural_mutate_old.h) under the OLD hook
# file, reached through the tvm_hook_override_old.h wrapper; it also compiles
# src/ffi/extra/structural_mutate_old.cc into the executable so the OLD engine's container
# descent uses OLD's own Array/Map hooks rather than the library's UC ones. `--only a,b`
# builds a subset, leaving the other executables exactly as they are.
#
# split_fuse_bench_<variant> is the two binary split-fuse fixtures; real_tvm_bench_<variant> is
# the full fixture set (split-fuse, call-split-fuse, seq), which the gold and uc hook files cover
# and the mixed hook file -- the round-1, binary-only control -- does not.
#
# The hook file is what selects the engine: it includes the engine header it was written
# against, and the driver refuses to build if the engine it finds is not the one named here.
# Nothing is selected at run time. The content hashes of the engine header, of expected.h and of
# the hook file are stamped into each executable's provenance, so a run's tables say which bytes
# they measured.
#
# libtvm_ffi and libtvm_compiler are one build, against structural_mutate.h (UC), shared by all
# three executables. No engine code crosses that boundary on a timed arm except `old`, which is
# TVM's own Substitute and is the same shared-library code in every executable; both libraries
# are built with hidden visibility, so nothing the executable defines is interposed into them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFI_ROOT="$(cd "${HERE}/../../.." && pwd)"
OUT="${OUT:-${FFI_ROOT}/build_bench}"
CXX="${CXX:-g++}"
STD="${STD:-c++20}"
FLAGS="${FLAGS:--O3 -DNDEBUG -std=${STD}}"
TVM_ROOT=""
TVM_BUILD=""
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tvm) TVM_ROOT="$(cd "$2" && pwd)"; shift 2 ;;
    --tvm-build) TVM_BUILD="$(cd "$2" && pwd)"; shift 2 ;;
    --only) ONLY=",$2,"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -n "${TVM_ROOT}" ]] || { echo "--tvm is required" >&2; exit 1; }
[[ -z "${TVM_BUILD}" ]] && TVM_BUILD="${TVM_ROOT}/build"
if [[ "$(readlink -f "${TVM_ROOT}/3rdparty/tvm-ffi")" != "$(readlink -f "${FFI_ROOT}")" ]]; then
  echo "${TVM_ROOT}/3rdparty/tvm-ffi is not this checkout; the libraries would be another engine's" >&2
  exit 1
fi
[[ -f "${TVM_BUILD}/lib/libtvm_compiler.so" ]] || { echo "${TVM_BUILD}/lib/libtvm_compiler.so is not built" >&2; exit 1; }

mkdir -p "${OUT}"

harness_commit="$(git -C "${FFI_ROOT}" rev-parse HEAD)"
harness_dirty="$(git -C "${FFI_ROOT}" status --porcelain -- benchmarks src include CMakeLists.txt | head -1)"
[[ -n "${harness_dirty}" ]] && harness_commit="${harness_commit}-dirty"
tvm_sha="$(git -C "${TVM_ROOT}" rev-parse HEAD)"
tvm_dirty="$(git -C "${TVM_ROOT}" status --porcelain -- src include CMakeLists.txt | head -1)"
[[ -n "${tvm_dirty}" ]] && tvm_sha="${tvm_sha}-dirty"

sha() { sha256sum "$1" | cut -c1-64; }

expected_sha="$(sha "${FFI_ROOT}/include/tvm/ffi/expected.h")"
echo "expected.h sha256:${expected_sha:0:12}"

build_variant() {
  local driver="$1" name="$2" code="$3" engine="$4" hooks="$5" hooks_sha_of="${6:-$5}" extra_src="${7:-}"
  if [[ -n "${ONLY}" && "${ONLY}" != *",${name},"* ]]; then return; fi
  local engine_path="${FFI_ROOT}/include/tvm/ffi/extra/${engine}"
  local hooks_path="${HERE}/${hooks_sha_of}"
  local engine_sha hooks_sha
  engine_sha="$(sha "${engine_path}")"
  hooks_sha="$(sha "${hooks_path}")"
  echo "${driver} ${name}: ${engine} sha256:${engine_sha:0:12}, ${hooks_sha_of} sha256:${hooks_sha:0:12}"
  ${CXX} ${FLAGS} \
    -I"${TVM_ROOT}/include" -I"${TVM_ROOT}/3rdparty/tvm-ffi/include" \
    -I"${TVM_ROOT}/3rdparty/tvm-ffi/3rdparty/dlpack/include" -I"${HERE}" \
    "-DTVM_FFI_BENCH_HARNESS_COMMIT=\"${harness_commit}\"" \
    "-DTVM_FFI_BENCH_CXX_FLAGS=\"${FLAGS}\"" \
    "-DTVM_FFI_BENCH_ENGINE_SHA=\"${harness_commit} (${name})\"" \
    "-DTVM_FFI_BENCH_TVM_SHA=\"${tvm_sha}\"" \
    "-DTVM_FFI_BENCH_TVM_FFI_PIN=\"${harness_commit}\"" \
    "-DTVM_FFI_BENCH_HOOK_HEADER=\"${hooks_sha_of}\"" \
    "-DTVM_FFI_BENCH_HOOK_INCLUDE=\"${hooks}\"" \
    "-DTVM_FFI_BENCH_VARIANT=\"${name}\"" \
    "-DTVM_FFI_BENCH_VARIANT_CODE=${code}" \
    "-DTVM_FFI_BENCH_ENGINE_HEADER=\"${engine}\"" \
    "-DTVM_FFI_BENCH_ENGINE_SHA256=\"${engine_sha}\"" \
    "-DTVM_FFI_BENCH_EXPECTED_SHA256=\"${expected_sha}\"" \
    "-DTVM_FFI_BENCH_HOOKS_SHA256=\"${hooks_sha}\"" \
    "${HERE}/${driver}.cc" ${extra_src} \
    -L"${TVM_BUILD}/lib" -ltvm_compiler -ltvm_ffi -Wl,-rpath,"${TVM_BUILD}/lib" \
    -o "${OUT}/${driver}_${name}"
  echo "  -> ${OUT}/${driver}_${name}"
}

build_variant split_fuse_bench gold  1 structural_mutate_gold.h  tvm_hook_override_gold.h
build_variant split_fuse_bench uc    2 structural_mutate.h       tvm_hook_override_uc.h
build_variant split_fuse_bench mixed 3 structural_mutate_mixed.h tvm_hook_override_mixed.h
build_variant real_tvm_bench   gold  1 structural_mutate_gold.h  tvm_hook_override_gold.h
build_variant real_tvm_bench   uc    2 structural_mutate.h       tvm_hook_override_uc.h
build_variant split_fuse_bench old   4 structural_mutate_old.h   tvm_hook_override_old.h tvm_hook_override.h "${FFI_ROOT}/src/ffi/extra/structural_mutate_old.cc"
build_variant real_tvm_bench   old   4 structural_mutate_old.h   tvm_hook_override_old.h tvm_hook_override.h "${FFI_ROOT}/src/ffi/extra/structural_mutate_old.cc"
echo "binaries in ${OUT}"
