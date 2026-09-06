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
# Build the structural-traversal benchmark harnesses and stamp their provenance in.
#
#   ./build.sh                       # mini-TIR only
#   ./build.sh --tvm /path/to/tvm    # mini-TIR and real TVM
#
# The tvm-ffi build this links against must be configured with
# -DTVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE=ON, which disables the write-once type-attribute
# guard so the harness can wrap registered hooks for its dispatch-count assertions.  For the
# real-TVM harness, that means TVM itself must be configured with that option, since TVM
# builds tvm-ffi through add_subdirectory(3rdparty/tvm-ffi).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFI_ROOT="$(cd "${HERE}/../../.." && pwd)"
FFI_BUILD="${FFI_BUILD:-${FFI_ROOT}/build}"
OUT="${OUT:-${FFI_ROOT}/build_bench}"
CXX="${CXX:-g++}"
STD="${STD:-c++20}"
FLAGS="${FLAGS:--O3 -DNDEBUG -std=${STD}}"
TVM_ROOT=""
TVM_BUILD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tvm) TVM_ROOT="$(cd "$2" && pwd)"; shift 2 ;;
    --tvm-build) TVM_BUILD="$(cd "$2" && pwd)"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -n "${TVM_ROOT}" && -z "${TVM_BUILD}" ]] && TVM_BUILD="${TVM_ROOT}/build"

mkdir -p "${OUT}"

harness_commit="$(git -C "${FFI_ROOT}" rev-parse HEAD)"
harness_dirty="$(git -C "${FFI_ROOT}" status --porcelain -- benchmarks src include CMakeLists.txt | head -1)"
[[ -n "${harness_dirty}" ]] && harness_commit="${harness_commit}-dirty"
engine_sha="${harness_commit}"
guard="disabled (TVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE=ON)"

common_defs=(
  "-DTVM_FFI_BENCH_HARNESS_COMMIT=\"${harness_commit}\""
  "-DTVM_FFI_BENCH_ENGINE_SHA=\"${engine_sha}\""
  "-DTVM_FFI_BENCH_CXX_FLAGS=\"${FLAGS}\""
  "-DTVM_FFI_BENCH_GUARD_DISABLED=\"${guard}\""
)

echo "building mini-TIR harness"
${CXX} ${FLAGS} \
  -I"${FFI_ROOT}/include" -I"${FFI_ROOT}/3rdparty/dlpack/include" -I"${HERE}" \
  "${common_defs[@]}" \
  -DTVM_FFI_BENCH_TVM_SHA="\"n/a (mini-TIR builds from tvm-ffi types alone)\"" \
  "${HERE}/mini_tir_bench.cc" \
  -L"${FFI_BUILD}/lib" -ltvm_ffi -Wl,-rpath,"${FFI_BUILD}/lib" \
  -o "${OUT}/mini_tir_bench"

if [[ -n "${TVM_ROOT}" ]]; then
  tvm_sha="$(git -C "${TVM_ROOT}" rev-parse HEAD)"
  tvm_dirty="$(git -C "${TVM_ROOT}" status --porcelain -- src include CMakeLists.txt | head -1)"
  [[ -n "${tvm_dirty}" ]] && tvm_sha="${tvm_sha}-dirty"
  ffi_pin="$(git -C "${TVM_ROOT}/3rdparty/tvm-ffi" rev-parse HEAD)"
  echo "building real-TVM harness against ${tvm_sha} (tvm-ffi ${ffi_pin})"
  ${CXX} ${FLAGS} \
    -I"${TVM_ROOT}/include" -I"${TVM_ROOT}/3rdparty/tvm-ffi/include" \
    -I"${TVM_ROOT}/3rdparty/tvm-ffi/3rdparty/dlpack/include" -I"${HERE}" \
    "-DTVM_FFI_BENCH_HARNESS_COMMIT=\"${harness_commit}\"" \
    "-DTVM_FFI_BENCH_ENGINE_SHA=\"${ffi_pin}\"" \
    "-DTVM_FFI_BENCH_CXX_FLAGS=\"${FLAGS}\"" \
    "-DTVM_FFI_BENCH_GUARD_DISABLED=\"${guard}\"" \
    "-DTVM_FFI_BENCH_TVM_SHA=\"${tvm_sha}\"" \
    "-DTVM_FFI_BENCH_TVM_FFI_PIN=\"${ffi_pin}\"" \
    "${HERE}/real_tvm_bench.cc" \
    -L"${TVM_BUILD}/lib" -ltvm_compiler -ltvm_ffi -Wl,-rpath,"${TVM_BUILD}/lib" \
    -o "${OUT}/real_tvm_bench"
fi
echo "binaries in ${OUT}"
