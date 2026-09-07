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
# SINGLE STATE -- one binary per harness, against the checkout you are sitting in:
#
#   ./build.sh                       # mini-TIR only
#   ./build.sh --tvm /path/to/tvm    # mini-TIR and real TVM
#
# TWO STATES -- one real-TVM binary per tvm-ffi ref, for an interleaved A/B run:
#
#   ./build.sh --tvm /path/to/tvm \
#       --state A=62df2f5,hooks=tvm_hook_override_pre753.h,shim=state_shims/pre753_visit_return_none.h \
#       --state B=897ece6
#
#   Each state gets its own tvm-ffi worktree at its ref, its own apache/tvm worktree at the
#   TVM revision you pass (the same one for both -- TVM is what is being held fixed), and its
#   own build tree.  `hooks=` selects that state's hook file: the harness's hooks have to
#   compile against the API the state actually has, so there is one whole file per state
#   rather than conditionals inside one.  `shim=` force-includes a header into apache/tvm's
#   own translation units for states whose engine is missing something TVM's sources use; it
#   is never on the benchmark's include path.  Both default to the current-state files, and
#   the empty shim is force-included even when a state needs nothing, so the two states are
#   compiled with the identical flag shape.
#
#   `reuse=LABEL` builds another binary against a state already built under LABEL, for the same
#   engine with a different hook file.  It is what separates the hook half of a change from the
#   engine half inside one build lineage; without it a delta between two separately compiled
#   binaries cannot be told from a code-layout artifact.
#
#   report.py --two-state then interleaves the processes A/B/A/B.  Do not compare absolutes
#   from two separately compiled binaries; only the interleaved delta is claimed.
#
# The tvm-ffi build this links against must be configured with
# -DTVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE=ON, which disables the write-once type-attribute
# guard.  The real-TVM harness registers its own structural hooks over the ones TVM installed
# from static init, and that second registration is what the guard would refuse.  TVM itself
# must be configured with the option too, since TVM builds tvm-ffi through
# add_subdirectory(3rdparty/tvm-ffi).  In two-state mode this script does that configuring.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFI_ROOT="$(cd "${HERE}/../../.." && pwd)"
FFI_BUILD="${FFI_BUILD:-${FFI_ROOT}/build}"
OUT="${OUT:-${FFI_ROOT}/build_bench}"
WORK="${WORK:-${FFI_ROOT}/build_states}"
CXX="${CXX:-g++}"
STD="${STD:-c++20}"
FLAGS="${FLAGS:--O3 -DNDEBUG -std=${STD}}"
JOBS="${JOBS:-$(nproc)}"
TVM_ROOT=""
TVM_BUILD=""
STATES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tvm) TVM_ROOT="$(cd "$2" && pwd)"; shift 2 ;;
    --tvm-build) TVM_BUILD="$(cd "$2" && pwd)"; shift 2 ;;
    --state) STATES+=("$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -n "${TVM_ROOT}" && -z "${TVM_BUILD}" ]] && TVM_BUILD="${TVM_ROOT}/build"

mkdir -p "${OUT}"

harness_commit="$(git -C "${FFI_ROOT}" rev-parse HEAD)"
harness_dirty="$(git -C "${FFI_ROOT}" status --porcelain -- benchmarks src include CMakeLists.txt | head -1)"
[[ -n "${harness_dirty}" ]] && harness_commit="${harness_commit}-dirty"

common_defs=(
  "-DTVM_FFI_BENCH_HARNESS_COMMIT=\"${harness_commit}\""
  "-DTVM_FFI_BENCH_CXX_FLAGS=\"${FLAGS}\""
)

# ---------------------------------------------------------------------------
# Two-state mode.
# ---------------------------------------------------------------------------

# The branch-local commit that adds TVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE. A state checkout
# is at an upstream ref and does not have it, so it is applied as a patch rather than assumed.
guard_commit() {
  git -C "${FFI_ROOT}" log --format=%H --grep='Allow branch-local type-attr override' HEAD |
    tail -1
}

# One tvm-ffi worktree at REF, with the two submodules linked from this checkout. The link is
# exact rather than approximate: the pins are compared and a mismatch stops the build, because
# a state built against a different dlpack is not the state it claims to be.
prepare_ffi_state() {
  local label="$1" ref="$2" dir="${WORK}/ffi-${label}"
  rm -rf "${dir}"
  git -C "${FFI_ROOT}" worktree prune
  git -C "${FFI_ROOT}" worktree add --detach "${dir}" "${ref}" >/dev/null
  local sub here there
  for sub in dlpack libbacktrace; do
    here="$(git -C "${FFI_ROOT}" ls-tree HEAD "3rdparty/${sub}" | awk '{print $3}')"
    there="$(git -C "${FFI_ROOT}" ls-tree "${ref}" "3rdparty/${sub}" | awk '{print $3}')"
    if [[ "${here}" != "${there}" ]]; then
      echo "state ${label}: 3rdparty/${sub} pin differs (${there} vs ${here} here);" >&2
      echo "  check it out separately rather than linking, or the states are not comparable" >&2
      exit 1
    fi
    rm -rf "${dir}/3rdparty/${sub}"
    ln -s "${FFI_ROOT}/3rdparty/${sub}" "${dir}/3rdparty/${sub}"
  done
  git -C "${FFI_ROOT}" show "$(guard_commit)" | git -C "${dir}" apply -
  echo "${dir}"
}

# One apache/tvm worktree at the revision under test, with 3rdparty/tvm-ffi pointed at the
# state's engine and every other 3rdparty entry linked from the checkout being mirrored.
prepare_tvm_state() {
  local label="$1" ffi_dir="$2" dir="${WORK}/tvm-${label}"
  local sha; sha="$(git -C "${TVM_ROOT}" rev-parse HEAD)"
  rm -rf "${dir}"
  git -C "${TVM_ROOT}" worktree prune
  git -C "${TVM_ROOT}" worktree add --detach "${dir}" "${sha}" >/dev/null
  local sub
  for sub in $(ls "${TVM_ROOT}/3rdparty"); do
    rm -rf "${dir}/3rdparty/${sub}"
    if [[ "${sub}" == "tvm-ffi" ]]; then
      ln -s "${ffi_dir}" "${dir}/3rdparty/tvm-ffi"
    else
      ln -s "${TVM_ROOT}/3rdparty/${sub}" "${dir}/3rdparty/${sub}"
    fi
  done
  echo "${dir}"
}

build_state() {
  local spec="$1"
  local label="${spec%%=*}" rest="${spec#*=}"
  local ref="${rest%%,*}"
  local hooks="tvm_hook_override.h" shim="state_shims/none.h" reuse="" opt
  IFS=',' read -ra opts <<< "${rest}"
  for opt in "${opts[@]:1}"; do
    case "${opt}" in
      hooks=*) hooks="${opt#hooks=}" ;;
      shim=*) shim="${opt#shim=}" ;;
      reuse=*) reuse="${opt#reuse=}" ;;
      *) echo "state ${label}: unknown option ${opt}" >&2; exit 1 ;;
    esac
  done
  local ffi_dir tvm_dir
  if [[ -n "${reuse}" ]]; then
    # Same engine as an already-built state, different hook file. The hook file is compiled
    # into the benchmark TU alone, so the engine build is bit-for-bit the one already made and
    # rebuilding TVM against it would only cost time. Used to separate the hook half of a
    # change from the engine half within one build lineage.
    ffi_dir="${WORK}/ffi-${reuse}"
    tvm_dir="${WORK}/tvm-${reuse}"
    [[ -d "${tvm_dir}/build/lib" ]] || { echo "state ${label}: reuse=${reuse} is not built" >&2; exit 1; }
    local reuse_ref; reuse_ref="$(git -C "${ffi_dir}" rev-parse HEAD)"
    if [[ "$(git -C "${FFI_ROOT}" rev-parse "${ref}")" != "${reuse_ref}" ]]; then
      echo "state ${label}: reuse=${reuse} is at ${reuse_ref}, not ${ref}" >&2
      exit 1
    fi
  else
    ffi_dir="$(prepare_ffi_state "${label}" "${ref}")"
    tvm_dir="$(prepare_tvm_state "${label}" "${ffi_dir}")"
  fi
  local ffi_sha tvm_sha
  ffi_sha="$(git -C "${ffi_dir}" rev-parse HEAD)"
  tvm_sha="$(git -C "${tvm_dir}" rev-parse HEAD)"
  echo "state ${label}: tvm-ffi ${ffi_sha:0:10}, apache/tvm ${tvm_sha:0:10}, hooks ${hooks}, shim ${shim}"

  if [[ -z "${reuse}" ]]; then
    cmake -S "${tvm_dir}" -B "${tvm_dir}/build" -DCMAKE_BUILD_TYPE=Release \
          -DTVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE=ON \
          -DCMAKE_CXX_FLAGS="-include ${HERE}/${shim}" > "${tvm_dir}/configure.log"
    cmake --build "${tvm_dir}/build" -j "${JOBS}" --target tvm_compiler > "${tvm_dir}/build.log"
  fi

  ${CXX} ${FLAGS} \
    -I"${tvm_dir}/include" -I"${tvm_dir}/3rdparty/tvm-ffi/include" \
    -I"${tvm_dir}/3rdparty/tvm-ffi/3rdparty/dlpack/include" -I"${HERE}" \
    "${common_defs[@]}" \
    "-DTVM_FFI_BENCH_ENGINE_SHA=\"${ffi_sha}\"" \
    "-DTVM_FFI_BENCH_TVM_SHA=\"${tvm_sha}\"" \
    "-DTVM_FFI_BENCH_TVM_FFI_PIN=\"${ffi_sha}\"" \
    "-DTVM_FFI_BENCH_HOOK_HEADER=\"${hooks}\"" \
    "${HERE}/real_tvm_bench.cc" \
    -L"${tvm_dir}/build/lib" -ltvm_compiler -ltvm_ffi -Wl,-rpath,"${tvm_dir}/build/lib" \
    -o "${OUT}/real_tvm_bench_${label}"
  echo "  -> ${OUT}/real_tvm_bench_${label}"
}

if [[ ${#STATES[@]} -gt 0 ]]; then
  [[ -n "${TVM_ROOT}" ]] || { echo "--state requires --tvm" >&2; exit 1; }
  mkdir -p "${WORK}"
  for spec in "${STATES[@]}"; do build_state "${spec}"; done
  echo "binaries in ${OUT}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Single-state mode.
# ---------------------------------------------------------------------------

echo "building mini-TIR harness"
${CXX} ${FLAGS} \
  -I"${FFI_ROOT}/include" -I"${FFI_ROOT}/3rdparty/dlpack/include" -I"${HERE}" \
  "${common_defs[@]}" \
  "-DTVM_FFI_BENCH_ENGINE_SHA=\"${harness_commit}\"" \
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
    "${common_defs[@]}" \
    "-DTVM_FFI_BENCH_ENGINE_SHA=\"${ffi_pin}\"" \
    "-DTVM_FFI_BENCH_TVM_SHA=\"${tvm_sha}\"" \
    "-DTVM_FFI_BENCH_TVM_FFI_PIN=\"${ffi_pin}\"" \
    "${HERE}/real_tvm_bench.cc" \
    -L"${TVM_BUILD}/lib" -ltvm_compiler -ltvm_ffi -Wl,-rpath,"${TVM_BUILD}/lib" \
    -o "${OUT}/real_tvm_bench"
fi
echo "binaries in ${OUT}"
