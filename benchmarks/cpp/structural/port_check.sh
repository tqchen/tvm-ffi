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
# Is a hook file still a faithful port of the TVM source it was taken from?
#
#   ./port_check.sh /path/to/tvm                          # tvm_hook_override.h, recorded sha
#   ./port_check.sh /path/to/tvm <sha>                    # against another revision
#   ./port_check.sh --header tvm_hook_override_pre753.h /path/to/tvm
#
# Extracts each ported function from the TVM revision and diffs it against the copy in the
# header.  Deviations the harness intends are marked `HARNESS DEVIATION` in the header and are
# expected to show up here; everything else is drift and should be re-ported.
#
# PER-STATE HOOK FILES.  A two-state run compiles each state's hooks against that state's own
# tvm-ffi API, so there is one hook file per state and only one of them can be byte-identical
# to TVM's source.  A state file declares what it had to change in an `API ADAPTATION` block,
# and this script checks it twice over:
#
#   1. the declared substitution is undone before diffing, so DRIFT still means real drift in
#      the body rather than the adaptation showing up on every visit hook; and
#   2. the replacement text is confirmed to be what apache/tvm itself used at the revision the
#      header names in `VERIFIED AGAINST`, so the adaptation is a port of the PR's own earlier
#      spelling rather than something invented in the harness.
#
# A state file whose adaptation cannot be checked that way -- because the state's API has no
# corresponding apache/tvm revision -- must say so in the block instead of naming a sha, and
# this script reports it as UNCHECKED rather than passing it silently.
#
# 20275 is under review and moving, so run this after any rebase of it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEADER="${HERE}/tvm_hook_override.h"
if [[ "${1:-}" == "--header" ]]; then
  HEADER="$2"
  [[ -f "${HEADER}" ]] || HEADER="${HERE}/$2"
  shift 2
fi
TVM="${1:?usage: port_check.sh [--header FILE] /path/to/tvm [sha]}"
SHA="${2:-$(grep -o 'head [0-9a-f]\{7,40\}' "${HEADER}" | head -1 | cut -d' ' -f2)}"

# function name -> "return type:source file", in the order the header groups them.
FUNCS=(
  "IntImmVisit:TVMFFIAny:src/ir/expr.cc"
  "IntImmMutate:TVMFFIAny:src/ir/expr.cc"
  "IntImmMaybeInplaceMutate:TVMFFIAny:src/ir/expr.cc"
  "VarVisit:TVMFFIAny:src/ir/expr.cc"
  "VarMutate:TVMFFIAny:src/ir/expr.cc"
  "VarMaybeInplaceMutate:TVMFFIAny:src/ir/expr.cc"
  "CallVisit:TVMFFIAny:src/ir/expr.cc"
  "CallMutate:TVMFFIAny:src/ir/expr.cc"
  "CallMaybeInplaceMutate:TVMFFIAny:src/ir/expr.cc"
  "BinaryVisit:TVMFFIAny:src/ir/prim/expr.cc"
  "BinaryMutate:TVMFFIAny:src/ir/prim/expr.cc"
  "BinaryMaybeInplaceMutate:TVMFFIAny:src/ir/prim/expr.cc"
  "SeqStmtVisit:TVMFFIAny:src/tirx/ir/stmt.cc"
  "IsSeqStmtNoOp:bool:src/tirx/ir/stmt.cc"
  "AppendSeqStmtResult:void:src/tirx/ir/stmt.cc"
  "MutateSeqStmtChanged:TVMFFIAny:src/tirx/ir/stmt.cc"
  "MutateSeqStmtRaw:TVMFFIAny:src/tirx/ir/stmt.cc"
  "InplaceSplice:TVMFFIAny:src/tirx/ir/stmt.cc"
  "MaybeInplaceMutateSeqStmtChanged:TVMFFIAny:src/tirx/ir/stmt.cc"
  "MaybeInplaceMutateSeqStmtRaw:TVMFFIAny:src/tirx/ir/stmt.cc"
  "EvaluateVisit:TVMFFIAny:src/tirx/ir/stmt.cc"
  "EvaluateMutate:TVMFFIAny:src/tirx/ir/stmt.cc"
  "EvaluateMaybeInplaceMutate:TVMFFIAny:src/tirx/ir/stmt.cc"
)

# Print the body of `<ret> <name>(` ... up to the closing brace at column 0.
extract() { awk -v sig="^$3 $2\\\\(" '
  $0 ~ sig { inside = 1 }
  inside { print }
  inside && /^}$/ { exit }
' "$1"; }

# The declared adaptation, if any: FROM is the line the TVM source uses, TO is the line this
# header uses instead. Undoing TO -> FROM is what makes the diff below meaningful.
ADAPT_FROM="$(sed -n 's|^//   \(TVM_FFI_S_VISIT_RETURN_NONE();\)$|\1|p' "${HEADER}" | head -1)"
ADAPT_TO="$(sed -n 's|^//   \(return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(nullptr));\)$|\1|p' "${HEADER}" | head -1)"
ADAPT_SHA="$(sed -n 's|^// VERIFIED AGAINST apache/tvm#20275 \([0-9a-f]\{7,40\}\).*|\1|p' "${HEADER}" | head -1)"

echo "checking ${HEADER} against ${TVM} @ ${SHA}"
status=0
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

if [[ -n "${ADAPT_TO}" ]]; then
  if [[ -z "${ADAPT_SHA}" ]]; then
    echo "UNCHECKED        API ADAPTATION declares no VERIFIED AGAINST revision"
    status=1
  elif ! git -C "${TVM}" show "${ADAPT_SHA}:src/ir/expr.cc" 2>/dev/null | grep -qF "${ADAPT_TO}"; then
    echo "BAD ADAPTATION   ${ADAPT_TO}"
    echo "                 not found in apache/tvm ${ADAPT_SHA}, which the header cites for it"
    status=1
  else
    echo "ok               API ADAPTATION matches apache/tvm ${ADAPT_SHA}"
  fi
fi

for entry in "${FUNCS[@]}"; do
  name="${entry%%:*}"; rest="${entry#*:}"
  ret="${rest%%:*}"; file="${rest##*:}"
  git -C "${TVM}" show "${SHA}:${file}" > "${tmp}/src.cc"
  extract "${tmp}/src.cc" "${name}" "${ret}" > "${tmp}/tvm.txt"
  extract "${HEADER}" "${name}" "${ret}" > "${tmp}/harness.txt"
  # Undo the state file's declared adaptation so only real drift shows.
  if [[ -n "${ADAPT_TO}" ]]; then
    python3 - "${tmp}/harness.txt" "${ADAPT_TO}" "${ADAPT_FROM}" <<'PY'
import sys
p, to, frm = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(p).read()
open(p, 'w').write(text.replace('  ' + to, '  ' + frm))
PY
  fi
  if [[ ! -s "${tmp}/tvm.txt" ]]; then
    echo "MISSING IN TVM   ${name} (${file}) -- the PR may have renamed or removed it"
    status=1
  elif ! diff -q "${tmp}/tvm.txt" "${tmp}/harness.txt" > /dev/null; then
    echo "DRIFT            ${name} (${file})"
    diff -u "${tmp}/tvm.txt" "${tmp}/harness.txt" | sed 's/^/    /'
    status=1
  else
    echo "ok               ${name}"
  fi
done
exit "${status}"
