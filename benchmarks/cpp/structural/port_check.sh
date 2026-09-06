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
# Is tvm_hook_override.h still a faithful port of the TVM source it was taken from?
#
#   ./port_check.sh /path/to/tvm            # against the recorded sha
#   ./port_check.sh /path/to/tvm <sha>      # against another revision, e.g. a rebased PR head
#
# Extracts each ported function from the TVM revision and diffs it against the copy in the
# header.  Deviations the harness intends are marked `HARNESS DEVIATION` in the header and are
# expected to show up here; everything else is drift and should be re-ported.
#
# apache/tvm#20275 is under review and moving, so run this after any rebase of it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEADER="${HERE}/tvm_hook_override.h"
TVM="${1:?usage: port_check.sh /path/to/tvm [sha]}"
SHA="${2:-$(grep -o 'head [0-9a-f]\{7,40\}' "${HEADER}" | head -1 | cut -d' ' -f2)}"

# function name -> source file, in the order the header groups them.
FUNCS=(
  "IntImmVisit:src/ir/expr.cc"
  "IntImmMutate:src/ir/expr.cc"
  "IntImmMaybeInplaceMutate:src/ir/expr.cc"
  "VarVisit:src/ir/expr.cc"
  "VarMutate:src/ir/expr.cc"
  "VarMaybeInplaceMutate:src/ir/expr.cc"
  "BinaryVisit:src/ir/prim/expr.cc"
  "BinaryMutate:src/ir/prim/expr.cc"
  "BinaryMaybeInplaceMutate:src/ir/prim/expr.cc"
  "SeqStmtVisit:src/tirx/ir/stmt.cc"
  "MutateSeqStmtRaw:src/tirx/ir/stmt.cc"
  "MaybeInplaceMutateSeqStmtRaw:src/tirx/ir/stmt.cc"
  "EvaluateVisit:src/tirx/ir/stmt.cc"
  "EvaluateMutate:src/tirx/ir/stmt.cc"
  "EvaluateMaybeInplaceMutate:src/tirx/ir/stmt.cc"
)

# Print the body of `TVMFFIAny <name>(` ... up to the closing brace at column 0.
extract() { awk -v fn="$2" '
  $0 ~ ("^TVMFFIAny " fn "\\(") { inside = 1 }
  inside { print }
  inside && /^}$/ { exit }
' "$1"; }

echo "checking ${HEADER} against ${TVM} @ ${SHA}"
status=0
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
for entry in "${FUNCS[@]}"; do
  name="${entry%%:*}"
  file="${entry##*:}"
  git -C "${TVM}" show "${SHA}:${file}" > "${tmp}/src.cc"
  extract "${tmp}/src.cc" "${name}" > "${tmp}/tvm.txt"
  extract "${HEADER}" "${name}" > "${tmp}/harness.txt"
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
