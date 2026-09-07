/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_STATE_SHIMS_PRE753_VISIT_RETURN_NONE_H_
#define TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_STATE_SHIMS_PRE753_VISIT_RETURN_NONE_H_

// BUILD SHIM for a two-state run whose state predates tvm-ffi #753.
//
// Force-included (`-include`) into apache/tvm's own translation units, never into the
// benchmark's.  It is not an engine change and it is not a hook: no tvm-ffi source is edited,
// the state checkout stays byte-identical to its ref, and this header is not on the include
// path of `real_tvm_bench.cc`.
//
// WHY IT IS NEEDED.  apache/tvm#20275 is written against post-#753 tvm-ffi and uses two
// things a pre-#753 engine does not have:
//
//   * `TVM_FFI_S_VISIT_RETURN_NONE()`, the tail of about twenty of its own `__s_visit__`
//     hooks across eight source files; and
//   * `details::SMutateDeclaredTypeError()`, called directly from four `__s_mutate__` hooks
//     in `src/tirx/ir/stmt.cc` and `src/tirx/ir/layout/tile_core.cc`.
//
// Only the hooks in `tvm_hook_override.h` are ported, and only those get a per-state rewrite.
// The rest are TVM's code, none of them is on any arm's dispatch path, and without these two
// TVM does not compile at all against a pre-#753 engine -- so there would be no state-A
// binary to measure.
//
// WHY IT IS MEASUREMENT-NEUTRAL.  The macro is sugar over a value the pre-#753 engine already
// produces and already spells this way itself.  Compare `StructuralVisitorObj::VisitImpl` in
// the pre-#753 `structural_visit.h`, which returns "no interrupt" as exactly the expression
// below.  #753's own commit message names this as the tail "every hook currently spells out by
// hand".  The expansion below is that hand-spelling; #753 replaced it with a named macro and
// changed no value.
//
// WHAT IT IS NOT.  It does not give state A the post-#753 `MaybeReturnHelper` proxy -- that
// proxy does not exist in state A and nothing here needs it.  It is deliberately *not* used by
// the harness's own state-A hook file: those fifteen hooks are re-spelled per state from
// apache/tvm#20275's own pre-#753 revision, which is a checkable port rather than a shim.  See
// `port_check.sh --header`.

// Force-inclusion reaches every translation unit CMake compiles, including the trivial one it
// uses to test the compiler, which has no include path. So the shim is a no-op wherever the
// engine headers are not reachable.
#if defined(__has_include) && __has_include(<tvm/ffi/extra/structural_visit.h>)

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>

#ifdef TVM_FFI_S_VISIT_RETURN_NONE
#error "state_shims/pre753_visit_return_none.h force-included into a post-#753 engine build"
#endif

#define TVM_FFI_S_VISIT_RETURN_NONE()                          \
  return ::tvm::ffi::details::ExpectedUnsafe::MoveToTVMFFIAny( \
      ::tvm::ffi::Expected<::tvm::ffi::Optional<::tvm::ffi::VisitInterrupt>>(::std::nullopt))

// The pre-#753 engine spells this `SMutateDeclaredTypeErrorRaw()` and has it return the raw
// `TVMFFIAny` rather than the `Expected<Any>` its callers now convert.  Same error, same
// message, different carrier -- and it is a cold path that no arm in this benchmark takes, so
// it cannot reach a timing.  The body below is byte-identical to the post-#753 engine's.
namespace tvm {
namespace ffi {
namespace details {
TVM_FFI_COLD_CODE inline Expected<Any> SMutateDeclaredTypeError() noexcept {
  return Unexpected(
      Error("TypeError", "structural mutate result does not match the declared type", ""));
}
}  // namespace details
}  // namespace ffi
}  // namespace tvm

#endif  // __has_include(<tvm/ffi/extra/structural_visit.h>)

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_STATE_SHIMS_PRE753_VISIT_RETURN_NONE_H_
