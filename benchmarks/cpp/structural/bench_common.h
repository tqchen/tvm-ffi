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
#ifndef TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_BENCH_COMMON_H_
#define TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_BENCH_COMMON_H_

// The little that both harnesses genuinely share: the two floor arms, which are type-agnostic
// by construction, the build-time provenance stamped in by build.sh, and printing.
//
// The binary builds fixtures, runs arms, times them and prints timings.  It collects no
// metadata about itself: node counts, occurrence counts and working sets are arithmetic
// stated in the report next to the builder that produces them, and rebuild counts are read
// off the hooks, which are local code in `mini_tir.h` and `tvm_hook_override.h`.
//
// Node types, hooks, fixtures and arms are not here.  Each harness owns its own, plainly and
// without factoring against the other.  The timing loops are in `timer.h`.

#include <tvm/ffi/any.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstdint>
#include <cstdio>
#include <string>

#include "timer.h"

namespace tvm {
namespace ffi {
namespace bench {

/*! \brief Ownership of the root handed to a map arm. Walk arms have no ownership axis. */
enum class Ownership {
  kRetained,  // the caller still holds the root: refcount >= 2, copy-on-write
  kMoved,     // std::move(root): uniquely owned, the in-place path
};

inline const char* OwnershipName(Ownership o) {
  return o == Ownership::kRetained ? "retained" : "moved";
}

/*!
 * \brief What a fixture's replacement arms replace.
 *
 * `kAllVars` swaps every variable occurrence.  On a small tree that is a reasonable shape, but
 * on a long body it is not a workload anyone runs, and it hides implementation quality: every
 * node changes, so a traversal that rebuilds only the changed path and one that rebuilds
 * everything do the same amount of work.  `kSingleVar` changes exactly one occurrence -- a
 * sparse update, which is what a real substitution pass does -- and there the necessary and
 * the actual work diverge, with the gap growing as the body grows.
 */
enum class ReplaceKind { kAllVars, kSingleVar };

/*!
 * \brief What a fixture is, declared by the builder that constructs it.
 *
 * Every field here is arithmetic the author wrote down next to the loop that produces it, not
 * something the binary discovers by walking its own output.  The builder knows it made 4L+4
 * nodes because it wrote the loop that made them, and the rebuild counts follow from the
 * fixture's shape and the hooks in `mini_tir.h` / `tvm_hook_override.h`, which are local code.
 */
struct FixtureInfo {
  const char* name;
  /*! \brief Unique node identities; the divisor for every `ns/node` in this fixture's row. */
  int64_t unique_nodes;
  /*! \brief Node occurrences, which is one hook dispatch each for an unmatched traversal. */
  int64_t occurrences;
  /*! \brief Unique nodes times node size, in bytes. */
  int64_t working_set;
  /*! \brief Identities `map_replace` introduces, per ownership variant. */
  int64_t rebuilt_retained;
  int64_t rebuilt_moved;
  /*! \brief Traversals per timed sample, fixed per fixture so a sample is milliseconds. */
  int repeats;
  /*! \brief True when an in-place rebuild un-shares the graph, so the arm consumes it. */
  bool has_sharing;
  ReplaceKind replace_kind;
  /*! \brief `map_old` runs on one fixture only: it is the same engine as `map_replace`. */
  bool run_old = false;
  /*! \brief The one-off remap probe, on the fixture where the remap has work to do. */
  bool run_noremap = false;
};

// ---------------------------------------------------------------------------
// Build-time provenance, stamped in by build.sh. Compile-time constants, so a report cannot
// omit what the tool prints and nothing has to be probed at runtime.
// ---------------------------------------------------------------------------

#ifndef TVM_FFI_BENCH_HARNESS_COMMIT
#define TVM_FFI_BENCH_HARNESS_COMMIT "unknown"
#endif
#ifndef TVM_FFI_BENCH_ENGINE_SHA
#define TVM_FFI_BENCH_ENGINE_SHA "unknown"
#endif
#ifndef TVM_FFI_BENCH_TVM_SHA
#define TVM_FFI_BENCH_TVM_SHA "unknown"
#endif
#ifndef TVM_FFI_BENCH_TVM_FFI_PIN
#define TVM_FFI_BENCH_TVM_FFI_PIN "n/a"
#endif
#ifndef TVM_FFI_BENCH_CXX_FLAGS
#define TVM_FFI_BENCH_CXX_FLAGS "unknown"
#endif

inline void Emit(const std::string& line) { std::printf("%s\n", line.c_str()); }

inline void EmitProvenance(const std::string& key, const std::string& value) {
  Emit("#provenance\t" + key + "\t" + value);
}

/*! \brief One fixture line: the builder's declared facts, printed as handed over. */
inline void EmitFixture(const std::string& harness, const FixtureInfo& f) {
  Emit("#fixture\t" + harness + "\t" + f.name + "\t" + std::to_string(f.unique_nodes) + "\t" +
       std::to_string(f.occurrences) + "\t" + std::to_string(f.working_set) + "\t" +
       std::to_string(f.rebuilt_retained) + "\t" + std::to_string(f.rebuilt_moved) + "\t" +
       (f.replace_kind == ReplaceKind::kSingleVar ? "single_var" : "all_vars"));
}

/*! \brief One timing line. */
inline void EmitResult(const std::string& harness, const std::string& fixture,
                       const std::string& ownership, const std::string& arm,
                       double ns_per_traversal) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.4f", ns_per_traversal);
  Emit("#result\t" + harness + "\t" + fixture + "\t" + ownership + "\t" + arm + "\t" + buf);
}

inline void EmitStandardProvenance(const std::string& harness) {
  EmitProvenance("harness", harness);
  EmitProvenance("harness_branch_commit", TVM_FFI_BENCH_HARNESS_COMMIT);
  EmitProvenance("tvm_ffi_engine_sha", TVM_FFI_BENCH_ENGINE_SHA);
  EmitProvenance("tvm_sha", TVM_FFI_BENCH_TVM_SHA);
  EmitProvenance("tvm_ffi_submodule_pin", TVM_FFI_BENCH_TVM_FFI_PIN);
  EmitProvenance("flags", TVM_FFI_BENCH_CXX_FLAGS);
  EmitProvenance("compiler",
#if defined(__clang__)
                 std::string("clang ") + __clang_version__
#elif defined(__GNUC__)
                 std::string("gcc ") + __VERSION__
#else
                 std::string("unknown")
#endif
  );
  EmitProvenance("method", kMethodDescription);
}

// ---------------------------------------------------------------------------
// Floor arms: minimal hand-written vtables. Type-agnostic, so they live here.
// ---------------------------------------------------------------------------

/*! \brief Lower bound for a hooked walk: attr-column lookup and hook dispatch, no callbacks. */
class MinimalVisitorObj final : public StructuralVisitorObj {
 public:
  MinimalVisitorObj() : StructuralVisitorObj(VTable()) {}

 private:
  static TVMFFIAny Visit(StructuralVisitorObj* self, AnyView value) noexcept {
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralVisit);
    AnyView attr = column[value.type_index()];
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      return (*reinterpret_cast<FStructuralVisit>(attr.cast<void*>()))(self, value);
    }
    return AnyView(nullptr).CopyToTVMFFIAny();
  }
  static const StructuralVisitorVTable* VTable() {
    static const StructuralVisitorVTable vtable{&MinimalVisitorObj::Visit};
    return &vtable;
  }
};

/*! \brief Lower bound for a hooked map: attr-column lookup and hook dispatch, no callbacks. */
class MinimalMutatorObj final : public StructuralMutatorObj {
 public:
  MinimalMutatorObj() : StructuralMutatorObj(VTable()) {}

 private:
  static TVMFFIAny Mutate(StructuralMutatorObj* self, AnyView value) noexcept {
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMutate);
    AnyView attr = column[value.type_index()];
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      return (*reinterpret_cast<FStructuralMutate>(attr.cast<void*>()))(self, value);
    }
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }
  static TVMFFIAny InplaceMutate(StructuralMutatorObj* self, AnyView value) noexcept {
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMaybeInplaceMutate);
    AnyView attr = column[value.type_index()];
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      return (*reinterpret_cast<FStructuralMutate>(attr.cast<void*>()))(self, value);
    }
    return Mutate(self, value);
  }
  static TVMFFIAny NoRemapGet(StructuralMutatorObj*, AnyView) noexcept {
    return AnyView(nullptr).CopyToTVMFFIAny();
  }
  static TVMFFIAny NoRemapSet(StructuralMutatorObj*, AnyView, AnyView) noexcept {
    return AnyView(nullptr).CopyToTVMFFIAny();
  }
  static const StructuralMutatorVTable* VTable() {
    static const StructuralMutatorVTable vtable{&MinimalMutatorObj::Mutate,
                                                &MinimalMutatorObj::InplaceMutate,
                                                &MinimalMutatorObj::NoRemapGet,
                                                &MinimalMutatorObj::NoRemapSet};
    return &vtable;
  }
};

}  // namespace bench
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_BENCH_COMMON_H_
