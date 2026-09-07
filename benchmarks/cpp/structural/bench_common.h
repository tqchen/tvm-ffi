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
#include <cstdlib>
#include <string>
#include <vector>

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
 * \brief Which mutating arm a fixture's checks and sweeps select.
 *
 * Named after the arm it selects, not after a property of the workload: an earlier
 * `kSingleVar` selected the Stmt-level element swap, which reads as its opposite.
 *
 * `kSubst` selects `map_subst`, the Expr-level `Var` substitution.  It goes through the remap,
 * so every occurrence of the substituted variable changes: on a small tree that is a
 * reasonable shape, but on a long body every node changes and a traversal that rebuilds only
 * the changed path and one that rebuilds everything do the same amount of work.
 *
 * `kSwap` selects `map_swap`, the Stmt-level `Evaluate` swap.  `Evaluate` is not a free
 * variable, so no remap is involved and exactly the intended elements change -- a sparse
 * update, which is what a real substitution pass does, and where necessary and actual work
 * diverge as the body grows.
 */
enum class ArmKind { kSubst, kSwap };

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
  /*! \brief Identities the fixture's selected mutating arm introduces, per ownership. */
  int64_t rebuilt_retained;
  int64_t rebuilt_moved;
  /*! \brief Traversals per timed sample, fixed per fixture so a sample is milliseconds. */
  int repeats;
  /*! \brief True when an in-place rebuild un-shares the graph, so the arm consumes it. */
  bool has_sharing;
  ArmKind arm_kind;
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

/*!
 * \brief Where a rebuilding arm parks its output so teardown falls outside the clock.
 *
 * Releasing a rebuilt subgraph walks and frees every node in it. Left in the timed region that
 * cost is charged to the arm that built it, which systematically inflates whichever arms
 * allocate most -- the exact axis these tables compare. Moving an `Any` in is a pointer copy
 * and a null-out, negligible against a graph teardown.
 */
inline std::vector<Any>& ResultSink() {
  static std::vector<Any> sink;
  return sink;
}
/*! \brief Reserve before timing so no growth happens inside the clock. */
inline void ReserveResultSink(size_t n) {
  ResultSink().clear();
  ResultSink().reserve(n);
}
/*!
 * \brief Free the batch's outputs, outside the clock, and check that there were any.
 *
 * The whole point of the sink is that a map arm's output is destroyed here rather than inside
 * the timed region. An arm that let its result fall out of scope locally would time its own
 * teardown and read as slower for a reason no reader could see -- silently, because the
 * timings would still look plausible. This makes that structural: every map batch parks at
 * least one result, so an empty sink at drain time means an arm dropped its output.
 */
inline void DrainResultSink(const char* what) {
  if (ResultSink().empty()) {
    std::fflush(stdout);
    std::fprintf(stderr,
                 "structural benchmark check failed: %s parked no result, so its teardown ran "
                 "inside the timed region\n",
                 what);
    std::exit(2);
  }
  ResultSink().clear();
}

inline void Emit(const std::string& line) { std::printf("%s\n", line.c_str()); }

inline void EmitProvenance(const std::string& key, const std::string& value) {
  Emit("#provenance\t" + key + "\t" + value);
}

/*! \brief One fixture line: the builder's declared facts, printed as handed over. */
inline void EmitFixture(const std::string& harness, const FixtureInfo& f) {
  Emit("#fixture\t" + harness + "\t" + f.name + "\t" + std::to_string(f.unique_nodes) + "\t" +
       std::to_string(f.occurrences) + "\t" + std::to_string(f.working_set) + "\t" +
       std::to_string(f.rebuilt_retained) + "\t" + std::to_string(f.rebuilt_moved) + "\t" +
       (f.arm_kind == ArmKind::kSwap ? "swap" : "subst"));
}

/*!
 * \brief One node-size line: `#nodesize <harness> <logical name> <bytes>`.
 *
 * The two harnesses emit the same logical names for counterpart node types, and `report.py`
 * fails a run in which a pair disagrees.  Layout parity between mini-TIR and real TVM is the
 * fidelity requirement -- the only permitted difference is which node types exist -- and this
 * is what makes it a checked fact rather than a claim in a comment.
 */
inline void EmitNodeSize(const std::string& harness, const std::string& node, size_t bytes) {
  Emit("#nodesize\t" + harness + "\t" + node + "\t" + std::to_string(bytes));
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
