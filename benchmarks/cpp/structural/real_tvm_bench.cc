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

// The real-TVM half of the structural-traversal benchmark.  Links apache/tvm and measures
// the hooks TVM registers today, against the shipping PostOrderVisit and Substitute.
//
// Everything except the node types and hook installation lives in bench_common.h; the same
// fixtures, arms, method, working-set accounting and assertions are shared with the
// mini-TIR harness so the two sets of numbers are structural counterparts.
//
// Hook override: TVM installs its structural hooks from static-init blocks, so this file
// can replace them from main(), which runs afterwards.  That requires the write-once
// type-attribute guard to be disabled, which the branch-local
// TVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE build of tvm-ffi does.  The override is used here
// only by the untimed dispatch-count assertion pass, which restores TVM's own hooks before
// anything is timed; a future measurement adding a prototype arm uses the same mechanism.

#include <tvm/ir/op.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <vector>

#include "bench_common.h"

namespace real_tvm {

using namespace tvm;              // NOLINT(build/namespaces)
using namespace tvm::tirx;        // NOLINT(build/namespaces)
using namespace tvm::ffi::bench;  // NOLINT(build/namespaces)
namespace ffi = tvm::ffi;

constexpr const char* kHarness = "real-tvm";

// ---------------------------------------------------------------------------
// Node sizes, so the working set is measured rather than assumed.
// ---------------------------------------------------------------------------

std::unordered_map<int32_t, int64_t>* NodeSizeTable() {
  static std::unordered_map<int32_t, int64_t> table;
  return &table;
}

template <typename T>
void RegisterNodeSize() {
  (*NodeSizeTable())[T::RuntimeTypeIndex()] = static_cast<int64_t>(sizeof(T));
}

int64_t NodeSize(int32_t type_index, const ffi::Object* obj) {
  if (type_index == ffi::TypeIndex::kTVMFFIArray) {
    const auto* array = static_cast<const ffi::ArrayObj*>(obj);
    return static_cast<int64_t>(sizeof(ffi::ArrayObj) + array->size() * sizeof(ffi::Any));
  }
  auto it = NodeSizeTable()->find(type_index);
  return it == NodeSizeTable()->end() ? 0 : it->second;
}

void RegisterNodeSizes() {
  RegisterNodeSize<VarNode>();
  RegisterNodeSize<IntImmNode>();
  RegisterNodeSize<FloatImmNode>();
  RegisterNodeSize<prim::AddNode>();
  RegisterNodeSize<prim::MulNode>();
  RegisterNodeSize<prim::FloorDivNode>();
  RegisterNodeSize<prim::FloorModNode>();
  RegisterNodeSize<CallNode>();
  RegisterNodeSize<PrimTypeNode>();
  RegisterNodeSize<EvaluateNode>();
  RegisterNodeSize<SeqStmtNode>();
  RegisterNodeSize<OpNode>();
}

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

// Function-local statics: constructed on first use, which is after every static-init block.
PrimVar& Outer() {
  static PrimVar v("outer");
  return v;
}
PrimVar& Inner() {
  static PrimVar v("inner");
  return v;
}
PrimVar& Replacement() {
  static PrimVar v("replacement");
  return v;
}

PrimExpr SplitFuse(bool shared) {
  PrimExpr q = Outer() * 16 + Inner();
  PrimExpr r = shared ? q : PrimExpr(Outer() * 16 + Inner());
  return floordiv(q, 32) * 32 + floormod(r, 32);
}

// Composition of the `seq-L` fixture, asserted before anything is timed.  Each element is
// Mul, IntImm, Add, Evaluate; the shared preamble is the SeqStmt and the two Vars.  The
// `Array` holding the sequence is not a visited value -- SeqStmt's hook iterates its
// elements -- so it is neither a unique node nor a rebuilt one.
constexpr int64_t kSeqSharedNodes = 3;
// A replacement rebuilds the Mul/Add/Evaluate spine of every element, the SeqStmt above
// them, and introduces the substituted-in Var, which is absent from the input graph.
constexpr int64_t kSeqRebuiltTail = 2;

/*! \brief Unique nodes in the `seq-L` fixture: 4 per element plus the shared preamble. */
constexpr int64_t kSeqUniqueNodes(int length) { return 4LL * length + kSeqSharedNodes; }
/*! \brief Nodes a single-variable replacement rebuilds in `seq-L` under retained ownership. */
constexpr int64_t kSeqRebuiltRetained(int length) { return 3LL * length + kSeqRebuiltTail; }
/*! \brief Nodes of `seq-L` itself that a single-variable replacement changes. */
constexpr int64_t kSeqChanged(int length) { return 3LL * length + 1; }
/*!
 * \brief Nodes a `seq-L` replacement rebuilds under *moved* ownership.
 *
 * Not zero, and that is a finding rather than a fixture quirk.  TVM's in-place SeqStmt hook
 * takes a handle copy of the sequence (`Array<Stmt> mapped_seq = self->seq`) and hands each
 * element to `MaybeInplaceMutateIfUniqueExpected` through `mapped_seq[i]`, so no element is
 * ever uniquely owned and every one of them takes the copy-on-write path.  The SeqStmt node
 * itself is mutated in place; its whole element spine is not.  So a moved traversal of this
 * fixture still rebuilds 3 nodes per element, plus the substituted-in Var.
 */
constexpr int64_t kSeqRebuiltMoved(int length) { return 3LL * length + 1; }

Stmt LongSeq(int length) {
  ffi::Array<Stmt> body;
  body.reserve(length);
  for (int i = 0; i < length; ++i) {
    // `i + 2`: a multiplier of one would constant-fold the Mul away.
    body.push_back(Evaluate(Outer() * (i + 2) + Inner()));
  }
  return SeqStmt(body);
}

// ---------------------------------------------------------------------------
// Arms.  The vocabulary is fixed in bench_common.h; this binds it to TVM types.
// ---------------------------------------------------------------------------

size_t g_sink = 0;

/*! \brief `map_replace` swaps the two variables, so repeated in-place application is
 *         stationary: the fixture never drifts into a different workload. */
ffi::Any SwapVars(const Var& var) {
  if (var.same_as(Outer())) return ffi::Any(Replacement());
  if (var.same_as(Replacement())) return ffi::Any(Outer());
  return ffi::Any(var);
}

ffi::Optional<Expr> SwapVarsOld(const Var& var) {
  if (var.same_as(Outer())) return ffi::Optional<Expr>(Replacement());
  if (var.same_as(Replacement())) return ffi::Optional<Expr>(Outer());
  return std::nullopt;
}

void RunWalkArm(ArmId id, ffi::AnyView root) {
  switch (id) {
    case ArmId::kWalkFloor: {
      MinimalVisitorObj visitor;
      g_sink += visitor.VisitExpected(root).is_err();
      break;
    }
    case ArmId::kWalk: {
      size_t matched = 0;
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          root,
          [&](const Var&) -> ffi::Expected<ffi::WalkResult> {
            ++matched;
            return ffi::WalkResult::Advance();
          },
          [&](const Expr&) -> ffi::Expected<ffi::WalkResult> {
            ++matched;
            return ffi::WalkResult::Advance();
          });
      g_sink += matched;
      break;
    }
    case ArmId::kWalkNever: {
      size_t matched = 0;
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          root,
          [&](const FloatImm&) -> ffi::Expected<ffi::WalkResult> {
            ++matched;
            return ffi::WalkResult::Advance();
          },
          [&](const Expr&) -> ffi::Expected<ffi::WalkResult> {
            ++matched;
            return ffi::WalkResult::Advance();
          });
      g_sink += matched;
      break;
    }
    case ArmId::kWalkOld: {
      size_t matched = 0;
      PostOrderVisit(root.cast<ffi::ObjectRef>(),
                     [&](const ffi::ObjectRef& node) { matched += node.as<VarNode>() != nullptr; });
      g_sink += matched;
      break;
    }
    default:
      std::abort();
  }
}

/*!
 * \brief Run one map arm on \p slot.
 *
 * `retained` keeps the caller's handle alive, so the root's refcount is at least two and
 * the engine takes the copy-on-write path.  `moved` hands the sole reference over, so the
 * engine takes the in-place path; the result is stored back into the slot, which keeps the
 * next repetition uniquely owned and the working set fixed at one fixture.
 */
void RunMapArm(ArmId id, Ownership ownership, ffi::Any* slot) {
  auto take = [&]() -> ffi::Any {
    return ownership == Ownership::kMoved ? std::move(*slot) : ffi::Any(*slot);
  };
  auto give = [&](ffi::Any result) {
    if (ownership == Ownership::kMoved) {
      *slot = std::move(result);
    } else {
      g_sink += result.type_index();
    }
  };
  switch (id) {
    case ArmId::kMapFloor: {
      MinimalMutatorObj mutator;
      ffi::Any input = take();
      ffi::Expected<ffi::Any> result = ownership == Ownership::kMoved
                                           ? mutator.MaybeInplaceMutateIfUniqueExpected(input)
                                           : mutator.MutateExpected(input);
      g_sink += result.is_err();
      give(result.value());
      break;
    }
    case ArmId::kMapNever:
      give(ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
          take(), [](const FloatImm& value) { return ffi::Any(value); }));
      break;
    case ArmId::kMapIdentity:
      give(ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
          take(), [](const Var& var) { return ffi::Any(var); }));
      break;
    case ArmId::kMapReplace:
      give(ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(take(), SwapVars));
      break;
    case ArmId::kMapOld: {
      ffi::Any input = take();
      if (auto stmt = input.as<Stmt>()) {
        give(ffi::Any(Substitute(*stmt, SwapVarsOld)));
      } else {
        give(ffi::Any(Substitute(input.cast<Expr>(), SwapVarsOld)));
      }
      break;
    }
    default:
      std::abort();
  }
}

// ---------------------------------------------------------------------------
// Per-fixture driver: declare, assert, then measure.
// ---------------------------------------------------------------------------


/*! \brief The harness policy the shared driver in bench_common.h is instantiated on. */
struct RealTvmPolicy {
  static constexpr const char* kName = kHarness;
  // TVM's PostOrderVisit and Substitute are themselves built on the structural engine.
  static constexpr bool kOldGoesThroughEngine = true;
  static int64_t NodeSize(int32_t type_index, const ffi::Object* obj) {
    return real_tvm::NodeSize(type_index, obj);
  }
  static void RunWalkArm(ArmId id, ffi::AnyView root) { real_tvm::RunWalkArm(id, root); }
  static void RunMapArm(ArmId id, Ownership ownership, ffi::Any* slot) {
    real_tvm::RunMapArm(id, ownership, slot);
  }
  static ffi::Any MapReplace(ffi::Any root) {
    return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(root), SwapVars);
  }
  static ffi::Any MapIdentity(ffi::Any root) {
    return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(root),
                                                          [](const Var& var) { return ffi::Any(var); });
  }
  static ffi::Any MapNever(ffi::Any root) {
    return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
        std::move(root), [](const FloatImm& value) { return ffi::Any(value); });
  }
  static ffi::Any MapOld(ffi::Any root) {
    if (auto stmt = root.as<Stmt>()) return ffi::Any(Substitute(*stmt, SwapVarsOld));
    return ffi::Any(Substitute(root.cast<Expr>(), SwapVarsOld));
  }
};

}  // namespace real_tvm

int main() {
  using namespace real_tvm;
  EmitStandardProvenance(kHarness);
  RegisterNodeSizes();

  std::vector<Fixture> fixtures;
  {
    Fixture f;
    f.name = "split-fuse-shared";
    f.build = [] { return ffi::Any(SplitFuse(true)); };
    f.has_sharing = true;
    fixtures.push_back(f);
  }
  {
    Fixture f;
    f.name = "split-fuse-distinct";
    f.build = [] { return ffi::Any(SplitFuse(false)); };
    fixtures.push_back(f);
  }
  for (int length : SeqSweepLengths()) {
    Fixture f;
    f.name = "seq-" + std::to_string(length);
    f.build = [length] { return ffi::Any(LongSeq(length)); };
    fixtures.push_back(f);
  }

  // Declared expectations. Measured against the fixture before anything is timed.
  auto declare = [&](const std::string& name, int64_t unique, int64_t rebuilt_retained,
                     int64_t rebuilt_moved, int64_t changed, int64_t remap_hits) {
    for (Fixture& fixture : fixtures) {
      if (fixture.name != name) continue;
      fixture.expect_unique_nodes = unique;
      fixture.expect_rebuilt_retained = rebuilt_retained;
      fixture.expect_rebuilt_moved = rebuilt_moved;
      fixture.expect_changed = changed;
      fixture.expect_remap_hits = remap_hits;
    }
  };
  // Declared from the fixture's own construction; the harness fails the run if a build
  // change or an engine change moves them.
  // Under moved ownership the only rebuilt identity is the substituted-in Var, except on
  // the shared fixture, where the first parent to reach the shared subtree cannot mutate it
  // in place and copies it and its own changed child.
  declare("split-fuse-shared", 12, 9, 3, 6, 2);
  declare("split-fuse-distinct", 15, 9, 1, 8, 2);
  // SeqStmt sweep: nodes = 4 per element (Mul, IntImm, Add, Evaluate) plus the SeqStmt, its
  // Array, the two Vars and the shared PrimType; a replacement rebuilds the Mul/Add/Evaluate
  // spine of every element plus the Array and the SeqStmt.
  for (int length : SeqSweepLengths()) {
    declare("seq-" + std::to_string(length), kSeqUniqueNodes(length), kSeqRebuiltRetained(length),
            kSeqRebuiltMoved(length), kSeqChanged(length), 2LL * (length - 1));
  }

  for (const Fixture& fixture : fixtures) RunFixture<RealTvmPolicy>(fixture);
  Emit("#sink\t" + std::to_string(g_sink));
  return 0;
}
