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

// Real-TVM half of the structural benchmark: fixtures, arms, main.
//
// The structural hooks are in `tvm_hook_override.h` and are installed from main() over the
// ones TVM registered from its static-init blocks -- which is what the branch-local
// TVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE build of tvm-ffi exists for.  Every number here is
// taken on those hooks, including `walk_old` and `map_old`, because `PostOrderVisit` and
// `Substitute` are built on the structural engine: the old-versus-new comparison then differs
// only in the traversal API and not in the hooks underneath it.  No number here describes
// TVM's own registered hook implementations, and the report says so.
//
// `walk_functor` and `map_functor` are the exception by construction: TVM's ExprFunctor and
// StmtFunctor machinery does not go through structural hooks at all, which is the point of
// having them as arms.

#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <unordered_set>
#include <vector>

#include "tvm_hook_override.h"

namespace real_tvm {

using namespace tvm;              // NOLINT(build/namespaces)
using namespace tvm::tirx;        // NOLINT(build/namespaces)
using namespace tvm::ffi::bench;  // NOLINT(build/namespaces)
namespace ffi = tvm::ffi;

constexpr const char* kHarness = "real-tvm";

// ---------------------------------------------------------------------------
// Fixtures.  The same shapes the mini harness builds, so the rows are counterparts.
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
/*! \brief The one variable a sparse `seq-L` update replaces; element `L/2` uses it. */
PrimVar& Target() {
  static PrimVar v("target");
  return v;
}

PrimExpr SplitFuse(bool shared) {
  PrimExpr q = Outer() * 16 + Inner();
  PrimExpr r = shared ? q : PrimExpr(Outer() * 16 + Inner());
  return floordiv(q, 32) * 32 + floormod(r, 32);
}

Stmt LongSeq(int length) {
  ffi::Array<Stmt> body;
  body.reserve(length);
  for (int i = 0; i < length; ++i) {
    // `i + 2`: a multiplier of one would constant-fold the Mul away.  The middle element uses
    // Target so a sparse update has exactly one occurrence to change.
    PrimVar lhs = (i == length / 2) ? Target() : Outer();
    body.push_back(Evaluate(lhs * (i + 2) + Inner()));
  }
  return SeqStmt(body);
}

constexpr int64_t kVarBytes = sizeof(VarNode);
constexpr int64_t kImmBytes = sizeof(IntImmNode);
constexpr int64_t kAddBytes = sizeof(prim::AddNode);
constexpr int64_t kMulBytes = sizeof(prim::MulNode);
constexpr int64_t kEvalBytes = sizeof(EvaluateNode);
constexpr int64_t kSeqBytes(int64_t l) {
  return sizeof(SeqStmtNode) + sizeof(ffi::ArrayObj) + l * static_cast<int64_t>(sizeof(ffi::Any));
}
constexpr int64_t kSplitFuseSharedBytes =
    2 * kVarBytes + 4 * kImmBytes + 2 * kMulBytes + 2 * kAddBytes +
    static_cast<int64_t>(sizeof(prim::FloorDivNode)) +
    static_cast<int64_t>(sizeof(prim::FloorModNode));
constexpr int64_t kSplitFuseDistinctBytes =
    kSplitFuseSharedBytes + kMulBytes + kImmBytes + kAddBytes;
constexpr int64_t kSeqWorkingSet(int64_t l) {
  return l * (kMulBytes + kImmBytes + kAddBytes + kEvalBytes) + 3 * kVarBytes + kSeqBytes(l);
}

constexpr int kSeqLengths[] = {16, 256, 16384};
constexpr int kSplitFuseRepeats = 20000;
constexpr int SeqRepeats(int length) {
  return length == 16 ? 5000 : (length == 256 ? 500 : 10);
}

// ---------------------------------------------------------------------------
// The two functor-era baselines, which do not go through structural hooks.
// ---------------------------------------------------------------------------

/*! \brief apache/tvm main's PostOrderVisit: tirx::IRApplyVisit over StmtExprVisitor. */
class FunctorApplyVisit : public StmtExprVisitor {
 public:
  explicit FunctorApplyVisit(std::function<void(const ffi::ObjectRef&)> f) : f_(std::move(f)) {}

  void VisitExpr(const Expr& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    ExprVisitor::VisitExpr(node);
    f_(node);
  }
  void VisitStmt(const Stmt& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    StmtVisitor::VisitStmt(node);
    f_(node);
  }

 private:
  std::function<void(const ffi::ObjectRef&)> f_;
  std::unordered_set<const ffi::Object*> visited_;
};

/*! \brief apache/tvm main's Substitute: tirx::IRSubstitute over StmtExprMutator. */
class FunctorSubstitute : public StmtExprMutator {
 public:
  explicit FunctorSubstitute(std::function<ffi::Optional<Expr>(const Var&)> vmap)
      : vmap_(std::move(vmap)) {}

  Expr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);
    if (ffi::Optional<Expr> ret = vmap_(var)) return ret.value();
    return var;
  }

 private:
  std::function<ffi::Optional<Expr>(const Var&)> vmap_;
};

// ---------------------------------------------------------------------------
// Arms.
// ---------------------------------------------------------------------------

size_t g_sink = 0;

ffi::Any SwapAllVars(const Var& var) {
  if (var.same_as(Outer())) return ffi::Any(Replacement());
  if (var.same_as(Replacement())) return ffi::Any(Outer());
  return ffi::Any(var);
}
ffi::Any SwapTargetVar(const Var& var) {
  if (var.same_as(Target())) return ffi::Any(Replacement());
  if (var.same_as(Replacement())) return ffi::Any(Target());
  return ffi::Any(var);
}
ffi::Any SwapVars(ReplaceKind kind, const Var& var) {
  return kind == ReplaceKind::kAllVars ? SwapAllVars(var) : SwapTargetVar(var);
}
ffi::Optional<Expr> SwapVarsFn(ReplaceKind kind, const Var& var) {
  Expr mapped = SwapVars(kind, var).cast<Expr>();
  if (mapped.same_as(var)) return std::nullopt;
  return ffi::Optional<Expr>(mapped);
}

void WalkFloor(ffi::AnyView root) {
  MinimalVisitorObj visitor;
  g_sink += visitor.VisitExpected(root).is_err();
}

void Walk(ffi::AnyView root) {
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
}

void WalkNever(ffi::AnyView root) {
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
}

void WalkFunctor(ffi::AnyView root) {
  size_t matched = 0;
  FunctorApplyVisit visitor(
      [&](const ffi::ObjectRef& node) { matched += node.as<VarNode>() != nullptr; });
  ffi::ObjectRef node = root.cast<ffi::ObjectRef>();
  if (auto stmt = node.as<Stmt>()) {
    visitor(*stmt);
  } else {
    visitor(root.cast<Expr>());
  }
  g_sink += matched;
}

void WalkOld(ffi::AnyView root) {
  size_t matched = 0;
  PostOrderVisit(root.cast<ffi::ObjectRef>(),
                 [&](const ffi::ObjectRef& node) { matched += node.as<VarNode>() != nullptr; });
  g_sink += matched;
}

/*! \brief The retained-versus-moved call site: one `std::move` separates the two columns. */
ffi::Any Take(Ownership ownership, ffi::Any* slot) {
  return ownership == Ownership::kMoved ? std::move(*slot) : ffi::Any(*slot);
}
void Give(Ownership ownership, ffi::Any* slot, ffi::Any result) {
  if (ownership == Ownership::kMoved) {
    *slot = std::move(result);
  } else {
    g_sink += result.type_index();
  }
}

void MapFloor(Ownership ownership, ReplaceKind, ffi::Any* slot) {
  MinimalMutatorObj mutator;
  ffi::Any input = Take(ownership, slot);
  ffi::Expected<ffi::Any> result = ownership == Ownership::kMoved
                                       ? mutator.MaybeInplaceMutateIfUniqueExpected(input)
                                       : mutator.MutateExpected(input);
  g_sink += result.is_err();
  Give(ownership, slot, result.value());
}

void MapNever(Ownership ownership, ReplaceKind, ffi::Any* slot) {
  Give(ownership, slot,
       ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
           Take(ownership, slot), [](const FloatImm& v) { return ffi::Any(v); }));
}

void MapIdentity(Ownership ownership, ReplaceKind, ffi::Any* slot) {
  Give(ownership, slot,
       ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
           Take(ownership, slot), [](const Var& var) { return ffi::Any(var); }));
}

void MapReplace(Ownership ownership, ReplaceKind kind, ffi::Any* slot) {
  Give(ownership, slot,
       ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
           Take(ownership, slot), [kind](const Var& var) { return SwapVars(kind, var); }));
}

void MapFunctor(Ownership ownership, ReplaceKind kind, ffi::Any* slot) {
  ffi::Any input = Take(ownership, slot);
  FunctorSubstitute mutator([kind](const Var& var) { return SwapVarsFn(kind, var); });
  if (auto stmt = input.as<Stmt>()) {
    Give(ownership, slot, ffi::Any(mutator(*stmt)));
  } else {
    Give(ownership, slot, ffi::Any(mutator(input.cast<Expr>())));
  }
}

void MapOld(Ownership ownership, ReplaceKind kind, ffi::Any* slot) {
  ffi::Any input = Take(ownership, slot);
  auto vmap = [kind](const Var& var) { return SwapVarsFn(kind, var); };
  if (auto stmt = input.as<Stmt>()) {
    Give(ownership, slot, ffi::Any(Substitute(*stmt, vmap)));
  } else {
    Give(ownership, slot, ffi::Any(Substitute(input.cast<Expr>(), vmap)));
  }
}

struct WalkArm {
  const char* name;
  void (*run)(ffi::AnyView);
};
constexpr WalkArm kWalkArms[] = {
    {"walk_floor", &WalkFloor},     {"walk", &Walk},        {"walk_never", &WalkNever},
    {"walk_functor", &WalkFunctor}, {"walk_old", &WalkOld},
};

struct MapArm {
  const char* name;
  void (*run)(Ownership, ReplaceKind, ffi::Any*);
  bool rebuilds;
};
constexpr MapArm kMapArms[] = {
    {"map_floor", &MapFloor, false},       {"map_never", &MapNever, false},
    {"map_identity", &MapIdentity, false}, {"map_replace", &MapReplace, true},
    {"map_functor", &MapFunctor, true},    {"map_old", &MapOld, true},
};

void RunFixture(const FixtureInfo& info, ffi::Any (*build)()) {
  EmitFixture(kHarness, info);

  for (const WalkArm& arm : kWalkArms) {
    ffi::Any root = build();
    double ns = MeasureStationary(info.repeats, [&] { arm.run(root); });
    EmitResult(kHarness, info.name, "-", arm.name, ns);
  }

  for (Ownership ownership : {Ownership::kRetained, Ownership::kMoved}) {
    for (const MapArm& arm : kMapArms) {
      double ns = 0;
      if (ownership == Ownership::kMoved && arm.rebuilds && info.has_sharing) {
        std::vector<ffi::Any> pool(kPoolSize);
        int passes = info.repeats / kPoolSize;
        if (passes < 1) passes = 1;
        ns = MeasurePooled(
            passes, [&] { for (ffi::Any& slot : pool) slot = build(); },
            [&](int i) { arm.run(ownership, info.replace_kind, &pool[i]); });
      } else {
        ffi::Any slot = build();
        ns = MeasureStationary(info.repeats,
                               [&] { arm.run(ownership, info.replace_kind, &slot); });
      }
      EmitResult(kHarness, info.name, OwnershipName(ownership), arm.name, ns);
    }
  }
}

}  // namespace real_tvm

int main() {
  using namespace real_tvm;  // NOLINT(build/namespaces)
  tvm_hooks::InstallAll();

  EmitStandardProvenance(kHarness);
  EmitProvenance("structural_hooks", "harness (tvm_hook_override.h), installed over TVM's");
  EmitProvenance("seqstmt_inplace_hook", TVM_SEQSTMT_INPLACE_FIX ? "repaired" : "original");

  RunFixture({"split-fuse-shared", 12, 17, kSplitFuseSharedBytes, 9, 3, kSplitFuseRepeats, true,
              ReplaceKind::kAllVars},
             [] { return ffi::Any(SplitFuse(true)); });
  RunFixture({"split-fuse-distinct", 15, 17, kSplitFuseDistinctBytes, 9, 1, kSplitFuseRepeats,
              false, ReplaceKind::kAllVars},
             [] { return ffi::Any(SplitFuse(false)); });

  constexpr int64_t kSeqRebuiltRetained = 5;
  constexpr int64_t kSeqRebuiltMoved = TVM_SEQSTMT_INPLACE_FIX ? 1 : 4;
  for (int length : kSeqLengths) {
    static int current_length = 0;
    current_length = length;
    std::string name = "seq-" + std::to_string(length);
    RunFixture({name.c_str(), 4LL * length + 4, 6LL * length + 1, kSeqWorkingSet(length),
                kSeqRebuiltRetained, kSeqRebuiltMoved, SeqRepeats(length), false,
                ReplaceKind::kSingleVar},
               [] { return ffi::Any(LongSeq(current_length)); });
  }

  Emit("#sink\t" + std::to_string(g_sink));
  return 0;
}
