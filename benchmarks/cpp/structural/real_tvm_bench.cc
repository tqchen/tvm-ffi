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

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
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

PrimExpr SplitFuse(bool shared) {
  PrimExpr q = Outer() * 16 + Inner();
  PrimExpr r = shared ? q : PrimExpr(Outer() * 16 + Inner());
  return floordiv(q, 32) * 32 + floormod(r, 32);
}

/*!
 * \brief The `Evaluate` elements a `seq` replacement swaps, by position.
 *
 * A swap rather than a one-way replacement, so the arm is an involution: applying it twice
 * restores the original graph, the fixture stays stationary under repetition, and the timed
 * loop needs no pool of copies.  `density` pairs are swapped, so the number of changed
 * elements is `2 * density` -- a controlled input, not a consequence of which identities
 * happen to repeat.
 *
 * Pairs are spread across the sequence and never touch the first or last element, in case
 * either is special-cased: pair `k` swaps positions `L/4 + k` and `3L/4 - k`.
 */
struct SwapPair {
  int lo, hi;
};
std::vector<SwapPair> SwapPairs(int length, int density) {
  std::vector<SwapPair> pairs;
  for (int k = 0; k < density; ++k) pairs.push_back({length / 4 + k, 3 * length / 4 - k});
  return pairs;
}

/*! \brief Partner table for the seq swap: raw element pointer -> the element it swaps with. */
std::unordered_map<const ffi::Object*, Stmt>* SwapTable() {
  static std::unordered_map<const ffi::Object*, Stmt> table;
  return &table;
}

Stmt LongSeq(int length, int density) {
  ffi::Array<Stmt> body;
  body.reserve(length);
  for (int i = 0; i < length; ++i) {
    // `i + 2`: a multiplier of one would constant-fold the Mul away.
    body.push_back(Evaluate(Outer() * (i + 2) + Inner()));
  }
  SwapTable()->clear();
  for (const SwapPair& pair : SwapPairs(length, density)) {
    Stmt lo = body[pair.lo], hi = body[pair.hi];
    (*SwapTable())[lo.get()] = hi;
    (*SwapTable())[hi.get()] = lo;
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
// Two Vars (Outer, Inner) are shared across every element; 20275's SeqStmtVisit visits the
// Array itself, so it is a node too and kSeqBytes covers it.
constexpr int64_t kSeqWorkingSet(int64_t l) {
  return l * (kMulBytes + kImmBytes + kAddBytes + kEvalBytes) + 2 * kVarBytes + kSeqBytes(l);
}

constexpr int kSeqLengths[] = {16, 256, 16384};
// The density sweep runs on the L2 fixture: the middle of the range, least likely to be
// dominated by either cache effects or fixed overhead. Each point swaps `k` pairs, so
// `2k` of the 256 elements change.
constexpr int kDensitySweepLength = 256;
// Pair k swaps positions L/4+k and 3L/4-k, so the two halves meet at k = L/4 and the largest
// valid density is L/4 -- half the elements changed. CheckSeqSwap fails the run if a pair
// collides, which is how the previous {1,8,32,64,128} was caught.
constexpr int kDensities[] = {1, 8, 32, 64};

// The seq replacement swaps two Evaluate nodes, so it rebuilds their two Mul/Add-free spines:
// each swapped element's parent chain is just the SeqStmt, and the Evaluate objects themselves
// are reused rather than rebuilt. Retained copies the SeqStmt and its Array; moved mutates
// both in place.
constexpr int64_t kSeqRebuiltRetained = 2;
constexpr int64_t kSeqRebuiltMoved = 0;
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
/*!
 * \brief The seq replacement: swap two `Evaluate` nodes, leave every other element alone.
 *
 * Matching `Evaluate` rather than `Var` is what makes the sparse claim exact.  A `Var`
 * callback would go through the remap, so replacing one occurrence of a `Var` that repeats
 * across elements would propagate to all of them and the "sparse" update would silently be a
 * dense one.  `Evaluate` is not a free variable, no remap table is involved, and each element
 * is exactly one `Evaluate`, so "swap k pairs" means what it says.
 */
ffi::Any SwapEvaluates(const Stmt& stmt) {
  auto it = SwapTable()->find(stmt.get());
  if (it != SwapTable()->end()) return ffi::Any(it->second);
  return ffi::Any(stmt);
}
ffi::Optional<Expr> SwapVarsFn(const Var& var) {
  Expr mapped = SwapAllVars(var).cast<Expr>();
  if (mapped.same_as(var)) return std::nullopt;
  return ffi::Optional<Expr>(mapped);
}
/*!
 * \brief `replace` without the var remap: returns a fresh Var instead of consulting the table.
 *
 * A cost probe, not a valid substitution -- without the remap the same `Var` reached twice
 * yields two independent replacements, so the output is not what a consistent substitution
 * produces.  Run once on split/fuse, where `Outer` repeats and the remap therefore has work
 * to do, to price the remap on its own.
 */
ffi::Any ReplaceNoRemap(const Var& var) {
  if (var.same_as(Outer())) return ffi::Any(PrimVar("fresh"));
  return ffi::Any(var);
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

// identity and replace match the node type their fixture's replacement acts on, so the two
// rungs of the ladder differ only in what the callback returns.
void MapIdentity(Ownership ownership, ReplaceKind kind, ffi::Any* slot) {
  if (kind == ReplaceKind::kSingleVar) {
    Give(ownership, slot,
         ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
             Take(ownership, slot), [](const Stmt& stmt) { return ffi::Any(stmt); }));
  } else {
    Give(ownership, slot,
         ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
             Take(ownership, slot), [](const Var& var) { return ffi::Any(var); }));
  }
}

void MapReplace(Ownership ownership, ReplaceKind kind, ffi::Any* slot) {
  if (kind == ReplaceKind::kSingleVar) {
    Give(ownership, slot, ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(Take(ownership, slot),
                                                                        SwapEvaluates));
  } else {
    Give(ownership, slot,
         ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(Take(ownership, slot), SwapAllVars));
  }
}

/*! \brief One-off probe; see ReplaceNoRemap. split/fuse only. */
void MapReplaceNoRemap(Ownership ownership, ReplaceKind, ffi::Any* slot) {
  Give(ownership, slot,
       ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(Take(ownership, slot), ReplaceNoRemap));
}

void MapFunctor(Ownership ownership, ReplaceKind, ffi::Any* slot) {
  ffi::Any input = Take(ownership, slot);
  FunctorSubstitute mutator([](const Var& var) { return SwapVarsFn(var); });
  if (auto stmt = input.as<Stmt>()) {
    Give(ownership, slot, ffi::Any(mutator(*stmt)));
  } else {
    Give(ownership, slot, ffi::Any(mutator(input.cast<Expr>())));
  }
}

void MapOld(Ownership ownership, ReplaceKind, ffi::Any* slot) {
  ffi::Any input = Take(ownership, slot);
  auto vmap = [](const Var& var) { return SwapVarsFn(var); };
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
    {"walk_floor", &WalkFloor},
    {"walk_var", &Walk},
    {"walk_never", &WalkNever},
    {"walk_functor", &WalkFunctor},
    {"walk_old", &WalkOld},
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

// ---------------------------------------------------------------------------
// Untimed checks.  These run once per fixture before anything is timed; nothing here executes
// inside a timed loop.  They exist because every `moved` number in the report rests on the
// in-place path actually firing, and because the sparse claim rests on exactly the intended
// elements changing.
// ---------------------------------------------------------------------------

void Fail(const std::string& what) {
  std::fflush(stdout);
  std::fprintf(stderr, "structural benchmark check failed: %s\n", what.c_str());
  std::exit(2);
}

/*!
 * \brief Confirm the in-place path fires where it should and not where it should not.
 *
 * Raw pointers only.  Holding an `ObjectRef` to the input would itself be a reference, make
 * `unique()` false, suppress the very thing being tested, and then report the suppression as
 * a result.  In-place mutation returns the same object; copy-on-write returns a different one.
 */
void CheckInplace(const FixtureInfo& info, ffi::Any (*build)()) {
  const std::string tag = std::string("inplace/") + info.name;
  {
    ffi::Any root = build();
    const ffi::Object* before = root.cast<ffi::ObjectRef>().get();
    ffi::Any slot = std::move(root);
    MapReplace(Ownership::kMoved, info.replace_kind, &slot);
    if (slot.cast<ffi::ObjectRef>().get() != before) Fail(tag + "/moved: root was not in place");
  }
  {
    // Retained keeps a second reference, so the input must be copied rather than mutated: the
    // opposite direction, pinned so the check catches in-place firing where it should not.
    ffi::Any root = build();
    const ffi::Object* before = root.cast<ffi::ObjectRef>().get();
    ffi::Any slot = root;  // the second reference
    MapReplace(Ownership::kRetained, info.replace_kind, &slot);
    if (root.cast<ffi::ObjectRef>().get() != before) Fail(tag + "/retained: input was mutated");
  }
}

/*!
 * \brief Confirm the seq replacement changes exactly the elements it claims, and is an
 *        involution.
 *
 * Captures every element's raw pointer, swaps, and requires that precisely the targeted
 * positions differ and every other position is pointer-identical to the input.  Then applies
 * it a second time and requires the original pointers back, which is what makes the arm safe
 * to repeat in a timed loop without a pool.
 */
void CheckSeqSwap(int length, int density) {
  const std::string tag = "seqswap/L=" + std::to_string(length) + "/d=" + std::to_string(density);
  ffi::Any root = ffi::Any(LongSeq(length, density));
  const auto* seq = static_cast<const SeqStmtNode*>(root.cast<ffi::ObjectRef>().get());
  std::vector<const ffi::Object*> before;
  for (int i = 0; i < length; ++i) before.push_back(seq->seq[i].get());

  std::vector<bool> expected(length, false);
  for (const SwapPair& pair : SwapPairs(length, density)) {
    expected[pair.lo] = true;
    expected[pair.hi] = true;
  }

  ffi::Any once = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(ffi::Any(root), SwapEvaluates);
  const auto* out = static_cast<const SeqStmtNode*>(once.cast<ffi::ObjectRef>().get());
  int changed = 0;
  for (int i = 0; i < length; ++i) {
    const bool differs = out->seq[i].get() != before[i];
    if (differs) ++changed;
    if (differs != expected[i]) {
      Fail(tag + ": element " + std::to_string(i) + (differs ? " changed" : " did not change") +
           " against expectation");
    }
  }
  if (changed != 2 * density) {
    Fail(tag + ": " + std::to_string(changed) + " elements changed, expected " +
         std::to_string(2 * density));
  }
  ffi::Any twice = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(once), SwapEvaluates);
  const auto* back = static_cast<const SeqStmtNode*>(twice.cast<ffi::ObjectRef>().get());
  for (int i = 0; i < length; ++i) {
    if (back->seq[i].get() != before[i]) Fail(tag + ": not an involution at element " +
                                              std::to_string(i));
  }
}

/*! \brief Every type a fixture reaches must have a harness hook, or it silently uses TVM's. */
void CheckHookCoverage(ffi::Any (*build)(), const char* name) {
  std::vector<int32_t> covered = tvm_hooks::CoveredTypes();
  ffi::Any root = build();
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(root, [&](ffi::AnyView v) {
    const int32_t index = v.type_index();
    if (index >= ffi::TypeIndex::kTVMFFIStaticObjectBegin &&
        index != ffi::TypeIndex::kTVMFFIArray &&
        std::find(covered.begin(), covered.end(), index) == covered.end()) {
      Fail(std::string("hook coverage/") + name + ": type " +
           ffi::TypeIndexToTypeKey(index) + " falls through to a TVM-registered hook");
    }
    return ffi::WalkResult::Advance();
  });
}

void RunFixture(const FixtureInfo& info, ffi::Any (*build)()) {
  EmitFixture(kHarness, info);

  for (const WalkArm& arm : kWalkArms) {
    ffi::Any root = build();
    double ns = MeasureStationary(info.repeats, [&] { arm.run(root); });
    EmitResult(kHarness, info.name, "-", arm.name, ns);
  }

  for (Ownership ownership : {Ownership::kRetained, Ownership::kMoved}) {
    for (const MapArm& arm : kMapArms) {
      // `map_old` is Substitute, which is the same engine as map_replace differing only in
      // API, so a full grid of it would spend columns confirming a null result. One fixture
      // states the agreement.
      if (std::string(arm.name) == "map_old" && !info.run_old) continue;
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
    if (info.run_noremap) {
      ffi::Any slot = build();
      double ns = MeasureStationary(
          info.repeats, [&] { MapReplaceNoRemap(ownership, info.replace_kind, &slot); });
      EmitResult(kHarness, info.name, OwnershipName(ownership), "map_replace_noremap", ns);
    }
  }
}

}  // namespace real_tvm

int main() {
  using namespace real_tvm;  // NOLINT(build/namespaces)
  tvm_hooks::InstallAll();

  EmitStandardProvenance(kHarness);
  EmitProvenance("structural_hooks",
                 "harness (tvm_hook_override.h), ported from apache/tvm#20275 b51da96381, "
                 "installed over TVM's");
  EmitProvenance("seqstmt_inplace_hook", TVM_SEQSTMT_INPLACE_FIX ? "repaired" : "20275 as shipped");

  // split/fuse replaces every Var, with the remap, which is the semantics Substitute provides
  // and the thing worth measuring on a small tree. Under moved ownership the only new identity
  // is the substituted-in Var, except on the shared fixture, where the first parent to reach
  // the shared subtree cannot mutate it in place and copies it and its own changed child.
  auto build_shared = [] { return ffi::Any(SplitFuse(true)); };
  auto build_distinct = [] { return ffi::Any(SplitFuse(false)); };
  FixtureInfo shared{"split-fuse-shared", 12, 17, kSplitFuseSharedBytes, 9, 3, kSplitFuseRepeats,
                     true, ReplaceKind::kAllVars};
  shared.run_old = true;       // the one fixture map_old runs on
  shared.run_noremap = true;   // Outer repeats here, so the remap has work to do
  FixtureInfo distinct{"split-fuse-distinct", 15, 17, kSplitFuseDistinctBytes, 9, 1,
                       kSplitFuseRepeats, false, ReplaceKind::kAllVars};

  CheckHookCoverage(build_shared, "split-fuse-shared");
  CheckInplace(shared, build_shared);
  CheckInplace(distinct, build_distinct);
  RunFixture(shared, build_shared);
  RunFixture(distinct, build_distinct);

  // seq swaps two Evaluate nodes: no remap is involved, and the change count is exactly two
  // regardless of L, so the update stays sparse as the body grows.
  for (int length : kSeqLengths) {
    static int current_length = 0;
    current_length = length;
    auto build = [] { return ffi::Any(LongSeq(current_length, 1)); };
    CheckHookCoverage(build, "seq");
    CheckSeqSwap(length, 1);
    std::string name = "seq-" + std::to_string(length);
    FixtureInfo info{name.c_str(),
                     4LL * length + 4,
                     6LL * length + 2,
                     kSeqWorkingSet(length),
                     kSeqRebuiltRetained,
                     kSeqRebuiltMoved,
                     SeqRepeats(length),
                     false,
                     ReplaceKind::kSingleVar};
    CheckInplace(info, build);
    RunFixture(info, build);
  }

  // Change-density sweep, on the L2 fixture only: does moved stop losing to retained once
  // enough of the body changes? The mechanism says the in-place path pays a uniqueness check
  // at every node it descends through, O(N), and saves a rebuild only where something
  // changes, O(changed), so there should be a crossover.
  {
    static int sweep_length = 0;
    sweep_length = kDensitySweepLength;
    for (int density : kDensities) {
      static int current_density = 0;
      current_density = density;
      auto build = [] { return ffi::Any(LongSeq(sweep_length, current_density)); };
      CheckSeqSwap(sweep_length, density);
      std::string name = "density-" + std::to_string(2 * density) + "of" +
                         std::to_string(sweep_length);
      for (Ownership ownership : {Ownership::kRetained, Ownership::kMoved}) {
        ffi::Any slot = build();
        double ns = MeasureStationary(SeqRepeats(sweep_length), [&] {
          MapReplace(ownership, ReplaceKind::kSingleVar, &slot);
        });
        EmitResult(kHarness, name, OwnershipName(ownership), "map_replace", ns);
      }
    }
  }

  Emit("#sink\t" + std::to_string(g_sink));
  return 0;
}
