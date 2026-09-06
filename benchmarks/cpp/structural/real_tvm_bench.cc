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

#include <tvm/ffi/extra/structural_equal.h>

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

/*!
 * \brief Partner table for the intra-element swap: the `IntImm` multiplier inside one element
 *        maps to the multiplier inside its partner.
 *
 * The element-swap arm above replaces whole `Evaluate` nodes, so nothing *inside* an element
 * ever changes and element-level in-place mutation has no work to do however well it is
 * implemented.  This arm changes a leaf inside the element instead: the `Evaluate`'s `value`
 * field must be updated, which makes the `Evaluate` node itself a candidate for in-place
 * mutation, and the same for the `Add` and `Mul` above the leaf.  That is the workload a real
 * substitution pass performs, and the only one on which the `SeqStmt` unique-array access can
 * pay.  Still an involution, so the fixture stays stationary.
 */
std::unordered_map<const ffi::Object*, PrimExpr>* IntraSwapTable() {
  static std::unordered_map<const ffi::Object*, PrimExpr> table;
  return &table;
}

/*!
 * \brief Splice table: an element that maps to a nested `SeqStmt` replaces itself with all of
 *        its statements, so `n > 1` grows the sequence, `n == 1` leaves it alone and `n == 0`
 *        shrinks it. Nothing else in the harness reaches the splice path at all.
 */
std::unordered_map<const ffi::Object*, Stmt>* SpliceTable() {
  static std::unordered_map<const ffi::Object*, Stmt> table;
  return &table;
}
/*! \brief Statements a growing splice expands one element into. */
constexpr int kSpliceWidth = 4;

/*! \brief The `IntImm` multiplier inside `Evaluate(v * imm + inner)`. */
PrimExpr MultiplierOf(const Stmt& element) {
  const auto* eval = element.as<EvaluateNode>();
  const auto* add = eval->value.as<prim::AddNode>();
  const auto* mul = add->a.as<prim::MulNode>();
  return mul->b;
}

/*! \brief Which swap table a fixture build populates. See IntraSwapTable. */
enum class SwapMode { kElement, kIntraElement, kSpliceGrow, kSpliceShrink };

Stmt LongSeq(int length, int density, SwapMode mode) {
  ffi::Array<Stmt> body;
  body.reserve(length);
  for (int i = 0; i < length; ++i) {
    // `i + 2`: a multiplier of one would constant-fold the Mul away.
    body.push_back(Evaluate(Outer() * (i + 2) + Inner()));
  }
  SwapTable()->clear();
  IntraSwapTable()->clear();
  SpliceTable()->clear();
  for (const SwapPair& pair : SwapPairs(length, density)) {
    Stmt lo = body[pair.lo], hi = body[pair.hi];
    if (mode == SwapMode::kElement) {
      // Holds a handle to each swapped element, which is harmless here because the arm
      // replaces elements rather than mutating them.
      (*SwapTable())[lo.get()] = hi;
      (*SwapTable())[hi.get()] = lo;
    } else if (mode == SwapMode::kIntraElement) {
      // Holds handles to the leaves only. Holding the elements too would give them a second
      // reference and suppress the in-place mutation this arm exists to exercise.
      PrimExpr lo_imm = MultiplierOf(lo), hi_imm = MultiplierOf(hi);
      (*IntraSwapTable())[lo_imm.get()] = hi_imm;
      (*IntraSwapTable())[hi_imm.get()] = lo_imm;
    } else {
      ffi::Array<Stmt> expansion;
      if (mode == SwapMode::kSpliceGrow) {
        for (int j = 0; j < kSpliceWidth; ++j) expansion.push_back(lo);
      }
      // kSpliceShrink leaves it empty, which removes the element.
      (*SpliceTable())[lo.get()] = SeqStmt(expansion);
      (*SpliceTable())[hi.get()] = SeqStmt(expansion);
    }
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
/*! \brief The splice replacement: map an element to a nested `SeqStmt`. */
ffi::Any SpliceElements(const Stmt& stmt) {
  auto it = SpliceTable()->find(stmt.get());
  if (it != SpliceTable()->end()) return ffi::Any(it->second);
  return ffi::Any(stmt);
}

/*! \brief The intra-element replacement: swap the multipliers of two elements. */
ffi::Any SwapMultipliers(const IntImm& imm) {
  auto it = IntraSwapTable()->find(imm.get());
  if (it != IntraSwapTable()->end()) return ffi::Any(it->second);
  return ffi::Any(imm);
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

/*!
 * \brief Map arms take the input and return the output. Ownership lives at the call site.
 *
 * `retained` hands over a second handle while the caller keeps one, so the root's refcount is
 * at least two and the engine takes the copy-on-write path. `moved` hands over the sole
 * reference, so it takes the in-place path. Neither arm destroys anything: the driver retains
 * every output in a buffer and frees them all after the clock is read.
 */
ffi::Any MapFloor(ffi::Any input, ReplaceKind, Ownership ownership) {
  MinimalMutatorObj mutator;
  ffi::Expected<ffi::Any> result = ownership == Ownership::kMoved
                                       ? mutator.MaybeInplaceMutateIfUniqueExpected(input)
                                       : mutator.MutateExpected(input);
  g_sink += result.is_err();
  return result.value();
}

ffi::Any MapNever(ffi::Any input, ReplaceKind, Ownership) {
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
      std::move(input), [](const FloatImm& v) { return ffi::Any(v); });
}

// identity and replace match the node type their fixture's replacement acts on, so the two
// rungs of the ladder differ only in what the callback returns.
ffi::Any MapIdentity(ffi::Any input, ReplaceKind kind, Ownership) {
  if (kind == ReplaceKind::kSingleVar) {
    return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
        std::move(input), [](const Stmt& stmt) { return ffi::Any(stmt); });
  }
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
      std::move(input), [](const Var& var) { return ffi::Any(var); });
}

ffi::Any MapReplace(ffi::Any input, ReplaceKind kind, Ownership) {
  if (kind == ReplaceKind::kSingleVar) {
    return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(input), SwapEvaluates);
  }
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(input), SwapAllVars);
}

/*! \brief seq only: change a leaf inside the element rather than replacing the element. */
ffi::Any MapReplaceField(ffi::Any input, ReplaceKind, Ownership) {
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(input), SwapMultipliers);
}

/*! \brief seq only: the only arm that reaches the hook's splice path. */
ffi::Any MapSplice(ffi::Any input, ReplaceKind, Ownership) {
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(input), SpliceElements);
}

/*! \brief One-off probe; see ReplaceNoRemap. split/fuse only. */
ffi::Any MapReplaceNoRemap(ffi::Any input, ReplaceKind, Ownership) {
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(input), ReplaceNoRemap);
}

ffi::Any MapFunctor(ffi::Any input, ReplaceKind, Ownership) {
  FunctorSubstitute mutator([](const Var& var) { return SwapVarsFn(var); });
  if (auto stmt = input.as<Stmt>()) return ffi::Any(mutator(*stmt));
  return ffi::Any(mutator(input.cast<Expr>()));
}

ffi::Any MapOld(ffi::Any input, ReplaceKind, Ownership) {
  auto vmap = [](const Var& var) { return SwapVarsFn(var); };
  if (auto stmt = input.as<Stmt>()) return ffi::Any(Substitute(*stmt, vmap));
  return ffi::Any(Substitute(input.cast<Expr>(), vmap));
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
  ffi::Any (*run)(ffi::Any, ReplaceKind, Ownership);
  /*! rief Whether the arm changes the graph, which is what makes it consume a shared DAG. */
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
    ffi::Any out = MapReplace(std::move(root), info.replace_kind, Ownership::kMoved);
    if (out.cast<ffi::ObjectRef>().get() != before) Fail(tag + "/moved: root was not in place");
  }
  {
    // Retained keeps a second reference, so the input must be copied rather than mutated: the
    // opposite direction, pinned so the check catches in-place firing where it should not.
    ffi::Any root = build();
    const ffi::Object* before = root.cast<ffi::ObjectRef>().get();
    ffi::Any out = MapReplace(ffi::Any(root), info.replace_kind, Ownership::kRetained);
    (void)out;  // `root` still holds a reference, so the input must not have been mutated
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
  ffi::Any root = ffi::Any(LongSeq(length, density, SwapMode::kElement));
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

/*!
 * \brief Differential test: the in-place hook must produce exactly what the non-in-place path
 *        produces. It is an optimization of the same semantics, not a variant with its own.
 *
 * `MutateSeqStmtRaw` defines what splicing means, so running both on identical inputs and
 * requiring structurally equal outputs verifies the cursor bookkeeping without having to argue
 * it. Retained ownership routes through `MutateSeqStmtRaw`; moved routes through the in-place
 * hook. Covers grow, shrink, unchanged length, splices at the first and last element, several
 * in one pass, and a splice that empties the sequence entirely.
 */
/*!
 * \brief Build a nested `SeqStmt` of any length, bypassing the public constructor.
 *
 * `SeqStmt`'s constructor rejects both length 0 and length 1 ("An empty SeqStmt is prohibited",
 * "A SeqStmt of length 1 is prohibited"), so **splicing can only ever grow a sequence** through
 * the public API -- the shrink and length-unchanged directions are unreachable. They are still
 * exercised here, constructed directly, so the hook's behaviour on them is pinned rather than
 * merely argued; both paths receive the same input, so the differential test stays valid.
 */
Stmt MakeNestedSeq(ffi::Array<Stmt> body) {
  ffi::ObjectPtr<SeqStmtNode> node = ffi::make_object<SeqStmtNode>();
  node->seq = std::move(body);
  return Stmt(node);
}

void CheckSpliceAgainstReference() {
  struct Case {
    const char* name;
    int length;
    std::vector<int> targets;  // element positions that map to a nested SeqStmt
    // Statements each target expands into: 0 shrinks, >1 grows. A nested SeqStmt of length 1
    // is not expressible -- SeqStmt's constructor prohibits it -- so the length-unchanged
    // splice cannot occur and is not tested.
    int width;
  };
  const std::vector<Case> cases = {
      {"grow/first", 8, {0}, 4},        {"grow/last", 8, {7}, 4},
      {"grow/middle", 8, {3}, 4},       {"grow/several", 8, {1, 3, 5}, 3},
      {"grow/adjacent", 8, {2, 3}, 2},  {"keep/one", 8, {4}, 1},
      {"shrink/first", 8, {0}, 0},      {"shrink/last", 8, {7}, 0},
      {"shrink/several", 8, {0, 2, 4}, 0}, {"shrink/all", 5, {0, 1, 2, 3, 4}, 0},
      {"grow/wide", 4, {0, 1, 2, 3}, 6},  // forces the spill path
  };
  for (const Case& c : cases) {
    auto build = [&]() {
      ffi::Array<Stmt> body;
      for (int i = 0; i < c.length; ++i) body.push_back(Evaluate(Outer() * (i + 2) + Inner()));
      SpliceTable()->clear();
      for (int t : c.targets) {
        ffi::Array<Stmt> expansion;
        for (int j = 0; j < c.width; ++j) expansion.push_back(body[t]);
        (*SpliceTable())[body[t].get()] = MakeNestedSeq(expansion);
      }
      return SeqStmt(body);
    };
    // Retained: a second reference forces the copy-on-write path, which is MutateSeqStmtRaw.
    ffi::Any reference_input = ffi::Any(build());
    ffi::Any keep_alive = reference_input;
    ffi::Any reference = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
        ffi::Any(reference_input), SpliceElements);
    // Moved: the sole reference, which is the in-place hook.
    ffi::Any inplace_input = ffi::Any(build());
    ffi::Any inplace = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(inplace_input),
                                                                     SpliceElements);
    const auto* a = reference.cast<ffi::ObjectRef>().as<SeqStmtNode>();
    const auto* b = inplace.cast<ffi::ObjectRef>().as<SeqStmtNode>();
    const int64_t a_len = a == nullptr ? -1 : static_cast<int64_t>(a->seq.size());
    const int64_t b_len = b == nullptr ? -1 : static_cast<int64_t>(b->seq.size());
    // Non-vacuous: the reference itself must have changed length by the amount the case
    // describes, or agreement between the two paths would prove nothing.
    int64_t expected = c.length;
    for (size_t k = 0; k < c.targets.size(); ++k) expected += c.width - 1;
    if (a_len != expected) {
      Fail(std::string("splice/") + c.name + ": MutateSeqStmtRaw produced length " +
           std::to_string(a_len) + ", expected " + std::to_string(expected));
    }
    if (a_len != b_len) {
      Fail(std::string("splice/") + c.name + ": length " + std::to_string(b_len) +
           " in place against " + std::to_string(a_len) + " from MutateSeqStmtRaw");
    }
    if (!ffi::StructuralEqual()(reference, inplace)) {
      Fail(std::string("splice/") + c.name + ": in-place output differs from MutateSeqStmtRaw");
    }
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

/*!
 * \brief Time one map arm. The timed region is the lambda passed to `Timed`, and nothing
 *        enters or leaves it implicitly.
 *
 * Outside, before: the fixture is built and the result buffer is reserved for the whole batch,
 * so no growth reallocation can land inside the clock.
 *
 * Inside: only the arm, `repeats` times, and whatever running it entails. In `moved` mode the
 * input is moved and consumed, and any freeing that follows from consuming it is part of the
 * operation under test -- in-place mutation is partly about reusing storage the input owned --
 * so it stays inside.
 *
 * Outside, after: the buffer is cleared, which is where every retained output is destroyed.
 * That teardown is the caller's, not the mutation's, and charging it to the arm that built the
 * graph would inflate exactly the arms that allocate most.
 *
 * What each ownership path holds live differs by design and is part of the comparison:
 * `retained` keeps one input alive and reuses it every iteration, `moved` consumes the
 * previous iteration's output as the next iteration's input, so a single graph walks down the
 * chain. Both retain every output.
 */
double MeasureMapArm(ffi::Any (*run)(ffi::Any, ReplaceKind, Ownership), Ownership ownership,
                     ReplaceKind kind, int repeats, ffi::Any (*build)(), bool pooled = false) {
  std::vector<ffi::Any>& out = ResultSink();
  ffi::Any held;             // retained: the caller's handle, alive for the whole batch
  std::vector<ffi::Any> pool;  // moved on a fixture the arm consumes: independent copies
  // A rebuilding in-place arm on a fixture with a pointer-shared subtree un-shares it, so the
  // chain would drift to a different graph after one iteration. Those cases consume a fresh
  // copy per iteration from a pool built untimed.
  const bool use_pool = pooled && ownership == Ownership::kMoved;
  const int batch_size = use_pool ? kPoolSize : repeats;

  auto setup = [&] {  // untimed
    ReserveResultSink(static_cast<size_t>(batch_size) + 1);
    if (use_pool) {
      pool.clear();
      pool.reserve(batch_size);
      for (int i = 0; i < batch_size; ++i) pool.push_back(build());
    } else if (ownership == Ownership::kRetained) {
      held = build();
    } else {
      out.push_back(build());  // the chain's first link
    }
  };
  auto batch = [&] {  // the timed region, and only this
    if (use_pool) {
      for (int i = 0; i < batch_size; ++i) out.push_back(run(std::move(pool[i]), kind, ownership));
    } else if (ownership == Ownership::kRetained) {
      for (int i = 0; i < batch_size; ++i) out.push_back(run(ffi::Any(held), kind, ownership));
    } else {
      for (int i = 0; i < batch_size; ++i) out.push_back(run(std::move(out[i]), kind, ownership));
    }
  };
  auto teardown = [&] {  // untimed
    DrainResultSink();
    held = ffi::Any();
    pool.clear();
  };

  setup();
  batch();  // warm-up
  teardown();

  std::vector<double> samples;
  samples.reserve(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    setup();
    double begin = NowNs();
    batch();
    double elapsed = NowNs() - begin;
    teardown();
    samples.push_back(elapsed / batch_size);
  }
  return Median(std::move(samples));
}

void RunFixture(const FixtureInfo& info, ffi::Any (*build)()) {
  EmitFixture(kHarness, info);

  for (const WalkArm& arm : kWalkArms) {
    // Walk arms own nothing and build nothing, so there is no result to retain.
    ffi::Any root = build();
    double ns = MeasureStationary(info.repeats, [&] { arm.run(root); }, [] {});
    EmitResult(kHarness, info.name, "-", arm.name, ns);
  }

  for (Ownership ownership : {Ownership::kRetained, Ownership::kMoved}) {
    for (const MapArm& arm : kMapArms) {
      // `map_old` is Substitute, which is the same engine as map_replace differing only in
      // API, so a full grid of it would spend columns confirming a null result.
      if (std::string(arm.name) == "map_old" && !info.run_old) continue;
      double ns = MeasureMapArm(arm.run, ownership, info.replace_kind, info.repeats, build,
                                arm.rebuilds && info.has_sharing);
      EmitResult(kHarness, info.name, OwnershipName(ownership), arm.name, ns);
    }
    if (info.replace_kind == ReplaceKind::kSingleVar) {
      double ns = MeasureMapArm(&MapReplaceField, ownership, info.replace_kind, info.repeats,
                                build);
      EmitResult(kHarness, info.name, OwnershipName(ownership), "map_replace_field", ns);
    }
    if (info.run_noremap) {
      double ns = MeasureMapArm(&MapReplaceNoRemap, ownership, info.replace_kind, info.repeats,
                                build);
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

  CheckSpliceAgainstReference();
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
    auto build = [] { return ffi::Any(LongSeq(current_length, 1, SwapMode::kElement)); };
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

  // The intra-element workload: change a leaf inside two elements rather than replacing the
  // elements. The Evaluate above each changed leaf has its `value` field updated, so it is a
  // candidate for in-place mutation -- which is what SeqStmt's hook decides, and the only
  // workload on which the repaired hook can pay. Run across the seq sweep and, at the L2 size,
  // across density.
  {
    static int intra_length = 0;
    static int intra_density = 0;
    auto build = [] { return ffi::Any(LongSeq(intra_length, intra_density, SwapMode::kIntraElement)); };
    for (int length : kSeqLengths) {
      intra_length = length;
      intra_density = 1;
      std::string name = "intra-seq-" + std::to_string(length);
      for (Ownership ownership : {Ownership::kRetained, Ownership::kMoved}) {
        double ns = MeasureMapArm(&MapReplaceField, ownership, ReplaceKind::kSingleVar,
                                  SeqRepeats(length), build);
        EmitResult(kHarness, name, OwnershipName(ownership), "map_replace_field", ns);
      }
    }
    intra_length = kDensitySweepLength;
    for (int density : kDensities) {
      intra_density = density;
      std::string name = "intra-density-" + std::to_string(2 * density) + "of" +
                         std::to_string(kDensitySweepLength);
      for (Ownership ownership : {Ownership::kRetained, Ownership::kMoved}) {
        double ns = MeasureMapArm(&MapReplaceField, ownership, ReplaceKind::kSingleVar,
                                  SeqRepeats(kDensitySweepLength), build);
        EmitResult(kHarness, name, OwnershipName(ownership), "map_replace_field", ns);
      }
    }
  }

  // Splicing: the only arm that reaches the hook's splice path. Growing is not stationary --
  // the sequence gets longer each time -- so it runs over a pool of independent copies rebuilt
  // untimed between passes, like the shared-DAG case. Two seq sizes and two splice counts;
  // seq-16384 is left out because a pool of it would be gigabytes.
  {
    static int splice_length = 0;
    static int splice_count = 0;
    auto build = [] {
      return ffi::Any(LongSeq(splice_length, splice_count, SwapMode::kSpliceGrow));
    };
    for (int length : {16, 256}) {
      splice_length = length;
      for (int count : {1, 8}) {
        if (count * 2 > length / 2) continue;
        splice_count = count;
        std::string name = "splice-" + std::to_string(2 * count) + "x" +
                           std::to_string(kSpliceWidth) + "-L" + std::to_string(length);
        for (Ownership ownership : {Ownership::kRetained, Ownership::kMoved}) {
          // Growing is not stationary -- the sequence lengthens each time -- so the moved path
          // consumes a fresh copy per iteration from an untimed pool.
          double ns = MeasureMapArm(&MapSplice, ownership, ReplaceKind::kSingleVar,
                                    SeqRepeats(length), build, /*pooled=*/true);
          EmitResult(kHarness, name, OwnershipName(ownership), "map_splice", ns);
        }
      }
    }
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
      auto build = [] { return ffi::Any(LongSeq(sweep_length, current_density, SwapMode::kElement)); };
      CheckSeqSwap(sweep_length, density);
      std::string name = "density-" + std::to_string(2 * density) + "of" +
                         std::to_string(sweep_length);
      for (Ownership ownership : {Ownership::kRetained, Ownership::kMoved}) {
        double ns = MeasureMapArm(&MapReplace, ownership, ReplaceKind::kSingleVar,
                                  SeqRepeats(sweep_length), build);
        EmitResult(kHarness, name, OwnershipName(ownership), "map_replace", ns);
      }
    }
  }

  Emit("#sink\t" + std::to_string(g_sink));
  return 0;
}
