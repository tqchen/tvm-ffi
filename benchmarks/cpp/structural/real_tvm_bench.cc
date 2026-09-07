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

// The hook file, selected per engine state by build.sh. A two-state run compiles each state's
// hooks against that state's own tvm-ffi API, so this is a whole file per state rather than a
// switch inside one -- see the banner in tvm_hook_override_pre753.h.
#ifndef TVM_FFI_BENCH_HOOK_HEADER
#define TVM_FFI_BENCH_HOOK_HEADER "tvm_hook_override.h"
#endif
#include TVM_FFI_BENCH_HOOK_HEADER

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
 * \brief The same split/fuse tree with every arithmetic node expressed as a `Call`.
 *
 * Row-for-row counterpart of `SplitFuse`: identical topology, identical `Var`s, the same six
 * binary operations in the same order, so the only variable between the two fixtures is the
 * node representation.  The delta between their rows is what a `Call` costs over a direct
 * node.
 *
 * It is the only fixture that reaches three hook decisions #20275 made and justified on
 * performance grounds, and that no other fixture can exercise:
 *
 *   * **descent into an `Array` inside a node** -- `Call.args`.  Every other fixture's
 *     children are direct typed fields, so `CallMutate`'s container path is otherwise dead.
 *   * **the empty `ty_args` skip.**  Nothing here has type arguments, so the guard is taken on
 *     every `Call`, on every arm, including `floor`.
 *   * **the interned `Op` operator skip.**
 *
 * The operator is a stand-in and its identity does not matter: the traversal never evaluates
 * the call, and `CallVisit`/`CallMutate` skip an `OpNode` operator without looking at which
 * one it is.  TIR has no builtin for `+` or `floordiv` -- those *are* nodes -- so four
 * distinct interned builtins stand in for the four node types, which keeps the `op` field
 * varying exactly as the direct form's node type does.
 */
PrimExpr CallOp(const Op& op, PrimExpr a, PrimExpr b) {
  return Call(PrimType::Int(32), op, {std::move(a), std::move(b)});
}
PrimExpr CallImm(int64_t v) { return IntImm(PrimType::Int(32), v); }
PrimExpr CallSplitFuse(bool shared) {
  const Op& mul = prim::builtin::shift_left();
  const Op& add = prim::builtin::bitwise_or();
  const Op& fdiv = prim::builtin::bitwise_and();
  const Op& fmod = prim::builtin::bitwise_xor();
  PrimExpr q = CallOp(add, CallOp(mul, Outer(), CallImm(16)), Inner());
  PrimExpr r = shared ? q : CallOp(add, CallOp(mul, Outer(), CallImm(16)), Inner());
  return CallOp(add, CallOp(mul, CallOp(fdiv, q, CallImm(32)), CallImm(32)),
                CallOp(fmod, r, CallImm(32)));
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

/*! \brief Which table a fixture build populates: element swaps, or splice targets. */
enum class SwapMode { kElement, kSpliceGrow, kSpliceShrink };

Stmt LongSeq(int length, int density, SwapMode mode) {
  ffi::Array<Stmt> body;
  body.reserve(length);
  for (int i = 0; i < length; ++i) {
    // `i + 2`: a multiplier of one would constant-fold the Mul away.
    body.push_back(Evaluate(Outer() * (i + 2) + Inner()));
  }
  SwapTable()->clear();
  SpliceTable()->clear();
  for (const SwapPair& pair : SwapPairs(length, density)) {
    Stmt lo = body[pair.lo], hi = body[pair.hi];
    if (mode == SwapMode::kElement) {
      // Holds a handle to each swapped element, which is harmless here because the arm
      // replaces elements rather than mutating them.
      (*SwapTable())[lo.get()] = hi;
      (*SwapTable())[hi.get()] = lo;
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
// The Call form: every binary node becomes a CallNode plus the two-element ArrayObj holding
// its args. `ty` is a PrimType and `ty_args` is empty, so neither is a node the traversal
// reaches -- that is what the two skip guards do.
constexpr int64_t kCallBytes = sizeof(CallNode);
constexpr int64_t kArgsBytes = sizeof(ffi::ArrayObj) + 2 * static_cast<int64_t>(sizeof(ffi::Any));
constexpr int64_t kCallSplitFuseSharedBytes =
    2 * kVarBytes + 4 * kImmBytes + 6 * (kCallBytes + kArgsBytes);
constexpr int64_t kCallSplitFuseDistinctBytes =
    kCallSplitFuseSharedBytes + kImmBytes + 2 * (kCallBytes + kArgsBytes);
// Two Vars (Outer, Inner) are shared across every element; 20275's SeqStmtVisit visits the
// Array itself, so it is a node too and kSeqBytes covers it.
constexpr int64_t kSeqWorkingSet(int64_t l) {
  return l * (kMulBytes + kImmBytes + kAddBytes + kEvalBytes) + 2 * kVarBytes + kSeqBytes(l);
}

constexpr int kSeqLengths[] = {16, 256, 16384};
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
  // Bound to `Stmt`, narrowed here.  A callback bound to `Evaluate` would let the engine's
  // link test reject `SeqStmt` before the callback ran; binding to the base type fires on
  // every statement and narrows in user code, which is how a Stmt-level pass is actually
  // written.  It also gives this arm and `map_identity_stmt` the same link, so the difference
  // between them is the rebuild and not how many nodes the link accepted.
  if (stmt.as<EvaluateNode>() == nullptr) return ffi::Any(stmt);
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


ffi::Optional<Expr> SwapVarsFn(const Var& var) {
  Expr mapped = SwapAllVars(var).cast<Expr>();
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

/*!
 * \brief Map arms take the input and return the output. Ownership lives at the call site.
 *
 * `retained` hands over a second handle while the caller keeps one, so the root's refcount is
 * at least two and the engine takes the copy-on-write path. `moved` hands over the sole
 * reference, so it takes the in-place path. Neither arm destroys anything: the driver retains
 * every output in a buffer and frees them all after the clock is read.
 */
// The root call goes through the hook file, not directly, because the minimal mutator's result
// type is the state's own: `Expected<Any>` where the ABI carries a value, `Expected<UnchangedOr
// <Any>>` where it can also carry `unchanged`. One `MinimalMutateRoot` per state, beside that
// state's hooks, keeps the conditional out of here and out of every hook body.
ffi::Any MapFloor(ffi::Any input, ArmKind, Ownership ownership) {
  MinimalMutatorObj mutator;
  ffi::Expected<ffi::Any> result =
      tvm_hooks::MinimalMutateRoot(&mutator, input, ownership == Ownership::kMoved);
  g_sink += result.is_err();
  return result.value();
}

ffi::Any MapNever(ffi::Any input, ArmKind, Ownership) {
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
      std::move(input), [](const FloatImm& v) { return ffi::Any(v); });
}

// The ladder has two mutating operations, and each one is named after what it does.
//
// `map_subst` is Expr-level: match `Var`, substitute through the remap.  That is exactly what
// `Substitute` and `FunctorSubstitute` do, so `map_functor` and `map_old` are baselines for it
// and for no other arm.  It runs on every fixture.
//
// `map_swap` is Stmt-level: match `Stmt`, swap two whole `Evaluate` nodes.  It is what
// exercises the SeqStmt hook's element in-place and splice paths, and it has no functor
// baseline -- `StmtExprMutator` hooks `VisitExpr_(const VarNode*)` and does different work on
// the same graph.  It runs on the seq fixtures only; split/fuse is an Expr tree with no Stmt
// nodes.
//
// Both are the same engine, `StructuralMap`, reached with a different callback.  They were
// both called `replace` and told apart only by a `_var` / `_stmt` suffix, which is how an
// earlier report came to print the Stmt-level one beside `functor`/`old` -- two operations
// read as one comparison.  One name, one operation.
ffi::Any MapIdentityVar(ffi::Any input, ArmKind, Ownership) {
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
      std::move(input), [](const Var& var) { return ffi::Any(var); });
}

ffi::Any MapSubst(ffi::Any input, ArmKind, Ownership) {
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(input), SwapAllVars);
}

ffi::Any MapIdentityStmt(ffi::Any input, ArmKind, Ownership) {
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
      std::move(input), [](const Stmt& stmt) { return ffi::Any(stmt); });
}

ffi::Any MapSwap(ffi::Any input, ArmKind, Ownership) {
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(input), SwapEvaluates);
}

/*! \brief The arm a fixture declares, used by the checks and the density sweep. */
ffi::Any MapFixtureArm(ffi::Any input, ArmKind kind, Ownership ownership) {
  return kind == ArmKind::kSwap ? MapSwap(std::move(input), kind, ownership)
                                : MapSubst(std::move(input), kind, ownership);
}

/*! \brief seq only: the only arm that reaches the hook's splice path. */
ffi::Any MapSplice(ffi::Any input, ArmKind, Ownership) {
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(input), SpliceElements);
}

ffi::Any MapFunctor(ffi::Any input, ArmKind, Ownership) {
  FunctorSubstitute mutator([](const Var& var) { return SwapVarsFn(var); });
  if (auto stmt = input.as<Stmt>()) return ffi::Any(mutator(*stmt));
  return ffi::Any(mutator(input.cast<Expr>()));
}

ffi::Any MapOld(ffi::Any input, ArmKind, Ownership) {
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
  ffi::Any (*run)(ffi::Any, ArmKind, Ownership);
  /*! rief Whether the arm changes the graph, which is what makes it consume a shared DAG. */
  bool rebuilds;
};
// Each fixture family carries the one operation it is shaped for, and its arms are the arms
// that operation has baselines for.
//
// split/fuse runs the Expr-level ladder: `Var` substitution, which is exactly what
// `Substitute` and `FunctorSubstitute` do, so `map_functor` and `map_old` are baselines here
// and every column is the same operation.
constexpr MapArm kExprMapArms[] = {
    {"map_floor", &MapFloor, false},
    {"map_never", &MapNever, false},
    {"map_identity_var", &MapIdentityVar, false},
    {"map_subst", &MapSubst, true},
    {"map_functor", &MapFunctor, true},
    {"map_old", &MapOld, true},
};
// seq runs the Stmt-level swap at scale. `StmtExprMutator` hooks `VisitExpr_(const VarNode*)`
// and does different work on the same graph, so there is no functor or shipping baseline for
// this operation -- and rather than a column of `n/a`, those two arms are simply not here. A
// baseline that cannot perform the operation is a column that belongs on another table.
constexpr MapArm kSeqMapArms[] = {
    {"map_floor", &MapFloor, false},
    {"map_never", &MapNever, false},
    {"map_identity_stmt", &MapIdentityStmt, false},
    {"map_swap", &MapSwap, true},
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
    // `floor` too, and not only the mutating arm. `floor` under `moved` pays a uniqueness
    // check at every node it descends through and rebuilds nothing, so if that check never
    // succeeded the cost would be real but the arm would be measuring a path no caller hits.
    ffi::Any root = build();
    const ffi::Object* before = root.cast<ffi::ObjectRef>().get();
    ffi::Any out = MapFloor(std::move(root), info.arm_kind, Ownership::kMoved);
    if (out.cast<ffi::ObjectRef>().get() != before) {
      Fail(tag + "/floor/moved: the in-place path did not fire");
    }
  }
  {
    ffi::Any root = build();
    const ffi::Object* before = root.cast<ffi::ObjectRef>().get();
    ffi::Any out = MapFixtureArm(std::move(root), info.arm_kind, Ownership::kMoved);
    if (out.cast<ffi::ObjectRef>().get() != before) Fail(tag + "/moved: root was not in place");
  }
  {
    // Retained keeps a second reference, so the input must be copied rather than mutated: the
    // opposite direction, pinned so the check catches in-place firing where it should not.
    ffi::Any root = build();
    const ffi::Object* before = root.cast<ffi::ObjectRef>().get();
    ffi::Any out = MapFixtureArm(ffi::Any(root), info.arm_kind, Ownership::kRetained);
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

/*!
 * \brief a1031a2177's normalization: what shape a result of \p expected statements must have.
 *
 * Zero statements is `Evaluate(0)`, one is that statement unwrapped, and anything else is a
 * `SeqStmt` of that length. Before a1031a2177 every result was a `SeqStmt`, so this is the
 * check the previous twelve cases could not make.
 */
bool CheckSeqShape(const std::string& tag, const ffi::Any& result, int64_t expected) {
  const ffi::ObjectRef ref = result.cast<ffi::ObjectRef>();
  const auto* seq = ref.as<SeqStmtNode>();
  if (expected == 0) {
    const auto* eval = ref.as<EvaluateNode>();
    const auto* imm = eval == nullptr ? nullptr : eval->value.as<IntImmNode>();
    if (imm == nullptr || imm->value != 0) {
      Fail(tag + ": an emptied sequence must normalize to Evaluate(0)");
      return false;
    }
    return true;
  }
  if (expected == 1) {
    if (seq != nullptr) {
      Fail(tag + ": a sequence of one must normalize to the element, not a length-one SeqStmt");
      return false;
    }
    return true;
  }
  if (seq == nullptr || static_cast<int64_t>(seq->seq.size()) != expected) {
    Fail(tag + ": expected a SeqStmt of " + std::to_string(expected) + " statements");
    return false;
  }
  return true;
}

/*!
 * \brief The deliberate asymmetry in a1031a2177's no-op dropping, pinned.
 *
 * `IsSeqStmtNoOp` drops `Evaluate(0)` from the output, but only from the first changed element
 * onward: the lead loop returns `self` untouched when nothing changed, and the rebuild path
 * copies the prefix before that point with `InitRange` rather than replaying it. So the same
 * `Evaluate(0)` survives or is dropped depending on where it sits relative to a change
 * somewhere else in the sequence, and on whether there is a change at all.
 *
 * This reads as a bug and is not one: dropping no-ops on an unchanged pass would rewrite every
 * sequence that contains one and destroy the `same_as` fast path, and replaying the prefix to
 * drop them would give up the `InitRange` copy. It is untested in TVM's own tests, so it is
 * pinned here.
 *
 * It also bounds the differential test: the two paths agree on a changed pass, which is what
 * `CheckSpliceAgainstReference` asserts, and on an unchanged pass both return the input.
 */
void CheckNoOpAsymmetry() {
  // [no-op, e0, e1, e2, no-op, e3] -- one no-op before the element that changes and one after.
  auto build = [&](bool splice) {
    ffi::Array<Stmt> body;
    ffi::Array<Stmt> real;
    for (int i = 0; i < 4; ++i) real.push_back(Evaluate(Outer() * (i + 2) + Inner()));
    body.push_back(Evaluate(0));
    body.push_back(real[0]);
    body.push_back(real[1]);
    body.push_back(real[2]);
    body.push_back(Evaluate(0));
    body.push_back(real[3]);
    SpliceTable()->clear();
    if (splice) (*SpliceTable())[real[2].get()] = MakeNestedSeq({real[2], real[2]});
    return SeqStmt(body);
  };
  auto noops = [](const SeqStmtNode* seq) {
    int64_t n = 0;
    for (const Stmt& stmt : seq->seq) {
      const auto* eval = stmt.as<EvaluateNode>();
      const auto* imm = eval == nullptr ? nullptr : eval->value.as<IntImmNode>();
      if (imm != nullptr && imm->value == 0) ++n;
    }
    return n;
  };
  {  // Nothing changes: the node itself comes back and both no-ops survive.
    ffi::Any root = ffi::Any(build(false));
    const ffi::Object* before = root.cast<ffi::ObjectRef>().get();
    ffi::Any out =
        ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(root), SpliceElements);
    const auto* seq = out.cast<ffi::ObjectRef>().as<SeqStmtNode>();
    if (out.cast<ffi::ObjectRef>().get() != before) {
      Fail("noop/unchanged: an unchanged pass must return the input node");
    }
    if (seq == nullptr || noops(seq) != 2) {
      Fail("noop/unchanged: both Evaluate(0)s must survive a pass that changes nothing");
    }
  }
  {  // Element 3 splices into two: the no-op after it is dropped, the one before survives.
    ffi::Any input = ffi::Any(build(true));
    ffi::Any out = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(ffi::Any(input),
                                                                 SpliceElements);
    const auto* seq = out.cast<ffi::ObjectRef>().as<SeqStmtNode>();
    // [no-op, e0, e1, e2, e2, e3]: the prefix is copied verbatim, the suffix is rebuilt.
    if (seq == nullptr || static_cast<int64_t>(seq->seq.size()) != 6 || noops(seq) != 1) {
      Fail("noop/changed: the Evaluate(0) before the first change must survive and the one "
           "after it must be dropped");
    }
    if (noops(static_cast<const SeqStmtNode*>(input.cast<ffi::ObjectRef>().get())) != 2) {
      Fail("noop/changed: the input must be unchanged under retained ownership");
    }
  }
}

void CheckSpliceAgainstReference() {
  struct Case {
    const char* name;
    int length;
    std::vector<int> targets;  // element positions that map to a nested SeqStmt
    // Statements each target expands into: 0 shrinks, >1 grows. A nested SeqStmt of length 1
    // is not expressible through the public constructor, so MakeNestedSeq builds it directly.
    int width;
  };
  const std::vector<Case> cases = {
      {"grow/first", 8, {0}, 4},           {"grow/last", 8, {7}, 4},
      {"grow/middle", 8, {3}, 4},          {"grow/several", 8, {1, 3, 5}, 3},
      {"grow/adjacent", 8, {2, 3}, 2},     {"keep/one", 8, {4}, 1},
      {"shrink/first", 8, {0}, 0},         {"shrink/last", 8, {7}, 0},
      {"shrink/several", 8, {0, 2, 4}, 0}, {"grow/wide", 4, {0, 1, 2, 3}, 6},
      // a1031a2177 normalizes the result, so these three pin the shapes that opens up: a
      // sequence that ends at one element returns that element unwrapped rather than a
      // length-one SeqStmt, and one that ends at zero returns Evaluate(0).
      {"normalize/to-one", 2, {0}, 0},     {"normalize/to-zero", 2, {0, 1}, 0},
      {"normalize/all-gone", 5, {0, 1, 2, 3, 4}, 0},
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
    // Non-vacuous: the reference itself must have the shape the case describes, or agreement
    // between the two paths would prove nothing.
    int64_t expected = c.length;
    for (size_t k = 0; k < c.targets.size(); ++k) expected += c.width - 1;
    if (!CheckSeqShape(std::string("splice/") + c.name + "/reference", reference, expected)) {
      return;
    }
    CheckSeqShape(std::string("splice/") + c.name + "/inplace", inplace, expected);
    if (!ffi::StructuralEqual()(reference, inplace)) {
      Fail(std::string("splice/") + c.name + ": in-place output differs from MutateSeqStmtRaw");
    }
  }
  CheckNoOpAsymmetry();
}

/*!
 * \brief The builder declares what it built; this confirms the declaration, untimed.
 *
 * The counts stay declared rather than measured -- making the binary re-derive them would put
 * counting inside the measurement, and the builder wrote the loop and knows them.  What it
 * cannot do is notice when a fixture is edited and its constants are not, which is a silent
 * error in every `ns/node` in the row.  So the declaration is checked once, before anything is
 * timed, against a walk of the graph it actually built, and the failure prints the numbers to
 * paste in.
 *
 * `rebuilt` is the pointer-identity set difference between input and output, which is what the
 * report says it is.  Under `moved` it is a lower bound: the input is consumed, so a
 * replacement can land on a just-freed address and the comparison undercounts.  The check
 * therefore requires the declared `moved` count to be no greater than the observed one.
 */
void CheckDeclaredCounts(const FixtureInfo& info, ffi::Any (*build)()) {
  const std::string tag = std::string("counts/") + info.name;
  std::unordered_set<const ffi::Object*> unique;
  int64_t occurrences = 0;
  ffi::Any root = build();
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(root, [&](ffi::AnyView v) {
    if (v.type_index() >= ffi::TypeIndex::kTVMFFIStaticObjectBegin) {
      unique.insert(v.cast<ffi::ObjectRef>().get());
      ++occurrences;
    }
    return ffi::WalkResult::Advance();
  });
  auto rebuilt = [&](Ownership ownership) {
    ffi::Any input = build();
    std::unordered_set<const ffi::Object*> before;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(input, [&](ffi::AnyView v) {
      if (v.type_index() >= ffi::TypeIndex::kTVMFFIStaticObjectBegin) {
        before.insert(v.cast<ffi::ObjectRef>().get());
      }
      return ffi::WalkResult::Advance();
    });
    ffi::Any out = ownership == Ownership::kMoved
                       ? MapFixtureArm(std::move(input), info.arm_kind, ownership)
                       : MapFixtureArm(ffi::Any(input), info.arm_kind, ownership);
    int64_t n = 0;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(out, [&](ffi::AnyView v) {
      if (v.type_index() >= ffi::TypeIndex::kTVMFFIStaticObjectBegin &&
          before.count(v.cast<ffi::ObjectRef>().get()) == 0) {
        ++n;
      }
      return ffi::WalkResult::Advance();
    });
    return n;
  };
  const int64_t observed_unique = static_cast<int64_t>(unique.size());
  const int64_t observed_retained = rebuilt(Ownership::kRetained);
  const int64_t observed_moved = rebuilt(Ownership::kMoved);
  if (observed_unique != info.unique_nodes || occurrences != info.occurrences ||
      observed_retained != info.rebuilt_retained || observed_moved < info.rebuilt_moved) {
    Fail(tag + ": declared unique=" + std::to_string(info.unique_nodes) + " occurrences=" +
         std::to_string(info.occurrences) + " rebuilt_retained=" +
         std::to_string(info.rebuilt_retained) + " rebuilt_moved=" +
         std::to_string(info.rebuilt_moved) + "; walked unique=" +
         std::to_string(observed_unique) + " occurrences=" + std::to_string(occurrences) +
         " rebuilt_retained=" + std::to_string(observed_retained) + " rebuilt_moved=" +
         std::to_string(observed_moved));
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
double MeasureMapArm(const char* arm, ffi::Any (*run)(ffi::Any, ArmKind, Ownership),
                     Ownership ownership, ArmKind kind, int repeats, ffi::Any (*build)(),
                     bool pooled = false) {
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
    DrainResultSink(arm);
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

void RunFixture(const FixtureInfo& info, ffi::Any (*build)(), const MapArm* arms,
                size_t arm_count) {
  EmitFixture(kHarness, info);

  for (const WalkArm& arm : kWalkArms) {
    // Walk arms own nothing and build nothing, so there is no result to retain.
    ffi::Any root = build();
    double ns = MeasureStationary(info.repeats, [&] { arm.run(root); }, [] {});
    EmitResult(kHarness, info.name, "-", arm.name, ns);
  }

  for (Ownership ownership : {Ownership::kRetained, Ownership::kMoved}) {
    for (size_t i = 0; i < arm_count; ++i) {
      const MapArm& arm = arms[i];
      double ns = MeasureMapArm(arm.name, arm.run, ownership, info.arm_kind, info.repeats, build,
                                arm.rebuilds && info.has_sharing);
      EmitResult(kHarness, info.name, OwnershipName(ownership), arm.name, ns);
    }
  }
}

/*! \brief Every object in a post-order walk of \p root, by raw pointer, in walk order. */
std::vector<const ffi::Object*> NodePointers(ffi::AnyView root) {
  std::vector<const ffi::Object*> pointers;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(root, [&](ffi::AnyView v) {
    if (const ffi::Object* obj = v.as<ffi::Object>()) pointers.push_back(obj);
    return ffi::WalkResult::Advance();
  });
  return pointers;
}

/*!
 * \brief Confirm an identity arm rebuilds nothing at all.
 *
 * On an identity arm every node is unchanged, so a hook set that short-circuits before
 * constructing anything leaves every node in the output the same object it came from, root
 * included, under both ownerships.  A hook that computes the changed result and only then
 * discovers nothing changed still returns the right answer and still passes every other check
 * here; this is what catches it.
 *
 * Raw `const Object*` throughout, never an `ObjectRef`: a retained handle is itself a
 * reference, would make `unique()` false, and would suppress the in-place path being checked.
 * Same discipline and the same machinery as `CheckInplace`.
 *
 * What it does not see: a hook that constructs a node and then discards it leaves the output
 * pointer-identical, so this does not catch that direction on its own.
 */
void CheckIdentityPointers(const FixtureInfo& info, ffi::Any (*build)()) {
  struct IdentityArm {
    const char* name;
    ffi::Any (*run)(ffi::Any, ArmKind, Ownership);
  };
  const IdentityArm arms[] = {{"map_identity_var", &MapIdentityVar},
                              {"map_identity_stmt", &MapIdentityStmt}};
  for (const IdentityArm& arm : arms) {
    for (Ownership ownership : {Ownership::kRetained, Ownership::kMoved}) {
      const std::string tag = std::string("identity/") + info.name + "/" + arm.name + "/" +
                              OwnershipName(ownership);
      ffi::Any root = build();
      const std::vector<const ffi::Object*> before = NodePointers(root);
      const ffi::Object* before_root = root.cast<ffi::ObjectRef>().get();
      ffi::Any out = arm.run(ownership == Ownership::kMoved ? std::move(root) : ffi::Any(root),
                             info.arm_kind, ownership);
      if (out.cast<ffi::ObjectRef>().get() != before_root) {
        Fail(tag + ": the output root is a different object, so the traversal rebuilt a node "
                   "on which every child reported unchanged");
      }
      const std::vector<const ffi::Object*> after = NodePointers(out);
      if (after.size() != before.size()) {
        Fail(tag + ": the output has " + std::to_string(after.size()) + " nodes against the "
                   "input's " + std::to_string(before.size()));
      }
      for (size_t i = 0; i < before.size(); ++i) {
        if (before[i] == after[i]) continue;
        Fail(tag + ": node " + std::to_string(i) + " of " + std::to_string(before.size()) +
             " (`" + ffi::TypeIndexToTypeKey(after[i]->type_index()) +
             "`) is a different object in the output, so its hook rebuilt it on a traversal "
             "where every node is unchanged");
      }
    }
  }
}

/*! \brief Everything a fixture needs before it is timed. */
void PrepareFixture(const FixtureInfo& info, ffi::Any (*build)(), const char* coverage_tag) {
  CheckHookCoverage(build, coverage_tag);
  CheckDeclaredCounts(info, build);
  CheckInplace(info, build);
  CheckIdentityPointers(info, build);
}

}  // namespace real_tvm

int main() {
  using namespace real_tvm;  // NOLINT(build/namespaces)
  tvm_hooks::InstallAll();

  EmitStandardProvenance(kHarness);
  EmitProvenance("structural_hooks",
                 "harness (" TVM_FFI_BENCH_HOOK_HEADER "), ported from apache/tvm#20275 "
                 "a1031a2177, installed over TVM's");

  CheckSpliceAgainstReference();

  // split/fuse substitutes every Var, with the remap, which is the semantics Substitute
  // provides and the thing worth measuring on a small Expr tree.  Under moved ownership the
  // only new identity is the substituted-in Var, except on the shared fixture, where the first
  // parent to reach the shared subtree cannot mutate it in place and copies it and its own
  // changed child.  Each shape is built twice: once from direct Add/Mul/FloorDiv/FloorMod
  // nodes and once from Calls, so the two rows differ only in node representation.
  struct ExprFixture {
    FixtureInfo info;
    ffi::Any (*build)();
  };
  const ExprFixture expr_fixtures[] = {
      {{"split-fuse-shared", 12, 17, kSplitFuseSharedBytes, 10, 4, kSplitFuseRepeats, true,
        ArmKind::kSubst},
       [] { return ffi::Any(SplitFuse(true)); }},
      {{"split-fuse-distinct", 15, 17, kSplitFuseDistinctBytes, 10, 2, kSplitFuseRepeats, false,
        ArmKind::kSubst},
       [] { return ffi::Any(SplitFuse(false)); }},
      {{"call-split-fuse-shared", 18, 25, kCallSplitFuseSharedBytes, 18, 6, kSplitFuseRepeats,
        true, ArmKind::kSubst},
       [] { return ffi::Any(CallSplitFuse(true)); }},
      {{"call-split-fuse-distinct", 23, 25, kCallSplitFuseDistinctBytes, 18, 2,
        kSplitFuseRepeats, false, ArmKind::kSubst},
       [] { return ffi::Any(CallSplitFuse(false)); }},
  };
  for (const ExprFixture& f : expr_fixtures) {
    PrepareFixture(f.info, f.build, f.info.name);
    RunFixture(f.info, f.build, kExprMapArms, sizeof(kExprMapArms) / sizeof(kExprMapArms[0]));
  }

  // seq swaps two Evaluate nodes: no remap is involved, and the change count is exactly two
  // regardless of L, so the update stays sparse as the body grows.
  for (int length : kSeqLengths) {
    static int current_length = 0;
    current_length = length;
    auto build = [] { return ffi::Any(LongSeq(current_length, 1, SwapMode::kElement)); };
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
                     ArmKind::kSwap};
    PrepareFixture(info, build, "seq");
    RunFixture(info, build, kSeqMapArms, sizeof(kSeqMapArms) / sizeof(kSeqMapArms[0]));
  }

  Emit("#sink\t" + std::to_string(g_sink));
  return 0;
}
