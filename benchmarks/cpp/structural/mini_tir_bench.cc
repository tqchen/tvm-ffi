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

// mini-TIR half of the structural benchmark: fixtures, arms, main.
//
// The node types and their hooks are in `mini_tir.h`; the timing loops in `timer.h`.  This
// file builds the fixtures, declares what it built, runs one function per arm and prints one
// line per measurement.  It discovers nothing about itself at runtime.

#include "mini_tir.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace mini_tir {

using namespace tvm::ffi;         // NOLINT(build/namespaces)
using namespace tvm::ffi::mini;   // NOLINT(build/namespaces)
using namespace tvm::ffi::bench;  // NOLINT(build/namespaces)

constexpr const char* kHarness = "mini-tir";

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

Any& Ty() {
  static Any ty = Any(HPrimType(make_object<HPrimTypeObj>(32)));
  return ty;
}
HVar& Outer() {
  static HVar v(make_object<HVarObj>(Ty(), 0));
  return v;
}
HVar& Inner() {
  static HVar v(make_object<HVarObj>(Ty(), 1));
  return v;
}
HVar& Replacement() {
  static HVar v(make_object<HVarObj>(Ty(), 2));
  return v;
}

template <typename T>
HPrimExpr Bin(HPrimExpr a, HPrimExpr b) {
  return HPrimExpr(make_object<T>(Ty(), std::move(a), std::move(b)));
}
HPrimExpr Imm(int64_t v) { return HPrimExpr(make_object<HIntImmObj>(Ty(), v)); }

/*!
 * \brief `floordiv(o*16+i, 32)*32 + floormod(o*16+i, 32)` -- the split/fuse index expression.
 *
 * `shared == true` reuses one pointer for both occurrences of `o*16+i`, so the fixture is a
 * DAG; `false` builds two structurally equal, pointer-distinct copies.  Twelve unique nodes
 * shared, fifteen distinct, seventeen occurrences either way.
 */
HPrimExpr SplitFuse(bool shared) {
  HPrimExpr q = Bin<HAddObj>(Bin<HMulObj>(Outer(), Imm(16)), Inner());
  HPrimExpr r = shared ? q : Bin<HAddObj>(Bin<HMulObj>(Outer(), Imm(16)), Inner());
  return Bin<HAddObj>(Bin<HMulObj>(Bin<HFloorDivObj>(q, Imm(32)), Imm(32)),
                      Bin<HFloorModObj>(r, Imm(32)));
}

/*!
 * \brief `SeqStmt` of `L` statements `Evaluate(v * (i+2) + inner)`.
 *
 * Four unique nodes per element -- Mul, IntImm, Add, Evaluate -- plus a shared preamble of
 * the SeqStmt and three Vars, so `N = 4L + 4`; six occurrences per element plus the SeqStmt,
 * so `occurrences = 6L + 1`.  `i + 2` because a multiplier of one would fold the Mul away.
 */
/*! \brief The `Evaluate` elements a `seq` replacement swaps, by position. See real_tvm_bench. */
struct SwapPair {
  int lo, hi;
};
std::vector<SwapPair> SwapPairs(int length, int density) {
  std::vector<SwapPair> pairs;
  for (int k = 0; k < density; ++k) pairs.push_back({length / 4 + k, 3 * length / 4 - k});
  return pairs;
}
std::unordered_map<const Object*, HStmt>* SwapTable() {
  static std::unordered_map<const Object*, HStmt> table;
  return &table;
}

HStmt LongSeq(int length, int density) {
  Array<HStmt> body;
  body.reserve(length);
  for (int i = 0; i < length; ++i) {
    body.push_back(
        HStmt(make_object<HEvaluateObj>(Bin<HAddObj>(Bin<HMulObj>(Outer(), Imm(i + 2)), Inner()))));
  }
  SwapTable()->clear();
  for (const SwapPair& pair : SwapPairs(length, density)) {
    HStmt lo = body[pair.lo], hi = body[pair.hi];
    (*SwapTable())[lo.get()] = hi;
    (*SwapTable())[hi.get()] = lo;
  }
  return HStmt(make_object<HSeqStmtObj>(body));
}

// Working sets, as sizeof sums over what the builders above construct.
constexpr int64_t kVarBytes = sizeof(HVarObj);
constexpr int64_t kImmBytes = sizeof(HIntImmObj);
constexpr int64_t kAddBytes = sizeof(HAddObj);
constexpr int64_t kMulBytes = sizeof(HMulObj);
constexpr int64_t kEvalBytes = sizeof(HEvaluateObj);
// A SeqStmt of L elements owns an Array whose payload is inline.
constexpr int64_t kSeqBytes(int64_t l) {
  return sizeof(HSeqStmtObj) + sizeof(ArrayObj) + l * static_cast<int64_t>(sizeof(Any));
}
// split/fuse: 2 Var, 4 IntImm, 2 Mul, 2 Add, 1 FloorDiv, 1 FloorMod (shared); the distinct
// variant adds one more Mul, IntImm and Add.
constexpr int64_t kSplitFuseSharedBytes =
    2 * kVarBytes + 4 * kImmBytes + 2 * kMulBytes + 2 * kAddBytes +
    static_cast<int64_t>(sizeof(HFloorDivObj)) + static_cast<int64_t>(sizeof(HFloorModObj));
constexpr int64_t kSplitFuseDistinctBytes =
    kSplitFuseSharedBytes + kMulBytes + kImmBytes + kAddBytes;
constexpr int64_t kSeqWorkingSet(int64_t l) {
  return l * (kMulBytes + kImmBytes + kAddBytes + kEvalBytes) + 2 * kVarBytes + kSeqBytes(l);
}

/*!
 * \brief The three `seq` lengths, one per cache level on the benchmark machine.
 *
 * 16 lands inside a 32 KiB L1d, 256 inside a 1 MiB L2, 16384 inside a 32 MiB L3.  The report
 * states the machine's geometry and names the level; the binary does not probe for it.
 */
constexpr int kSeqLengths[] = {16, 256, 16384};

/*!
 * \brief Traversals per timed sample, per fixture.
 *
 * Chosen so a sample lands in the milliseconds, far above the two clock reads bracketing it,
 * without making the large fixtures take minutes.  Plain constants rather than a calibration
 * loop: #367 ran a fixed 20,000 traversals and this harness reproduces its numbers.
 */
constexpr int kSplitFuseRepeats = 20000;
constexpr int SeqRepeats(int length) {
  return length == 16 ? 5000 : (length == 256 ? 500 : 10);
}

// ---------------------------------------------------------------------------
// Arms.  One plain function each; the tables below are what main() iterates.
// ---------------------------------------------------------------------------

size_t g_sink = 0;

// The replacement callbacks swap rather than replace one-way, so repeating an arm in place is
// stationary and the timed loop never drifts into a different workload.
Any SwapAllVars(const HVar& var) {
  if (var->id == 0) return Any(Replacement());
  if (var->id == 2) return Any(Outer());
  return Any(var);
}
/*! \brief The seq replacement: swap two `Evaluate` nodes; no remap is involved. */
Any SwapEvaluates(const HStmt& stmt) {
  auto it = SwapTable()->find(stmt.get());
  if (it != SwapTable()->end()) return Any(it->second);
  return Any(stmt);
}
Optional<HPrimExpr> SwapVarsFn(const HVar& var) {
  HPrimExpr mapped = SwapAllVars(var).cast<HPrimExpr>();
  if (mapped.same_as(HPrimExpr(var))) return std::nullopt;
  return Optional<HPrimExpr>(mapped);
}

void WalkFloor(AnyView root) {
  MinimalVisitorObj visitor;
  g_sink += visitor.VisitExpected(root).is_err();
}

void Walk(AnyView root) {
  size_t matched = 0;
  StructuralWalk<WalkOrder::kPostOrder>(
      root,
      [&](const HVar&) -> Expected<WalkResult> {
        ++matched;
        return WalkResult::Advance();
      },
      [&](const HExpr&) -> Expected<WalkResult> {
        ++matched;
        return WalkResult::Advance();
      });
  g_sink += matched;
}

/*! \brief The same shape whose first link can never match: prices link testing alone. */
void WalkNever(AnyView root) {
  size_t matched = 0;
  StructuralWalk<WalkOrder::kPostOrder>(
      root,
      [&](const HPrimType&) -> Expected<WalkResult> {
        ++matched;
        return WalkResult::Advance();
      },
      [&](const HExpr&) -> Expected<WalkResult> {
        ++matched;
        return WalkResult::Advance();
      });
  g_sink += matched;
}

/*! \brief The functor-era PostOrderVisit: ExprFunctor/StmtFunctor vtable + IRApplyVisit. */
void WalkFunctor(AnyView root) {
  size_t matched = 0;
  FunctorPostOrderVisit(root.cast<ObjectRef>(),
                        [&](const ObjectRef& node) { matched += node.as<HVarObj>() != nullptr; });
  g_sink += matched;
}

/*! \brief What the pinned TVM ships: StructuralWalk plus a dedup set. */
void WalkOld(AnyView root) {
  size_t matched = 0;
  ShippingPostOrderVisit(
      root, [&](const ObjectRef& node) { matched += node.as<HVarObj>() != nullptr; });
  g_sink += matched;
}

/*!
 * \brief Take the root the way this ownership variant means.
 *
 * `retained` keeps the caller's handle alive, so the root's refcount is at least two and the
 * engine takes the copy-on-write path.  `moved` hands the sole reference over, so the engine
 * takes the in-place path; the result goes back into the slot, which keeps the next
 * repetition uniquely owned and the working set fixed at one fixture.  The difference between
 * a whole column of this report and the next is this one `std::move`.
 */
Any Take(Ownership ownership, Any* slot) {
  return ownership == Ownership::kMoved ? std::move(*slot) : Any(*slot);
}
void Give(Ownership ownership, Any* slot, Any result) {
  if (ownership == Ownership::kMoved) {
    *slot = std::move(result);
  } else {
    g_sink += result.type_index();
  }
}

void MapFloor(Ownership ownership, ReplaceKind, Any* slot) {
  MinimalMutatorObj mutator;
  Any input = Take(ownership, slot);
  Expected<Any> result = ownership == Ownership::kMoved
                             ? mutator.MaybeInplaceMutateIfUniqueExpected(input)
                             : mutator.MutateExpected(input);
  g_sink += result.is_err();
  Give(ownership, slot, result.value());
}

void MapNever(Ownership ownership, ReplaceKind, Any* slot) {
  Give(ownership, slot,
       StructuralMap<WalkOrder::kPostOrder>(Take(ownership, slot),
                                            [](const HPrimType& v) { return Any(v); }));
}

void MapIdentity(Ownership ownership, ReplaceKind kind, Any* slot) {
  if (kind == ReplaceKind::kSingleVar) {
    Give(ownership, slot,
         StructuralMap<WalkOrder::kPostOrder>(Take(ownership, slot),
                                              [](const HStmt& stmt) { return Any(stmt); }));
  } else {
    Give(ownership, slot,
         StructuralMap<WalkOrder::kPostOrder>(Take(ownership, slot),
                                              [](const HVar& var) { return Any(var); }));
  }
}

void MapReplace(Ownership ownership, ReplaceKind kind, Any* slot) {
  if (kind == ReplaceKind::kSingleVar) {
    Give(ownership, slot,
         StructuralMap<WalkOrder::kPostOrder>(Take(ownership, slot), SwapEvaluates));
  } else {
    Give(ownership, slot,
         StructuralMap<WalkOrder::kPostOrder>(Take(ownership, slot), SwapAllVars));
  }
}

/*! \brief The functor-era Substitute: ExprMutator/StmtMutator vtable + IRSubstitute. */
/*! \brief The functor-era Substitute: ExprMutator/StmtMutator vtable + IRSubstitute. */
void MapFunctor(Ownership ownership, ReplaceKind, Any* slot) {
  Give(ownership, slot, FunctorSubstitute(Take(ownership, slot), SwapVarsFn));
}

/*! \brief What the pinned TVM ships: a StructuralMutatorObj owning its own var remap. */
/*! \brief What the pinned TVM ships: a StructuralMutatorObj owning its own var remap. */
void MapOld(Ownership ownership, ReplaceKind, Any* slot) {
  Give(ownership, slot, ShippingSubstitute(Take(ownership, slot), SwapVarsFn));
}

struct WalkArm {
  const char* name;
  void (*run)(AnyView);
};
constexpr WalkArm kWalkArms[] = {
    {"walk_floor", &WalkFloor},     {"walk_var", &Walk},    {"walk_never", &WalkNever},
    {"walk_functor", &WalkFunctor}, {"walk_old", &WalkOld},
};

struct MapArm {
  const char* name;
  void (*run)(Ownership, ReplaceKind, Any*);
  /*! \brief Whether the arm changes the graph, which is what makes it consume a shared DAG. */
  bool rebuilds;
};
constexpr MapArm kMapArms[] = {
    {"map_floor", &MapFloor, false},     {"map_never", &MapNever, false},
    {"map_identity", &MapIdentity, false}, {"map_replace", &MapReplace, true},
    {"map_functor", &MapFunctor, true},  {"map_old", &MapOld, true},
};

// ---------------------------------------------------------------------------
// main: build, declare, time, print.
// ---------------------------------------------------------------------------

void RunFixture(const FixtureInfo& info, Any (*build)()) {
  EmitFixture(kHarness, info);

  for (const WalkArm& arm : kWalkArms) {
    Any root = build();
    double ns = MeasureStationary(info.repeats, [&] { arm.run(root); });
    EmitResult(kHarness, info.name, "-", arm.name, ns);
  }

  for (Ownership ownership : {Ownership::kRetained, Ownership::kMoved}) {
    for (const MapArm& arm : kMapArms) {
      double ns = 0;
      // An in-place rebuilding arm on a fixture with a pointer-shared subtree un-shares it,
      // so the fixture is not stationary: it runs over a pool of independent copies rebuilt
      // untimed between passes.
      if (ownership == Ownership::kMoved && arm.rebuilds && info.has_sharing) {
        std::vector<Any> pool(kPoolSize);
        int passes = info.repeats / kPoolSize;
        if (passes < 1) passes = 1;
        ns = MeasurePooled(
            passes, [&] { for (Any& slot : pool) slot = build(); },
            [&](int i) { arm.run(ownership, info.replace_kind, &pool[i]); });
      } else {
        Any slot = build();
        ns = MeasureStationary(info.repeats,
                               [&] { arm.run(ownership, info.replace_kind, &slot); });
      }
      EmitResult(kHarness, info.name, OwnershipName(ownership), arm.name, ns);
    }
  }
}

}  // namespace mini_tir

int main() {
  using namespace mini_tir;  // NOLINT(build/namespaces)
  EmitStandardProvenance(kHarness);
  EmitProvenance("structural_hooks",
                 "mini-TIR's own (mini_tir.h), shaped after apache/tvm#20275 b51da96381");
  EmitProvenance("seqstmt_inplace_hook",
                 MINI_SEQSTMT_INPLACE_FIX ? "repaired" : "20275 as shipped");

  auto build_shared = [] { return Any(SplitFuse(true)); };
  auto build_distinct = [] { return Any(SplitFuse(false)); };
  FixtureInfo shared{"split-fuse-shared", 12, 17, kSplitFuseSharedBytes, 9, 3, kSplitFuseRepeats,
                     true, ReplaceKind::kAllVars};
  shared.run_old = true;
  FixtureInfo distinct{"split-fuse-distinct", 15, 17, kSplitFuseDistinctBytes, 9, 1,
                       kSplitFuseRepeats, false, ReplaceKind::kAllVars};
  RunFixture(shared, build_shared);
  RunFixture(distinct, build_distinct);

  // seq swaps two Evaluate nodes, so nothing below the SeqStmt is rebuilt: retained copies the
  // SeqStmt and its Array, moved mutates both in place.
  constexpr int64_t kSeqRebuiltRetained = 2;
  constexpr int64_t kSeqRebuiltMoved = 0;
  for (int length : kSeqLengths) {
    static int current_length = 0;
    current_length = length;
    std::string name = "seq-" + std::to_string(length);
    RunFixture({name.c_str(), 4LL * length + 4, 6LL * length + 2, kSeqWorkingSet(length),
                kSeqRebuiltRetained, kSeqRebuiltMoved, SeqRepeats(length), false,
                ReplaceKind::kSingleVar},
               [] { return Any(LongSeq(current_length, 1)); });
  }

  Emit("#sink\t" + std::to_string(g_sink));
  return 0;
}
