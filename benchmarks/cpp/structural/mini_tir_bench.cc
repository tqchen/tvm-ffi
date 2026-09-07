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
#include <unordered_set>
#include <vector>

namespace mini_tir {

using namespace tvm::ffi;         // NOLINT(build/namespaces)
using namespace tvm::ffi::mini;   // NOLINT(build/namespaces)
using namespace tvm::ffi::bench;  // NOLINT(build/namespaces)

constexpr const char* kHarness = "mini-tir";

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

HPrimType& Ty() {
  static HPrimType ty = HPrimType::Int(32);
  return ty;
}
HVar& Outer() {
  static HVar v(make_object<HVarObj>(Ty(), String("outer")));
  return v;
}
HVar& Inner() {
  static HVar v(make_object<HVarObj>(Ty(), String("inner")));
  return v;
}
HVar& Replacement() {
  static HVar v(make_object<HVarObj>(Ty(), String("replacement")));
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
 * \brief The same split/fuse tree with every arithmetic node expressed as a `Call`.
 *
 * Row-for-row counterpart of `SplitFuse` and of real_tvm_bench.cc's `CallSplitFuse`: identical
 * topology, identical `Var`s, the same six binary operations in the same order, with each
 * expressed as a `Call` to an interned operator instead of a direct node.  The only variable
 * between the two fixtures is the node representation.
 *
 * It is the only fixture that reaches the three skip guards 20275's `Call` hooks carry -- an
 * `Array` field (`args`), an empty `ty_args`, and an interned operator -- and now it exists in
 * both harnesses rather than only in real TVM.  The operator's identity does not matter: the
 * traversal never evaluates the call and the hooks skip an `HOpObj` operator without looking
 * at which one, so four distinct interned names stand in for the four node types, which keeps
 * the `op` field varying exactly as the direct form's node type does.
 */
HPrimExpr CallOp(const HOp& op, HPrimExpr a, HPrimExpr b) {
  Array<HExpr> args;
  args.reserve(2);
  args.push_back(HExpr(std::move(a)));
  args.push_back(HExpr(std::move(b)));
  return HPrimExpr(make_object<HCallObj>(Ty(), HExpr(op), std::move(args)));
}
HPrimExpr CallSplitFuse(bool shared) {
  const HOp& mul = HOp::Get("h.shift_left");
  const HOp& add = HOp::Get("h.bitwise_or");
  const HOp& fdiv = HOp::Get("h.bitwise_and");
  const HOp& fmod = HOp::Get("h.bitwise_xor");
  HPrimExpr q = CallOp(add, CallOp(mul, Outer(), Imm(16)), Inner());
  HPrimExpr r = shared ? q : CallOp(add, CallOp(mul, Outer(), Imm(16)), Inner());
  return CallOp(add, CallOp(mul, CallOp(fdiv, q, Imm(32)), Imm(32)), CallOp(fmod, r, Imm(32)));
}

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

/*!
 * \brief `SeqStmt` of `L` statements `Evaluate(v * (i+2) + inner)`.
 *
 * Four unique nodes per element -- Mul, IntImm, Add, Evaluate -- plus a shared preamble of
 * the SeqStmt and three Vars, so `N = 4L + 4`; six occurrences per element plus the SeqStmt,
 * so `occurrences = 6L + 1`.  `i + 2` because a multiplier of one would fold the Mul away.
 */
HStmt LongSeq(int length, int density) {
  Array<HStmt> body;
  body.reserve(length);
  for (int i = 0; i < length; ++i) {
    body.push_back(HStmt(make_object<HEvaluateObj>(
        HExpr(Bin<HAddObj>(Bin<HMulObj>(Outer(), Imm(i + 2)), Inner())))));
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
// The Call form: every binary node becomes an HCallObj plus the two-element ArrayObj holding
// its args. `ty` is an HPrimType and `ty_args` is empty, so neither is a node the traversal
// reaches -- that is what the two skip guards do.
constexpr int64_t kCallBytes = sizeof(HCallObj);
constexpr int64_t kArgsBytes = sizeof(ArrayObj) + 2 * static_cast<int64_t>(sizeof(Any));
constexpr int64_t kCallSplitFuseSharedBytes =
    2 * kVarBytes + 4 * kImmBytes + 6 * (kCallBytes + kArgsBytes);
constexpr int64_t kCallSplitFuseDistinctBytes =
    kCallSplitFuseSharedBytes + kImmBytes + 2 * (kCallBytes + kArgsBytes);
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
  if (var.same_as(Outer())) return Any(Replacement());
  if (var.same_as(Replacement())) return Any(Outer());
  return Any(var);
}
/*!
 * \brief The seq replacement: swap two `Evaluate` nodes; no remap is involved.
 *
 * Bound to `HStmt` and narrowed here, the same way real_tvm_bench.cc binds and narrows: that
 * is how a Stmt-level pass is written, and it gives this arm and `map_identity_stmt` the same
 * link so the difference between them is the rebuild rather than what the link accepted.
 */
Any SwapEvaluates(const HStmt& stmt) {
  if (stmt.as<HEvaluateObj>() == nullptr) return Any(stmt);
  auto it = SwapTable()->find(stmt.get());
  if (it != SwapTable()->end()) return Any(it->second);
  return Any(stmt);
}
Optional<HExpr> SwapVarsFn(const HVar& var) {
  HExpr mapped = SwapAllVars(var).cast<HExpr>();
  if (mapped.same_as(var)) return std::nullopt;
  return Optional<HExpr>(mapped);
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
      [&](const HFloatImm&) -> Expected<WalkResult> {
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

/*! \brief What the pinned TVM ships: PostOrderVisit, which is IRApplyVisit at this pin. */
void WalkOld(AnyView root) {
  size_t matched = 0;
  ShippingPostOrderVisit(
      root.cast<ObjectRef>(),
      [&](const ObjectRef& node) { matched += node.as<HVarObj>() != nullptr; });
  g_sink += matched;
}

/*!
 * \brief Map arms take the input and return the output. Ownership lives at the call site.
 *
 * `retained` hands over a second handle while the caller keeps one, so the engine takes the
 * copy-on-write path; `moved` hands over the sole reference, so it takes the in-place path.
 * Neither destroys anything: the driver retains every output and frees them after the clock.
 */
Any MapFloor(Any input, ArmKind, Ownership ownership) {
  MinimalMutatorObj mutator;
  Expected<Any> result = ownership == Ownership::kMoved
                             ? mutator.MaybeInplaceMutateIfUniqueExpected(input)
                             : mutator.MutateExpected(input);
  g_sink += result.is_err();
  return result.value();
}

Any MapNever(Any input, ArmKind, Ownership) {
  return StructuralMap<WalkOrder::kPostOrder>(std::move(input),
                                              [](const HFloatImm& v) { return Any(v); });
}

// Two named operations rather than one selected by fixture; see real_tvm_bench.cc for why.
// `*_var` is Expr-level Var substitution, which is what map_functor and map_old do, so those
// two are baselines for these arms and for no others. `*_stmt` swaps whole HEvaluate nodes and
// has no functor baseline. Var arms run everywhere, Stmt arms on the seq fixtures only.
Any MapIdentityVar(Any input, ArmKind, Ownership) {
  return StructuralMap<WalkOrder::kPostOrder>(std::move(input),
                                              [](const HVar& var) { return Any(var); });
}

Any MapSubst(Any input, ArmKind, Ownership) {
  return StructuralMap<WalkOrder::kPostOrder>(std::move(input), SwapAllVars);
}

Any MapIdentityStmt(Any input, ArmKind, Ownership) {
  return StructuralMap<WalkOrder::kPostOrder>(std::move(input),
                                              [](const HStmt& stmt) { return Any(stmt); });
}

Any MapSwap(Any input, ArmKind, Ownership) {
  return StructuralMap<WalkOrder::kPostOrder>(std::move(input), SwapEvaluates);
}

/*! \brief The functor-era Substitute: ExprMutator/StmtMutator vtable + IRSubstitute. */
Any MapFunctor(Any input, ArmKind, Ownership) {
  return FunctorSubstitute(std::move(input), [](const HVar& var) { return SwapVarsFn(var); });
}

/*! \brief What the pinned TVM ships: Substitute, which is IRSubstitute at this pin. */
Any MapOld(Any input, ArmKind, Ownership) {
  return ShippingSubstitute(std::move(input), [](const HVar& var) { return SwapVarsFn(var); });
}

/*!
 * \brief The arm a fixture declares, used by the pre-timing checks.
 *
 * Mirrors real_tvm_bench.cc's `MapFixtureArm`, which is what its own checks run.
 */
Any MapFixtureArm(Any input, ArmKind kind, Ownership ownership) {
  return kind == ArmKind::kSwap ? MapSwap(std::move(input), kind, ownership)
                                : MapSubst(std::move(input), kind, ownership);
}

void Fail(const std::string& what) {
  std::fflush(stdout);
  std::fprintf(stderr, "structural benchmark check failed: %s\n", what.c_str());
  std::exit(2);
}

/*!
 * \brief Confirm the in-place path fires where it should and not where it should not.
 *
 * Ported from real_tvm_bench.cc's `CheckInplace`.  Raw pointers only: holding an `ObjectRef`
 * to the input would itself be a reference, make `unique()` false, suppress the thing being
 * tested, and report the suppression as a result.
 */
void CheckInplace(const FixtureInfo& info, Any (*build)()) {
  const std::string tag = std::string("inplace/") + info.name;
  {
    Any root = build();
    const Object* before = root.cast<ObjectRef>().get();
    Any out = MapFloor(std::move(root), info.arm_kind, Ownership::kMoved);
    if (out.cast<ObjectRef>().get() != before) {
      Fail(tag + "/floor/moved: the in-place path did not fire");
    }
  }
  {
    Any root = build();
    const Object* before = root.cast<ObjectRef>().get();
    Any out = MapFixtureArm(std::move(root), info.arm_kind, Ownership::kMoved);
    if (out.cast<ObjectRef>().get() != before) Fail(tag + "/moved: root was not in place");
  }
  {
    Any root = build();
    const Object* before = root.cast<ObjectRef>().get();
    Any out = MapFixtureArm(Any(root), info.arm_kind, Ownership::kRetained);
    (void)out;
    if (root.cast<ObjectRef>().get() != before) Fail(tag + "/retained: input was mutated");
  }
}

/*!
 * \brief The builder declares what it built; this confirms the declaration, untimed.
 *
 * Ported from real_tvm_bench.cc's `CheckDeclaredCounts` and run for the same reason: the
 * counts stay declared rather than measured, but a fixture edited without its constants is a
 * silent error in every `ns/node` in the row.  It is also what keeps the two harnesses'
 * fixture rows counterparts -- both declare the same numbers, and both are checked against a
 * walk of what they actually built.
 */
void CheckDeclaredCounts(const FixtureInfo& info, Any (*build)()) {
  const std::string tag = std::string("counts/") + info.name;
  std::unordered_set<const Object*> unique;
  int64_t occurrences = 0;
  Any root = build();
  StructuralWalk<WalkOrder::kPostOrder>(root, [&](AnyView v) {
    if (v.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
      unique.insert(v.cast<ObjectRef>().get());
      ++occurrences;
    }
    return WalkResult::Advance();
  });
  auto rebuilt = [&](Ownership ownership) {
    Any input = build();
    std::unordered_set<const Object*> before;
    StructuralWalk<WalkOrder::kPostOrder>(input, [&](AnyView v) {
      if (v.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
        before.insert(v.cast<ObjectRef>().get());
      }
      return WalkResult::Advance();
    });
    Any out = ownership == Ownership::kMoved
                  ? MapFixtureArm(std::move(input), info.arm_kind, ownership)
                  : MapFixtureArm(Any(input), info.arm_kind, ownership);
    int64_t n = 0;
    StructuralWalk<WalkOrder::kPostOrder>(out, [&](AnyView v) {
      if (v.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin &&
          before.count(v.cast<ObjectRef>().get()) == 0) {
        ++n;
      }
      return WalkResult::Advance();
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

/*! \brief Everything a fixture needs before it is timed. */
void PrepareFixture(const FixtureInfo& info, Any (*build)()) {
  CheckDeclaredCounts(info, build);
  CheckInplace(info, build);
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
  Any (*run)(Any, ArmKind, Ownership);
  bool rebuilds;
};
// Same split as real_tvm_bench.cc: each fixture family carries the one operation it is shaped
// for, with the baselines that operation has. See the arm vocabulary in README.md.
constexpr MapArm kExprMapArms[] = {
    {"map_floor", &MapFloor, false},              {"map_never", &MapNever, false},
    {"map_identity_var", &MapIdentityVar, false}, {"map_subst", &MapSubst, true},
    {"map_functor", &MapFunctor, true},           {"map_old", &MapOld, true},
};
constexpr MapArm kSeqMapArms[] = {
    {"map_floor", &MapFloor, false},
    {"map_never", &MapNever, false},
    {"map_identity_stmt", &MapIdentityStmt, false},
    {"map_swap", &MapSwap, true},
};

// ---------------------------------------------------------------------------
// main: build, declare, time, print.
// ---------------------------------------------------------------------------

/*!
 * \brief Time one map arm. The timed region is `batch`, and nothing enters or leaves it
 *        implicitly. See real_tvm_bench.cc for the full statement of the boundary.
 */
double MeasureMapArm(const char* arm, Any (*run)(Any, ArmKind, Ownership),
                     Ownership ownership, ArmKind kind, int repeats, Any (*build)(),
                     bool pooled = false) {
  std::vector<Any>& out = ResultSink();
  Any held;
  std::vector<Any> pool;
  const bool use_pool = pooled && ownership == Ownership::kMoved;
  const int batch_size = use_pool ? kPoolSize : repeats;

  auto setup = [&] {  // untimed: fixtures built, buffer reserved for the whole batch
    ReserveResultSink(static_cast<size_t>(batch_size) + 1);
    if (use_pool) {
      pool.clear();
      pool.reserve(batch_size);
      for (int i = 0; i < batch_size; ++i) pool.push_back(build());
    } else if (ownership == Ownership::kRetained) {
      held = build();
    } else {
      out.push_back(build());
    }
  };
  auto batch = [&] {  // timed: only this
    if (use_pool) {
      for (int i = 0; i < batch_size; ++i) out.push_back(run(std::move(pool[i]), kind, ownership));
    } else if (ownership == Ownership::kRetained) {
      for (int i = 0; i < batch_size; ++i) out.push_back(run(Any(held), kind, ownership));
    } else {
      for (int i = 0; i < batch_size; ++i) out.push_back(run(std::move(out[i]), kind, ownership));
    }
  };
  auto teardown = [&] {  // untimed: every retained output destroyed here
    DrainResultSink(arm);
    held = Any();
    pool.clear();
  };

  setup();
  batch();
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

/*!
 * \brief The layout-parity evidence.
 *
 * `real_tvm_bench.cc` emits the same logical names for its counterparts, and `report.py` fails
 * a run in which any pair disagrees.  This is what keeps "mini-TIR's nodes are apache/tvm's
 * nodes" a checked claim rather than a comment.
 */
void EmitNodeSizes() {
  EmitNodeSize(kHarness, "Span", sizeof(HSpanObj));
  EmitNodeSize(kHarness, "Type", sizeof(HTypeObj));
  EmitNodeSize(kHarness, "PrimType", sizeof(HPrimTypeObj));
  EmitNodeSize(kHarness, "Expr", sizeof(HExprObj));
  EmitNodeSize(kHarness, "Var", sizeof(HVarObj));
  EmitNodeSize(kHarness, "IntImm", sizeof(HIntImmObj));
  EmitNodeSize(kHarness, "FloatImm", sizeof(HFloatImmObj));
  EmitNodeSize(kHarness, "Call", sizeof(HCallObj));
  EmitNodeSize(kHarness, "Add", sizeof(HAddObj));
  EmitNodeSize(kHarness, "Mul", sizeof(HMulObj));
  EmitNodeSize(kHarness, "FloorDiv", sizeof(HFloorDivObj));
  EmitNodeSize(kHarness, "FloorMod", sizeof(HFloorModObj));
  EmitNodeSize(kHarness, "Stmt", sizeof(HStmtObj));
  EmitNodeSize(kHarness, "Evaluate", sizeof(HEvaluateObj));
  EmitNodeSize(kHarness, "SeqStmt", sizeof(HSeqStmtObj));
}

void RunFixture(const FixtureInfo& info, Any (*build)()) {
  PrepareFixture(info, build);
  EmitFixture(kHarness, info);

  for (const WalkArm& arm : kWalkArms) {
    Any root = build();
    double ns = MeasureStationary(info.repeats, [&] { arm.run(root); }, [] {});
    EmitResult(kHarness, info.name, "-", arm.name, ns);
  }
  const bool seq = info.arm_kind == ArmKind::kSwap;
  const MapArm* arms = seq ? kSeqMapArms : kExprMapArms;
  const size_t arm_count = seq ? sizeof(kSeqMapArms) / sizeof(kSeqMapArms[0])
                               : sizeof(kExprMapArms) / sizeof(kExprMapArms[0]);
  for (Ownership ownership : {Ownership::kRetained, Ownership::kMoved}) {
    for (size_t i = 0; i < arm_count; ++i) {
      const MapArm& arm = arms[i];
      double ns = MeasureMapArm(arm.name, arm.run, ownership, info.arm_kind, info.repeats, build,
                                arm.rebuilds && info.has_sharing);
      EmitResult(kHarness, info.name, OwnershipName(ownership), arm.name, ns);
    }
  }
}

}  // namespace mini_tir

int main() {
  using namespace mini_tir;  // NOLINT(build/namespaces)
  EmitStandardProvenance(kHarness);
  EmitProvenance("structural_hooks",
                 "mini-TIR's own (mini_tir.h), ported from apache/tvm#20275 a1031a2177 over "
                 "layout-identical node types");
  EmitNodeSizes();

  struct ExprFixture {
    FixtureInfo info;
    Any (*build)();
  };
  const ExprFixture expr_fixtures[] = {
      {{"split-fuse-shared", 12, 17, kSplitFuseSharedBytes, 10, 4, kSplitFuseRepeats, true,
        ArmKind::kSubst},
       [] { return Any(SplitFuse(true)); }},
      {{"split-fuse-distinct", 15, 17, kSplitFuseDistinctBytes, 10, 2, kSplitFuseRepeats, false,
        ArmKind::kSubst},
       [] { return Any(SplitFuse(false)); }},
      {{"call-split-fuse-shared", 18, 25, kCallSplitFuseSharedBytes, 18, 6, kSplitFuseRepeats,
        true, ArmKind::kSubst},
       [] { return Any(CallSplitFuse(true)); }},
      {{"call-split-fuse-distinct", 23, 25, kCallSplitFuseDistinctBytes, 18, 2, kSplitFuseRepeats,
        false, ArmKind::kSubst},
       [] { return Any(CallSplitFuse(false)); }},
  };
  for (const ExprFixture& f : expr_fixtures) {
    RunFixture(f.info, f.build);
  }

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
                ArmKind::kSwap},
               [] { return Any(LongSeq(current_length, 1)); });
  }

  Emit("#sink\t" + std::to_string(g_sink));
  return 0;
}
