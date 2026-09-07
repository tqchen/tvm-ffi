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

// The split-fuse half of `real_tvm_bench.cc`, one engine per executable.
//
// Three structural-mutation engines live in this tree -- structural_mutate_gold.h (GOLD 730d6fc),
// structural_mutate.h (UC 31f6948) and structural_mutate_mixed.h (GOLD carrying UC's protocol
// alongside its own) -- and each has a hook file written against it. This driver is compiled
// once per engine: `TVM_FFI_BENCH_HOOK_HEADER` names the hook file, the hook file includes its
// own engine header, and nothing here selects an engine at run time. build_mix.sh stamps the
// content hash of both files into the provenance.
//
// Everything else is `real_tvm_bench.cc`, cut down to what the two binary split-fuse fixtures
// reach: the same builders, the same arms (`floor`, `never`, `identity`, `subst`, `functor`,
// `old`, and the walk arms), the same untimed checks, the same timing loops and the same output
// lines, so `report.py` reads it unchanged. The Call, seq, swap and splice machinery is gone,
// because the hook files carry only the six node types these fixtures dispatch on -- Var,
// IntImm, Add, Mul, FloorDiv, FloorMod -- and the coverage assertion holds every executable to
// that.

#include <tvm/ir/prim/expr.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

// The hook file first: it includes the engine it was written against, and the engine headers
// share one include guard, so the first one in wins and bench_common.h's own include of
// <tvm/ffi/extra/structural_mutate.h> becomes a no-op. The checks below make sure the engine
// that won is the one this executable was asked for.
#ifndef TVM_FFI_BENCH_HOOK_HEADER
#error "build_mix.sh selects the hook file with -DTVM_FFI_BENCH_HOOK_HEADER"
#endif
#include TVM_FFI_BENCH_HOOK_HEADER

// Which engine header is in this translation unit, read off the macros each one defines:
// GOLD's assign macro is built on TVM_FFI_S_MUTATE_DATA_OR_RETURN_IMPL_, UC's on
// TVM_FFI_S_MUTATE_RETURN_UNCHANGED, and the mixed engine carries both.
#if defined(TVM_FFI_S_MUTATE_DATA_OR_RETURN_IMPL_) && defined(TVM_FFI_S_MUTATE_RETURN_UNCHANGED)
#define TVM_FFI_BENCH_DETECTED_ENGINE_CODE 3
#elif defined(TVM_FFI_S_MUTATE_DATA_OR_RETURN_IMPL_)
#define TVM_FFI_BENCH_DETECTED_ENGINE_CODE 1
#elif defined(TVM_FFI_S_MUTATE_RETURN_UNCHANGED)
#define TVM_FFI_BENCH_DETECTED_ENGINE_CODE 2
#elif defined(TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN)
// EXPERIMENT (task #387): the pre-unchanged engine, base 0909bb4, defines only the assign macros.
#define TVM_FFI_BENCH_DETECTED_ENGINE_CODE 4
#else
#error "no structural-mutation engine header is in this translation unit"
#endif
#ifndef TVM_FFI_BENCH_VARIANT_CODE
#error "build_mix.sh states the expected engine with -DTVM_FFI_BENCH_VARIANT_CODE (1 gold, 2 uc, 3 mixed)"
#endif
#if TVM_FFI_BENCH_VARIANT_CODE != TVM_FFI_BENCH_DETECTED_ENGINE_CODE
#error "the hook file pulled in a different engine header than this executable was asked for"
#endif
#ifndef TVM_FFI_BENCH_VARIANT
#define TVM_FFI_BENCH_VARIANT "unnamed"
#endif
#ifndef TVM_FFI_BENCH_ENGINE_HEADER
#define TVM_FFI_BENCH_ENGINE_HEADER "unstamped"
#endif
#ifndef TVM_FFI_BENCH_ENGINE_SHA256
#define TVM_FFI_BENCH_ENGINE_SHA256 "unstamped"
#endif
#ifndef TVM_FFI_BENCH_HOOKS_SHA256
#define TVM_FFI_BENCH_HOOKS_SHA256 "unstamped"
#endif

#include "bench_common.h"
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

constexpr int64_t kVarBytes = sizeof(VarNode);
constexpr int64_t kImmBytes = sizeof(IntImmNode);
constexpr int64_t kAddBytes = sizeof(prim::AddNode);
constexpr int64_t kMulBytes = sizeof(prim::MulNode);
constexpr int64_t kSplitFuseSharedBytes =
    2 * kVarBytes + 4 * kImmBytes + 2 * kMulBytes + 2 * kAddBytes +
    static_cast<int64_t>(sizeof(prim::FloorDivNode)) +
    static_cast<int64_t>(sizeof(prim::FloorModNode));
constexpr int64_t kSplitFuseDistinctBytes =
    kSplitFuseSharedBytes + kMulBytes + kImmBytes + kAddBytes;
constexpr int kSplitFuseRepeats = 20000;
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

/*! \brief The arm a fixture declares, used by the checks and the density sweep. */
ffi::Any MapFixtureArm(ffi::Any input, ArmKind kind, Ownership ownership) {
  return MapSubst(std::move(input), kind, ownership);
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
  const IdentityArm arms[] = {{"map_identity_var", &MapIdentityVar}};
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
  EmitProvenance("variant", TVM_FFI_BENCH_VARIANT);
  EmitProvenance("engine_header",
                 TVM_FFI_BENCH_ENGINE_HEADER " sha256:" TVM_FFI_BENCH_ENGINE_SHA256);
  EmitProvenance("hook_header", TVM_FFI_BENCH_HOOK_HEADER " sha256:" TVM_FFI_BENCH_HOOKS_SHA256);
  EmitProvenance("structural_hooks",
                 "harness (" TVM_FFI_BENCH_HOOK_HEADER "), the six split-fuse hooks extracted "
                 "from the 220f363 hook files, installed over TVM's");

  // split/fuse substitutes every Var, with the remap, which is the semantics Substitute
  // provides and the thing worth measuring on a small Expr tree.  Under moved ownership the
  // only new identity is the substituted-in Var, except on the shared fixture, where the first
  // parent to reach the shared subtree cannot mutate it in place and copies it and its own
  // changed child.
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
  };
  for (const ExprFixture& f : expr_fixtures) {
    PrepareFixture(f.info, f.build, f.info.name);
    RunFixture(f.info, f.build, kExprMapArms, sizeof(kExprMapArms) / sizeof(kExprMapArms[0]));
  }

  Emit("#sink\t" + std::to_string(g_sink));
  return 0;
}
