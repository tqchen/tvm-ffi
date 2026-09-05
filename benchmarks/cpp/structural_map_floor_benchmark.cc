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
/*!
 * \file structural_map_floor_benchmark.cc
 * \brief Floor benchmark for the StructuralMap engine, with no external dependency.
 *
 * Measures three points on the same distinct (unshared) tree, all performing a no-op
 * mutation whose result must be pointer-identical to the input:
 *
 * - plain recursive descent, hand written, as the absolute floor;
 * - a minimal mutator whose vtable dispatch is only an attr-column lookup and an
 *   indirect call into the registered hook, driving the same hooks as StructuralMap.
 *   This is the lower bound for what any engine over these hooks can cost;
 * - StructuralMap itself.
 *
 * The gap between the last two is the engine's own overhead, which is what this
 * benchmark exists to keep honest. Build standalone, for example:
 *
 *   g++ -O3 -DNDEBUG -std=c++17 -I<ffi>/include -I<ffi>/3rdparty/dlpack/include \\
 *       structural_map_floor_benchmark.cc -o floor_bench \\
 *       -L<build>/lib -ltvm_ffi -Wl,-rpath,<build>/lib
 *
 * Report medians of several runs: a single run is not reliable to better than 10-20%.
 */
// recursive descent, on distinct (unshared) binary trees.
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace tvm {
namespace ffi {

// ------------------------------------------------- minimal mutator lower bound
// A standalone mutator whose vtable dispatch is nothing but an attr-column
// lookup and an indirect call into the registered hook: no callback chain, no
// identity remap, no Expected plumbing. It drives the same registered hooks as
// StructuralMap, so the delta against StructuralMap is the engine's own cost.
// This is a yardstick, not a proposal.
class MinimalMutatorObj : public StructuralMutatorObj {
 public:
  MinimalMutatorObj() : StructuralMutatorObj(VTable()) {}

  static TVMFFIAny Mutate(StructuralMutatorObj* self, AnyView value) noexcept {
    static reflection::TypeAttrColumn col(reflection::type_attr::kStructuralMutate);
    AnyView attr = col[value.type_index()];
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      return (*reinterpret_cast<FStructuralMutate>(attr.cast<void*>()))(self, value);
    }
    return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(value)));
  }
  static TVMFFIAny NoRemapGet(StructuralMutatorObj*, AnyView) noexcept {
    TVMFFIAny a;
    a.type_index = TypeIndex::kTVMFFINone;
    a.zero_padding = 0;
    a.v_int64 = 0;
    return a;
  }
  static TVMFFIAny NoRemapSet(StructuralMutatorObj*, AnyView, AnyView) noexcept {
    TVMFFIAny a;
    a.type_index = TypeIndex::kTVMFFINone;
    a.zero_padding = 0;
    a.v_int64 = 0;
    return a;
  }
  static const StructuralMutatorVTable* VTable() {
    static const StructuralMutatorVTable v{&MinimalMutatorObj::Mutate, &MinimalMutatorObj::Mutate,
                                           &MinimalMutatorObj::NoRemapGet,
                                           &MinimalMutatorObj::NoRemapSet};
    return &v;
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("bench.MinimalMutator", MinimalMutatorObj,
                                    StructuralMutatorObj);
};

// ---------------------------------------------------------------- node types
class BinObj : public Object {
 public:
  ObjectRef lhs;
  ObjectRef rhs;
  BinObj(ObjectRef lhs, ObjectRef rhs) : lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  explicit BinObj(UnsafeInit) {}

  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    const auto* self = value.cast<const BinObj*>();
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ObjectRef, lhs, mutator->MutateExpected(self->lhs));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ObjectRef, rhs, mutator->MutateExpected(self->rhs));
    if (lhs.same_as(self->lhs) && rhs.same_as(self->rhs))
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(
        Any(make_object<BinObj>(std::move(lhs), std::move(rhs))));
  }
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BinObj>().def_ro("lhs", &BinObj::lhs).def_ro("rhs", &BinObj::rhs);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::TypeAttrDef<BinObj>().attr(
        refl::type_attr::kStructuralMutate,
        reinterpret_cast<void*>(static_cast<FStructuralMutate>(&BinObj::StructuralMutate)));
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("bench.Bin", BinObj, Object);
};

class LeafObj : public Object {
 public:
  int64_t v;
  explicit LeafObj(int64_t v) : v(v) {}
  explicit LeafObj(UnsafeInit) {}
  static TVMFFIAny StructuralMutate(StructuralMutatorObj*, AnyView value) noexcept {
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LeafObj>().def_ro("v", &LeafObj::v);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::TypeAttrDef<LeafObj>().attr(
        refl::type_attr::kStructuralMutate,
        reinterpret_cast<void*>(static_cast<FStructuralMutate>(&LeafObj::StructuralMutate)));
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("bench.Leaf", LeafObj, Object);
};

// A type that never occurs in the tree: gives a callback that matches nothing.
class NeverObj : public Object {
 public:
  explicit NeverObj(UnsafeInit) {}
  NeverObj() {}
  static void RegisterReflection() { reflection::ObjectDef<NeverObj>(); }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("bench.Never", NeverObj, Object);
};
class Never : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Never, ObjectRef, NeverObj);
};

// -------------------------------------------- PrimExpr-shaped typed hierarchy
// TVM's PrimExpr is not a node type but a view: any node of a base expression type whose `ty`
// field holds a PrimType. Checking one therefore costs an IsObjectInstance range test, a
// dereference of the node to reach `ty`, and a second type check on that field -- far more than
// the single boundary compare an ObjectRef-typed field costs. TVM's mutate hooks declare their
// fields with exactly this shape, so measuring the assign macro on ObjectRef alone understates
// what it costs there.
//
// TBinExprObj and TBinPrimObj are deliberately identical: same base, same fields, same layout,
// and their trees hold the same values. The only difference is the type their hook declares in
// TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN, so the gap between them is the cost of the richer check
// and nothing else.

class TPrimTypeObj : public Object {
 public:
  TPrimTypeObj() {}
  explicit TPrimTypeObj(UnsafeInit) {}
  static void RegisterReflection() { reflection::ObjectDef<TPrimTypeObj>(); }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("bench.PrimType", TPrimTypeObj, Object);
};
class TPrimType : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TPrimType, ObjectRef, TPrimTypeObj);
};

// Non-final base reserving child slots, mirroring tvm::ExprNode.
class TExprObj : public Object {
 public:
  Any ty;
  TExprObj() {}
  explicit TExprObj(Any ty) : ty(std::move(ty)) {}
  explicit TExprObj(UnsafeInit) {}
  static void RegisterReflection() {
    reflection::ObjectDef<TExprObj>().def_ro("ty", &TExprObj::ty);
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr uint32_t _type_child_slots = 64;
  TVM_FFI_DECLARE_OBJECT_INFO("bench.Expr", TExprObj, Object);
};
class TExpr : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TExpr, ObjectRef, TExprObj);
};

// The view type: any TExprObj whose `ty` is a TPrimType. Mirrors tvm::PrimExpr.
class TPrimExpr : public TExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TPrimExpr, TExpr, TExprObj);
};

template <>
inline constexpr bool use_default_type_traits_v<TPrimExpr> = false;

template <>
struct TypeTraits<TPrimExpr> : public ObjectRefTypeTraitsBase<TPrimExpr> {
  using Base = ObjectRefTypeTraitsBase<TPrimExpr>;
  using Base::CopyFromAnyViewAfterCheck;
  using Base::CopyToAnyView;
  using Base::GetMismatchTypeInfo;
  using Base::MoveFromAnyAfterCheck;
  using Base::MoveToAny;
  using Base::TypeStr;

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return TPrimExpr::_type_is_nullable;
    if (src->type_index < TypeIndex::kTVMFFIStaticObjectBegin ||
        !details::IsObjectInstance<TExprObj>(src->type_index)) {
      return false;
    }
    const auto* expr = static_cast<const TExprObj*>(
        details::ObjectUnsafe::RawObjectPtrFromUnowned<Object>(src->v_obj));
    return details::AnyUnsafe::CheckAnyStrict<TPrimType>(expr->ty);
  }

  TVM_FFI_INLINE static std::optional<TPrimExpr> TryCastFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) return CopyFromAnyViewAfterCheck(src);
    return std::nullopt;
  }
};

// Declares TExpr: the check is an IsObjectInstance range test.
class TBinExprObj : public TExprObj {
 public:
  TExpr lhs, rhs;
  TBinExprObj(Any ty, TExpr lhs, TExpr rhs)
      : TExprObj(std::move(ty)), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  explicit TBinExprObj(UnsafeInit) : TExprObj(UnsafeInit{}) {}

  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    const auto* self = value.cast<const TBinExprObj*>();
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(TExpr, lhs, mutator->MutateExpected(self->lhs));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(TExpr, rhs, mutator->MutateExpected(self->rhs));
    if (lhs.same_as(self->lhs) && rhs.same_as(self->rhs))
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(
        Any(make_object<TBinExprObj>(self->ty, std::move(lhs), std::move(rhs))));
  }
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TBinExprObj>()
        .def_ro("lhs", &TBinExprObj::lhs)
        .def_ro("rhs", &TBinExprObj::rhs);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::TypeAttrDef<TBinExprObj>().attr(
        refl::type_attr::kStructuralMutate,
        reinterpret_cast<void*>(static_cast<FStructuralMutate>(&TBinExprObj::StructuralMutate)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("bench.BinExpr", TBinExprObj, TExprObj);
};

// Identical to TBinExprObj except that it declares TPrimExpr, so each assignment also
// dereferences the node and type-checks its `ty`.
class TBinPrimObj : public TExprObj {
 public:
  TExpr lhs, rhs;
  TBinPrimObj(Any ty, TExpr lhs, TExpr rhs)
      : TExprObj(std::move(ty)), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  explicit TBinPrimObj(UnsafeInit) : TExprObj(UnsafeInit{}) {}

  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    const auto* self = value.cast<const TBinPrimObj*>();
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(TPrimExpr, lhs, mutator->MutateExpected(self->lhs));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(TPrimExpr, rhs, mutator->MutateExpected(self->rhs));
    if (lhs.same_as(self->lhs) && rhs.same_as(self->rhs))
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(
        Any(make_object<TBinPrimObj>(self->ty, std::move(lhs), std::move(rhs))));
  }
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TBinPrimObj>()
        .def_ro("lhs", &TBinPrimObj::lhs)
        .def_ro("rhs", &TBinPrimObj::rhs);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::TypeAttrDef<TBinPrimObj>().attr(
        refl::type_attr::kStructuralMutate,
        reinterpret_cast<void*>(static_cast<FStructuralMutate>(&TBinPrimObj::StructuralMutate)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("bench.BinPrim", TBinPrimObj, TExprObj);
};

// Same again, but skipping the check, to price it directly.
class TBinSkipObj : public TExprObj {
 public:
  TExpr lhs, rhs;
  TBinSkipObj(Any ty, TExpr lhs, TExpr rhs)
      : TExprObj(std::move(ty)), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  explicit TBinSkipObj(UnsafeInit) : TExprObj(UnsafeInit{}) {}

  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    const auto* self = value.cast<const TBinSkipObj*>();
    TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN_SKIP_CHECK(TPrimExpr, lhs,
                                                        mutator->MutateExpected(self->lhs));
    TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN_SKIP_CHECK(TPrimExpr, rhs,
                                                        mutator->MutateExpected(self->rhs));
    if (lhs.same_as(self->lhs) && rhs.same_as(self->rhs))
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(
        Any(make_object<TBinSkipObj>(self->ty, std::move(lhs), std::move(rhs))));
  }
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TBinSkipObj>()
        .def_ro("lhs", &TBinSkipObj::lhs)
        .def_ro("rhs", &TBinSkipObj::rhs);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::TypeAttrDef<TBinSkipObj>().attr(
        refl::type_attr::kStructuralMutate,
        reinterpret_cast<void*>(static_cast<FStructuralMutate>(&TBinSkipObj::StructuralMutate)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("bench.BinSkip", TBinSkipObj, TExprObj);
};

class TLeafExprObj : public TExprObj {
 public:
  int64_t v;
  TLeafExprObj(Any ty, int64_t v) : TExprObj(std::move(ty)), v(v) {}
  explicit TLeafExprObj(UnsafeInit) : TExprObj(UnsafeInit{}) {}
  static TVMFFIAny StructuralMutate(StructuralMutatorObj*, AnyView value) noexcept {
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TLeafExprObj>().def_ro("v", &TLeafExprObj::v);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::TypeAttrDef<TLeafExprObj>().attr(
        refl::type_attr::kStructuralMutate,
        reinterpret_cast<void*>(static_cast<FStructuralMutate>(&TLeafExprObj::StructuralMutate)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("bench.LeafExpr", TLeafExprObj, TExprObj);
};

// One shared `ty` for the whole tree, so the check hits an already-hot object.
inline Any TypedTy() {
  static Any ty = Any(TPrimType(make_object<TPrimTypeObj>()));
  return ty;
}
template <typename TBin>
TExpr MakeTypedTree(int depth, int64_t* counter) {
  if (depth == 0) {
    return TExpr(make_object<TLeafExprObj>(TypedTy(), (*counter)++));
  }
  TExpr l = MakeTypedTree<TBin>(depth - 1, counter);
  TExpr r = MakeTypedTree<TBin>(depth - 1, counter);
  return TExpr(make_object<TBin>(TypedTy(), std::move(l), std::move(r)));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  BinObj::RegisterReflection();
  LeafObj::RegisterReflection();
  NeverObj::RegisterReflection();
  TPrimTypeObj::RegisterReflection();
  TExprObj::RegisterReflection();
  TBinExprObj::RegisterReflection();
  TBinPrimObj::RegisterReflection();
  TBinSkipObj::RegisterReflection();
  TLeafExprObj::RegisterReflection();
}

// ------------------------------------------------------------------- fixture
int64_t g_counter = 0;
ObjectRef MakeTree(int depth) {
  if (depth == 0) return ObjectRef(make_object<LeafObj>(g_counter++));
  ObjectRef l = MakeTree(depth - 1);
  ObjectRef r = MakeTree(depth - 1);
  return ObjectRef(make_object<BinObj>(std::move(l), std::move(r)));
}
size_t CountNodes(const ObjectRef& n) {
  if (const auto* b = n.as<BinObj>()) return 1 + CountNodes(b->lhs) + CountNodes(b->rhs);
  return 1;
}

// -------------------------------------------------- baseline recursive descent
// The "old visitor/mutator" shape: descend, rebuild only on change.
__attribute__((noinline)) ObjectRef PlainMutate(const ObjectRef& n) {
  if (const auto* b = n.as<BinObj>()) {
    ObjectRef l = PlainMutate(b->lhs);
    ObjectRef r = PlainMutate(b->rhs);
    if (l.same_as(b->lhs) && r.same_as(b->rhs)) return n;
    return ObjectRef(make_object<BinObj>(std::move(l), std::move(r)));
  }
  return n;
}

__attribute__((noinline)) ObjectRef MinimalNoOp(const ObjectRef& root) {
  ObjectPtr<MinimalMutatorObj> m = make_object<MinimalMutatorObj>();
  TVMFFIAny out = MinimalMutatorObj::Mutate(m.get(), AnyView(root));
  return ObjectRef(details::ObjectUnsafe::ObjectPtrFromOwned<Object>(out.v_obj));
}
// Matches every leaf, roughly half the nodes, and returns it unchanged. A callback that
// matches nothing measures only the unmatched path; real substitutions match a substantial
// fraction, so this variant is what any inline-vs-outline decision must be judged against.
__attribute__((noinline)) Any StructuralMatchLeaf(const ObjectRef& root) {
  return StructuralMap<WalkOrder::kPostOrder>(
      AnyView(root),
      [](const LeafObj* node, TVMFFIDefRegionKind) -> Any { return Any(GetRef<ObjectRef>(node)); });
}
// Matches every typed leaf, about half the nodes, so the binary hooks -- and their assign
// macros -- run on the changed path the same way a real substitution makes them run.
__attribute__((noinline)) Any StructuralTypedLeaf(const TExpr& root) {
  return StructuralMap<WalkOrder::kPostOrder>(
      AnyView(root), [](const TLeafExprObj* node, TVMFFIDefRegionKind) -> Any {
        return Any(GetRef<ObjectRef>(node));
      });
}

__attribute__((noinline)) Any StructuralNoOp(const ObjectRef& root) {
  return StructuralMap<WalkOrder::kPostOrder>(AnyView(root), [](Never v) -> Any { return Any(v); });
}

// ---------------------------------------------------------------------- timing
template <typename F>
double BestNs(F&& f, int iters, int reps, int warmup) {
  for (int i = 0; i < warmup; ++i) f();
  std::vector<double> samples;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) f();
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / iters);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

}  // namespace ffi
}  // namespace tvm

int main(int argc, char** argv) {
  using namespace tvm::ffi;
  int depth = argc > 1 ? atoi(argv[1]) : 7;
  int iters = argc > 2 ? atoi(argv[2]) : 2000;
  int reps = 15, warmup = 500;

  ObjectRef root = MakeTree(depth);
  size_t n = CountNodes(root);

  // correctness: both must return the identical root (pure no-op)
  ObjectRef plain = PlainMutate(root);
  Any structural = StructuralNoOp(root);
  bool plain_same = plain.same_as(root);
  bool struct_same = structural.as<Object>() == root.get();
  printf("depth=%d nodes=%zu plain_noop_identity=%d structural_noop_identity=%d\n", depth, n,
         (int)plain_same, (int)struct_same);

  ObjectRef minimal = MinimalNoOp(root);
  printf("minimal_noop_identity=%d\n", (int)minimal.same_as(root));
  double p = BestNs(
      [&] {
        ObjectRef r = PlainMutate(root);
        asm volatile("" ::"r"(r.get()));
      },
      iters, reps, warmup);
  double mn = BestNs(
      [&] {
        ObjectRef r = MinimalNoOp(root);
        asm volatile("" ::"r"(r.get()));
      },
      iters, reps, warmup);
  double s = BestNs(
      [&] {
        Any r = StructuralNoOp(root);
        asm volatile("" ::"r"(r.as<Object>()));
      },
      iters, reps, warmup);
  Any match_chk = StructuralMatchLeaf(root);
  printf("match_leaf_identity=%d\n", (int)(match_chk.as<Object>() == root.get()));
  double ml = BestNs(
      [&] {
        Any r = StructuralMatchLeaf(root);
        asm volatile("" ::"r"(r.as<Object>()));
      },
      iters, reps, warmup);
  int64_t tc = 0;
  TExpr texpr_root = MakeTypedTree<TBinExprObj>(depth, &tc);
  tc = 0;
  TExpr tprim_root = MakeTypedTree<TBinPrimObj>(depth, &tc);
  tc = 0;
  TExpr tskip_root = MakeTypedTree<TBinSkipObj>(depth, &tc);
  Any te_chk = StructuralTypedLeaf(texpr_root);
  Any tp_chk = StructuralTypedLeaf(tprim_root);
  Any ts_chk = StructuralTypedLeaf(tskip_root);
  printf("typed_identity=%d%d%d\n", (int)(te_chk.as<Object>() == texpr_root.get()),
         (int)(tp_chk.as<Object>() == tprim_root.get()),
         (int)(ts_chk.as<Object>() == tskip_root.get()));
  double te = BestNs(
      [&] {
        Any r = StructuralTypedLeaf(texpr_root);
        asm volatile("" ::"r"(r.as<Object>()));
      },
      iters, reps, warmup);
  double tp = BestNs(
      [&] {
        Any r = StructuralTypedLeaf(tprim_root);
        asm volatile("" ::"r"(r.as<Object>()));
      },
      iters, reps, warmup);
  double ts = BestNs(
      [&] {
        Any r = StructuralTypedLeaf(tskip_root);
        asm volatile("" ::"r"(r.as<Object>()));
      },
      iters, reps, warmup);

  printf("plain      : %10.1f ns/op  %8.2f ns/node\n", p, p / n);
  printf("minimal    : %10.1f ns/op  %8.2f ns/node  (lower bound)\n", mn, mn / n);
  printf("structural : %10.1f ns/op  %8.2f ns/node  (0%% match)\n", s, s / n);
  printf("match-leaf : %10.1f ns/op  %8.2f ns/node  (~50%% match)\n", ml, ml / n);
  printf("ratio      : %10.2fx        delta %6.2f ns/node\n", s / p, (s - p) / n);
  // Same tree, same layout, same callback; only the type the hook declares differs.
  printf("typed-expr : %10.1f ns/op  %8.2f ns/node  (declares TExpr)\n", te, te / n);
  printf("typed-prim : %10.1f ns/op  %8.2f ns/node  (declares TPrimExpr)\n", tp, tp / n);
  printf("typed-skip : %10.1f ns/op  %8.2f ns/node  (TPrimExpr, SKIP_CHECK)\n", ts, ts / n);
  printf("check cost : %8.2f ns/node vs TExpr   %8.2f ns/node vs SKIP_CHECK\n", (tp - te) / n,
         (tp - ts) / n);
  return 0;
}
