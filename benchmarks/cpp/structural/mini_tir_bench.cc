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

// The mini-TIR half of the structural-traversal benchmark.  Builds a standalone mirror of
// TVM's TIR node hierarchy out of tvm-ffi's own types, registers its own structural hooks,
// and runs the same fixtures, arms, method and assertions as real_tvm_bench.cc.  It needs
// no TVM checkout and no hook override.
//
// Mirrored deliberately, because these are what the engine's per-node cost is made of:
//
//   * HExprObj is a non-final base with a `ty` field, TreeNode kind and 64 child slots;
//   * HVarObj derives from it and is FreeVar kind, so every matched Var takes an identity
//     remap get *and* set, the way tvm::Var does;
//   * HPrimExpr is a *view* over any HExprObj whose `ty` holds an HPrimType, with the same
//     TypeTraits shape as tvm::PrimExpr: checking a field costs an IsObjectInstance range
//     test, a dereference to reach `ty`, and a second type check on that field;
//   * binary ops are final with two HPrimExpr operands and MutateBinary's guard that skips
//     result-type inference when neither operand's type changed;
//   * HSeqStmtObj's in-place hook copies the sequence handle before mapping elements, the
//     same idiom tvm::SeqStmtNode uses -- including its consequence for in-place mutation.
//
// It is a model of the shape, not a clone: no Span, result types compare a single `bits`
// field rather than full dtype logic, and only four binary ops are mirrored.

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>

#include <string>
#include <unordered_set>
#include <vector>

#include "bench_common.h"

namespace tvm {
namespace ffi {
namespace mini {

// ---------------------------------------------------------------- PrimType
class HPrimTypeObj : public Object {
 public:
  int64_t bits = 32;
  HPrimTypeObj() {}
  explicit HPrimTypeObj(int64_t bits) : bits(bits) {}
  explicit HPrimTypeObj(UnsafeInit) {}
  static void RegisterReflection() {
    reflection::ObjectDef<HPrimTypeObj>().def_ro("bits", &HPrimTypeObj::bits);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.PrimType", HPrimTypeObj, Object);
};
class HPrimType : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HPrimType, ObjectRef, HPrimTypeObj);
};

// ---------------------------------------------------------------- Expr
class HExprObj : public Object {
 public:
  Any ty;
  HExprObj() {}
  explicit HExprObj(Any ty) : ty(std::move(ty)) {}
  explicit HExprObj(UnsafeInit) {}
  static void RegisterReflection() {
    reflection::ObjectDef<HExprObj>().def_ro("ty", &HExprObj::ty);
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr uint32_t _type_child_slots = 64;
  TVM_FFI_DECLARE_OBJECT_INFO("h.Expr", HExprObj, Object);
};
class HExpr : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HExpr, ObjectRef, HExprObj);
};

/*! \brief A view over any HExprObj whose `ty` is an HPrimType. Mirrors tvm::PrimExpr. */
class HPrimExpr : public HExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HPrimExpr, HExpr, HExprObj);
};

}  // namespace mini

template <>
inline constexpr bool use_default_type_traits_v<mini::HPrimExpr> = false;

template <>
struct TypeTraits<mini::HPrimExpr> : public ObjectRefTypeTraitsBase<mini::HPrimExpr> {
  using Base = ObjectRefTypeTraitsBase<mini::HPrimExpr>;
  using Base::CopyFromAnyViewAfterCheck;
  using Base::CopyToAnyView;
  using Base::GetMismatchTypeInfo;
  using Base::MoveFromAnyAfterCheck;
  using Base::MoveToAny;
  using Base::TypeStr;
  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return mini::HPrimExpr::_type_is_nullable;
    if (src->type_index < TypeIndex::kTVMFFIStaticObjectBegin ||
        !details::IsObjectInstance<mini::HExprObj>(src->type_index)) {
      return false;
    }
    const auto* e = details::ObjectUnsafe::RawObjectPtrFromUnowned<mini::HExprObj>(src->v_obj);
    return details::AnyUnsafe::CheckAnyStrict<mini::HPrimType>(e->ty);
  }
  TVM_FFI_INLINE static std::optional<mini::HPrimExpr> TryCastFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) return CopyFromAnyViewAfterCheck(src);
    return std::nullopt;
  }
};

namespace mini {

/*! \brief `ObjectRef::as` for a raw Object pointer, which the model traversals work in. */
template <typename T>
TVM_FFI_INLINE const T* ObjAs(const Object* node) {
  return (node != nullptr && details::IsObjectInstance<T>(node->type_index()))
             ? static_cast<const T*>(node)
             : nullptr;
}

// Helpers shared by every hook body.
TVM_FFI_INLINE TVMFFIAny VisitDone() {
  return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Optional<VisitInterrupt>>(std::nullopt));
}
TVM_FFI_INLINE TVMFFIAny KeepValue(AnyView value) {
  return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
}

// ---------------------------------------------------------------- Var (FreeVar)
class HVarObj : public HExprObj {
 public:
  int64_t id = 0;
  HVarObj(Any ty, int64_t id) : HExprObj(std::move(ty)), id(id) {}
  explicit HVarObj(UnsafeInit) : HExprObj(UnsafeInit{}) {}

  static TVMFFIAny StructuralVisit(StructuralVisitorObj*, AnyView) noexcept { return VisitDone(); }
  /*! \brief Mirrors tvm::MutateVar: consult the remap first, record on the way out. */
  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    Expected<Any> cached = mutator->VarRemapGetExpected(value);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(cached);
    if (details::ExpectedUnsafe::GetData(cached).type_index() != TypeIndex::kTVMFFINone) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(cached));
    }
    Expected<void> set_result = mutator->VarRemapSetExpected(value, AnyView(value));
    if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(set_result).error()));
    }
    return KeepValue(value);
  }
  static void RegisterReflection();
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_final = true;
  static constexpr const char* _type_key = "h.Var";
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(HVarObj, HExprObj);
};
class HVar : public HPrimExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HVar, HPrimExpr, HVarObj);
};

// ---------------------------------------------------------------- IntImm
class HIntImmObj : public HExprObj {
 public:
  int64_t value_ = 0;
  HIntImmObj(Any ty, int64_t v) : HExprObj(std::move(ty)), value_(v) {}
  explicit HIntImmObj(UnsafeInit) : HExprObj(UnsafeInit{}) {}
  static TVMFFIAny StructuralVisit(StructuralVisitorObj*, AnyView) noexcept { return VisitDone(); }
  static TVMFFIAny StructuralMutate(StructuralMutatorObj*, AnyView value) noexcept {
    return KeepValue(value);
  }
  static void RegisterReflection();
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_final = true;
  static constexpr const char* _type_key = "h.IntImm";
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(HIntImmObj, HExprObj);
};

// ------------------------------------------------- binary ops (Add/Mul/FloorDiv/FloorMod)
template <typename T>
class HBinOpObj : public HExprObj {
 public:
  HPrimExpr a, b;
  HBinOpObj(Any ty, HPrimExpr a, HPrimExpr b)
      : HExprObj(std::move(ty)), a(std::move(a)), b(std::move(b)) {}
  explicit HBinOpObj(UnsafeInit) : HExprObj(UnsafeInit{}) {}

  /*! \brief Mirrors tvm::MutateBinary's guard: skip type inference when neither type moved. */
  static Expected<HPrimType> ResultType(const HPrimExpr& a, const HPrimExpr& b) noexcept {
    auto at = a->ty.as<HPrimType>();
    auto bt = b->ty.as<HPrimType>();
    if (!at.has_value() || !bt.has_value() || at.value()->bits != bt.value()->bits) {
      return Unexpected(Error("TypeError", "mismatched types", ""));
    }
    return at.value();
  }
  static TVMFFIAny StructuralVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
    const auto* self = value.cast<const T*>();
    auto a_result = visitor->VisitExpected(self->a);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(a_result);
    auto b_result = visitor->VisitExpected(self->b);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(b_result);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(b_result));
  }
  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    const auto* self = value.cast<const T*>();
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimExpr, a, mutator->MutateExpected(self->a));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimExpr, b, mutator->MutateExpected(self->b));
    if (a.same_as(self->a) && b.same_as(self->b)) return KeepValue(value);
    ObjectPtr<T> copy = make_object<T>(*static_cast<const T*>(self));
    if (!a->ty.same_as(self->a->ty) || !b->ty.same_as(self->b->ty)) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimType, rty, ResultType(a, b));
      copy->HExprObj::ty = std::move(rty);
    }
    copy->a = std::move(a);
    copy->b = std::move(b);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(copy)));
  }
  static TVMFFIAny StructuralMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                                AnyView value) noexcept {
    T* self = const_cast<T*>(value.cast<const T*>());
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimExpr, a,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->a));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimExpr, b,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->b));
    if (a.same_as(self->a) && b.same_as(self->b)) return KeepValue(value);
    if (!a->ty.same_as(self->a->ty) || !b->ty.same_as(self->b->ty)) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimType, rty, ResultType(a, b));
      self->HExprObj::ty = std::move(rty);
    }
    self->a = std::move(a);
    self->b = std::move(b);
    return KeepValue(value);
  }
  static void RegisterReflection();
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_final = true;
};
#define H_DECL_BINOP(Name, Key)                                      \
  class Name : public HBinOpObj<Name> {                              \
   public:                                                           \
    using HBinOpObj<Name>::HBinOpObj;                                \
    static constexpr const char* _type_key = Key;                    \
    TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(Name, HExprObj); \
  }
H_DECL_BINOP(HAddObj, "h.Add");
H_DECL_BINOP(HMulObj, "h.Mul");
H_DECL_BINOP(HFloorDivObj, "h.FloorDiv");
H_DECL_BINOP(HFloorModObj, "h.FloorMod");

// ---------------------------------------------------------------- Stmt
class HStmtObj : public Object {
 public:
  HStmtObj() {}
  explicit HStmtObj(UnsafeInit) {}
  static void RegisterReflection() { reflection::ObjectDef<HStmtObj>(); }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr uint32_t _type_child_slots = 32;
  TVM_FFI_DECLARE_OBJECT_INFO("h.Stmt", HStmtObj, Object);
};
class HStmt : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HStmt, ObjectRef, HStmtObj);
};

class HEvaluateObj : public HStmtObj {
 public:
  HPrimExpr value;
  explicit HEvaluateObj(HPrimExpr value) : value(std::move(value)) {}
  explicit HEvaluateObj(UnsafeInit) : HStmtObj(UnsafeInit{}) {}
  static TVMFFIAny StructuralVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
    const auto* self = value.cast<const HEvaluateObj*>();
    auto result = visitor->VisitExpected(self->value);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(result);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }
  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    const auto* self = value.cast<const HEvaluateObj*>();
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimExpr, mapped, mutator->MutateExpected(self->value));
    if (mapped.same_as(self->value)) return KeepValue(value);
    ObjectPtr<HEvaluateObj> copy = make_object<HEvaluateObj>(*self);
    copy->value = std::move(mapped);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(copy)));
  }
  static TVMFFIAny StructuralMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                                AnyView value) noexcept {
    auto* self = const_cast<HEvaluateObj*>(value.cast<const HEvaluateObj*>());
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimExpr, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
    if (mapped.same_as(self->value)) return KeepValue(value);
    self->value = std::move(mapped);
    return KeepValue(value);
  }
  static void RegisterReflection();
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_final = true;
  static constexpr const char* _type_key = "h.Evaluate";
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(HEvaluateObj, HStmtObj);
};

class HSeqStmtObj : public HStmtObj {
 public:
  Array<HStmt> seq;
  explicit HSeqStmtObj(Array<HStmt> seq) : seq(std::move(seq)) {}
  explicit HSeqStmtObj(UnsafeInit) : HStmtObj(UnsafeInit{}) {}
  static TVMFFIAny StructuralVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
    const auto* self = value.cast<const HSeqStmtObj*>();
    Expected<Optional<VisitInterrupt>> result = std::nullopt;
    for (const HStmt& statement : self->seq) {
      result = visitor->VisitExpected(statement);
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(result);
    }
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(result);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }
  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    const auto* self = value.cast<const HSeqStmtObj*>();
    Array<HStmt> mapped = self->seq;
    for (size_t i = 0; i < mapped.size(); ++i) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HStmt, statement, mutator->MutateExpected(mapped[i]));
      if (!statement.same_as(mapped[i])) mapped.Set(i, std::move(statement));
    }
    if (mapped.same_as(self->seq)) return KeepValue(value);
    ObjectPtr<HSeqStmtObj> copy = make_object<HSeqStmtObj>(*self);
    copy->seq = std::move(mapped);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(copy)));
  }
  // Deliberately the same idiom as tvm::MaybeInplaceMutateSeqStmt, including its handle copy
  // of the sequence: mirroring it is what makes the two harnesses comparable on this fixture.
  static TVMFFIAny StructuralMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                                AnyView value) noexcept {
    auto* self = const_cast<HSeqStmtObj*>(value.cast<const HSeqStmtObj*>());
    Array<HStmt> mapped = self->seq;
    for (size_t i = 0; i < mapped.size(); ++i) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HStmt, statement,
                                        mutator->MaybeInplaceMutateIfUniqueExpected(mapped[i]));
      if (!statement.same_as(mapped[i])) mapped.Set(i, std::move(statement));
    }
    if (mapped.same_as(self->seq)) return KeepValue(value);
    self->seq = std::move(mapped);
    return KeepValue(value);
  }
  static void RegisterReflection();
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_final = true;
  static constexpr const char* _type_key = "h.SeqStmt";
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(HSeqStmtObj, HStmtObj);
};

// ---------------------------------------------------------------- registration
template <typename T>
void RegisterHooks() {
  namespace refl = tvm::ffi::reflection;
  refl::TypeAttrDef<T>()
      .attr(refl::type_attr::kStructuralVisit,
            reinterpret_cast<void*>(static_cast<FStructuralVisit>(&T::StructuralVisit)))
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&T::StructuralMutate)));
}
template <typename T>
void RegisterInplaceHook() {
  namespace refl = tvm::ffi::reflection;
  refl::TypeAttrDef<T>().attr(
      refl::type_attr::kStructuralMaybeInplaceMutate,
      reinterpret_cast<void*>(static_cast<FStructuralMutate>(&T::StructuralMaybeInplaceMutate)));
}

void HVarObj::RegisterReflection() {
  reflection::ObjectDef<HVarObj>().def_ro("id", &HVarObj::id);
  RegisterHooks<HVarObj>();
  // A Var has no children, so its in-place hook is its ordinary hook.
  reflection::TypeAttrDef<HVarObj>().attr(
      reflection::type_attr::kStructuralMaybeInplaceMutate,
      reinterpret_cast<void*>(static_cast<FStructuralMutate>(&HVarObj::StructuralMutate)));
}
void HIntImmObj::RegisterReflection() {
  reflection::ObjectDef<HIntImmObj>().def_ro("value_", &HIntImmObj::value_);
  RegisterHooks<HIntImmObj>();
  reflection::TypeAttrDef<HIntImmObj>().attr(
      reflection::type_attr::kStructuralMaybeInplaceMutate,
      reinterpret_cast<void*>(static_cast<FStructuralMutate>(&HIntImmObj::StructuralMutate)));
}
template <typename T>
void HBinOpObj<T>::RegisterReflection() {
  reflection::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
  RegisterHooks<T>();
  RegisterInplaceHook<T>();
}
void HEvaluateObj::RegisterReflection() {
  reflection::ObjectDef<HEvaluateObj>().def_ro("value", &HEvaluateObj::value);
  RegisterHooks<HEvaluateObj>();
  RegisterInplaceHook<HEvaluateObj>();
}
void HSeqStmtObj::RegisterReflection() {
  reflection::ObjectDef<HSeqStmtObj>().def_ro("seq", &HSeqStmtObj::seq);
  RegisterHooks<HSeqStmtObj>();
  RegisterInplaceHook<HSeqStmtObj>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralVisit);
  refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
  refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMaybeInplaceMutate);
  HPrimTypeObj::RegisterReflection();
  HExprObj::RegisterReflection();
  HStmtObj::RegisterReflection();
  HVarObj::RegisterReflection();
  HIntImmObj::RegisterReflection();
  HBinOpObj<HAddObj>::RegisterReflection();
  HBinOpObj<HMulObj>::RegisterReflection();
  HBinOpObj<HFloorDivObj>::RegisterReflection();
  HBinOpObj<HFloorModObj>::RegisterReflection();
  HEvaluateObj::RegisterReflection();
  HSeqStmtObj::RegisterReflection();
}

// ---------------------------------------------------------------------------
// The shipping-API models: what `walk_old` and `map_old` stand in for here.
// ---------------------------------------------------------------------------

/*! \brief Mirrors tvm::PostOrderVisit: recursive, deduplicated by identity, post-order. */
class MinimalPostOrderModel {
 public:
  template <typename F>
  void Run(const ObjectRef& root, F&& callback) {
    Visit(root.get(), callback);
  }

 private:
  template <typename F>
  void Visit(const Object* node, F& callback) {
    if (node == nullptr || !visited_.insert(node).second) return;
    if (const auto* bin = TryBinary(node)) {
      Visit(bin->first, callback);
      Visit(bin->second, callback);
    } else if (const auto* eval = ObjAs<HEvaluateObj>(node)) {
      Visit(eval->value.get(), callback);
    } else if (const auto* seq = ObjAs<HSeqStmtObj>(node)) {
      for (const HStmt& statement : seq->seq) Visit(statement.get(), callback);
    }
    callback(node);
  }
  struct Pair {
    const Object* first;
    const Object* second;
  };
  const Pair* TryBinary(const Object* node) {
    if (const auto* v = ObjAs<HAddObj>(node)) return Store(v);
    if (const auto* v = ObjAs<HMulObj>(node)) return Store(v);
    if (const auto* v = ObjAs<HFloorDivObj>(node)) return Store(v);
    if (const auto* v = ObjAs<HFloorModObj>(node)) return Store(v);
    return nullptr;
  }
  template <typename T>
  const Pair* Store(const T* v) {
    scratch_.first = v->a.get();
    scratch_.second = v->b.get();
    return &scratch_;
  }
  Pair scratch_{nullptr, nullptr};
  std::unordered_set<const Object*> visited_;
};

/*!
 * \brief Mirrors the pre-structural tvm::Substitute shape: a recursive, non-memoizing
 *        mutator with no variable remapping.
 *
 * As in the earlier reports, this is deliberately leaner than what the engine has to do,
 * which is why the mini-TIR map delta reads as a regression against it while the real-TVM
 * map delta reads as an improvement against `Substitute`.  The two answer different
 * questions and must not be read as contradicting each other.
 */
class MinimalSubstituteModel {
 public:
  explicit MinimalSubstituteModel(std::function<Optional<HPrimExpr>(const HVarObj*)> vmap)
      : vmap_(std::move(vmap)) {}

  Any Run(const Any& root) {
    if (auto stmt = root.as<HStmt>()) return Any(MutateStmt(*stmt));
    return Any(MutateExpr(root.cast<HPrimExpr>()));
  }

 private:
  HPrimExpr MutateExpr(const HPrimExpr& expr) {
    if (const auto* var = expr.as<HVarObj>()) {
      Optional<HPrimExpr> replaced = vmap_(var);
      return replaced.has_value() ? replaced.value() : expr;
    }
    if (const auto* v = expr.as<HAddObj>()) return MutateBinary<HAddObj>(expr, v);
    if (const auto* v = expr.as<HMulObj>()) return MutateBinary<HMulObj>(expr, v);
    if (const auto* v = expr.as<HFloorDivObj>()) return MutateBinary<HFloorDivObj>(expr, v);
    if (const auto* v = expr.as<HFloorModObj>()) return MutateBinary<HFloorModObj>(expr, v);
    return expr;
  }
  template <typename T>
  HPrimExpr MutateBinary(const HPrimExpr& expr, const T* self) {
    HPrimExpr a = MutateExpr(self->a);
    HPrimExpr b = MutateExpr(self->b);
    if (a.same_as(self->a) && b.same_as(self->b)) return expr;
    ObjectPtr<T> copy = make_object<T>(*self);
    copy->a = std::move(a);
    copy->b = std::move(b);
    return HPrimExpr(std::move(copy));
  }
  HStmt MutateStmt(const HStmt& stmt) {
    if (const auto* eval = stmt.as<HEvaluateObj>()) {
      HPrimExpr mapped = MutateExpr(eval->value);
      if (mapped.same_as(eval->value)) return stmt;
      ObjectPtr<HEvaluateObj> copy = make_object<HEvaluateObj>(*eval);
      copy->value = std::move(mapped);
      return HStmt(std::move(copy));
    }
    if (const auto* seq = stmt.as<HSeqStmtObj>()) {
      Array<HStmt> mapped = seq->seq;
      for (size_t i = 0; i < mapped.size(); ++i) {
        HStmt statement = MutateStmt(mapped[i]);
        if (!statement.same_as(mapped[i])) mapped.Set(i, std::move(statement));
      }
      if (mapped.same_as(seq->seq)) return stmt;
      ObjectPtr<HSeqStmtObj> copy = make_object<HSeqStmtObj>(*seq);
      copy->seq = std::move(mapped);
      return HStmt(std::move(copy));
    }
    return stmt;
  }
  std::function<Optional<HPrimExpr>(const HVarObj*)> vmap_;
};

}  // namespace mini
}  // namespace ffi
}  // namespace tvm

// ---------------------------------------------------------------------------

namespace mini_tir {

using namespace tvm::ffi;         // NOLINT(build/namespaces)
using namespace tvm::ffi::mini;   // NOLINT(build/namespaces)
using namespace tvm::ffi::bench;  // NOLINT(build/namespaces)

constexpr const char* kHarness = "mini-tir";

// ---- node sizes -----------------------------------------------------------
std::unordered_map<int32_t, int64_t>* NodeSizeTable() {
  static std::unordered_map<int32_t, int64_t> table;
  return &table;
}
template <typename T>
void RegisterNodeSize() {
  (*NodeSizeTable())[T::RuntimeTypeIndex()] = static_cast<int64_t>(sizeof(T));
}
int64_t NodeSize(int32_t type_index, const Object* obj) {
  if (type_index == TypeIndex::kTVMFFIArray) {
    const auto* array = static_cast<const ArrayObj*>(obj);
    return static_cast<int64_t>(sizeof(ArrayObj) + array->size() * sizeof(Any));
  }
  auto it = NodeSizeTable()->find(type_index);
  return it == NodeSizeTable()->end() ? 0 : it->second;
}
void RegisterNodeSizes() {
  RegisterNodeSize<HPrimTypeObj>();
  RegisterNodeSize<HVarObj>();
  RegisterNodeSize<HIntImmObj>();
  RegisterNodeSize<HAddObj>();
  RegisterNodeSize<HMulObj>();
  RegisterNodeSize<HFloorDivObj>();
  RegisterNodeSize<HFloorModObj>();
  RegisterNodeSize<HEvaluateObj>();
  RegisterNodeSize<HSeqStmtObj>();
}

// ---- fixtures -------------------------------------------------------------
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

// Composition of the `seq-L` fixture, identical in shape to the real-TVM one.
constexpr int64_t kSeqSharedNodes = 3;
constexpr int64_t kSeqRebuiltTail = 2;
constexpr int64_t kSeqUniqueNodes(int length) { return 4LL * length + kSeqSharedNodes; }
constexpr int64_t kSeqRebuiltRetained(int length) { return 3LL * length + kSeqRebuiltTail; }
constexpr int64_t kSeqChanged(int length) { return 3LL * length + 1; }
constexpr int64_t kSeqRebuiltMoved(int length) { return 3LL * length + 1; }

/*! \brief floordiv(o*16+i, 32)*32 + floormod(o*16+i, 32) -- the split/fuse index expression. */
HPrimExpr SplitFuse(bool shared) {
  HPrimExpr q = Bin<HAddObj>(Bin<HMulObj>(Outer(), Imm(16)), Inner());
  HPrimExpr r = shared ? q : Bin<HAddObj>(Bin<HMulObj>(Outer(), Imm(16)), Inner());
  return Bin<HAddObj>(Bin<HMulObj>(Bin<HFloorDivObj>(q, Imm(32)), Imm(32)),
                      Bin<HFloorModObj>(r, Imm(32)));
}

HStmt LongSeq(int length) {
  Array<HStmt> body;
  body.reserve(length);
  for (int i = 0; i < length; ++i) {
    body.push_back(HStmt(make_object<HEvaluateObj>(
        Bin<HAddObj>(Bin<HMulObj>(Outer(), Imm(i + 2)), Inner()))));
  }
  return HStmt(make_object<HSeqStmtObj>(body));
}

// ---- arms -----------------------------------------------------------------
size_t g_sink = 0;

Any SwapVars(const HVar& var) {
  if (var->id == 0) return Any(Replacement());
  if (var->id == 2) return Any(Outer());
  return Any(var);
}
Optional<HPrimExpr> SwapVarsOld(const HVarObj* var) {
  if (var->id == 0) return Optional<HPrimExpr>(Replacement());
  if (var->id == 2) return Optional<HPrimExpr>(Outer());
  return std::nullopt;
}

void RunWalkArm(ArmId id, AnyView root) {
  switch (id) {
    case ArmId::kWalkFloor: {
      MinimalVisitorObj visitor;
      g_sink += visitor.VisitExpected(root).is_err();
      break;
    }
    case ArmId::kWalk: {
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
      break;
    }
    case ArmId::kWalkNever: {
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
      break;
    }
    case ArmId::kWalkOld: {
      size_t matched = 0;
      MinimalPostOrderModel model;
      model.Run(root.cast<ObjectRef>(),
                [&](const Object* node) { matched += ObjAs<HVarObj>(node) != nullptr; });
      g_sink += matched;
      break;
    }
    default:
      std::abort();
  }
}

Any MapOldOnce(Any root) {
  MinimalSubstituteModel model(SwapVarsOld);
  return model.Run(root);
}

void RunMapArm(ArmId id, Ownership ownership, Any* slot) {
  auto take = [&]() -> Any { return ownership == Ownership::kMoved ? std::move(*slot) : Any(*slot); };
  auto give = [&](Any result) {
    if (ownership == Ownership::kMoved) {
      *slot = std::move(result);
    } else {
      g_sink += result.type_index();
    }
  };
  switch (id) {
    case ArmId::kMapFloor: {
      MinimalMutatorObj mutator;
      Any input = take();
      Expected<Any> result = ownership == Ownership::kMoved
                                 ? mutator.MaybeInplaceMutateIfUniqueExpected(input)
                                 : mutator.MutateExpected(input);
      g_sink += result.is_err();
      give(result.value());
      break;
    }
    case ArmId::kMapNever:
      give(StructuralMap<WalkOrder::kPostOrder>(
          take(), [](const HPrimType& value) { return Any(value); }));
      break;
    case ArmId::kMapIdentity:
      give(StructuralMap<WalkOrder::kPostOrder>(take(), [](const HVar& var) { return Any(var); }));
      break;
    case ArmId::kMapReplace:
      give(StructuralMap<WalkOrder::kPostOrder>(take(), SwapVars));
      break;
    case ArmId::kMapOld:
      give(MapOldOnce(take()));
      break;
    default:
      std::abort();
  }
}

/*! \brief The harness policy the shared driver in bench_common.h is instantiated on. */
struct MiniTirPolicy {
  static constexpr const char* kName = kHarness;
  static constexpr bool kOldGoesThroughEngine = false;
  static int64_t NodeSize(int32_t type_index, const Object* obj) {
    return mini_tir::NodeSize(type_index, obj);
  }
  static void RunWalkArm(ArmId id, AnyView root) { mini_tir::RunWalkArm(id, root); }
  static void RunMapArm(ArmId id, Ownership ownership, Any* slot) {
    mini_tir::RunMapArm(id, ownership, slot);
  }
  static Any MapReplace(Any root) {
    return StructuralMap<WalkOrder::kPostOrder>(std::move(root), SwapVars);
  }
  static Any MapIdentity(Any root) {
    return StructuralMap<WalkOrder::kPostOrder>(std::move(root),
                                                [](const HVar& var) { return Any(var); });
  }
  static Any MapNever(Any root) {
    return StructuralMap<WalkOrder::kPostOrder>(std::move(root),
                                                [](const HPrimType& value) { return Any(value); });
  }
  static Any MapOld(Any root) { return MapOldOnce(std::move(root)); }
};

}  // namespace mini_tir

int main() {
  using namespace mini_tir;  // NOLINT(build/namespaces)
  EmitStandardProvenance(kHarness);
  RegisterNodeSizes();

  std::vector<Fixture> fixtures;
  {
    Fixture f;
    f.name = "split-fuse-shared";
    f.build = [] { return Any(SplitFuse(true)); };
    f.has_sharing = true;
    fixtures.push_back(f);
  }
  {
    Fixture f;
    f.name = "split-fuse-distinct";
    f.build = [] { return Any(SplitFuse(false)); };
    fixtures.push_back(f);
  }
  for (int length : SeqSweepLengths()) {
    Fixture f;
    f.name = "seq-" + std::to_string(length);
    f.build = [length] { return Any(LongSeq(length)); };
    fixtures.push_back(f);
  }

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
  declare("split-fuse-shared", 12, 9, 3, 6, 2);
  declare("split-fuse-distinct", 15, 9, 1, 8, 2);
  for (int length : SeqSweepLengths()) {
    declare("seq-" + std::to_string(length), kSeqUniqueNodes(length), kSeqRebuiltRetained(length),
            kSeqRebuiltMoved(length), kSeqChanged(length), 2LL * (length - 1));
  }

  for (const Fixture& fixture : fixtures) RunFixture<MiniTirPolicy>(fixture);
  Emit("#sink\t" + std::to_string(g_sink));
  return 0;
}
