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
#ifndef TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_MINI_TIR_H_
#define TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_MINI_TIR_H_

// mini-TIR: a mirror of TVM's TIR node shape built from tvm-ffi types alone, with its own
// structural hooks, plus the traversal machinery the baseline arms measure.
//
// Deliberately not factored against `tvm_hook_override.h`.  The two node sets' hooks read
// almost identically and are written out twice on purpose: a reader can follow either without
// mentally instantiating a template, and a change to one cannot silently reshape the other.
//
// What the node set mirrors, and where it is a model rather than a clone:
//
//   * HExprObj carries a `ty` field, so a PrimExpr-shaped view costs the same checks;
//   * HPrimExpr is a *view* over any HExprObj whose `ty` holds an HPrimType, with the same
//     TypeTraits shape as tvm::PrimExpr: checking a field costs an IsObjectInstance range
//     test, a dereference to reach `ty`, and a second type check on that field;
//   * binary ops are final with two HPrimExpr operands and MutateBinary's guard that skips
//     result-type inference when neither operand's type changed;
//   * HSeqStmtObj's in-place hook has the same two shapes as TVM's, selected by
//     MINI_SEQSTMT_INPLACE_FIX, so before and after are measured on the same footing.
//
// No Span, result types compare a single `bits` field rather than full dtype logic, and only
// four binary ops are mirrored.
//
// The second half of this file is the pre-structural functor machinery -- tvm::NodeFunctor,
// tirx::ExprFunctor/StmtFunctor and the IRApplyVisit / IRSubstitute shapes built on them --
// ported over these types, plus ports of what the pinned apache/tvm ships today.  Those four
// are what the `*_functor` and `*_old` arms measure.

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstring>
#include <functional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bench_common.h"

// Whether HSeqStmtObj's in-place hook is the repaired shape.  The same switch exists in
// `tvm_hook_override.h`, so the before/after is measured identically on both node sets.
#ifndef MINI_SEQSTMT_INPLACE_FIX
#define MINI_SEQSTMT_INPLACE_FIX 1
#endif

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
  // The same idiom as tvm::MaybeInplaceMutateSeqStmt, so the two harnesses stay comparable on
  // this fixture -- including the fix.  Built with -DMINI_SEQSTMT_INPLACE_FIX=0 this is the
  // original shape, whose handle copy of the sequence leaves no element uniquely owned and
  // forces every one of them onto the copy-on-write path; with 1 it is the repaired shape.
  static TVMFFIAny StructuralMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                                AnyView value) noexcept {
    auto* self = const_cast<HSeqStmtObj*>(value.cast<const HSeqStmtObj*>());
#if MINI_SEQSTMT_INPLACE_FIX
    if (self->seq.unique()) {
      Any failure;
      bool failed = false;
      self->seq.MutateByApply([&](HStmt statement) -> HStmt {
        if (TVM_FFI_PREDICT_FALSE(failed)) return statement;
        Expected<Any> mapped = mutator->MaybeInplaceMutateIfUniqueExpected(statement);
        if (TVM_FFI_PREDICT_FALSE(mapped.is_err())) {
          failed = true;
          failure = Any(std::move(mapped).error());
          return statement;
        }
        return details::AnyUnsafe::MoveFromAnyAfterCheck<HStmt>(std::move(mapped).value());
      });
      if (TVM_FFI_PREDICT_FALSE(failed)) {
        return details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(failure));
      }
      return KeepValue(value);
    }
#endif
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

// The four baseline models, ported from apache/tvm rather than approximated.  Included from
// mini_tir_bench.cc inside namespace tvm::ffi::mini.
//
//   walk_functor / map_functor  the pre-structural functor machinery: tvm::NodeFunctor's
//                               static vtable of function pointers indexed by type index,
//                               tirx::ExprFunctor / StmtFunctor's virtual VisitExpr_ /
//                               VisitStmt_ overload sets, and the IRApplyVisit / IRSubstitute
//                               shapes built on them.  This is what apache/tvm main still
//                               ships as PostOrderVisit and Substitute.
//   walk_old / map_old          what the pinned apache/tvm ships *today*: PostOrderVisit is
//                               StructuralWalk plus a dedup set, and Substitute is a custom
//                               StructuralMutatorObj that owns a var remap.  Both are ports
//                               of the current bodies, not of the functor-era ones.
//
// Ported faithfully in mechanism, not in breadth: the node set stays the mini seven.

// ---------------------------------------------------------------------------
// tvm::NodeFunctor -- a vtable of plain function pointers indexed by type index.
// ---------------------------------------------------------------------------

template <typename FType>
class HNodeFunctor;

template <typename R, typename... Args>
class HNodeFunctor<R(const ObjectRef& n, Args...)> {
 private:
  typedef R (*FPointer)(const ObjectRef& n, Args...);
  using TSelf = HNodeFunctor<R(const ObjectRef& n, Args...)>;
  std::vector<FPointer> func_;
  uint32_t begin_type_index_{0};

 public:
  R operator()(const ObjectRef& n, Args... args) const {
    uint32_t type_index = n->type_index();
    if (type_index >= begin_type_index_) {
      uint32_t index = type_index - begin_type_index_;
      if (index < func_.size() && func_[index] != nullptr) {
        return (*func_[index])(n, std::forward<Args>(args)...);
      }
    }
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
    for (int32_t i = type_info->type_depth - 1; i >= 0; --i) {
      type_index = type_info->type_ancestors[i]->type_index;
      if (type_index >= begin_type_index_) {
        uint32_t index = type_index - begin_type_index_;
        if (index < func_.size() && func_[index] != nullptr) {
          return (*func_[index])(n, std::forward<Args>(args)...);
        }
      }
    }
    std::abort();
  }
  template <typename TNode>
  TSelf& set_dispatch(FPointer f) {  // NOLINT(*)
    uint32_t tindex = TNode::RuntimeTypeIndex();
    if (func_.size() <= tindex) func_.resize(tindex + 1, nullptr);
    func_[tindex] = f;
    return *this;
  }
  /*! \brief Compact the table so dispatch is one bounds test plus one load. */
  void Finalize() {
    while (begin_type_index_ < func_.size() && func_[begin_type_index_] == nullptr) {
      ++begin_type_index_;
    }
    size_t new_size = func_.size() - begin_type_index_;
    if (begin_type_index_ != 0) {
      std::memmove(func_.data(), func_.data() + begin_type_index_, new_size * sizeof(FPointer));
    }
    func_.resize(new_size);
    func_.shrink_to_fit();
  }
};

// ---------------------------------------------------------------------------
// tirx::ExprFunctor / StmtFunctor.
// ---------------------------------------------------------------------------

#define H_EXPR_FUNCTOR_DEFAULT \
  { return VisitExprDefault_(op, std::forward<Args>(args)...); }
#define H_EXPR_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) { \
    return self->VisitExpr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });
#define H_FOR_EACH_EXPR(F) \
  F(HVarObj) F(HIntImmObj) F(HAddObj) F(HMulObj) F(HFloorDivObj) F(HFloorModObj)
#define H_STMT_FUNCTOR_DEFAULT \
  { return VisitStmtDefault_(op, std::forward<Args>(args)...); }
#define H_STMT_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) { \
    return self->VisitStmt_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });
#define H_FOR_EACH_STMT(F) F(HEvaluateObj) F(HSeqStmtObj)

template <typename FType>
class HExprFunctor;

template <typename R, typename... Args>
class HExprFunctor<R(const HPrimExpr& n, Args...)> {
 private:
  using TSelf = HExprFunctor<R(const HPrimExpr& n, Args...)>;
  using FType = HNodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  using result_type = R;
  virtual ~HExprFunctor() {}
  R operator()(const HPrimExpr& n, Args... args) { return VisitExpr(n, std::forward<Args>(args)...); }
  virtual R VisitExpr(const HPrimExpr& n, Args... args) {
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
#define H_DECL_VISIT_EXPR(OP) virtual R VisitExpr_(const OP* op, Args... args) H_EXPR_FUNCTOR_DEFAULT;
  H_FOR_EACH_EXPR(H_DECL_VISIT_EXPR)
#undef H_DECL_VISIT_EXPR
  virtual R VisitExprDefault_(const Object*, Args...) { std::abort(); }

 private:
  static FType InitVTable() {
    FType vtable;
    H_FOR_EACH_EXPR(H_EXPR_FUNCTOR_DISPATCH)
    vtable.Finalize();
    return vtable;
  }
};

template <typename FType>
class HStmtFunctor;

template <typename R, typename... Args>
class HStmtFunctor<R(const HStmt& n, Args...)> {
 private:
  using TSelf = HStmtFunctor<R(const HStmt& n, Args...)>;
  using FType = HNodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  using result_type = R;
  virtual ~HStmtFunctor() {}
  R operator()(const HStmt& n, Args... args) { return VisitStmt(n, std::forward<Args>(args)...); }
  virtual R VisitStmt(const HStmt& n, Args... args) {
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
#define H_DECL_VISIT_STMT(OP) virtual R VisitStmt_(const OP* op, Args... args) H_STMT_FUNCTOR_DEFAULT;
  H_FOR_EACH_STMT(H_DECL_VISIT_STMT)
#undef H_DECL_VISIT_STMT
  virtual R VisitStmtDefault_(const Object*, Args...) { std::abort(); }

 private:
  static FType InitVTable() {
    FType vtable;
    H_FOR_EACH_STMT(H_STMT_FUNCTOR_DISPATCH)
    vtable.Finalize();
    return vtable;
  }
};

// ---------------------------------------------------------------------------
// tirx::ExprVisitor / StmtVisitor / StmtExprVisitor.
// ---------------------------------------------------------------------------

class HExprVisitor : public HExprFunctor<void(const HPrimExpr&)> {
 public:
  using HExprFunctor::operator();

 protected:
  using HExprFunctor::VisitExpr;
  void VisitExpr_(const HVarObj*) override {}
  void VisitExpr_(const HIntImmObj*) override {}
#define H_DEFINE_BINOP_VISIT(OP)                 \
  void VisitExpr_(const OP* op) override {       \
    this->VisitExpr(op->a);                      \
    this->VisitExpr(op->b);                      \
  }
  H_DEFINE_BINOP_VISIT(HAddObj)
  H_DEFINE_BINOP_VISIT(HMulObj)
  H_DEFINE_BINOP_VISIT(HFloorDivObj)
  H_DEFINE_BINOP_VISIT(HFloorModObj)
#undef H_DEFINE_BINOP_VISIT
};

class HStmtVisitor : public HStmtFunctor<void(const HStmt&)> {
 public:
  using HStmtFunctor::operator();

 protected:
  using HStmtFunctor::VisitStmt;
  virtual void VisitExpr(const HPrimExpr&) {}
  void VisitStmt_(const HEvaluateObj* op) override { this->VisitExpr(op->value); }
  void VisitStmt_(const HSeqStmtObj* op) override {
    for (const HStmt& statement : op->seq) this->VisitStmt(statement);
  }
};

class HStmtExprVisitor : public HExprVisitor, public HStmtVisitor {
 public:
  using HExprVisitor::operator();
  using HStmtVisitor::operator();

 protected:
  using HExprVisitor::VisitExpr;
  using HStmtVisitor::VisitStmt;
  void VisitExpr(const HPrimExpr& e) override { return HExprVisitor::VisitExpr(e); }
};

// ---------------------------------------------------------------------------
// tirx::ExprMutator / StmtMutator / StmtExprMutator.
// ---------------------------------------------------------------------------

class HExprMutator : public HExprFunctor<HPrimExpr(const HPrimExpr&)> {
 public:
  using HExprFunctor::operator();

 protected:
  using HExprFunctor::VisitExpr;
  HPrimExpr VisitExpr_(const HVarObj* op) override { return GetRef<HPrimExpr>(op); }
  HPrimExpr VisitExpr_(const HIntImmObj* op) override { return GetRef<HPrimExpr>(op); }
  // Mirrors DEFINE_BIOP_EXPR_MUTATE_: on a change the operator's own constructor rebuilds the
  // node, which infers the result type -- there is no MutateBinary guard on this path.
#define H_DEFINE_BINOP_MUTATE(OP)                                          \
  HPrimExpr VisitExpr_(const OP* op) override {                            \
    HPrimExpr a = this->VisitExpr(op->a);                                  \
    HPrimExpr b = this->VisitExpr(op->b);                                  \
    if (a.same_as(op->a) && b.same_as(op->b)) return GetRef<HPrimExpr>(op); \
    Expected<HPrimType> rty = OP::ResultType(a, b);                        \
    if (rty.is_err()) std::abort();                                        \
    return HPrimExpr(make_object<OP>(Any(rty.value()), std::move(a), std::move(b))); \
  }
  H_DEFINE_BINOP_MUTATE(HAddObj)
  H_DEFINE_BINOP_MUTATE(HMulObj)
  H_DEFINE_BINOP_MUTATE(HFloorDivObj)
  H_DEFINE_BINOP_MUTATE(HFloorModObj)
#undef H_DEFINE_BINOP_MUTATE
};

class HStmtMutator : public HStmtFunctor<HStmt(const HStmt&)> {
 public:
  /*! \brief Mirrors tirx::StmtMutator::operator()(Stmt): by value, so COW can trigger. */
  HStmt operator()(HStmt stmt) {
    allow_copy_on_write_ = true;
    return VisitStmt(stmt);
  }

 protected:
  bool allow_copy_on_write_{false};

  template <typename TNode>
  ObjectPtr<TNode> CopyOnWrite(const TNode* node) {
    if (allow_copy_on_write_) return GetObjectPtr<TNode>(const_cast<TNode*>(node));
    return make_object<TNode>(*node);
  }
  /*! \brief Mirrors StmtMutator::Internal::MutateArray, including its COW discipline. */
  template <typename T, typename F>
  Array<T> MutateArray(const Array<T>& arr, F fmutate) {
    if (allow_copy_on_write_ && arr.unique()) {
      const_cast<Array<T>&>(arr).MutateByApply(fmutate);
      return arr;
    }
    bool allow_cow = false;
    std::swap(allow_cow, allow_copy_on_write_);
    Array<T> copy = arr.Map(fmutate);
    std::swap(allow_cow, allow_copy_on_write_);
    return copy;
  }
  HStmt VisitStmt(const HStmt& stmt) override {
    if (allow_copy_on_write_ && !stmt.unique()) {
      allow_copy_on_write_ = false;
      HStmt ret = HStmtFunctor::VisitStmt(stmt);
      allow_copy_on_write_ = true;
      return ret;
    }
    return HStmtFunctor::VisitStmt(stmt);
  }
  virtual HPrimExpr VisitExpr(const HPrimExpr& e) { return e; }
  HStmt VisitStmt_(const HEvaluateObj* op) override {
    HPrimExpr value = this->VisitExpr(op->value);
    if (value.same_as(op->value)) return GetRef<HStmt>(op);
    ObjectPtr<HEvaluateObj> n = CopyOnWrite(op);
    n->value = std::move(value);
    return HStmt(n);
  }
  HStmt VisitStmt_(const HSeqStmtObj* op) override {
    Array<HStmt> seq = MutateArray(op->seq, [this](const HStmt& s) { return this->VisitStmt(s); });
    if (seq.same_as(op->seq)) return GetRef<HStmt>(op);
    ObjectPtr<HSeqStmtObj> n = CopyOnWrite(op);
    n->seq = std::move(seq);
    return HStmt(n);
  }
};

class HStmtExprMutator : public HExprMutator, public HStmtMutator {
 public:
  using HStmtMutator::operator();
  using HExprMutator::operator();

 protected:
  using HExprMutator::VisitExpr;
  using HStmtMutator::VisitStmt;
  HPrimExpr VisitExpr(const HPrimExpr& e) override { return HExprMutator::VisitExpr(e); }
};

// ---------------------------------------------------------------------------
// walk_functor: tirx::IRApplyVisit, apache/tvm main's PostOrderVisit.
// ---------------------------------------------------------------------------

class HIRApplyVisit : public HStmtExprVisitor {
 public:
  explicit HIRApplyVisit(std::function<void(const ObjectRef&)> f) : f_(f) {}

  void VisitExpr(const HPrimExpr& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    HExprVisitor::VisitExpr(node);
    f_(node);
  }
  void VisitStmt(const HStmt& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    HStmtVisitor::VisitStmt(node);
    f_(node);
  }

 private:
  std::function<void(const ObjectRef&)> f_;
  std::unordered_set<const Object*> visited_;
};

inline void FunctorPostOrderVisit(const ObjectRef& node,
                                  std::function<void(const ObjectRef&)> fvisit) {
  HIRApplyVisit visitor(std::move(fvisit));
  if (auto stmt = node.as<HStmt>()) {
    visitor(*stmt);
  } else {
    visitor(HPrimExpr(details::ObjectUnsafe::ObjectPtrFromObjectRef<HExprObj>(node)));
  }
}

// ---------------------------------------------------------------------------
// map_functor: tirx::IRSubstitute, apache/tvm main's Substitute.
// ---------------------------------------------------------------------------

class HIRSubstitute : public HStmtExprMutator {
 public:
  explicit HIRSubstitute(std::function<Optional<HPrimExpr>(const HVar&)> vmap)
      : vmap_(std::move(vmap)) {}

  HPrimExpr VisitExpr_(const HVarObj* op) final {
    HVar var = GetRef<HVar>(op);
    if (Optional<HPrimExpr> ret = vmap_(var)) return ret.value();
    return HPrimExpr(var);
  }

 private:
  std::function<Optional<HPrimExpr>(const HVar&)> vmap_;
};

inline Any FunctorSubstitute(Any root, std::function<Optional<HPrimExpr>(const HVar&)> vmap) {
  HIRSubstitute mutator(std::move(vmap));
  if (auto stmt = root.as<HStmt>()) return Any(mutator(*stmt));
  return Any(mutator(root.cast<HPrimExpr>()));
}

// ---------------------------------------------------------------------------
// walk_old: the pinned apache/tvm's PostOrderVisit, which is the structural engine
// plus a dedup set -- not the functor machinery above.
// ---------------------------------------------------------------------------

template <typename F>
inline void ShippingPostOrderVisit(AnyView node, F fvisit) {
  std::unordered_set<const Object*> visited;
  StructuralWalk<WalkOrder::kPostOrder>(
      node,
      [&](const HStmt& current) -> WalkResult {
        if (!visited.insert(current.get()).second) return WalkResult::Advance();
        fvisit(current);
        return WalkResult::Advance();
      },
      [&](const HExpr& current) -> WalkResult {
        if (!visited.insert(current.get()).second) return WalkResult::Advance();
        fvisit(current);
        return WalkResult::Advance();
      });
}

// ---------------------------------------------------------------------------
// map_old: the pinned apache/tvm's Substitute -- StructuralSubstituteMutatorObj, a custom
// StructuralMutatorObj owning its own var remap, dispatched through the registered hooks.
// ---------------------------------------------------------------------------

class HStructuralSubstituteMutatorObj final : public StructuralMutatorObj {
 public:
  explicit HStructuralSubstituteMutatorObj(std::function<Optional<HPrimExpr>(const HVar&)> vmap)
      : StructuralMutatorObj(VTable()), vmap_(std::move(vmap)) {}

 private:
  static const StructuralMutatorVTable* VTable() {
    static const StructuralMutatorVTable vtable{
        &HStructuralSubstituteMutatorObj::DispatchMutate,
        &HStructuralSubstituteMutatorObj::DispatchMaybeInplaceMutate,
        &HStructuralSubstituteMutatorObj::DispatchVarRemapGet,
        &HStructuralSubstituteMutatorObj::DispatchVarRemapSet,
    };
    return &vtable;
  }
  static TVMFFIAny DispatchMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    auto* self = static_cast<HStructuralSubstituteMutatorObj*>(mutator);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(self->MutateImpl(value, false));
  }
  static TVMFFIAny DispatchMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                              AnyView value) noexcept {
    auto* self = static_cast<HStructuralSubstituteMutatorObj*>(mutator);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(self->MutateImpl(value, true));
  }
  static TVMFFIAny DispatchVarRemapGet(StructuralMutatorObj* mutator, AnyView var) noexcept {
    auto* self = static_cast<HStructuralSubstituteMutatorObj*>(mutator);
    try {
      if (self->def_region_kind() != kTVMFFIDefRegionKindNone) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(nullptr)));
      }
      std::optional<Any> mapped = self->var_remap_.Get(var.cast<ObjectRef>());
      return details::ExpectedUnsafe::MoveToTVMFFIAny(
          Expected<Any>(mapped.has_value() ? *std::move(mapped) : Any(nullptr)));
    } catch (const Error& error) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Unexpected(error)));
    }
  }
  static TVMFFIAny DispatchVarRemapSet(StructuralMutatorObj* mutator, AnyView var,
                                       AnyView mapped_value) noexcept {
    auto* self = static_cast<HStructuralSubstituteMutatorObj*>(mutator);
    try {
      if (self->def_region_kind() != kTVMFFIDefRegionKindNone) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<void>());
      }
      self->var_remap_.Set(var.cast<ObjectRef>(), Any(mapped_value));
      return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<void>());
    } catch (const Error& error) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<void>(Unexpected(error)));
    }
  }

  Expected<Any> MutateImpl(AnyView value, bool maybe_inplace) noexcept {
    try {
      std::optional<HVar> var = value.as<HVar>();
      if (!var.has_value() || def_region_kind() != kTVMFFIDefRegionKindNone) {
        return maybe_inplace ? DefaultMaybeInplaceMutateExpected(value)
                             : DefaultMutateExpected(value);
      }
      Expected<Any> cached = VarRemapGetExpected(value);
      if (TVM_FFI_PREDICT_FALSE(cached.is_err())) return cached;
      if (details::ExpectedUnsafe::GetData(cached).type_index() != TypeIndex::kTVMFFINone) {
        return cached;
      }
      Expected<Any> mapped = maybe_inplace ? DefaultMaybeInplaceMutateExpected(value)
                                           : DefaultMutateExpected(value);
      if (TVM_FFI_PREDICT_FALSE(mapped.is_err())) return mapped;
      Any result = details::ExpectedUnsafe::GetData(mapped);
      if (Optional<HPrimExpr> replacement = vmap_(*var)) {
        result = Any(replacement.value());
      }
      Expected<void> set_result = VarRemapSetExpected(value, result);
      if (set_result.is_err()) return Unexpected(std::move(set_result).error());
      return result;
    } catch (const Error& error) {
      return Unexpected(error);
    }
  }

  std::function<Optional<HPrimExpr>(const HVar&)> vmap_;
  Map<ObjectRef, Any> var_remap_;
};

inline Any ShippingSubstitute(AnyView root,
                              std::function<Optional<HPrimExpr>(const HVar&)> vmap) {
  StructuralMutator mutator(make_object<HStructuralSubstituteMutatorObj>(std::move(vmap)));
  return mutator->MaybeInplaceMutateIfUniqueExpected(root).value();
}

}  // namespace mini
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_MINI_TIR_H_
