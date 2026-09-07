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

// mini-TIR: apache/tvm's TIR node layouts rebuilt from tvm-ffi types alone, with ports of
// their structural hooks, plus the traversal machinery the baseline arms measure.
//
// THE FIDELITY REQUIREMENT.  mini-TIR is not a sketch of TVM's node shape; every node it does
// have is its apache/tvm counterpart's layout -- same fields, same order, same types -- and
// every hook it does have is a port of that counterpart's hook body.  The ONLY difference
// permitted between the two harnesses is WHICH node types exist: mini-TIR carries a reduced
// set (four binary operators, one statement pair, no dialect nodes) and nothing else.  Where a
// fixture exists in both, a node must cost the same to size, lay out and traverse in either,
// and the acceptance test is that mini and real agree WITHIN A SINGLE HOST.
//
// That requirement is enforced rather than asserted in prose: both binaries emit `#nodesize`
// lines for the counterpart pairs and `report.py` fails a run in which any pair disagrees.
//
// mini-TIR keeping its own hook file is not a divergence.  It is the same arrangement as
// `tvm_hook_override.h`, which also writes TVM's hooks out locally and always overrides; the
// two files are deliberately unfactored so a reader can follow either without instantiating a
// template and a change to one cannot silently reshape the other.
//
// The second half of this file is the pre-structural functor machinery -- tvm::NodeFunctor,
// tirx::ExprFunctor/StmtFunctor and the IRApplyVisit / IRSubstitute shapes built on them --
// ported over these types, plus ports of what the pinned apache/tvm ships today.  Those four
// are what the `*_functor` and `*_old` arms measure.

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bench_common.h"

namespace tvm {
namespace ffi {
namespace mini {

// ===========================================================================================
// THE NODE SET.  Every node below is field-for-field, order-for-order and type-for-type its
// apache/tvm counterpart at a1031a2177, so a node costs the same to size, to lay out and to
// traverse in either harness.  The only permitted difference between the two harnesses is
// WHICH node types exist -- mini-TIR has a reduced set -- never what one of them contains.
//
//   mini-TIR              apache/tvm a1031a2177          declared in
//   -------------------   ---------------------------    -----------------------------------
//   HSourceNameObj        SourceNameNode                 include/tvm/ir/source_map.h
//   HSpanObj              SpanNode                       include/tvm/ir/source_map.h
//   HTypeObj              TypeNode                       include/tvm/ir/base_expr.h
//   HOpaqueTypeObj        OpaqueTypeNode                 include/tvm/ir/base_expr.h
//   HPrimTypeObj          PrimTypeNode                   include/tvm/ir/base_expr.h
//   HAttrsObj             AttrsNode                      include/tvm/ir/attrs.h
//   HExprObj              ExprNode                       include/tvm/ir/base_expr.h
//   HExpr / HPrimExpr     Expr / PrimExpr                include/tvm/ir/base_expr.h
//   HVarObj               VarNode                        include/tvm/ir/expr.h
//   HIntImmObj            IntImmNode                     include/tvm/ir/expr.h
//   HOpObj                OpNode                         include/tvm/ir/op.h
//   HCallObj              CallNode                       include/tvm/ir/expr.h
//   HBinOpObj<T>          prim::BinaryOpNode<T>          include/tvm/ir/prim/expr.h
//   HStmtObj              tirx::StmtNode                 include/tvm/tirx/stmt.h
//   HEvaluateObj          tirx::EvaluateNode             include/tvm/tirx/stmt.h
//   HSeqStmtObj           tirx::SeqStmtNode              include/tvm/tirx/stmt.h
//
// Fields a fixture never populates are still declared, because they are still paid for: a
// null `span` is eight bytes in every Expr, Stmt and Type, and leaving it out made mini's
// nodes a different size from real TVM's.  `_type_child_slots`, `_type_final` and
// `_type_s_eq_hash_kind` are copied too -- they decide whether an `IsObjectInstance` check is
// an equality test or a range test, which is on the hot path of every field cast.
//
// The reduced set is the modelling licence and the whole of it: four binary operators rather
// than TVM's full arithmetic set, one statement pair, and no dialect nodes.  A node that does
// exist here is not a model of its counterpart, it is a copy of its layout.
//
// `mini_tir_bench.cc` emits `#nodesize` for each of these and `real_tvm_bench.cc` emits the
// same lines for its counterparts; `report.py` fails a run in which any counterpart pair
// disagrees, so this table cannot rot silently.
// ===========================================================================================

// ---------------------------------------------------------------- SourceName (ir.SourceName)
class HSourceNameObj final : public Object {
 public:
  String name;

  static void RegisterReflection() {
    reflection::ObjectDef<HSourceNameObj>().def_ro("name", &HSourceNameObj::name);
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.SourceName", HSourceNameObj, Object);
};
class HSourceName : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HSourceName, ObjectRef, HSourceNameObj);
};

// ---------------------------------------------------------------- Span (ir.Span)
// Never constructed by any fixture, exactly as in TVM, where `Span()` is a null ObjectRef.
// It is declared because the null handle occupies a word in every Expr, Stmt and Type.
class HSpanObj : public Object {
 public:
  HSourceName source_name;
  int line;
  int column;
  int end_line;
  int end_column;

  static void RegisterReflection() {
    reflection::ObjectDef<HSpanObj>()
        .def_ro("source_name", &HSpanObj::source_name)
        .def_ro("line", &HSpanObj::line)
        .def_ro("column", &HSpanObj::column)
        .def_ro("end_line", &HSpanObj::end_line)
        .def_ro("end_column", &HSpanObj::end_column);
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("h.Span", HSpanObj, Object);
};
class HSpan : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HSpan, ObjectRef, HSpanObj);
};

// ---------------------------------------------------------------- Type (ir.Type)
class HTypeObj : public Object {
 public:
  mutable HSpan span;

  static void RegisterReflection() {
    reflection::ObjectDef<HTypeObj>().def_ro("span", &HTypeObj::span,
                                             reflection::DefaultValue(HSpan()),
                                             reflection::AttachFieldFlag::SEqHashIgnore());
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr bool _type_s_eq_hash_subclass_kind_fixed = true;
  static constexpr uint32_t _type_child_slots = 14;
  TVM_FFI_DECLARE_OBJECT_INFO("h.Type", HTypeObj, Object);
};
class HType : public ObjectRef {
 public:
  /*! \brief Sentinel for a type that has not been populated yet; mirrors Type::Missing(). */
  static HType Missing();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(HType, ObjectRef, HTypeObj);
};

class HOpaqueTypeObj final : public HTypeObj {
 public:
  static void RegisterReflection() { reflection::ObjectDef<HOpaqueTypeObj>(); }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.OpaqueType", HOpaqueTypeObj, HTypeObj);
};
class HOpaqueType final : public HType {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(HOpaqueType, HType, HOpaqueTypeObj);
};

/*! \brief Mirrors PrimTypeNode: a DLDataType, not a bit width. */
class HPrimTypeObj final : public HTypeObj {
 public:
  DLDataType dtype;

  static void RegisterReflection() {
    reflection::ObjectDef<HPrimTypeObj>().def_ro("dtype", &HPrimTypeObj::dtype);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.PrimType", HPrimTypeObj, HTypeObj);
};
class HPrimType : public HType {
 public:
  /*! \brief The interned int type the fixtures use, mirroring PrimType::Int(bits). */
  static HPrimType Int(int bits);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(HPrimType, HType, HPrimTypeObj);
};

// ---------------------------------------------------------------- Attrs (ir.Attrs)
// Never constructed, exactly as in the Call fixture, where `attrs` is a null handle. Declared
// because CallNode pays a word for it and both Call hooks skip it.
class HAttrsObj : public Object {
 public:
  static void RegisterReflection() { reflection::ObjectDef<HAttrsObj>(); }
  static constexpr uint32_t _type_child_slots = 1;
  TVM_FFI_DECLARE_OBJECT_INFO("h.Attrs", HAttrsObj, Object);
};
class HAttrs : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HAttrs, ObjectRef, HAttrsObj);
};

// ---------------------------------------------------------------- Expr (ir.Expr)
class HExprObj : public Object {
 public:
  mutable HSpan span;
  mutable HType ty = HType::Missing();

  HExprObj() {}
  explicit HExprObj(HType ty) : ty(std::move(ty)) {}

  static void RegisterReflection() {
    reflection::ObjectDef<HExprObj>()
        .def_ro("span", &HExprObj::span, reflection::DefaultValue(HSpan()),
                reflection::AttachFieldFlag::SEqHashIgnore())
        .def_ro("ty", &HExprObj::ty, reflection::DefaultValue(HType::Missing()));
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr uint32_t _type_child_slots = 64;
  TVM_FFI_DECLARE_OBJECT_INFO("h.Expr", HExprObj, Object);
};
class HExpr : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HExpr, ObjectRef, HExprObj);
};

/*!
 * \brief A view over any HExprObj whose `ty` is an HPrimType.  Mirrors tvm::PrimExpr, which is
 *        `TypedExpr<PrimType>`: a type category rather than a node category.
 */
class HPrimExpr : public HExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HPrimExpr, HExpr, HExprObj);
};

}  // namespace mini

template <>
inline constexpr bool use_default_type_traits_v<mini::HPrimType> = false;

/*! \brief Mirrors TypeTraits<PrimType>: an ObjectRef with a DLDataType fallback. */
template <>
struct TypeTraits<mini::HPrimType>
    : public ObjectRefWithFallbackTraitsBase<mini::HPrimType, DLDataType> {
  TVM_FFI_INLINE static mini::HPrimType ConvertFallbackValue(DLDataType dtype);
};

template <>
inline constexpr bool use_default_type_traits_v<mini::HPrimExpr> = false;

/*!
 * \brief Mirrors TypeTraits<TypedExpr<ExpectedType>>: an IsObjectInstance range test, a
 *        dereference to reach `ty`, and a second strict check on that field.
 */
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
    // Non-owning: this only reads `ty`, and the owning form's incref/decref pair costs two
    // atomics per check on a path every typed field assignment takes.
    const auto* e = details::ObjectUnsafe::RawObjectPtrFromUnowned<mini::HExprObj>(src->v_obj);
    return details::AnyUnsafe::CheckAnyStrict<mini::HPrimType>(e->ty);
  }
  TVM_FFI_INLINE static std::optional<mini::HPrimExpr> TryCastFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) return CopyFromAnyViewAfterCheck(src);
    return std::nullopt;
  }
};

namespace mini {

// `Type` and `PrimType` are NOTNULLABLE in TVM, and their constructors assign `data_` from
// inside a .cc.  A header-only mirror reaches the same place through ObjectUnsafe.
template <typename TRef, typename TObj>
TVM_FFI_INLINE TRef MakeNotNullableRef(ObjectPtr<TObj> node) {
  return details::ObjectUnsafe::ObjectRefFromObjectPtr<TRef>(std::move(node));
}

inline HType HType::Missing() {
  static HType missing = MakeNotNullableRef<HType>(make_object<HOpaqueTypeObj>());
  return missing;
}

inline HPrimType MakePrimType(DLDataType dtype) {
  ObjectPtr<HPrimTypeObj> node = make_object<HPrimTypeObj>();
  node->dtype = dtype;
  return MakeNotNullableRef<HPrimType>(std::move(node));
}

inline HPrimType HPrimType::Int(int bits) {
  // Interned exactly as TVM interns its common PrimTypes, so every node in a fixture shares
  // one handle and the fixture's working set counts it once.
  static HPrimType int32_type = MakePrimType(DLDataType{kDLInt, 32, 1});
  if (bits == 32) return int32_type;
  return MakePrimType(DLDataType{kDLInt, static_cast<uint8_t>(bits), 1});
}

}  // namespace mini

TVM_FFI_INLINE mini::HPrimType TypeTraits<mini::HPrimType>::ConvertFallbackValue(
    DLDataType dtype) {
  return mini::MakePrimType(dtype);
}

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

// ---------------------------------------------------------------- Var (ir.Var)
class HVarObj : public HExprObj {
 public:
  String name;

  HVarObj(HType ty, String name) : HExprObj(std::move(ty)), name(std::move(name)) {}

  /*!
   * \brief Ported from VarVisit.  The `ty` guard is not decoration: a Var pays an
   *        `as<HPrimTypeObj>()` on every visit, and omitting it made mini's Var cheaper than
   *        TVM's by exactly that check.
   */
  static TVMFFIAny StructuralVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
    // skips: name
    const HVarObj* self = details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HVarObj>(value);
    // A PrimType carries only a dtype, so it has nothing to visit.
    if (!self->ty.as<HPrimTypeObj>()) {
      if (visitor->def_region_kind() == kTVMFFIDefRegionKindNonRecursive) {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
            kTVMFFIDefRegionKindNone, [&]() { return visitor->VisitExpected(self->ty); }));
      } else {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
      }
    }
    return VisitDone();
  }
  /*! \brief Ported from VarMutate: consult the remap first, record on the way out. */
  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    // skips: name
    const HVarObj* self = details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HVarObj>(value);
    Expected<Any> remap_result = mutator->VarRemapGetExpected(value);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap_result);
    if (details::ExpectedUnsafe::GetData(remap_result).type_index() != TypeIndex::kTVMFFINone) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(remap_result));
    }
    auto return_mapped_var = [&](Any mapped_var) -> TVMFFIAny {
      auto set_result = mutator->VarRemapSetExpected(value, mapped_var);
      if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
        return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(set_result).error()));
      }
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(mapped_var));
    };
    if (self->ty.as<HPrimTypeObj>()) {
      return return_mapped_var(Any(self));
    }
    auto mutate_ty = [&]() { return mutator->MutateExpected(self->ty); };
    Expected<Any> mapped_ty_result =
        mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
            ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_ty)
            : mutate_ty();
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HType, mapped_ty, std::move(mapped_ty_result));
    Any mapped_var = Any(self);
    if (!mapped_ty.same_as(self->ty)) {
      ObjectPtr<HVarObj> copy = make_object<HVarObj>(*self);
      copy->ty = std::move(mapped_ty);
      mapped_var = Any(std::move(copy));
    }
    return return_mapped_var(std::move(mapped_var));
  }
  /*! \brief Ported from VarMaybeInplaceMutate. */
  static TVMFFIAny StructuralMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                                AnyView value) noexcept {
    // skips: name
    HVarObj* self = const_cast<HVarObj*>(
        details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HVarObj>(value));
    Expected<Any> remap_result = mutator->VarRemapGetExpected(value);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap_result);
    if (details::ExpectedUnsafe::GetData(remap_result).type_index() != TypeIndex::kTVMFFINone) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(remap_result));
    }
    auto return_mapped_var = [&](Any mapped_var) -> TVMFFIAny {
      auto set_result = mutator->VarRemapSetExpected(value, mapped_var);
      if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
        return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(set_result).error()));
      }
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(mapped_var));
    };
    if (self->ty.as<HPrimTypeObj>()) {
      return return_mapped_var(Any(self));
    }
    auto mutate_ty = [&]() { return mutator->MaybeInplaceMutateIfUniqueExpected(self->ty); };
    Expected<Any> mapped_ty_result =
        mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
            ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_ty)
            : mutate_ty();
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HType, mapped_ty, std::move(mapped_ty_result));
    if (!mapped_ty.same_as(self->ty)) self->ty = std::move(mapped_ty);
    return return_mapped_var(Any(self));
  }
  static void RegisterReflection();
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  // Mirrors VarNode exactly: one reserved child slot that cannot overflow, so an
  // IsObjectInstance check on a Var is a two-slot range test rather than an equality test.
  static constexpr uint32_t _type_child_slots = 1;
  static constexpr bool _type_child_slots_can_overflow = false;
  static constexpr const char* _type_key = "h.Var";
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(HVarObj, HExprObj);
};
class HVar : public HPrimExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HVar, HPrimExpr, HVarObj);
};

// ---------------------------------------------------------------- IntImm (ir.IntImm)
class HIntImmObj : public HExprObj {
 public:
  int64_t value;

  HIntImmObj(HType ty, int64_t value) : HExprObj(std::move(ty)), value(value) {}

  static TVMFFIAny StructuralVisit(StructuralVisitorObj*, AnyView) noexcept {
    // skips: value
    return VisitDone();
  }
  static TVMFFIAny StructuralMutate(StructuralMutatorObj*, AnyView value) noexcept {
    // skips: value
    const HIntImmObj* self =
        details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HIntImmObj>(value);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
  }
  static TVMFFIAny StructuralMaybeInplaceMutate(StructuralMutatorObj*, AnyView value) noexcept {
    // skips: value
    const HIntImmObj* self =
        details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HIntImmObj>(value);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
  }
  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.IntImm", HIntImmObj, HExprObj);
};

// ---------------------------------------------------------------- OpaqueExpr (ir.OpaqueExpr)
/*!
 * \brief Declared and never built, like HFloatImmObj.  The functor-era Call visit and mutate
 *        test the operator against it before descending, so it has to exist for that test to
 *        cost what it costs in TVM.
 */
class HOpaqueExprObj : public HExprObj {
 public:
  static void RegisterReflection() { reflection::ObjectDef<HOpaqueExprObj>(); }
  static constexpr uint32_t _type_child_slots = 2;
  TVM_FFI_DECLARE_OBJECT_INFO("h.OpaqueExpr", HOpaqueExprObj, HExprObj);
};
class HOpaqueExpr : public HExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HOpaqueExpr, HExpr, HOpaqueExprObj);
};

// ---------------------------------------------------------------- FloatImm (ir.FloatImm)
/*!
 * \brief Declared, registered, and never built by any fixture -- exactly like FloatImmNode in
 *        the real harness, whose only role is to be the `never` arms' link target.
 *
 * The link target has to be a real final Expr subtype for the link test to cost what it costs
 * in TVM: an equality test against one type index on a node that is never that type.
 */
class HFloatImmObj : public HExprObj {
 public:
  double value;

  static void RegisterReflection() {
    reflection::ObjectDef<HFloatImmObj>().def_ro("value", &HFloatImmObj::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.FloatImm", HFloatImmObj, HExprObj);
};
class HFloatImm : public HPrimExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HFloatImm, HPrimExpr, HFloatImmObj);
};

// ---------------------------------------------------------------- Op (ir.Op)
/*!
 * \brief The interned operator singleton a Call names.  Both Call hooks skip an operator that
 *        is one of these without looking at which, so only its identity and its type index
 *        matter -- but its layout is still TVM's, because it is still an Expr in the graph.
 */
class HOpObj : public HExprObj {
 public:
  String name;
  String description;
  Array<Any> arguments;
  String attrs_type_key;
  uint32_t attrs_type_index{0};
  int32_t num_inputs = -1;
  int32_t support_level = 10;

  static void RegisterReflection() {
    reflection::ObjectDef<HOpObj>()
        .def_ro("name", &HOpObj::name)
        .def_ro("description", &HOpObj::description, reflection::AttachFieldFlag::SEqHashIgnore())
        .def_ro("arguments", &HOpObj::arguments, reflection::AttachFieldFlag::SEqHashIgnore())
        .def_ro("attrs_type_key", &HOpObj::attrs_type_key,
                reflection::AttachFieldFlag::SEqHashIgnore())
        .def_ro("num_inputs", &HOpObj::num_inputs, reflection::AttachFieldFlag::SEqHashIgnore())
        .def_ro("support_level", &HOpObj::support_level,
                reflection::AttachFieldFlag::SEqHashIgnore());
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUniqueInstance;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.Op", HOpObj, HExprObj);
};
class HOp : public HExpr {
 public:
  /*! \brief One interned singleton per name, mirroring TVM's Op registry. */
  static const HOp& Get(const char* name);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HOp, HExpr, HOpObj);
};

// ---------------------------------------------------------------- Call (ir.Call)
/*!
 * \brief Ported from CallNode and its three hooks.
 *
 * The only mini node with a container field (`args`), and the only one with skip guards: a
 * `PrimType` result type, an interned `Op` operator and an empty `ty_args` are each skipped
 * rather than descended.  `call-split-fuse` exists to reach all three, in both harnesses.
 */
class HCallObj : public HExprObj {
 public:
  HExpr op;
  Array<HExpr> args;
  HAttrs attrs;
  Array<HType> ty_args;

  HCallObj(HType ty, HExpr op, Array<HExpr> args)
      : HExprObj(std::move(ty)), op(std::move(op)), args(std::move(args)) {}

  static TVMFFIAny StructuralVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
    // skips: attrs, constant metadata left untouched like the classic Expr functors.
    const HCallObj* self =
        details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HCallObj>(value);
    // A PrimType carries only a dtype, so it has nothing to visit.
    if (!self->ty.as<HPrimTypeObj>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
    }
    // An Op is an interned registry singleton, so it has nothing to visit.
    if (!self->op.as<HOpObj>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->op));
    }
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->args));
    // An empty ty_args has no element to traverse.
    if (!self->ty_args.empty()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty_args));
    }
    return VisitDone();
  }

  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    // skips: attrs, constant metadata left untouched like the classic Expr functors.
    const HCallObj* self =
        details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HCallObj>(value);
    // Deliberate copy: avoids Any boxing on the dominant primitive skip path.
    HType mapped_ty = self->ty;
    if (!self->ty.as<HPrimTypeObj>()) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HType, descended_ty, mutator->MutateExpected(self->ty));
      mapped_ty = std::move(descended_ty);
    }
    HExpr mapped_op = self->op;
    if (!self->op.as<HOpObj>()) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HExpr, descended_op, mutator->MutateExpected(self->op));
      mapped_op = std::move(descended_op);
    }
    using HExprArray = Array<HExpr>;
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HExprArray, mapped_args,
                                      mutator->MutateExpected(self->args));
    Array<HType> mapped_ty_args = self->ty_args;
    if (!self->ty_args.empty()) {
      using HTypeArray = Array<HType>;
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HTypeArray, descended_ty_args,
                                        mutator->MutateExpected(self->ty_args));
      mapped_ty_args = std::move(descended_ty_args);
    }
    if (mapped_ty.same_as(self->ty) && mapped_op.same_as(self->op) &&
        mapped_args.same_as(self->args) && mapped_ty_args.same_as(self->ty_args)) {
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
    }
    ObjectPtr<HCallObj> copy = make_object<HCallObj>(*self);
    copy->ty = std::move(mapped_ty);
    copy->op = std::move(mapped_op);
    copy->args = std::move(mapped_args);
    copy->ty_args = std::move(mapped_ty_args);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(copy)));
  }

  static TVMFFIAny StructuralMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                                AnyView value) noexcept {
    // skips: attrs, constant metadata left untouched like the classic Expr functors.
    HCallObj* self = const_cast<HCallObj*>(
        details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HCallObj>(value));
    // Deliberate copy: avoids Any boxing on the dominant primitive skip path.
    HType mapped_ty = self->ty;
    if (!self->ty.as<HPrimTypeObj>()) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HType, descended_ty,
                                        mutator->MaybeInplaceMutateIfUniqueExpected(self->ty));
      mapped_ty = std::move(descended_ty);
    }
    HExpr mapped_op = self->op;
    if (!self->op.as<HOpObj>()) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HExpr, descended_op,
                                        mutator->MaybeInplaceMutateIfUniqueExpected(self->op));
      mapped_op = std::move(descended_op);
    }
    using HExprArray = Array<HExpr>;
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HExprArray, mapped_args,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->args));
    Array<HType> mapped_ty_args = self->ty_args;
    if (!self->ty_args.empty()) {
      using HTypeArray = Array<HType>;
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
          HTypeArray, descended_ty_args,
          mutator->MaybeInplaceMutateIfUniqueExpected(self->ty_args));
      mapped_ty_args = std::move(descended_ty_args);
    }
    if (mapped_ty.same_as(self->ty) && mapped_op.same_as(self->op) &&
        mapped_args.same_as(self->args) && mapped_ty_args.same_as(self->ty_args)) {
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
    }
    self->ty = std::move(mapped_ty);
    self->op = std::move(mapped_op);
    self->args = std::move(mapped_args);
    self->ty_args = std::move(mapped_ty_args);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
  }

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.Call", HCallObj, HExprObj);
};
class HCall : public HExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HCall, HExpr, HCallObj);
};

// ------------------------------------------------- binary ops (Add/Mul/FloorDiv/FloorMod)
template <typename T>
class HBinOpObj : public HExprObj {
 public:
  HPrimExpr a;
  HPrimExpr b;

  HBinOpObj() {}
  HBinOpObj(HType ty, HPrimExpr a, HPrimExpr b)
      : HExprObj(std::move(ty)), a(std::move(a)), b(std::move(b)) {}

  /*! \brief Ported from BinaryVisit. */
  static TVMFFIAny StructuralVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
    // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
    const T* self = details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const T>(value);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->a));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->b));
    return VisitDone();
  }
  /*! \brief Ported from BinaryMutate: 20275 dropped result-type re-inference. */
  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
    const T* self = details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const T>(value);
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimExpr, a, mutator->MutateExpected(self->a));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimExpr, b, mutator->MutateExpected(self->b));
    if (a.same_as(self->a) && b.same_as(self->b)) {
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
    }
    ObjectPtr<T> copy = make_object<T>(*self);
    copy->a = std::move(a);
    copy->b = std::move(b);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(copy)));
  }
  /*! \brief Ported from BinaryMaybeInplaceMutate: no same_as early return, no re-inference. */
  static TVMFFIAny StructuralMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                                AnyView value) noexcept {
    // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
    T* self = const_cast<T*>(details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const T>(value));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimExpr, a,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->a));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimExpr, b,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->b));
    if (!a.same_as(self->a)) self->a = std::move(a);
    if (!b.same_as(self->b)) self->b = std::move(b);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
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

// ---------------------------------------------------------------- Stmt (tirx.Stmt)
class HStmtObj : public Object {
 public:
  mutable HSpan span;

  static void RegisterReflection() {
    reflection::ObjectDef<HStmtObj>().def_ro("span", &HStmtObj::span,
                                             reflection::AttachFieldFlag::SEqHashIgnore());
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr bool _type_s_eq_hash_subclass_kind_fixed = true;
  static constexpr uint32_t _type_child_slots = 15;
  TVM_FFI_DECLARE_OBJECT_INFO("h.Stmt", HStmtObj, Object);
};
class HStmt : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HStmt, ObjectRef, HStmtObj);
};

// ---------------------------------------------------------------- Evaluate (tirx.Evaluate)
class HEvaluateObj : public HStmtObj {
 public:
  // `HExpr`, not `HPrimExpr`, exactly as EvaluateNode holds `Expr`.  The difference is not
  // cosmetic: a cast to HPrimExpr dereferences the node to check `ty`, and holding the
  // narrower type here charged mini one dependent load per element that real TVM never paid.
  HExpr value;

  explicit HEvaluateObj(HExpr value) : value(std::move(value)) {}

  static TVMFFIAny StructuralVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
    const HEvaluateObj* self =
        details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HEvaluateObj>(value);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
    return VisitDone();
  }
  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    const HEvaluateObj* self =
        details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HEvaluateObj>(value);
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HExpr, mapped_value, mutator->MutateExpected(self->value));
    if (mapped_value.same_as(self->value)) {
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
    }
    ObjectPtr<HEvaluateObj> copy = make_object<HEvaluateObj>(*self);
    copy->value = std::move(mapped_value);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(copy)));
  }
  static TVMFFIAny StructuralMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                                AnyView value) noexcept {
    HEvaluateObj* self = const_cast<HEvaluateObj*>(
        details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HEvaluateObj>(value));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HExpr, mapped_value,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
    if (mapped_value.same_as(self->value)) {
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
    }
    self->value = std::move(mapped_value);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
  }
  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.Evaluate", HEvaluateObj, HStmtObj);
};
class HEvaluate : public HStmt {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HEvaluate, HStmt, HEvaluateObj);
};

// ---------------------------------------------------------------- SeqStmt (tirx.SeqStmt)
class HSeqStmtObj : public HStmtObj {
 public:
  Array<HStmt> seq;

  explicit HSeqStmtObj(Array<HStmt> seq) : seq(std::move(seq)) {}

  static TVMFFIAny StructuralVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
    const HSeqStmtObj* self =
        details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HSeqStmtObj>(value);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->seq));
    return VisitDone();
  }
  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept;
  static TVMFFIAny StructuralMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                                AnyView value) noexcept;
  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.SeqStmt", HSeqStmtObj, HStmtObj);
};

/*! \brief The `Evaluate(0)` a fully dropped sequence collapses to. Mirrors `Evaluate(0)`. */
inline HStmt EvaluateZero() {
  return HStmt(make_object<HEvaluateObj>(HExpr(make_object<HIntImmObj>(HPrimType::Int(32), 0))));
}

// ---------------------------------------------------------------------------
// The SeqStmt mutate pair, ported from a1031a2177's src/tirx/ir/stmt.cc.
//
// This is the whole of it -- IsSeqStmtNoOp, AppendSeqStmtResult, MutateSeqStmtChanged,
// MutateSeqStmtRaw, InplaceSplice, MaybeInplaceMutateSeqStmtChanged and
// MaybeInplaceMutateSeqStmtRaw -- and not a summary of it.  mini has no splice arm, so the
// splice branches are never taken here; they are present because a hook body's branches are
// part of what a node costs to traverse, and an abbreviated version of these functions is a
// different function.
// ---------------------------------------------------------------------------

inline bool IsSeqStmtNoOp(const HStmt& stmt) {
  const auto* evaluate = stmt.as<HEvaluateObj>();
  const auto* value = evaluate == nullptr ? nullptr : evaluate->value.as<HIntImmObj>();
  return value != nullptr && value->value == 0;
}

inline void AppendSeqStmtResult(Array<HStmt>* output, HStmt mapped) {
  if (IsSeqStmtNoOp(mapped)) {
    return;
  }
  if (const auto* nested = mapped.as<HSeqStmtObj>()) {
    for (const HStmt& stmt : nested->seq) {
      output->push_back(stmt);
    }
  } else {
    output->push_back(std::move(mapped));
  }
}

inline TVMFFIAny MutateSeqStmtChanged(StructuralMutatorObj* mutator, const HSeqStmtObj* self,
                                      int64_t index, HStmt mapped) noexcept {
  const int64_t size = static_cast<int64_t>(self->seq.size());
  ObjectPtr<ArrayObj> output_obj = ArrayObj::CreateRepeated(size, Any());
  output_obj->InitRange(0, self->seq.begin(), self->seq.begin() + index);
  output_obj->resize(index);
  Array<HStmt> output(std::move(output_obj));
  AppendSeqStmtResult(&output, std::move(mapped));
  for (int64_t i = index + 1; i < size; ++i) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HStmt, mapped, mutator->MutateExpected(self->seq[i]));
    AppendSeqStmtResult(&output, std::move(mapped));
  }
  if (output.empty()) {
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(EvaluateZero()));
  }
  if (output.size() == 1) {
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(output[0]));
  }
  ObjectPtr<HSeqStmtObj> copy = make_object<HSeqStmtObj>(*self);
  copy->seq = std::move(output);
  return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(copy)));
}

inline TVMFFIAny MutateSeqStmtRaw(StructuralMutatorObj* mutator, AnyView value) noexcept {
  const HSeqStmtObj* self =
      details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HSeqStmtObj>(value);
  const int64_t size = static_cast<int64_t>(self->seq.size());
  for (int64_t i = 0; i < size; ++i) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HStmt, mapped, mutator->MutateExpected(self->seq[i]));
    if (!self->seq[i].same_as(mapped)) {
      return MutateSeqStmtChanged(mutator, self, i, std::move(mapped));
    }
  }
  return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
}

inline TVMFFIAny InplaceSplice(StructuralMutatorObj* mutator, HSeqStmtObj* self, ArrayObj* seq,
                               int64_t total, int64_t index, const HSeqStmtObj* nested) noexcept {
  const int64_t size = static_cast<int64_t>(seq->size());
  Array<HStmt> output;
  output.reserve(size + static_cast<int64_t>(nested->seq.size()) - 1);
  for (int64_t i = 0; i < total; ++i) {
    output.push_back(seq->begin()[i].cast<HStmt>());
  }
  for (const HStmt& stmt : nested->seq) {
    output.push_back(stmt);
  }
  for (int64_t i = index + 1; i < size; ++i) {
    const Any& item = seq->begin()[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HStmt, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));
    AppendSeqStmtResult(&output, std::move(mapped));
  }
  if (output.empty()) {
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(EvaluateZero()));
  }
  if (output.size() == 1) {
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(output[0]));
  }
  self->seq = std::move(output);
  return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
}

inline TVMFFIAny MaybeInplaceMutateSeqStmtChanged(StructuralMutatorObj* mutator,
                                                  HSeqStmtObj* self, ArrayObj* seq, int64_t index,
                                                  HStmt mapped) noexcept {
  const int64_t size = static_cast<int64_t>(seq->size());
  int64_t total = index;
  if (IsSeqStmtNoOp(mapped)) {
    // Drop Evaluate(0), matching SeqStmt::Flatten.
  } else if (const auto* nested = mapped.as<HSeqStmtObj>()) {
    const int64_t nested_size = static_cast<int64_t>(nested->seq.size());
    if (total + nested_size > index + 1) {
      return InplaceSplice(mutator, self, seq, total, index, nested);
    }
    for (const HStmt& stmt : nested->seq) {
      seq->SetItemAfterCheck(total++, Any(stmt));
    }
  } else {
    seq->SetItemAfterCheck(total++, Any(std::move(mapped)));
  }
  for (int64_t i = index + 1; i < size; ++i) {
    const Any& item = seq->begin()[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HStmt, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));
    if (IsSeqStmtNoOp(mapped)) {
      continue;
    }
    if (const auto* nested = mapped.as<HSeqStmtObj>()) {
      const int64_t nested_size = static_cast<int64_t>(nested->seq.size());
      if (total + nested_size > i + 1) {
        return InplaceSplice(mutator, self, seq, total, i, nested);
      }
      for (const HStmt& stmt : nested->seq) {
        seq->SetItemAfterCheck(total++, Any(stmt));
      }
      continue;
    }
    if (total != i || !item.same_as(mapped)) {
      seq->SetItemAfterCheck(total, Any(std::move(mapped)));
    }
    ++total;
  }
  seq->resize(total);
  if (total == 0) {
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(EvaluateZero()));
  }
  if (total == 1) {
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(seq->begin()[0].cast<HStmt>()));
  }
  return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
}

inline TVMFFIAny MaybeInplaceMutateSeqStmtRaw(StructuralMutatorObj* mutator,
                                              AnyView value) noexcept {
  HSeqStmtObj* self = const_cast<HSeqStmtObj*>(
      details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const HSeqStmtObj>(value));
  // The engine establishes ownership of the SeqStmt, but seq is a field and needs its own check.
  if (!self->seq.unique()) {
    return MutateSeqStmtRaw(mutator, value);
  }
  ArrayObj* seq = self->seq.GetArrayObj();
  const int64_t size = static_cast<int64_t>(seq->size());
  for (int64_t i = 0; i < size; ++i) {
    // Borrow the storage slot so the element stays unique during in-place dispatch.
    const Any& item = seq->begin()[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HStmt, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));
    if (!item.same_as(mapped)) {
      return MaybeInplaceMutateSeqStmtChanged(mutator, self, seq, i, std::move(mapped));
    }
  }
  return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(self));
}

inline TVMFFIAny HSeqStmtObj::StructuralMutate(StructuralMutatorObj* mutator,
                                               AnyView value) noexcept {
  return MutateSeqStmtRaw(mutator, value);
}
inline TVMFFIAny HSeqStmtObj::StructuralMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                                           AnyView value) noexcept {
  return MaybeInplaceMutateSeqStmtRaw(mutator, value);
}

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

inline void HVarObj::RegisterReflection() {
  reflection::ObjectDef<HVarObj>().def_ro("name", &HVarObj::name,
                                          reflection::AttachFieldFlag::SEqHashIgnore());
  RegisterHooks<HVarObj>();
  RegisterInplaceHook<HVarObj>();
}
inline void HIntImmObj::RegisterReflection() {
  reflection::ObjectDef<HIntImmObj>().def_ro("value", &HIntImmObj::value);
  RegisterHooks<HIntImmObj>();
  RegisterInplaceHook<HIntImmObj>();
}
inline void HCallObj::RegisterReflection() {
  reflection::ObjectDef<HCallObj>()
      .def_ro("op", &HCallObj::op)
      .def_ro("args", &HCallObj::args)
      .def_ro("attrs", &HCallObj::attrs)
      .def_ro("ty_args", &HCallObj::ty_args);
  RegisterHooks<HCallObj>();
  RegisterInplaceHook<HCallObj>();
}
template <typename T>
void HBinOpObj<T>::RegisterReflection() {
  reflection::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
  RegisterHooks<T>();
  RegisterInplaceHook<T>();
}
inline void HEvaluateObj::RegisterReflection() {
  reflection::ObjectDef<HEvaluateObj>().def_ro("value", &HEvaluateObj::value);
  RegisterHooks<HEvaluateObj>();
  RegisterInplaceHook<HEvaluateObj>();
}
inline void HSeqStmtObj::RegisterReflection() {
  reflection::ObjectDef<HSeqStmtObj>().def_ro("seq", &HSeqStmtObj::seq);
  RegisterHooks<HSeqStmtObj>();
  RegisterInplaceHook<HSeqStmtObj>();
}

inline const HOp& HOp::Get(const char* name) {
  // Node-stable, so a reference handed out stays valid as later operators are interned --
  // the same contract tvm::Op::Get has, and what lets a fixture hold four of them at once.
  static std::map<std::string, HOp>* registry = new std::map<std::string, HOp>();
  auto it = registry->find(name);
  if (it != registry->end()) return it->second;
  ObjectPtr<HOpObj> node = make_object<HOpObj>();
  node->ty = HPrimType::Int(32);
  node->name = String(name);
  node->num_inputs = 2;
  return registry->emplace(std::string(name), HOp(node)).first->second;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralVisit);
  refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
  refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMaybeInplaceMutate);
  HSourceNameObj::RegisterReflection();
  HSpanObj::RegisterReflection();
  HTypeObj::RegisterReflection();
  HOpaqueTypeObj::RegisterReflection();
  HPrimTypeObj::RegisterReflection();
  HAttrsObj::RegisterReflection();
  HExprObj::RegisterReflection();
  HStmtObj::RegisterReflection();
  HVarObj::RegisterReflection();
  HIntImmObj::RegisterReflection();
  HOpaqueExprObj::RegisterReflection();
  HFloatImmObj::RegisterReflection();
  HOpObj::RegisterReflection();
  HCallObj::RegisterReflection();
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
  F(HVarObj) F(HCallObj) F(HAddObj) F(HMulObj) F(HFloorDivObj) F(HFloorModObj) F(HIntImmObj)
#define H_STMT_FUNCTOR_DEFAULT \
  { return VisitStmtDefault_(op, std::forward<Args>(args)...); }
#define H_STMT_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) { \
    return self->VisitStmt_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });
#define H_FOR_EACH_STMT(F) F(HEvaluateObj) F(HSeqStmtObj)

/*!
 * \brief Mirrors tirx::ExprFunctor, which is keyed on `Expr` and not on `PrimExpr`.
 *
 * Keying it on the narrow type made every functor dispatch in mini skip the `ty` dereference
 * that TVM's `VisitPrimExpr` pays, which is why the `*_functor` arms disagreed between the
 * harnesses in the opposite direction from the `*_old` ones.
 */
template <typename FType>
class HExprFunctor;

template <typename R, typename... Args>
class HExprFunctor<R(const HExpr& n, Args...)> {
 private:
  using TSelf = HExprFunctor<R(const HExpr& n, Args...)>;
  using FType = HNodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  using result_type = R;
  virtual ~HExprFunctor() {}
  R operator()(const HExpr& n, Args... args) { return VisitExpr(n, std::forward<Args>(args)...); }
  virtual R VisitExpr(const HExpr& n, Args... args) {
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

/*! \brief Mirrors functor_common.h's VisitArray: an index loop, not a range-for. */
template <typename T, typename F>
inline void HVisitArray(const Array<T>& arr, F fvisit) {
  for (size_t i = 0; i < arr.size(); i++) {
    fvisit(arr[i]);
  }
}

// ---------------------------------------------------------------------------
// Why every `VisitExpr_` / `VisitStmt_` body below carries `H_LIBRARY_BODY`.
//
// apache/tvm declares `ExprVisitor`, `ExprMutator`, `StmtVisitor` and `StmtMutator` `TVM_DLL`
// and compiles every one of their `VisitExpr_` / `VisitStmt_` bodies -- and the node
// constructors those bodies reach, `TVM_DEFINE_BINOP_CONSTRUCTOR` -- into
// libtvm_compiler.so.  An arm in `real_tvm_bench.cc` calls them across that boundary, so not
// one of them inlines into the arm, and none is visible to the optimizer that compiles the
// arm.  mini-TIR is a single translation unit, so at -O3 every one of them inlines unless it
// is told not to.  That is a difference in the compiled shape of the thing being measured,
// not in its algorithm, and it is the same arrangement `ShippingPostOrderVisit` and
// `ShippingSubstitute` below already use for the two entry points.
//
// It is worth what it costs: with the bodies inlined, mini's `map_functor` and `map_old` ran
// 25-32% under real-TVM's on every Expr fixture, which is enough to reverse the report's
// headline -- real reads `subst` against `old` at -26% to -51% on those rows and inlined mini
// read +4% to -24%.
//
// Not applied to `HIRSubstitute` / `HIRApplyVisit`, whose apache/tvm counterparts
// (`FunctorSubstitute`, `FunctorApplyVisit`) are written in `real_tvm_bench.cc` itself and so
// are inlinable on both sides; it is applied to `HShippingSubstitute` /
// `HShippingApplyVisit`, whose counterparts are `tirx::IRSubstitute` and `tirx::IRApplyVisit`
// inside the library.
//
// `-DMINI_TIR_INLINE_LIBRARY_BODIES` builds the control: same algorithm, bodies inlinable.
// It exists so the choice above stays a measurement rather than an assertion.
// ---------------------------------------------------------------------------
#ifdef MINI_TIR_INLINE_LIBRARY_BODIES
#define H_LIBRARY_BODY
#else
#define H_LIBRARY_BODY TVM_FFI_NO_INLINE
#endif

class HExprVisitor : public HExprFunctor<void(const HExpr&)> {
 public:
  using HExprFunctor::operator();

 protected:
  using HExprFunctor::VisitExpr;
  H_LIBRARY_BODY void VisitExpr_(const HVarObj*) override {}
  H_LIBRARY_BODY void VisitExpr_(const HIntImmObj*) override {}
  /*! \brief Ported from ExprVisitor::VisitExpr_(const CallNode*). */
  H_LIBRARY_BODY void VisitExpr_(const HCallObj* op) override {
    if (op->op.as<HOpaqueExprObj>()) {
      this->VisitExpr(op->op);
    }
    HVisitArray(op->args, [this](const HExpr& e) { this->VisitExpr(e); });
  }
#define H_DEFINE_BINOP_VISIT(OP)                          \
  H_LIBRARY_BODY void VisitExpr_(const OP* op) override { \
    this->VisitExpr(op->a);                               \
    this->VisitExpr(op->b);                               \
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
  virtual void VisitExpr(const HExpr&) {}
  H_LIBRARY_BODY void VisitStmt_(const HEvaluateObj* op) override {
    this->VisitExpr(op->value);
  }
  H_LIBRARY_BODY void VisitStmt_(const HSeqStmtObj* op) override {
    HVisitArray(op->seq, [this](const HStmt& s) { this->VisitStmt(s); });
  }
};

class HStmtExprVisitor : public HExprVisitor, public HStmtVisitor {
 public:
  using HExprVisitor::operator();
  using HStmtVisitor::operator();

 protected:
  using HExprVisitor::VisitExpr;
  using HStmtVisitor::VisitStmt;
  void VisitExpr(const HExpr& e) override { return HExprVisitor::VisitExpr(e); }
};

// ---------------------------------------------------------------------------
// tirx::ExprMutator / StmtMutator / StmtExprMutator.
// ---------------------------------------------------------------------------

/*!
 * \brief Ported from TVM_DEFINE_BINOP_CONSTRUCTOR, which is what DEFINE_BIOP_EXPR_MUTATE_
 *        reaches when an operand changed: the operand types are re-read and compared, and the
 *        result type is copied from `a`.
 */
template <typename T>
H_LIBRARY_BODY inline HPrimExpr HMakeBinOp(HPrimExpr a, HPrimExpr b) {
  const HPrimTypeObj* a_ty = a.get()->ty.as<HPrimTypeObj>();
  const HPrimTypeObj* b_ty = b.get()->ty.as<HPrimTypeObj>();
  if (a_ty == nullptr || b_ty == nullptr || a_ty->dtype.code != b_ty->dtype.code ||
      a_ty->dtype.bits != b_ty->dtype.bits || a_ty->dtype.lanes != b_ty->dtype.lanes) {
    std::abort();
  }
  ObjectPtr<T> node = make_object<T>();
  node->ty = a.get()->ty;
  node->a = std::move(a);
  node->b = std::move(b);
  return HPrimExpr(node);
}

class HExprMutator : public HExprFunctor<HExpr(const HExpr&)> {
 public:
  using HExprFunctor::operator();

 protected:
  using HExprFunctor::VisitExpr;
  /*! \brief Mirrors ExprMutator::VisitPrimExpr, including the cast back to the narrow type. */
  HPrimExpr VisitPrimExpr(const HPrimExpr& expr) { return VisitExpr(expr).as_or_throw<HPrimExpr>(); }
  H_LIBRARY_BODY HExpr VisitExpr_(const HVarObj* op) override { return GetRef<HExpr>(op); }
  H_LIBRARY_BODY HExpr VisitExpr_(const HIntImmObj* op) override { return GetRef<HExpr>(op); }
  /*! \brief Ported from ExprMutator::VisitExpr_(const CallNode*). */
  H_LIBRARY_BODY HExpr VisitExpr_(const HCallObj* op) override {
    HExpr call_op = op->op;
    if (op->op.as<HOpaqueExprObj>()) {
      call_op = this->VisitExpr(op->op);
    }
    Array<HExpr> args = op->args.Map([this](const HExpr& arg) -> HExpr { return this->VisitExpr(arg); });
    if (call_op.same_as(op->op) && args.same_as(op->args)) {
      return GetRef<HExpr>(op);
    }
    ObjectPtr<HCallObj> node = make_object<HCallObj>(op->ty, std::move(call_op), std::move(args));
    node->attrs = op->attrs;
    node->ty_args = op->ty_args;
    return HExpr(node);
  }
  // Mirrors DEFINE_BIOP_EXPR_MUTATE_: on a change the operator's own constructor rebuilds the
  // node, which re-reads and compares the operand types.
#define H_DEFINE_BINOP_MUTATE(OP)                                    \
  H_LIBRARY_BODY HExpr VisitExpr_(const OP* op) override {           \
    HPrimExpr a = this->VisitPrimExpr(op->a);                        \
    HPrimExpr b = this->VisitPrimExpr(op->b);                        \
    if (a.same_as(op->a) && b.same_as(op->b)) {                      \
      return GetRef<HExpr>(op);                                      \
    }                                                                \
    return HMakeBinOp<OP>(std::move(a), std::move(b));               \
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
  virtual HExpr VisitExpr(const HExpr& e) { return e; }
  HPrimExpr VisitPrimExpr(const HPrimExpr& e) { return VisitExpr(e).as_or_throw<HPrimExpr>(); }
  H_LIBRARY_BODY HStmt VisitStmt_(const HEvaluateObj* op) override {
    HExpr value = this->VisitExpr(op->value);
    if (value.same_as(op->value)) return GetRef<HStmt>(op);
    ObjectPtr<HEvaluateObj> n = CopyOnWrite(op);
    n->value = std::move(value);
    return HStmt(n);
  }
  H_LIBRARY_BODY HStmt VisitStmt_(const HSeqStmtObj* op) override {
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
  using HExprMutator::VisitPrimExpr;
  using HStmtMutator::VisitStmt;
  HExpr VisitExpr(const HExpr& e) override { return HExprMutator::VisitExpr(e); }
};

// ---------------------------------------------------------------------------
// walk_functor: tirx::IRApplyVisit, apache/tvm main's PostOrderVisit.
// ---------------------------------------------------------------------------

class HIRApplyVisit : public HStmtExprVisitor {
 public:
  explicit HIRApplyVisit(std::function<void(const ObjectRef&)> f) : f_(f) {}

  void VisitExpr(const HExpr& node) final {
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
  if (node.as<HStmtObj>()) {
    visitor(node.as_or_throw<HStmt>());
  } else {
    visitor(node.as_or_throw<HExpr>());
  }
}

// ---------------------------------------------------------------------------
// map_functor: tirx::IRSubstitute, apache/tvm main's Substitute.
// ---------------------------------------------------------------------------

class HIRSubstitute : public HStmtExprMutator {
 public:
  explicit HIRSubstitute(std::function<Optional<HExpr>(const HVar&)> vmap)
      : vmap_(std::move(vmap)) {}

  HExpr VisitExpr_(const HVarObj* op) final {
    HVar var = GetRef<HVar>(op);
    if (Optional<HExpr> ret = vmap_(var)) return ret.value();
    return HExpr(var);
  }

 private:
  std::function<Optional<HExpr>(const HVar&)> vmap_;
};

inline Any FunctorSubstitute(Any root, std::function<Optional<HExpr>(const HVar&)> vmap) {
  HIRSubstitute mutator(std::move(vmap));
  if (auto stmt = root.as<HStmt>()) return Any(mutator(*stmt));
  return Any(mutator(root.cast<HExpr>()));
}

// ---------------------------------------------------------------------------
// walk_old / map_old: what the pinned apache/tvm ships TODAY, at a1031a2177.
//
// Re-ported here.  These were ports of an older TVM, in which PostOrderVisit and Substitute
// were built on the structural engine; at a1031a2177 both are functor-era again --
// `PostOrderVisit` is `IRApplyVisit` and `Substitute` is `IRSubstitute`, a StmtExprMutator
// with a per-substitution type check.  Leaving the engine-based ports in place made mini's
// `old` arms a different algorithm from real TVM's, which is most of why `old` ran 6.6% to
// 44.9% slower in mini on every fixture.
//
// In real TVM these two entry points live in libtvm_compiler and are called across a
// shared-library boundary, so their bodies cannot inline into the arm.  mini has no such
// library, so each is its own out-of-line function over its own copy of the shape -- separate
// from `HIRApplyVisit` and `HIRSubstitute` above, exactly as TVM's copies are separate from
// the harness's `FunctorApplyVisit` and `FunctorSubstitute`.
// ---------------------------------------------------------------------------

/*! \brief A second, separate copy of IRApplyVisit: what TVM's PostOrderVisit runs. */
class HShippingApplyVisit : public HStmtExprVisitor {
 public:
  explicit HShippingApplyVisit(std::function<void(const ObjectRef&)> f) : f_(f) {}

  H_LIBRARY_BODY void VisitExpr(const HExpr& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    HExprVisitor::VisitExpr(node);
    f_(node);
  }
  H_LIBRARY_BODY void VisitStmt(const HStmt& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    HStmtVisitor::VisitStmt(node);
    f_(node);
  }

 private:
  std::function<void(const ObjectRef&)> f_;
  std::unordered_set<const Object*> visited_;
};

/*! \brief Ported from PostOrderVisit.  Out of line, as TVM's is across the .so boundary. */
TVM_FFI_NO_INLINE inline void ShippingPostOrderVisit(
    const ObjectRef& node, std::function<void(const ObjectRef&)> fvisit) {
  if (node.as<HStmtObj>()) {
    HShippingApplyVisit visitor(fvisit);
    visitor(node.as_or_throw<HStmt>());
  } else {
    HShippingApplyVisit visitor(fvisit);
    visitor(node.as_or_throw<HExpr>());
  }
}

/*!
 * \brief Ported from IRSubstitute, which is what `Substitute` is at this pin.
 *
 * The per-substitution check is the point of the port and not incidental: TVM runs a
 * StructuralEqual over the substituted and original result types on every replaced Var, in
 * release builds too, and it is most of the ~20% between `old` and the lean `functor`
 * baseline.  Dropping it would make mini's `old` a different function from TVM's.
 */
class HShippingSubstitute : public HStmtExprMutator {
 public:
  explicit HShippingSubstitute(std::function<Optional<HExpr>(const HVar&)> vmap)
      : vmap_(std::move(vmap)) {}

  H_LIBRARY_BODY HExpr VisitExpr_(const HVarObj* op) final {
    HVar var = GetRef<HVar>(op);
    auto ret = vmap_(var);
    if (ret.has_value()) {
      // Allow substitution of void variables with any expression.
      if (auto var_prim_type = var->ty.as<HPrimType>();
          !var_prim_type.has_value() || !IsVoidPrimType(var_prim_type.value())) {
        if (!StructuralEqual()(ret.value()->ty, var->ty)) std::abort();
      }
      return ret.value();
    }
    return HStmtExprMutator::VisitExpr_(op);
  }

 private:
  static bool IsVoidPrimType(const HPrimType& ty) {
    DLDataType dtype = ty->dtype;
    return dtype.code == static_cast<uint8_t>(kDLOpaqueHandle) && dtype.bits == 0 &&
           static_cast<int16_t>(dtype.lanes) == 0;
  }
  std::function<Optional<HExpr>(const HVar&)> vmap_;
};

/*! \brief Ported from Substitute.  Out of line, as TVM's is across the .so boundary. */
TVM_FFI_NO_INLINE inline Any ShippingSubstitute(Any root,
                                                std::function<Optional<HExpr>(const HVar&)> vmap) {
  HShippingSubstitute mutator(std::move(vmap));
  if (auto stmt = root.as<HStmt>()) return Any(mutator(*stmt));
  return Any(mutator(root.cast<HExpr>()));
}

}  // namespace mini
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_MINI_TIR_H_
