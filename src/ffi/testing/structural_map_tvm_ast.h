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
 * \file src/ffi/testing/structural_map_tvm_ast.h
 * \brief The TVM primitive-expression node set, copied into tvm-ffi so the
 *        structural traversal cost ladder can be measured without a TVM build.
 *
 * Every class, field, type-attribute value and hook body below is a copy of the
 * corresponding TVM declaration rather than an approximation.  Sources:
 *
 *   Span / SpanNode                      include/tvm/ir/source_map.h
 *   Type / TypeNode / PrimType           include/tvm/ir/base_expr.h, src/ir/type.cc
 *   ExprNode / Expr / TypedExpr /
 *     PrimExpr and their TypeTraits      include/tvm/ir/base_expr.h
 *   VarNode / Var, IntImmNode / IntImm   include/tvm/ir/expr.h, src/ir/expr.cc
 *   BinaryOpNode / Add / Mul /
 *     FloorDiv / FloorMod                include/tvm/ir/prim/expr.h, src/ir/prim/expr.cc
 *   VisitVar / MutateVar /
 *     MaybeInplaceMutateVar,
 *     VisitIntImm / MutateIntImm /
 *     MaybeInplaceMutateIntImm           src/ir/expr.cc
 *   VisitBinary / MutateBinary /
 *     MaybeInplaceMutateBinary           src/ir/prim/expr.cc
 *
 * Only the type keys are renamed, so the copied nodes can coexist with any other
 * registered type in the same process.  Deviations that could not be copied are
 * listed in tests/benchmark/structural_map_tvm_ast_calibration.md.
 */
#ifndef TVM_FFI_SRC_FFI_TESTING_STRUCTURAL_MAP_TVM_AST_H_
#define TVM_FFI_SRC_FFI_TESTING_STRUCTURAL_MAP_TVM_AST_H_

#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace tvm::ffi::testing::tvmast {

// ---------------------------------------------------------------------------
// Span, copied from include/tvm/ir/source_map.h.  Fixtures never populate a
// span; it exists so ExprNode carries the same non-traversed ObjectRef field.
// ---------------------------------------------------------------------------
class SpanNode : public Object {
 public:
  /*! \brief The source name. */
  String source_name;
  /*! \brief The line number. */
  int line;
  /*! \brief The column offset. */
  int column;
  /*! \brief The end line number. */
  int end_line;
  /*! \brief The end column number. */
  int end_column;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.tvmast.Span", SpanNode, Object);
};

class Span : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Span, ObjectRef, SpanNode);
};

// ---------------------------------------------------------------------------
// Type and PrimType, copied from include/tvm/ir/base_expr.h and src/ir/type.cc.
// ---------------------------------------------------------------------------
class TypeNode : public Object {
 public:
  /*! \brief Span that points to the original source code. */
  mutable Span span;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr const uint32_t _type_child_slots = 14;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.tvmast.Type", TypeNode, Object);
};

class Type : public ObjectRef {
 public:
  /*! \brief Sentinel for a type that has not been populated yet. */
  static Type Missing() {
    static Type missing = []() {
      Type type(UnsafeInit{});
      type.data_ = make_object<TypeNode>();
      return type;
    }();
    return missing;
  }

  /*! \return whether this is the missing-type sentinel. */
  bool IsMissing() const { return this->same_as(Type::Missing()); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Type, ObjectRef, TypeNode);
};

/*!
 * \brief Stand-in for tirx.BufferType.
 *
 * The copied Var hooks compare `ty` against this type index exactly as
 * src/ir/expr.cc compares against `tirx.BufferType`.  The split/fuse fixtures
 * only ever build PrimType vars, so the branch is false in both the copy and
 * in TVM; the type must exist so the one-time key lookup resolves.
 */
class BufferTypeNode final : public TypeNode {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.tvmast.BufferType", BufferTypeNode, TypeNode);
};

class PrimTypeNode final : public TypeNode {
 public:
  /*! \brief The raw DLPack dtype represented by this primitive type. */
  DLDataType dtype;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.tvmast.PrimType", PrimTypeNode, TypeNode);
};

class PrimType final : public Type {
 public:
  explicit PrimType(DLDataType dtype) : Type(UnsafeInit{}) { data_ = GetCachedPrimTypeNode(dtype); }

  PrimType(DLDataTypeCode code, int bits, int lanes = 1)
      : PrimType(DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits),
                            static_cast<uint16_t>(lanes)}) {}

  static PrimType Int(int bits, int lanes = 1) {
    if (lanes == 1) {
      if (bits == 32) {
        thread_local PrimType i32_ty(DLDataType{kDLInt, 32, 1});
        return i32_ty;
      }
      if (bits == 64) {
        thread_local PrimType i64_ty(DLDataType{kDLInt, 64, 1});
        return i64_ty;
      }
    }
    return PrimType(DLDataType{kDLInt, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
  }

  static PrimType Bool(int lanes = 1) {
    return PrimType(DLDataType{kDLBool, 8, static_cast<uint16_t>(lanes)});
  }

  DLDataTypeCode code() const { return static_cast<DLDataTypeCode>(get()->dtype.code); }
  int bits() const { return get()->dtype.bits; }
  int lanes() const { return get()->dtype.lanes; }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimType, Type, PrimTypeNode);

 private:
  static ObjectPtr<PrimTypeNode> GetCachedPrimTypeNode(DLDataType dtype) {
    thread_local std::unordered_map<uint32_t, ObjectPtr<PrimTypeNode>> cache;
    uint32_t key = (static_cast<uint32_t>(dtype.code) << 24) |
                   (static_cast<uint32_t>(dtype.bits) << 16) | static_cast<uint32_t>(dtype.lanes);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second;
    }
    ObjectPtr<PrimTypeNode> node = make_object<PrimTypeNode>();
    node->dtype = dtype;
    return cache.emplace(key, std::move(node)).first->second;
  }
};

inline bool operator==(const PrimType& lhs, const PrimType& rhs) {
  return lhs->dtype == rhs->dtype;
}
inline bool operator!=(const PrimType& lhs, const PrimType& rhs) { return !(lhs == rhs); }

// ---------------------------------------------------------------------------
// ExprNode and the Expr / TypedExpr / PrimExpr reference hierarchy, copied from
// include/tvm/ir/base_expr.h.
// ---------------------------------------------------------------------------
class ExprNode : public Object {
 public:
  /*! \brief Span that points to the original source code.  Never traversed. */
  mutable Span span;
  /*! \brief The deduced or annotated type of the expression.  Never traversed. */
  mutable Type ty = Type::Missing();

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr const uint32_t _type_child_slots = 64;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.tvmast.Expr", ExprNode, Object);
};

class Expr : public ObjectRef {
 public:
  bool operator==(const Expr& other) const = delete;
  bool operator!=(const Expr& other) const = delete;
  bool operator<(const Expr& other) const = delete;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Expr, ObjectRef, ExprNode);
};

template <typename ExpectedType>
class TypedExpr : public Expr {
 public:
  ExpectedType ty() const {
    const auto* node = get();
    return GetRef<ExpectedType>(
        node->ExprNode::ty.template as<typename ExpectedType::ContainerType>());
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TypedExpr, Expr, ExprNode);
  static constexpr bool _type_container_is_exact = false;
};

class PrimExprConvertibleNode : public Object {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO("testing.tvmast.PrimExprConvertible", PrimExprConvertibleNode,
                              Object);
};

class PrimExprConvertible : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PrimExprConvertible, ObjectRef,
                                             PrimExprConvertibleNode);
};

class PrimExpr : public TypedExpr<PrimType> {
 public:
  using TypedExpr<PrimType>::ty;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PrimExpr, TypedExpr<PrimType>, ExprNode);
  static constexpr bool _type_container_is_exact = false;
};

}  // namespace tvm::ffi::testing::tvmast

namespace tvm::ffi {

// TypeTraits copied from include/tvm/ir/base_expr.h.  The nested `ty` check in
// TypedExpr<...>::CheckAnyStrict and the fallback chain length on PrimExpr are
// the load-bearing parts: every `.cast<PrimExpr>()` in a mutate hook pays them.
template <>
inline constexpr bool use_default_type_traits_v<testing::tvmast::PrimType> = false;

template <>
struct TypeTraits<testing::tvmast::PrimType>
    : public ObjectRefWithFallbackTraitsBase<testing::tvmast::PrimType, DLDataType> {
  TVM_FFI_INLINE static testing::tvmast::PrimType ConvertFallbackValue(DLDataType dtype) {
    return testing::tvmast::PrimType(dtype);
  }
};

template <typename ExpectedType>
inline constexpr bool use_default_type_traits_v<testing::tvmast::TypedExpr<ExpectedType>> = false;

template <typename ExpectedType>
struct TypeTraits<testing::tvmast::TypedExpr<ExpectedType>>
    : public ObjectRefTypeTraitsBase<testing::tvmast::TypedExpr<ExpectedType>> {
  using Base = ObjectRefTypeTraitsBase<testing::tvmast::TypedExpr<ExpectedType>>;
  using Base::CopyFromAnyViewAfterCheck;
  using Base::CopyToAnyView;
  using Base::GetMismatchTypeInfo;
  using Base::MoveFromAnyAfterCheck;
  using Base::MoveToAny;
  using Base::TypeSchema;
  using Base::TypeStr;

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return testing::tvmast::TypedExpr<ExpectedType>::_type_is_nullable;
    }
    if (src->type_index < TypeIndex::kTVMFFIStaticObjectBegin ||
        !details::IsObjectInstance<testing::tvmast::ExprNode>(src->type_index)) {
      return false;
    }
    const auto* expr = static_cast<const testing::tvmast::ExprNode*>(
        details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj).get());
    return details::AnyUnsafe::CheckAnyStrict<ExpectedType>(expr->ty);
  }

  TVM_FFI_INLINE static std::optional<testing::tvmast::TypedExpr<ExpectedType>> TryCastFromAnyView(
      const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) {
      if (src->type_index == TypeIndex::kTVMFFINone) {
        return details::ObjectUnsafe::ObjectRefFromObjectPtr<
            testing::tvmast::TypedExpr<ExpectedType>>(nullptr);
      }
      return details::ObjectUnsafe::ObjectRefFromObjectPtr<
          testing::tvmast::TypedExpr<ExpectedType>>(
          details::ObjectUnsafe::ObjectPtrFromUnowned<testing::tvmast::ExprNode>(src->v_obj));
    }
    return std::nullopt;
  }
};

template <>
inline constexpr bool use_default_type_traits_v<testing::tvmast::PrimExpr> = false;

template <typename ObjectRefType, typename ExpectedType, typename... FallbackTypes>
struct TvmAstTypedExprWithFallbackTraitsBase
    : public ObjectRefWithFallbackTraitsBase<ObjectRefType, FallbackTypes...> {
  using Base = ObjectRefWithFallbackTraitsBase<ObjectRefType, FallbackTypes...>;

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return TypeTraits<testing::tvmast::TypedExpr<ExpectedType>>::CheckAnyStrict(src);
  }

  TVM_FFI_INLINE static std::optional<ObjectRefType> TryCastFromAnyView(const TVMFFIAny* src) {
    if (TypeTraits<testing::tvmast::TypedExpr<ExpectedType>>::TryCastFromAnyView(src)) {
      return details::ObjectUnsafe::ObjectRefFromObjectPtr<ObjectRefType>(
          details::ObjectUnsafe::ObjectPtrFromUnowned<testing::tvmast::ExprNode>(src->v_obj));
    }
    return Base::template TryFallbackTypes<FallbackTypes...>(src);
  }
};

template <>
struct TypeTraits<testing::tvmast::PrimExpr>
    : public TvmAstTypedExprWithFallbackTraitsBase<
          testing::tvmast::PrimExpr, testing::tvmast::PrimType, StrictBool, int64_t, double, String,
          testing::tvmast::PrimExprConvertible> {
  using Base =
      TvmAstTypedExprWithFallbackTraitsBase<testing::tvmast::PrimExpr, testing::tvmast::PrimType,
                                            StrictBool, int64_t, double, String,
                                            testing::tvmast::PrimExprConvertible>;
  using Base::CheckAnyStrict;
  using Base::CopyFromAnyViewAfterCheck;
  using Base::CopyToAnyView;
  using Base::GetMismatchTypeInfo;
  using Base::MoveFromAnyAfterCheck;
  using Base::MoveToAny;
  using Base::TryCastFromAnyView;
  using Base::TypeSchema;
  using Base::TypeStr;

  // The fallback converters exist so the fallback chain has TVM's shape and
  // length.  The benchmark never takes them: every value handed to a hook is
  // already a copied-AST expression node.
  static testing::tvmast::PrimExpr ConvertFallbackValue(StrictBool value);
  static testing::tvmast::PrimExpr ConvertFallbackValue(int64_t value);
  static testing::tvmast::PrimExpr ConvertFallbackValue(double value);
  static testing::tvmast::PrimExpr ConvertFallbackValue(String value);
  static testing::tvmast::PrimExpr ConvertFallbackValue(testing::tvmast::PrimExprConvertible value);
};

template <>
inline constexpr bool use_default_type_traits_v<testing::tvmast::Expr> = false;

template <>
struct TypeTraits<testing::tvmast::Expr>
    : public ObjectRefWithFallbackTraitsBase<testing::tvmast::Expr, testing::tvmast::PrimExpr> {
  TVM_FFI_INLINE static testing::tvmast::Expr ConvertFallbackValue(
      testing::tvmast::PrimExpr value) {
    return value;
  }
};

}  // namespace tvm::ffi

namespace tvm::ffi::testing::tvmast {

// ---------------------------------------------------------------------------
// VarNode / Var, copied from include/tvm/ir/expr.h and src/ir/expr.cc.
// ---------------------------------------------------------------------------
class VarNode : public ExprNode {
 public:
  /*! \brief The variable name. */
  String name;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  static constexpr const uint32_t _type_child_slots = 1;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.tvmast.Var", VarNode, ExprNode);
};

class Var : public Expr {
 public:
  explicit Var(String name, std::optional<Type> ty_annotation, Span span = Span()) {
    ObjectPtr<VarNode> n = make_object<VarNode>();
    n->name = std::move(name);
    if (ty_annotation.has_value()) {
      n->ty = ty_annotation.value();
    }
    n->span = std::move(span);
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Var, Expr, VarNode);
};

/*! \brief PrimVar, copied from include/tvm/tirx/var.h. */
class PrimVar : public PrimExpr {
 public:
  explicit PrimVar(String name, PrimType dtype = PrimType::Int(32), Span span = Span())
      : PrimExpr(Var(std::move(name), std::move(dtype), std::move(span)).as_or_throw<PrimExpr>()) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PrimVar, PrimExpr, VarNode);
  static constexpr bool _type_container_is_exact = false;
};

// ---------------------------------------------------------------------------
// IntImmNode / IntImm, copied from include/tvm/ir/expr.h and src/ir/expr.cc.
// ---------------------------------------------------------------------------
class IntImmNode : public ExprNode {
 public:
  /*! \brief the Internal value. */
  int64_t value;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.tvmast.IntImm", IntImmNode, ExprNode);
};

class IntImm : public PrimExpr {
 public:
  IntImm(PrimType value_ty, int64_t value, Span span = Span()) {
    ObjectPtr<IntImmNode> node = make_object<IntImmNode>();
    node->ty = std::move(value_ty);
    node->value = value;
    node->span = std::move(span);
    data_ = std::move(node);
  }

  static IntImm Int32(int64_t value, Span span = Span()) {
    return IntImm(PrimType::Int(32), value, std::move(span));
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntImm, PrimExpr, IntImmNode);
};

// ---------------------------------------------------------------------------
// BinaryOpNode and the four binary nodes the split/fuse fixtures need, copied
// from include/tvm/ir/prim/expr.h and src/ir/prim/expr.cc.
// ---------------------------------------------------------------------------
template <typename T>
class BinaryOpNode : public ExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  static const constexpr int _type_child_slots [[maybe_unused]] = 0;
  static const constexpr bool _type_final [[maybe_unused]] = true;
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(T, ExprNode);
};

#define TVM_FFI_TVMAST_DEFINE_BINOP(Name)                                   \
  class Name##Node : public BinaryOpNode<Name##Node> {                      \
   public:                                                                  \
    static constexpr const char* _type_key = "testing.tvmast." #Name;       \
  };                                                                        \
  class Name : public PrimExpr {                                            \
   public:                                                                  \
    Name(PrimExpr a, PrimExpr b, Span span = Span()) {                      \
      ObjectPtr<Name##Node> node = make_object<Name##Node>();               \
      node->ExprNode::ty = a.get()->ExprNode::ty;                           \
      node->a = std::move(a);                                               \
      node->b = std::move(b);                                               \
      node->span = std::move(span);                                         \
      data_ = std::move(node);                                              \
    }                                                                       \
    TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Name, PrimExpr, Name##Node); \
  }

TVM_FFI_TVMAST_DEFINE_BINOP(Add);
TVM_FFI_TVMAST_DEFINE_BINOP(Mul);
TVM_FFI_TVMAST_DEFINE_BINOP(FloorDiv);
TVM_FFI_TVMAST_DEFINE_BINOP(FloorMod);
#undef TVM_FFI_TVMAST_DEFINE_BINOP

// ---------------------------------------------------------------------------
// Structural hooks, copied from src/ir/expr.cc and src/ir/prim/expr.cc.
//
// TVM writes the binary hooks against the raw TVMFFIAny mutate entry points
// (MutateRaw / MaybeInplaceMutateIfUniqueRaw) that tvm-ffi exposes at the
// commit TVM vendors.  Later tvm-ffi main removed them in favour of the
// Expected<Any> wrappers.  Detect which the checked-out engine has, so the
// verbatim TVM body is used wherever it compiles and the wrapper form is used
// only where the raw entry points no longer exist.  The two forms are not
// interchangeable for measurement: see
// tests/benchmark/structural_map_tvm_ast_calibration.md.
// ---------------------------------------------------------------------------
namespace detail {
template <typename T, typename = void>
struct HasRawMutate : std::false_type {};
template <typename T>
struct HasRawMutate<T,
                    std::void_t<decltype(std::declval<T*>()->MutateRaw(std::declval<AnyView>()))>>
    : std::true_type {};
/*! \brief Whether the engine exposes TVM's raw TVMFFIAny mutate entry points. */
inline constexpr bool kHasRawMutate = HasRawMutate<StructuralMutatorObj>::value;
/*!
 * \brief StructuralMutatorObj, spelled so it depends on the caller's node type.
 *
 * Without this the discarded `if constexpr` branch below would still be name
 * checked against an engine that no longer declares the raw entry points.
 */
template <typename TNode>
using DependentMutatorObj =
    std::conditional_t<(sizeof(TNode) > 0), StructuralMutatorObj, StructuralMutatorObj>;
}  // namespace detail

/*! \brief Copied from src/ir/expr.cc: IsTIRXBufferType. */
inline bool IsTIRXBufferType(const Type& ty) {
  static const int32_t buffer_type_index = TypeKeyToIndex("testing.tvmast.BufferType");
  return ty->type_index() == buffer_type_index;
}

// Copied from src/ir/expr.cc.
inline TVMFFIAny VisitIntImm(StructuralVisitorObj*, AnyView) noexcept {
  return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Optional<VisitInterrupt>>(std::nullopt));
}

inline TVMFFIAny MutateIntImm(StructuralMutatorObj*, AnyView value) noexcept {
  return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(value)));
}

inline TVMFFIAny MaybeInplaceMutateIntImm(StructuralMutatorObj*, AnyView value) noexcept {
  return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(value)));
}

// Copied from src/ir/expr.cc.
inline TVMFFIAny VisitVar(StructuralVisitorObj* visitor, AnyView value) noexcept {
  const VarNode* self = value.cast<const VarNode*>();
  if (self->ty.as<PrimType>().has_value() ||
      (IsTIRXBufferType(self->ty) && visitor->def_region_kind() == kTVMFFIDefRegionKindNone)) {
    return details::ExpectedUnsafe::MoveToTVMFFIAny(
        Expected<Optional<VisitInterrupt>>(std::nullopt));
  }
  auto visit_ty = [&]() { return visitor->VisitExpected(self->ty); };
  auto result = visitor->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
                    ? visitor->WithDefRegionKind(kTVMFFIDefRegionKindNone, visit_ty)
                    : visit_ty();
  return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

// Copied from src/ir/expr.cc.
inline TVMFFIAny MutateVar(StructuralMutatorObj* mutator, AnyView value) noexcept {
  try {
    const VarNode* self = value.cast<const VarNode*>();
    const bool is_buffer_var = IsTIRXBufferType(self->ty);
    if (self->ty.as<PrimType>().has_value() ||
        (is_buffer_var && mutator->def_region_kind() == kTVMFFIDefRegionKindNone)) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(value)));
    }
    auto mutate_ty = [&]() { return mutator->MutateExpected(self->ty); };
    auto ty_result = mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
                         ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_ty)
                         : mutate_ty();
    if (TVM_FFI_PREDICT_FALSE(ty_result.is_err())) {
      AnyView error_context(self->ty);
      if (error_context.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
        Error error = ty_result.error();
        details::UpdateVisitErrorContext(error, error_context.cast<ObjectRef>());
      }
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(ty_result));
    }
    Type mapped_ty = std::move(details::ExpectedUnsafe::GetData(ty_result)).cast<Type>();
    Any mapped_var = Any(value);
    if (!mapped_ty.same_as(self->ty)) {
      ObjectPtr<VarNode> copy = make_object<VarNode>(*self);
      copy->ty = std::move(mapped_ty);
      mapped_var = Any(ObjectRef(std::move(copy)));
    }
    if (is_buffer_var && mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive) {
      auto set_result = mutator->VarRemapSetExpected(value, mapped_var);
      if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(
            Expected<Any>(Unexpected(std::move(set_result).error())));
      }
    }
    return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(std::move(mapped_var)));
  } catch (const Error& err) {
    return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Unexpected(err)));
  }
}

// Copied from src/ir/expr.cc.
inline TVMFFIAny MaybeInplaceMutateVar(StructuralMutatorObj* mutator, AnyView value) noexcept {
  try {
    VarNode* self = const_cast<VarNode*>(value.cast<const VarNode*>());
    const bool is_buffer_var = IsTIRXBufferType(self->ty);
    if (self->ty.as<PrimType>().has_value() ||
        (is_buffer_var && mutator->def_region_kind() == kTVMFFIDefRegionKindNone)) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(value)));
    }
    auto mutate_ty = [&]() { return mutator->MaybeInplaceMutateIfUniqueExpected(self->ty); };
    auto ty_result = mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
                         ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_ty)
                         : mutate_ty();
    if (TVM_FFI_PREDICT_FALSE(ty_result.is_err())) {
      AnyView error_context(self->ty);
      if (error_context.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
        Error error = ty_result.error();
        details::UpdateVisitErrorContext(error, error_context.cast<ObjectRef>());
      }
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(ty_result));
    }
    Type mapped_ty = std::move(details::ExpectedUnsafe::GetData(ty_result)).cast<Type>();
    if (!mapped_ty.same_as(self->ty)) self->ty = std::move(mapped_ty);
    if (is_buffer_var && mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive) {
      auto set_result = mutator->VarRemapSetExpected(value, value);
      if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(
            Expected<Any>(Unexpected(std::move(set_result).error())));
      }
    }
    return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(value)));
  } catch (const Error& err) {
    return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Unexpected(err)));
  }
}

// Copied from src/ir/prim/expr.cc.
template <typename TNode>
TVMFFIAny VisitBinary(StructuralVisitorObj* visitor, AnyView value) noexcept {
  const TNode* self = value.cast<const TNode*>();
  auto a_result = visitor->VisitExpected(self->a);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(a_result, AnyView(self->a));
  auto b_result = visitor->VisitExpected(self->b);
  return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(b_result));
}

// Copied from src/ir/prim/expr.cc.
template <typename TNode>
TVMFFIAny MutateBinary(StructuralMutatorObj* mutator, AnyView value) noexcept {
  try {
    const TNode* self = value.cast<const TNode*>();
    if constexpr (detail::kHasRawMutate) {
      TVMFFIAny mapped_a =
          static_cast<detail::DependentMutatorObj<TNode>*>(mutator)->MutateRaw(self->a);
      if (TVM_FFI_PREDICT_FALSE(mapped_a.type_index == TypeIndex::kTVMFFIError)) {
        auto error_result = details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(mapped_a);
        AnyView error_context(self->a);
        if (error_context.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
          Error error = error_result.error();
          details::UpdateVisitErrorContext(error, error_context.cast<ObjectRef>());
        }
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(error_result));
      }
      TVMFFIAny mapped_b =
          static_cast<detail::DependentMutatorObj<TNode>*>(mutator)->MutateRaw(self->b);
      if (TVM_FFI_PREDICT_FALSE(mapped_b.type_index == TypeIndex::kTVMFFIError)) {
        // Release the already-owned left result before propagating the right-child error.
        details::AnyUnsafe::MoveTVMFFIAnyToAny(&mapped_a);
        auto error_result = details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(mapped_b);
        AnyView error_context(self->b);
        if (error_context.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
          Error error = error_result.error();
          details::UpdateVisitErrorContext(error, error_context.cast<ObjectRef>());
        }
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(error_result));
      }
      Any owned_a = details::AnyUnsafe::MoveTVMFFIAnyToAny(&mapped_a);
      Any owned_b = details::AnyUnsafe::MoveTVMFFIAnyToAny(&mapped_b);
      PrimExpr a = std::move(owned_a).cast<PrimExpr>();
      PrimExpr b = std::move(owned_b).cast<PrimExpr>();
      if (a.same_as(self->a) && b.same_as(self->b)) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(value)));
      }
      ObjectPtr<TNode> copy = make_object<TNode>(*self);
      copy->a = std::move(a);
      copy->b = std::move(b);
      return details::ExpectedUnsafe::MoveToTVMFFIAny(
          Expected<Any>(Any(ObjectRef(std::move(copy)))));
    } else {
      Expected<Any> result_a = mutator->MutateExpected(self->a);
      if (TVM_FFI_PREDICT_FALSE(result_a.is_err())) {
        AnyView error_context(self->a);
        if (error_context.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
          Error error = result_a.error();
          details::UpdateVisitErrorContext(error, error_context.cast<ObjectRef>());
        }
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result_a));
      }
      Expected<Any> result_b = mutator->MutateExpected(self->b);
      if (TVM_FFI_PREDICT_FALSE(result_b.is_err())) {
        AnyView error_context(self->b);
        if (error_context.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
          Error error = result_b.error();
          details::UpdateVisitErrorContext(error, error_context.cast<ObjectRef>());
        }
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result_b));
      }
      PrimExpr a = std::move(details::ExpectedUnsafe::GetData(result_a)).cast<PrimExpr>();
      PrimExpr b = std::move(details::ExpectedUnsafe::GetData(result_b)).cast<PrimExpr>();
      if (a.same_as(self->a) && b.same_as(self->b)) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(value)));
      }
      ObjectPtr<TNode> copy = make_object<TNode>(*self);
      copy->a = std::move(a);
      copy->b = std::move(b);
      return details::ExpectedUnsafe::MoveToTVMFFIAny(
          Expected<Any>(Any(ObjectRef(std::move(copy)))));
    }
  } catch (const Error& err) {
    return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Unexpected(err)));
  }
}

// Copied from src/ir/prim/expr.cc.
template <typename TNode>
TVMFFIAny MaybeInplaceMutateBinary(StructuralMutatorObj* mutator, AnyView value) noexcept {
  try {
    TNode* self = const_cast<TNode*>(value.cast<const TNode*>());
    if constexpr (detail::kHasRawMutate) {
      TVMFFIAny mapped_a =
          static_cast<detail::DependentMutatorObj<TNode>*>(mutator)->MaybeInplaceMutateIfUniqueRaw(
              self->a);
      if (TVM_FFI_PREDICT_FALSE(mapped_a.type_index == TypeIndex::kTVMFFIError)) {
        auto error_result = details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(mapped_a);
        AnyView error_context(self->a);
        if (error_context.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
          Error error = error_result.error();
          details::UpdateVisitErrorContext(error, error_context.cast<ObjectRef>());
        }
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(error_result));
      }
      TVMFFIAny mapped_b =
          static_cast<detail::DependentMutatorObj<TNode>*>(mutator)->MaybeInplaceMutateIfUniqueRaw(
              self->b);
      if (TVM_FFI_PREDICT_FALSE(mapped_b.type_index == TypeIndex::kTVMFFIError)) {
        // Release the already-owned left result before propagating the right-child error.
        details::AnyUnsafe::MoveTVMFFIAnyToAny(&mapped_a);
        auto error_result = details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(mapped_b);
        AnyView error_context(self->b);
        if (error_context.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
          Error error = error_result.error();
          details::UpdateVisitErrorContext(error, error_context.cast<ObjectRef>());
        }
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(error_result));
      }
      Any owned_a = details::AnyUnsafe::MoveTVMFFIAnyToAny(&mapped_a);
      Any owned_b = details::AnyUnsafe::MoveTVMFFIAnyToAny(&mapped_b);
      PrimExpr a = std::move(owned_a).cast<PrimExpr>();
      PrimExpr b = std::move(owned_b).cast<PrimExpr>();
      if (!a.same_as(self->a)) self->a = std::move(a);
      if (!b.same_as(self->b)) self->b = std::move(b);
      return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(value)));
    } else {
      Expected<Any> result_a = mutator->MaybeInplaceMutateIfUniqueExpected(self->a);
      if (TVM_FFI_PREDICT_FALSE(result_a.is_err())) {
        AnyView error_context(self->a);
        if (error_context.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
          Error error = result_a.error();
          details::UpdateVisitErrorContext(error, error_context.cast<ObjectRef>());
        }
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result_a));
      }
      Expected<Any> result_b = mutator->MaybeInplaceMutateIfUniqueExpected(self->b);
      if (TVM_FFI_PREDICT_FALSE(result_b.is_err())) {
        AnyView error_context(self->b);
        if (error_context.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
          Error error = result_b.error();
          details::UpdateVisitErrorContext(error, error_context.cast<ObjectRef>());
        }
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result_b));
      }
      PrimExpr a = std::move(details::ExpectedUnsafe::GetData(result_a)).cast<PrimExpr>();
      PrimExpr b = std::move(details::ExpectedUnsafe::GetData(result_b)).cast<PrimExpr>();
      if (!a.same_as(self->a)) self->a = std::move(a);
      if (!b.same_as(self->b)) self->b = std::move(b);
      return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(value)));
    }
  } catch (const Error& err) {
    return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Unexpected(err)));
  }
}

/*!
 * \brief Register the copied node set: reflection fields first, then all three
 *        structural type-attribute columns for every node type.
 */
inline void RegisterTvmAstTypes() {
  static bool registered = [] {
    namespace refl = reflection;
    refl::ObjectDef<SpanNode>()
        .def_ro("source_name", &SpanNode::source_name)
        .def_ro("line", &SpanNode::line)
        .def_ro("column", &SpanNode::column)
        .def_ro("end_line", &SpanNode::end_line)
        .def_ro("end_column", &SpanNode::end_column);
    refl::ObjectDef<TypeNode>().def_ro("span", &TypeNode::span, refl::DefaultValue(Span()),
                                       refl::AttachFieldFlag::SEqHashIgnore());
    refl::ObjectDef<BufferTypeNode>();
    refl::ObjectDef<PrimTypeNode>().def_ro("dtype", &PrimTypeNode::dtype);
    refl::ObjectDef<PrimExprConvertibleNode>();
    refl::ObjectDef<ExprNode>()
        .def_ro("span", &ExprNode::span, refl::DefaultValue(Span()),
                refl::AttachFieldFlag::SEqHashIgnore())
        .def_ro("ty", &ExprNode::ty, refl::DefaultValue(Type::Missing()));
    refl::ObjectDef<VarNode>().def_ro("name", &VarNode::name,
                                      refl::AttachFieldFlag::SEqHashIgnore());
    refl::ObjectDef<IntImmNode>().def_ro("value", &IntImmNode::value);
    refl::ObjectDef<AddNode>().def_ro("a", &AddNode::a).def_ro("b", &AddNode::b);
    refl::ObjectDef<MulNode>().def_ro("a", &MulNode::a).def_ro("b", &MulNode::b);
    refl::ObjectDef<FloorDivNode>().def_ro("a", &FloorDivNode::a).def_ro("b", &FloorDivNode::b);
    refl::ObjectDef<FloorModNode>().def_ro("a", &FloorModNode::a).def_ro("b", &FloorModNode::b);

    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralVisit);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMaybeInplaceMutate);
    refl::TypeAttrDef<VarNode>()
        .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&VisitVar))
        .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&MutateVar))
        .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
              reinterpret_cast<void*>(&MaybeInplaceMutateVar));
    refl::TypeAttrDef<IntImmNode>()
        .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&VisitIntImm))
        .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&MutateIntImm))
        .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
              reinterpret_cast<void*>(&MaybeInplaceMutateIntImm));
#define TVM_FFI_TVMAST_REGISTER_BINOP(Name)                                                       \
  refl::TypeAttrDef<Name##Node>()                                                                 \
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&VisitBinary<Name##Node>)) \
      .attr(refl::type_attr::kStructuralMutate,                                                   \
            reinterpret_cast<void*>(&MutateBinary<Name##Node>))                                   \
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,                                       \
            reinterpret_cast<void*>(&MaybeInplaceMutateBinary<Name##Node>))
    TVM_FFI_TVMAST_REGISTER_BINOP(Add);
    TVM_FFI_TVMAST_REGISTER_BINOP(Mul);
    TVM_FFI_TVMAST_REGISTER_BINOP(FloorDiv);
    TVM_FFI_TVMAST_REGISTER_BINOP(FloorMod);
#undef TVM_FFI_TVMAST_REGISTER_BINOP
    return true;
  }();
  (void)registered;
}

}  // namespace tvm::ffi::testing::tvmast

namespace tvm::ffi {

// Definitions for the never-taken fallback converters declared above.  They
// exist only so TypeTraits<PrimExpr> has TVM's fallback chain; a copied-AST
// benchmark never hands a hook anything but an expression node.
inline testing::tvmast::PrimExpr TypeTraits<testing::tvmast::PrimExpr>::ConvertFallbackValue(
    StrictBool value) {
  return testing::tvmast::IntImm(testing::tvmast::PrimType::Bool(), value.operator bool());
}
inline testing::tvmast::PrimExpr TypeTraits<testing::tvmast::PrimExpr>::ConvertFallbackValue(
    int64_t value) {
  return testing::tvmast::IntImm(testing::tvmast::PrimType::Int(64), value);
}
inline testing::tvmast::PrimExpr TypeTraits<testing::tvmast::PrimExpr>::ConvertFallbackValue(
    double value) {
  TVM_FFI_THROW(TypeError) << "copied AST has no FloatImm: " << value;
  TVM_FFI_UNREACHABLE();
}
inline testing::tvmast::PrimExpr TypeTraits<testing::tvmast::PrimExpr>::ConvertFallbackValue(
    String value) {
  TVM_FFI_THROW(TypeError) << "copied AST has no StringImm: " << value;
  TVM_FFI_UNREACHABLE();
}
inline testing::tvmast::PrimExpr TypeTraits<testing::tvmast::PrimExpr>::ConvertFallbackValue(
    testing::tvmast::PrimExprConvertible value) {
  TVM_FFI_THROW(TypeError) << "copied AST has no PrimExprConvertible conversion";
  TVM_FFI_UNREACHABLE();
}

}  // namespace tvm::ffi

#endif  // TVM_FFI_SRC_FFI_TESTING_STRUCTURAL_MAP_TVM_AST_H_
