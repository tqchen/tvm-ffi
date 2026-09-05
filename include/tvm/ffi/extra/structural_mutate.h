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
 * \file tvm/ffi/extra/structural_mutate.h
 * \brief Structural mutation API with optional in-place optimization.
 */
#ifndef TVM_FFI_EXTRA_STRUCTURAL_MUTATE_H_
#define TVM_FFI_EXTRA_STRUCTURAL_MUTATE_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/expected.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/function_details.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/accessor.h>

#include <cstddef>
#include <exception>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

class StructuralMutatorObj;

/*!
 * \brief ABI callback type for structural mutation.
 *
 * \param mutator The active structural mutator.
 * \param value The borrowed value to mutate.
 * \return Raw ``TVMFFIAny`` containing the mutated value or an Error.
 *
 * \note The hook is exception-free like \ref FStructuralVisit. Representable failures must be
 *       returned as an Error. Hook implementations should use non-throwing accessors when the
 *       engine's type dispatch has already established the type; allocation failure and violated
 *       container invariants remain fatal.
 */
using FStructuralMutate = TVMFFIAny (*)(StructuralMutatorObj* mutator, AnyView value) noexcept;

/*!
 * \brief ABI callback type for looking up an identity substitution.
 *
 * \param mutator The active structural mutator.
 * \param var The borrowed variable identity to look up.
 * \return Raw ``TVMFFIAny`` containing the owning mapped value, FFI None when no mapping exists,
 *         or an Error.
 */
using FStructuralVarRemapGet = TVMFFIAny (*)(StructuralMutatorObj* mutator, AnyView var) noexcept;

/*!
 * \brief ABI callback type for recording an identity substitution.
 *
 * \param mutator The active structural mutator.
 * \param var The borrowed variable identity to bind.
 * \param mapped_value The borrowed replacement value.
 * \return Raw ``TVMFFIAny`` containing FFI None on success or an Error.
 */
using FStructuralVarRemapSet = TVMFFIAny (*)(StructuralMutatorObj* mutator, AnyView var,
                                             AnyView mapped_value) noexcept;

namespace details {

// Copy and structurally mutate the reflected fields of an object-backed value.
TVM_FFI_INLINE static Expected<Any> MutateReflectedFieldsExpected(StructuralMutatorObj* mutator,
                                                                  AnyView value) noexcept;

}  // namespace details

/*!
 * \brief VTable ABI for \ref StructuralMutator dispatch.
 */
struct StructuralMutatorVTable {
  /*!
   * \brief Mutate a value without modifying the source in place.
   *
   * \param mutator The active structural mutator.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` carrying the mutated value or Error.
   */
  FStructuralMutate mutate = nullptr;
  /*!
   * \brief Mutate a value, permitting an in-place implementation when it is safe.
   *
   * \param mutator The active structural mutator.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` carrying the mutated value or Error.
   *
   * The returned value may refer to the same object as \p value when the implementation mutates
   * that object in place.
   */
  FStructuralMutate maybe_inplace_mutate = nullptr;
  /*!
   * \brief Look up the replacement for a variable identity.
   *
   * \param mutator The active structural mutator.
   * \param var The borrowed variable identity to look up.
   * \return Raw ``TVMFFIAny`` carrying the owning replacement, FFI None on a miss, or Error.
   */
  FStructuralVarRemapGet var_remap_get = nullptr;
  /*!
   * \brief Record the replacement for a variable identity.
   *
   * \param mutator The active structural mutator.
   * \param var The borrowed variable identity to bind.
   * \param mapped_value The borrowed replacement value.
   * \return Raw ``TVMFFIAny`` carrying None or Error.
   */
  FStructuralVarRemapSet var_remap_set = nullptr;
};

/*!
 * \brief Object node of a structural mutator.
 */
class StructuralMutatorObj : public Object {
 public:
  /*!
   * \brief Mutate a value through the mutator vtable.
   *
   * \param value The value to mutate.
   * \return The mutated owning value.
   * \throws Error if mutation fails.
   *
   * This entry point never intentionally mutates \p value in place. Recursive mutations
   * also use \ref Mutate.
   */
  TVM_FFI_INLINE Any Mutate(AnyView value) { return MutateExpected(value).value(); }

  /*!
   * \brief Exception-free form of \ref Mutate.
   *
   * \param value The value to mutate.
   * \return The mutated owning value, or an Error if mutation failed.
   */
  TVM_FFI_INLINE Expected<Any> MutateExpected(AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>((*vtable_->mutate)(this, value));
  }

  /*!
   * \brief Mutate a value, permitting an in-place implementation when it is safe.
   *
   * \param value The borrowed value to mutate.
   * \return The mutated owning value.
   * \throws Error if mutation fails.
   *
   * The returned value may refer to the same object as \p value. Callers must use the return value
   * as the result of the mutation rather than assuming that the input object was reused.
   */
  TVM_FFI_INLINE Any MaybeInplaceMutate(AnyView value) {
    return MaybeInplaceMutateExpected(value).value();
  }

  /*!
   * \brief Exception-free form of \ref MaybeInplaceMutate.
   *
   * \param value The borrowed value to mutate.
   * \return The mutated owning value, or an Error if mutation failed.
   */
  TVM_FFI_INLINE Expected<Any> MaybeInplaceMutateExpected(AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(
        (*vtable_->maybe_inplace_mutate)(this, value));
  }

  /*!
   * \brief Mutate a value, using in-place mutation only for a uniquely owned object.
   *
   * \param value The borrowed value to mutate.
   * \return The mutated owning value, or an Error if mutation failed.
   */
  TVM_FFI_INLINE Expected<Any> MaybeInplaceMutateIfUniqueExpected(AnyView value) noexcept {
    const Object* obj = value.as<Object>();
    if (obj != nullptr && obj->unique()) {
      return MaybeInplaceMutateExpected(value);
    }
    return MutateExpected(value);
  }

  /*!
   * \brief Apply the default structural mutation with copy-on-write behavior.
   *
   * \param value The value to mutate.
   * \return The mutated value, or an Error if hook dispatch, copying, or field mutation failed.
   *
   * \note A registered ``__s_mutate__`` hook is dispatched before the reflected fallback and is
   *       responsible for variable-remap lookup and insertion when it represents a FreeVar or DAG
   *       identity. Automatic remapping applies only to the reflected fallback.
   */

  TVM_FFI_INLINE Expected<Any> DefaultMutateExpected(AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(DefaultMutateRaw(value));
  }

  /*!
   * \brief Apply custom maybe-in-place mutation, or fall back to non-in-place mutation.
   *
   * \param value The borrowed value to mutate.
   * \return The mutated owning value, or an Error if mutation failed. In-place changes
   *         completed before an Error are not rolled back.
   *
   * \note In-place mutation is explicitly opt-in. A registered
   *       ``__s_maybe_inplace_mutate__`` hook may rely on its input being safe to mutate and owns
   *       any variable-remap handling. When the hook is absent, this method calls
   *       \ref DefaultMutateExpected.
   */
  TVM_FFI_INLINE Expected<Any> DefaultMaybeInplaceMutateExpected(AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(DefaultMaybeInplaceMutateRaw(value));
  }

  /*!
   * \brief Look up the replacement recorded for a variable identity.
   *
   * \param var The borrowed variable identity to look up.
   * \return The owning replacement, FFI None if no replacement exists, or an Error if lookup
   *         fails.
   *
   * \note The identity must have ``kTVMFFISEqHashKindFreeVar`` or
   *       ``kTVMFFISEqHashKindDAGNode`` structural-equality metadata.
   */
  TVM_FFI_INLINE Expected<Any> VarRemapGetExpected(AnyView var) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>((*vtable_->var_remap_get)(this, var));
  }

  /*!
   * \brief Record the replacement for a variable identity.
   *
   * \param var The borrowed variable identity to bind.
   * \param mapped_value The borrowed replacement value.
   * \return Successful completion, or an Error if the binding is invalid or cannot be stored.
   *
   * \note The identity must have ``kTVMFFISEqHashKindFreeVar`` or
   *       ``kTVMFFISEqHashKindDAGNode`` structural-equality metadata.
   */
  TVM_FFI_INLINE Expected<void> VarRemapSetExpected(AnyView var, AnyView mapped_value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<void>(
        (*vtable_->var_remap_set)(this, var, mapped_value));
  }

  /*!
   * \brief Return the current def-region context.
   * \return The active def-region kind.
   */
  TVM_FFI_INLINE TVMFFIDefRegionKind def_region_kind() const { return def_region_mode_; }

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   *
   * \param kind The def-region kind to set during the callback.
   * \param callback A nullary callable that performs recursive mutation.
   * \return The value returned by \p callback.
   */
  template <typename Callback>
  TVM_FFI_INLINE auto WithDefRegionKind(TVMFFIDefRegionKind kind, Callback&& callback)
      -> decltype(std::forward<Callback>(callback)()) {
    class Scope {
     public:
      Scope(StructuralMutatorObj* mutator, TVMFFIDefRegionKind kind)
          : mutator_(mutator), old_kind_(mutator->def_region_mode_) {
        mutator_->def_region_mode_ = kind;
      }
      ~Scope() { mutator_->def_region_mode_ = old_kind_; }
      Scope(const Scope&) = delete;
      Scope& operator=(const Scope&) = delete;

     private:
      StructuralMutatorObj* mutator_;
      TVMFFIDefRegionKind old_kind_;
    };
    Scope scope(this, kind);
    return std::forward<Callback>(callback)();
  }

  /// \cond Doxygen_Suppress
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.StructuralMutator", StructuralMutatorObj, Object);
  /// \endcond

 protected:
  /*!
   * \brief Diagnostic for a malformed ``__s_mutate__`` registration.
   *
   * Kept out of line and cold: it can only fire for a type whose registered attribute is
   * neither an opaque function pointer nor an ffi.Function, so it is unreachable for any
   * correctly registered type. Inlined, its three string literals and Error construction
   * land in the traversal's hot path for no reason.
   */
  TVM_FFI_COLD_CODE static Expected<Any> BadStructuralMutateHookError() noexcept {
    return Unexpected(
        Error("TypeError", "__s_mutate__ must be an opaque function pointer or ffi.Function", ""));
  }

  // Convention: the ABI boundary is a raw TVMFFIAny; everything inside a callback or hook body
  // works in Expected<Any> and moves out to TVMFFIAny at that boundary.
  //
  // The Raw forms below exist because that boundary is also the default path. A hook is a C-ABI
  // function pointer returning TVMFFIAny, a 16-byte POD that stays in registers; wrapping the
  // result in Expected<Any> would force it to memory, since Expected<Any> is not trivially
  // destructible and is therefore classified MEMORY. Descent through an unmatched node calls a
  // hook and returns its result unchanged, so keeping that path raw removes the round trip
  // entirely. Only a matched callback pays for an Expected<Any>, and it materializes an Any
  // anyway to record its remap entry.
  //
  // Engine-internal: subclasses call the Expected forms above.
  /*! \brief Raw default mutation: attr lookup then hook, favouring the fn-ptr case. */
  TVM_FFI_INLINE TVMFFIAny DefaultMutateRaw(AnyView value) noexcept {
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMutate);
    AnyView attr = column[value.type_index()];
    // Exactly one frame per node: hooks propagate errors untouched, and this is the engine
    // dispatching into `value`, so both exits below name it here and nowhere else.
    TVMFFIAny result;
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      result = (*reinterpret_cast<FStructuralMutate>(attr.cast<void*>()))(this, value);
    } else {
      result = DefaultMutateRawTail(value, attr);
    }
    if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
      details::UpdateVisitErrorContext(result, value);
    }
    return result;
  }

  // The cold remainder of DefaultMutateRaw, out of line so that always-inlined caller stays
  // small at every inlining site. `attr` is the attribute the caller already read, so the
  // column is never looked up twice.
  /*!
   * \brief Whether a node's identity is remappable, so it maps once and reuses that result.
   * \param type_index The node's runtime type index.
   * \return True for a FreeVar or DAG node.
   */
  TVM_FFI_INLINE static bool IsRemappableIdentity(int32_t type_index) noexcept {
    if (type_index < TypeIndex::kTVMFFIStaticObjectBegin) return false;
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
    return type_info->metadata != nullptr &&
           (type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindFreeVar ||
            type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindDAGNode);
  }

  /*! \brief The cold remainder of DefaultMutateRaw: an ffi.Function hook, or no hook at all. */
  TVMFFIAny DefaultMutateRawTail(AnyView value, AnyView attr) noexcept {
    if (attr.type_index() != TypeIndex::kTVMFFINone) {
      // Registered, but as an ffi.Function rather than an opaque pointer.
      if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(
            attr.cast<Function>().CallExpected<Any>(this, value));
      }
      // Registered as neither: a malformed hook.
      return details::ExpectedUnsafe::MoveToTVMFFIAny(BadStructuralMutateHookError());
    }
    // No hook at all. A POD carries through unchanged; an object walks its reflected fields.
    if (value.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) {
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
    }
    // A FreeVar or DAG node maps once and every later occurrence reuses that result, so the
    // reflected walk runs under an identity remap.
    const bool remappable = IsRemappableIdentity(value.type_index());
    if (remappable) {
      Expected<Any> mapped = VarRemapGetExpected(value);
      if (TVM_FFI_PREDICT_FALSE(mapped.is_err()) ||
          details::ExpectedUnsafe::GetData(mapped).type_index() != TypeIndex::kTVMFFINone) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(mapped));
      }
    }
    Expected<Any> result = details::MutateReflectedFieldsExpected(this, value);
    if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
    }
    if (remappable) {
      Expected<void> set_result =
          VarRemapSetExpected(value, details::ExpectedUnsafe::GetData(result));
      if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(
            Expected<Any>(Unexpected(std::move(set_result).error())));
      }
    }
    return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }
  /*!
   * \brief Raw default maybe-in-place mutation.
   *
   * \note A registered opaque hook is the expected case here too, so the attribute is read once
   *       and every other shape is handed to the out-of-line remainder.
   */
  TVM_FFI_INLINE TVMFFIAny DefaultMaybeInplaceMutateRaw(AnyView value) noexcept {
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMaybeInplaceMutate);
    AnyView attr = column[value.type_index()];
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      // This is the engine dispatching into `value`; hooks propagate errors untouched, so the
      // node is named here. The fall-through re-dispatches the same node through
      // DefaultMutateRaw, which names it there instead -- exactly one frame either way.
      TVMFFIAny result = (*reinterpret_cast<FStructuralMutate>(attr.cast<void*>()))(this, value);
      if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
        details::UpdateVisitErrorContext(result, value);
      }
      return result;
    }
    return DefaultMaybeInplaceMutateRawTail(value, attr);
  }

  /*!
   * \brief The cold remainder of DefaultMaybeInplaceMutateRaw: an ffi.Function in-place hook, or
   *        no in-place hook at all, in which case the ordinary mutate path runs.
   */
  TVMFFIAny DefaultMaybeInplaceMutateRawTail(AnyView value, AnyView attr) noexcept {
    if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
      TVMFFIAny result = details::ExpectedUnsafe::MoveToTVMFFIAny(
          attr.cast<Function>().CallExpected<Any>(this, value));
      if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
        details::UpdateVisitErrorContext(result, value);
      }
      return result;
    }
    return DefaultMutateRaw(value);
  }

  /*!
   * \brief Construct a structural mutator from an immutable dispatch vtable.
   * \param vtable The non-null dispatch table for this mutator. It must outlive this object.
   */
  explicit StructuralMutatorObj(const StructuralMutatorVTable* vtable) : vtable_(vtable) {}

  /*!
   * \brief Non-owning pointer to the required ABI dispatch table.
   */
  const StructuralMutatorVTable* vtable_ = nullptr;

  /*!
   * \brief Current def-region context for def-region-aware structural mutation.
   */
  TVMFFIDefRegionKind def_region_mode_ = kTVMFFIDefRegionKindNone;
};

/*!
 * \brief ObjectRef wrapper for \ref StructuralMutatorObj.
 *
 * \sa StructuralMutatorObj
 */
class StructuralMutator : public ObjectRef {
 public:
  /*!
   * \brief Construct from an existing mutator object pointer.
   * \param n The object pointer to wrap.
   */
  explicit StructuralMutator(ObjectPtr<StructuralMutatorObj> n) : ObjectRef(std::move(n)) {}

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralMutator, ObjectRef, StructuralMutatorObj);
  /// \endcond
};

namespace details {

/*!
 * \brief Mutate the reflected structural fields of an object-backed value.
 *
 * \param mutator The active structural mutator.
 * \param value The object-backed value to mutate.
 * \return The original value when no field changes, a mutated shallow copy otherwise, or an
 *         Error if copying or mutation failed.
 */
TVM_FFI_INLINE static Expected<Any> MutateReflectedFieldsExpected(StructuralMutatorObj* mutator,
                                                                  AnyView value) noexcept {
  const Object* obj = value.as<Object>();
  int32_t type_index = obj->type_index();

  static reflection::TypeAttrColumn column(reflection::type_attr::kShallowCopy);
  AnyView attr = column[type_index];
  if (TVM_FFI_PREDICT_FALSE(attr.type_index() != TypeIndex::kTVMFFIFunction)) {
    return Unexpected(Error("TypeError", "__ffi_shallow_copy__ must be an ffi.Function", ""));
  }

  Expected<Any> result = attr.cast<Function>().CallExpected<Any>(value);
  if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
    return result;
  }

  const Any& result_value = details::ExpectedUnsafe::GetData(result);
  Object* new_obj = const_cast<Object*>(result_value.as<Object>());
  // Copy-on-write mutation requires a distinct target so partial updates cannot modify the source.
  if (TVM_FFI_PREDICT_FALSE(new_obj == nullptr || result.type_index() != value.type_index() ||
                            new_obj == obj)) {
    return Unexpected(Error(
        "TypeError",
        "Shallow copy callback must return a distinct object with the same type as its input", ""));
  }

  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(new_obj->type_index());
  bool field_changed = false;
  auto mutate_fields = [&]() {
    reflection::ForEachFieldInfoWithEarlyStop(
        type_info, [&](const TVMFFIFieldInfo* field_info) -> bool {
          if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashIgnore) {
            return false;
          }

          Any field_value;
          void* field_addr = reinterpret_cast<char*>(new_obj) + field_info->offset;
          int ret_code = field_info->getter(field_addr, reinterpret_cast<TVMFFIAny*>(&field_value));
          if (TVM_FFI_PREDICT_FALSE(ret_code != 0)) {
            result = Unexpected(details::MoveFromSafeCallRaised());
            return true;
          }

          Expected<Any> mutated_field = [&]() -> Expected<Any> {
            if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive) {
              return mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
                return mutator->MutateExpected(field_value);
              });
            } else if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefRecursive) {
              return mutator->WithDefRegionKind(kTVMFFIDefRegionKindRecursive, [&]() {
                return mutator->MutateExpected(field_value);
              });
            } else {
              return mutator->MutateExpected(field_value);
            }
          }();
          if (TVM_FFI_PREDICT_FALSE(mutated_field.is_err())) {
            result = Unexpected(std::move(mutated_field).error());
            return true;
          }
          const Any& new_field = details::ExpectedUnsafe::GetData(mutated_field);
          if (field_value.same_as(new_field)) {
            return false;
          }

          if (TVM_FFI_PREDICT_FALSE(field_info->setter == nullptr)) {
            result = Unexpected(Error(
                "TypeError",
                "Cannot structurally mutate field `" +
                    std::string(field_info->name.data, field_info->name.size) + "` of type `" +
                    std::string(type_info->type_key.data, type_info->type_key.size) +
                    "` because it does not define a setter",
                ""));
            return true;
          }

          ret_code = reflection::CallFieldSetter(field_info, field_addr,
                                                 reinterpret_cast<const TVMFFIAny*>(&new_field));
          if (TVM_FFI_PREDICT_FALSE(ret_code != 0)) {
            result = Unexpected(details::MoveFromSafeCallRaised());
            return true;
          }
          field_changed = true;
          return false;
        });
  };

  // A non-recursive definition applies to the FreeVar itself, but its fields are uses. The
  // complete field traversal are clamped to None, then the definition region is restored.
  if (mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive &&
      type_info->metadata != nullptr &&
      type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindFreeVar) {
    mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_fields);
  } else {
    mutate_fields();
  }

  if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
    return result;
  }
  if (!field_changed) {
    return Any(value);
  }
  return result;
}

}  // namespace details

// ---------------------------------------------------------------------------
// Structural Map API.
// ---------------------------------------------------------------------------

namespace details {

/// \cond Doxygen_Suppress
// Return from the current mutation function if Result is an Error.
// Append Node to the mutate error context before returning. Node is required: dropping it
// silently degrades every error message produced below this frame.
// A raw pointer Node must be non-null; pass nullable nodes as ObjectRef or Any so None is skipped.
#define TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result)                                           \
  do {                                                                                        \
    auto&& tvm_ffi_res_ = (Result);                                                           \
    if (TVM_FFI_PREDICT_FALSE(tvm_ffi_res_.is_err())) {                                       \
      return ::tvm::ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(::std::move(tvm_ffi_res_)); \
    }                                                                                         \
  } while (0)

/// \endcond

// Out of line so its strings and Error construction stay out of the hot path of whatever hook
// body TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN expands into. Same reason as
// BadStructuralMutateHookError.
TVM_FFI_COLD_CODE inline TVMFFIAny SMutateDeclaredTypeErrorRaw() noexcept {
  return AnyUnsafe::MoveAnyToTVMFFIAny(
      Any(Error("TypeError", "structural mutate result does not match the declared type", "")));
}

/// \cond Doxygen_Suppress
#define TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(Result, Type, Name, ResultExpr)    \
  auto Result = (ResultExpr); /* NOLINT(bugprone-macro-parentheses) */             \
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result);                                     \
  if (TVM_FFI_PREDICT_FALSE(!::tvm::ffi::details::AnyUnsafe::CheckAnyStrict<Type>( \
          ::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))) {                \
    return ::tvm::ffi::details::SMutateDeclaredTypeErrorRaw();                     \
  }                                                                                \
  Type Name = /* NOLINT(bugprone-macro-parentheses) */                             \
      ::tvm::ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<Type>(                 \
          ::std::move(::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))
/// \endcond

/*!
 * \brief Unwrap a successful mutation result into a newly declared value or return its error.
 *
 * ``Type`` must be concrete; use a type alias when it contains a top-level comma. A type mismatch
 * returns ``Unexpected(TypeError)`` through the surrounding ``Expected`` function without
 * throwing, reported with a fixed string so a correct hook pays only one predicted-not-taken
 * branch per field. This macro declares ``Name`` into the enclosing scope and must be used in a
 * braced block, never as an unbraced control-flow body. A raw pointer node must be non-null; pass
 * nullable nodes as ``ObjectRef`` or ``Any`` so ``None`` is skipped when constructing error
 * context.
 *
 * Example:
 * \code{.cpp}
 * TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ObjectRef, child, mutator->MutateExpected(self->child), self);
 * \endcode
 *
 * \param Type The concrete type of the successful value.
 * \param Name The name of the value declared in the enclosing scope.
 * \param ResultExpr An expression producing the ``Expected`` value to unwrap.
 */
#define TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Type, Name, ResultExpr)                                  \
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(TVM_FFI_STR_CONCAT(tvm_ffi_mutate_result_, __COUNTER__), \
                                          Type, Name, ResultExpr)

/// \cond Doxygen_Suppress
#define TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN_SKIP_CHECK_IMPL_(Result, Type, Name, ResultExpr) \
  auto Result = (ResultExpr); /* NOLINT(bugprone-macro-parentheses) */                            \
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result);                                                    \
  TVM_FFI_DCHECK(::tvm::ffi::details::AnyUnsafe::CheckAnyStrict<Type>(                            \
      ::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))                                      \
      << "unchecked structural-mutate assign: result is not of the declared type";                \
  Type Name = /* NOLINT(bugprone-macro-parentheses) */                                            \
      ::tvm::ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<Type>(                                \
          ::std::move(::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))
/// \endcond

/*!
 * \brief \ref TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN without the type check.
 *
 * Same signature and same error propagation; the difference is only what happens to a
 * successful result that is not of type \p Type.
 *
 * The caller must guarantee the result has the declared type; a mismatch is undefined behavior
 * in a release build, and debug builds catch it with ``TVM_FFI_DCHECK``.
 *
 * \param Type The concrete type of the successful value.
 * \param Name The name of the value declared in the enclosing scope.
 * \param ResultExpr An expression producing the ``Expected`` value to unwrap.
 */
#define TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN_SKIP_CHECK(Type, Name, ResultExpr) \
  TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN_SKIP_CHECK_IMPL_(                        \
      TVM_FFI_STR_CONCAT(tvm_ffi_mutate_result_, __COUNTER__), Type, Name, ResultExpr)

/// \cond Doxygen_Suppress
/// \endcond

/*!
 * \brief Structural mutator that invokes typed callbacks during recursive mapping.
 *
 * \tparam order Callback placement relative to child mapping.
 * \tparam Dispatch Callback dispatcher.
 *                  \sa StructuralMapCallbackChain
 */
/*!
 * \brief A runtime table of Function callbacks, usable as a single link.
 *
 * The typed links are matched at compile time from their argument type. The Python-driven API
 * instead carries a runtime list keyed by type index, so it appears to the mutator as one link
 * that performs its own lookup. That keeps a single traversal for both dispatch strategies.
 */

/*!
 * \brief Base of both the static and the dynamic StructuralMapMutator.
 *
 * Carries the identity-substitution environment they share. The dispatch thunks downcast only
 * as far as this class, so both reuse them regardless of how they store their callbacks.
 *
 */
class StructuralMapMutatorBaseObj : public StructuralMutatorObj {
 public:
  explicit StructuralMapMutatorBaseObj(const StructuralMutatorVTable* vtable)
      : StructuralMutatorObj(vtable) {}

 protected:
  /// \cond Doxygen_Suppress
  // Out of line so its strings stay out of the per-node dispatch function, which TryLink inlines
  // into. Shared by both map mutators: the typed one here and the dynamic one in the .cc.
  TVM_FFI_COLD_CODE static Expected<Any> SMutateDescentTypeError() noexcept {
    return Unexpected(Error("TypeError", "structural mutate: descent changed the node type", ""));
  }
  /// \endcond

  /*!
   * \brief Dispatch variable-remap lookup through the mutator vtable.
   * \param mutator The erased callback-aware mutator.
   * \param var The borrowed variable identity to look up.
   * \return Raw ``TVMFFIAny`` containing the owning replacement, FFI None, or Error.
   */
  static TVMFFIAny DispatchVarRemapGet(StructuralMutatorObj* mutator, AnyView var) noexcept {
    auto* self = static_cast<StructuralMapMutatorBaseObj*>(mutator);
    return ExpectedUnsafe::MoveToTVMFFIAny(self->VarRemapGetImpl(var));
  }

  /*!
   * \brief Dispatch variable-remap insertion through the mutator vtable.
   * \param mutator The erased callback-aware mutator.
   * \param var The borrowed variable identity to bind.
   * \param mapped_value The borrowed replacement value.
   * \return Raw ``TVMFFIAny`` containing FFI None or Error.
   */
  static TVMFFIAny DispatchVarRemapSet(StructuralMutatorObj* mutator, AnyView var,
                                       AnyView mapped_value) noexcept {
    auto* self = static_cast<StructuralMapMutatorBaseObj*>(mutator);
    return ExpectedUnsafe::MoveToTVMFFIAny(self->VarRemapSetImpl(var, mapped_value));
  }

  /*!
   * \brief Invoke a matched callback with optional def-region context.
   * \tparam Callback Callable returning a value implicitly convertible to ``Expected<Any>``.
   * \tparam Value Type of the converted value passed to the callback.
   * \param callback The matched callback.
   * \param value The converted value passed to the callback.
   * \param kind The active def-region kind.
   * \return The callback result normalized to ``Expected<Any>``.
   *
   * The callback may return a different type than the one that selected the link; only the
   * caller holding a field's static type can check that.
   */
  template <typename Callback, typename Value>
  TVM_FFI_INLINE static Expected<Any> InvokeCallbackLink(Callback& callback, Value&& value,
                                                         TVMFFIDefRegionKind kind) {
    using FuncInfo = FunctionInfo<std::decay_t<Callback>>;
    static_assert(std::is_convertible_v<typename FuncInfo::RetType, Expected<Any>>,
                  "StructuralMap callbacks must return a replacement value, Error, Unexpected, "
                  "or Expected<U> implicitly convertible to Expected<Any>");
    try {
      if constexpr (FuncInfo::num_args == 1) {
        return callback(std::forward<Value>(value));
      } else {
        return callback(std::forward<Value>(value), kind);
      }
    } catch (const Error& err) {
      return Unexpected(err);
    }
  }

  /*!
   * \brief Append \p node to a failed result's mutate error context.
   * \param result The failed result whose Error is annotated.
   * \param node The borrowed node to name in the context.
   */
  TVM_FFI_COLD_CODE static void UpdateVisitErrorContext(const Expected<Any>& result,
                                                        AnyView node) noexcept {
    // The Error is refcounted, so annotating the local handle annotates the object the result
    // holds. A non-object node has no context to add.
    if (node.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
      Error err = result.error();
      ::tvm::ffi::details::UpdateVisitErrorContext(err, node.cast<ObjectRef>());
    }
  }

  /*!
   * \brief Look up a replacement in the identity-substitution environment.
   * \param var The borrowed variable identity to look up.
   * \return The owning replacement, FFI None on a miss, or an Error.
   */
  Expected<Any> VarRemapGetImpl(AnyView var) noexcept {
    if (var.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) {
      return Unexpected(
          Error("TypeError", "Variable-remap key must be an object-backed value", ""));
    }
    try {
      ObjectRef var_ref = var.cast<ObjectRef>();
      std::optional<Any> result = var_remap_.Get(var_ref);
      if (!result.has_value()) {
        return Any(nullptr);
      }
      return *std::move(result);
    } catch (const Error& err) {
      return Unexpected(err);
    }
  }

  /*!
   * \brief Record a replacement in the identity-substitution environment.
   * \param var The borrowed variable identity to bind.
   * \param mapped_value The borrowed replacement value.
   * \return Successful completion, or an Error if the binding cannot be stored.
   */
  Expected<void> VarRemapSetImpl(AnyView var, AnyView mapped_value) noexcept {
    if (var.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) {
      return Unexpected(
          Error("TypeError", "Variable-remap key must be an object-backed value", ""));
    }
    try {
      ObjectRef var_ref = var.cast<ObjectRef>();
      Any owned_mapped_value(mapped_value);
      var_remap_.Set(var_ref, owned_mapped_value);
      return Expected<void>();
    } catch (const Error& err) {
      return Unexpected(err);
    }
  }

 private:
  /*! \brief Identity-substitution environment, shared across the whole traversal. */
  Map<ObjectRef, Any> var_remap_;
};

/*!
 * \brief Structural mutator that invokes statically typed callbacks during recursive mapping.
 *
 * Each callback is an ordinary callable and the engine selects it on its first argument's
 * type, so selection is a compile-time-known ``as<TSub>()`` on the input node.
 *
 * \tparam order Callback placement relative to child mapping.
 * \tparam Callbacks The callbacks, tested in declaration order.
 */
template <WalkOrder order, typename... Callbacks>
class StructuralMapMutatorObj : public StructuralMapMutatorBaseObj {
 public:
  /*!
   * \brief Construct a callback-aware mutator that owns its callbacks.
   * \param callbacks The typed callback links, tested in declaration order.
   */
  explicit StructuralMapMutatorObj(Callbacks... callbacks)
      : StructuralMapMutatorBaseObj(VTable()), callbacks_(std::move(callbacks)...) {}

 private:
  /*!
   * \brief Return the shared callback-aware mutator vtable.
   * \return Pointer to the immutable mutator vtable for this specialization.
   */
  static const StructuralMutatorVTable* VTable() {
    static const StructuralMutatorVTable vtable{
        &StructuralMapMutatorObj::DispatchMutate,
        &StructuralMapMutatorObj::DispatchMaybeInplaceMutate,
        &StructuralMapMutatorObj::DispatchVarRemapGet,
        &StructuralMapMutatorObj::DispatchVarRemapSet,
    };
    return &vtable;
  }

  /*!
   * \brief Dispatch callback-aware optional in-place mutation through the ABI vtable.
   * \param mutator The erased callback-aware mutator.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` containing the mutated value or Error.
   */
  static TVMFFIAny DispatchMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                              AnyView value) noexcept {
    auto* self = static_cast<StructuralMapMutatorObj*>(mutator);
    return self->MaybeInplaceMutateImplRaw(value);
  }

  /*!
   * \brief Dispatch callback-aware mutation through the ABI vtable.
   * \param mutator The erased callback-aware mutator.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` containing the mutated value or Error.
   */
  static TVMFFIAny DispatchMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    auto* self = static_cast<StructuralMapMutatorObj*>(mutator);
    return self->MutateImplRaw(value);
  }

  /*!
   * \brief Test one link against \p value and, if it matches, mutate the node through it.
   *
   * \tparam kMaybeInplace Whether the caller may mutate a uniquely owned node in place.
   * \tparam Callback The link's callback type.
   * \param callback The link's callback.
   * \param value The borrowed value to test and mutate.
   * \param out Receives the mutated value or Error when the link matched.
   * \return Whether the link matched, in which case \p out was written.
   */
  template <bool kMaybeInplace, typename Callback>
  TVM_FFI_INLINE bool TryLink(Callback& callback, AnyView value, Expected<Any>* out) noexcept {
    using FuncInfo = FunctionInfo<std::decay_t<Callback>>;
    static_assert(FuncInfo::num_args == 1 || FuncInfo::num_args == 2,
                  "StructuralMap callbacks must take one argument (value) or two arguments "
                  "(value, def-region kind)");
    using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
    using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;

    // Deliberately duplicated by StructuralMapDynMutatorObj::TryLink in structural_mutate.cc,
    // which differs only in how a link is found and called; keep the two in step.
    //
    // The match test and the matched-node path live together rather than in separate functions:
    // selecting the link already computes value.as<TSub>(), and a pre-order walk invokes on that
    // very node, so the converted value is reused instead of converted twice.
    //
    // Selection is on the input node, before any descent. A post-order walk must not select on
    // the descended node: the reflected fallback writes its own remap entry, so testing after
    // descent lets that entry swallow the callback.
    std::optional<TSub> matched;
    if constexpr (!std::is_same_v<TSub, AnyView> && !std::is_same_v<TSub, Any>) {
      matched = value.template as<TSub>();
      if (!matched.has_value()) return false;
    }

    // A FreeVar or DAG node maps once and every later occurrence reuses that result, so if this
    // node already has a cached remap entry, return it instead of mutating it again.
    const bool remappable = IsRemappableIdentity(value.type_index());
    if (remappable) {
      Expected<Any> mapped = VarRemapGetExpected(value);
      if (mapped.is_err()) {
        *out = std::move(mapped);
        return true;
      }
      if (ExpectedUnsafe::GetData(mapped).type_index() != TypeIndex::kTVMFFINone) {
        *out = std::move(mapped);
        return true;
      }
    }

    const TVMFFIDefRegionKind kind = def_region_kind();
    if constexpr (order == WalkOrder::kPreOrder) {
      // Pre-order: the callback rewrites this node first, then descent runs over whatever it
      // produced, so a replacement subtree is itself mapped.
      Expected<Any> callback_result = [&]() -> Expected<Any> {
        if constexpr (std::is_same_v<TSub, AnyView>) {
          return InvokeCallbackLink(callback, value, kind);
        } else if constexpr (std::is_same_v<TSub, Any>) {
          return InvokeCallbackLink(callback, Any(value), kind);
        } else {
          // Reuses the conversion the match already performed.
          return InvokeCallbackLink(callback, *std::move(matched), kind);
        }
      }();
      if (TVM_FFI_PREDICT_FALSE(callback_result.is_err())) {
        UpdateVisitErrorContext(callback_result, value);
        *out = std::move(callback_result);
        return true;
      }
      // Own the callback's result: it is the only reference from here on, and moving it out
      // beats holding a reference into an Expected that stays alive across the descent below.
      Any mapped_value = ExpectedUnsafe::GetData(callback_result);
      // Each descent names the node it actually ran on in the error context.
      *out = [&]() -> Expected<Any> {
        if constexpr (kMaybeInplace) {
          // A pre-order result can be mutated in place if unchanged or uniquely owned.
          const TVMFFIAny* mapped_data = AnyUnsafe::TVMFFIAnyPtrFromAny(mapped_value);
          const TVMFFIAny input_data = value.CopyToTVMFFIAny();
          if (mapped_data->type_index == input_data.type_index &&
              mapped_data->zero_padding == input_data.zero_padding &&
              mapped_data->v_int64 == input_data.v_int64) {
            return DefaultMaybeInplaceMutateExpected(value);
          }
          const Object* mapped_obj = mapped_value.as<Object>();
          bool can_inplace = mapped_obj != nullptr && mapped_obj->unique();
          return can_inplace ? DefaultMaybeInplaceMutateExpected(mapped_value)
                             : DefaultMutateExpected(mapped_value);
        } else {
          return DefaultMutateExpected(mapped_value);
        }
      }();
      if (TVM_FFI_PREDICT_FALSE(out->is_err())) return true;
    } else {
      // Post-order: children are mapped first and the callback sees the rebuilt node, so it
      // observes its operands already substituted.
      Expected<Any> descended =
          kMaybeInplace ? DefaultMaybeInplaceMutateExpected(value) : DefaultMutateExpected(value);
      if (TVM_FFI_PREDICT_FALSE(descended.is_err())) {
        *out = std::move(descended);
        return true;
      }
      // Held by reference, not moved out: the error path below names this node, so it has to
      // survive the callback. Only a storage-enabled TSub could gain from a move, and that is
      // exactly the case whose move would empty it.
      const Any& mapped_value = ExpectedUnsafe::GetData(descended);
      *out = [&]() -> Expected<Any> {
        if constexpr (std::is_same_v<TSub, AnyView>) {
          return InvokeCallbackLink(callback, AnyView(mapped_value), kind);
        } else if constexpr (std::is_same_v<TSub, Any>) {
          return InvokeCallbackLink(callback, Any(mapped_value), kind);
        } else {
          // Re-converted rather than reusing the match: the callback is invoked on the node
          // descent handed back, and must only see the type it asked for. Default mutation is
          // required to preserve the type, so failing here means some hook broke that.
          std::optional<TSub> descended_sub = mapped_value.template as<TSub>();
          if (TVM_FFI_PREDICT_FALSE(!descended_sub.has_value())) {
            return SMutateDescentTypeError();
          }
          return InvokeCallbackLink(callback, *std::move(descended_sub), kind);
        }
      }();
      if (TVM_FFI_PREDICT_FALSE(out->is_err())) {
        UpdateVisitErrorContext(*out, mapped_value);
        return true;
      }
    }

    // Bind this node's identity to its final result, so every later occurrence reuses it.
    if (remappable) {
      Expected<void> set_result = VarRemapSetExpected(value, ExpectedUnsafe::GetData(*out));
      if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
        *out = Unexpected(std::move(set_result).error());
      }
    }
    return true;
  }

  /*!
   * \brief Test every link in declaration order, stopping at the first that matches.
   * \return Whether some link matched, in which case \p out was written.
   */
  template <bool kMaybeInplace, size_t... Is>
  TVM_FFI_INLINE bool TryLinks(AnyView value, Expected<Any>* out,
                               std::index_sequence<Is...>) noexcept {
    return (TryLink<kMaybeInplace>(std::get<Is>(callbacks_), value, out) || ...);
  }

  /*!
   * \brief Mutate a value, invoking the first matching callback link.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` containing the mutated value or Error.
   */
  TVM_FFI_INLINE TVMFFIAny MutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if (TryLinks<false>(value, &out, std::index_sequence_for<Callbacks...>{})) {
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    }
    return DefaultMutateRaw(value);
  }

  /*!
   * \brief Mutate a value in place when safe, invoking the first matching callback link.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` containing the mutated value or Error.
   */
  TVM_FFI_INLINE TVMFFIAny MaybeInplaceMutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if (TryLinks<true>(value, &out, std::index_sequence_for<Callbacks...>{})) {
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    }
    return DefaultMaybeInplaceMutateRaw(value);
  }

  /*! \brief The callback links, tested in declaration order. */
  std::tuple<Callbacks...> callbacks_;
};

}  // namespace details

/*!
 * \brief Map a structured value graph and invoke typed replacement callbacks.
 *
 * Each callback is selected by the type of its first argument. The argument may be ``AnyView``,
 * ``Any``, an object reference type, an object pointer type, or another FFI-convertible POD type. A
 * callback may optionally take a second ``TVMFFIDefRegionKind`` argument. Callbacks are tested in
 * declaration order and only the first strict type match is invoked. An ``AnyView`` callback
 * argument is borrowed and must not be retained after the callback returns.
 *
 * Each callback should follow map semantics: it must not mutate the input in place and should
 * return a bare Any-convertible replacement or ``Expected<U>`` where ``U`` is Any-convertible.
 * A callback may instead return an error value or throw ``Error`` to stop the mapping. In
 * pre-order, an unchanged input or uniquely owned replacement may
 * continue through ``MaybeInplaceMutate``; a shared replacement uses ``Mutate``. In post-order,
 * the callback runs after the node's optional in-place mutation. In-place mutation is available
 * only through an explicit ``__s_maybe_inplace_mutate__`` hook.
 *
 * Objects marked ``kTVMFFISEqHashKindFreeVar`` or ``kTVMFFISEqHashKindDAGNode`` are
 * identity-substituted. A callback is invoked only for the first occurrence of each identity; its
 * final result, including an unchanged result, is reused for every later occurrence in the same
 * structural-map invocation.
 *
 * \sa WalkOrder, StructuralMutator
 *
 * Example:
 *
 * \code{.cpp}
 * Expected<Any> result = StructuralMapExpected<WalkOrder::kPostOrder>(
 *     root,
 *     [](const IntImm& value) -> Expected<Any> {
 *       if (value->value < 0) {
 *         return Unexpected(Error("ValueError", "negative constant", ""));
 *       }
 *       return Any(IntImm(value->value + 1));
 *     },
 *     [](const Add& add, TVMFFIDefRegionKind kind) -> Expected<Any> {
 *       // In post-order, add->lhs and add->rhs have already been mapped.
 *       return Any(add);
 *     });
 * \endcode
 *
 * \tparam order Whether callbacks run before or after recursively mapping children.
 * \tparam Callbacks Callback types whose first parameters select matching values.
 * \param root The borrowed root value to map.
 * \param callbacks Callbacks tested in declaration order. Each accepts ``(value)`` or
 *        ``(value, def_region_kind)`` and returns a bare Any-convertible replacement,
 *        ``Expected<U>`` where ``U`` is Any-convertible, or an error value.
 * \return The mapped owning value, or an Error if mapping or a callback fails.
 *
 * \note Returning ``Expected<U>`` expresses errors as values; throwing ``Error`` is also
 *       supported and is converted to the error state.
 */
template <WalkOrder order, typename... Callbacks>
Expected<Any> StructuralMapExpected(AnyView root, Callbacks&&... callbacks) noexcept {
  static_assert(sizeof...(Callbacks) != 0, "StructuralMap requires at least one callback");
  using Mutator = details::StructuralMapMutatorObj<order, std::decay_t<Callbacks>...>;
  StructuralMutator mutator(make_object<Mutator>(std::forward<Callbacks>(callbacks)...));
  return mutator->MaybeInplaceMutateIfUniqueExpected(root);
}

/*!
 * \brief Throwing form of \ref tvm::ffi::StructuralMapExpected.
 *
 * See \ref tvm::ffi::StructuralMapExpected for callback dispatch, ordering, and ownership
 * semantics.
 *
 * \tparam order Whether callbacks run before or after recursively mapping children.
 * \tparam Callbacks Callback types whose first parameters select matching values.
 * \param root The borrowed root value to map.
 * \param callbacks Callbacks tested in declaration order. Each accepts ``(value)`` or
 *        ``(value, def_region_kind)`` and returns a bare Any-convertible replacement,
 *        ``Expected<U>`` where ``U`` is Any-convertible, or an error value.
 * \return The mapped owning value.
 * \throws Error if mapping or a callback fails.
 *
 * \note Returning ``Expected<U>`` expresses errors as values; throwing ``Error`` is also
 *       supported and is rethrown by this interface.
 */
template <WalkOrder order, typename... Callbacks>
Any StructuralMap(AnyView root, Callbacks&&... callbacks) {
  return StructuralMapExpected<order>(root, std::forward<Callbacks>(callbacks)...).value();
}

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_STRUCTURAL_MUTATE_H_
