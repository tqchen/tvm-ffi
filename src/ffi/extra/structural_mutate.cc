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
 * \file src/ffi/extra/structural_mutate.cc
 * \brief Structural mutator and structural map registration.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>

#include <utility>

namespace tvm {
namespace ffi {

namespace details {

/*!
 * \brief Runtime structural map for callback arrays.
 *
 * \param root The root value to map.
 * \param callbacks Runtime callback entries of ``(type_index, ffi::Function)`` invoked as
 *                  ``callback(value)``.
 * \param callbacks_with_def_region_kind Runtime callback entries of ``(type_index, ffi::Function)``
 *                                       invoked as ``callback(value, def_region_kind)``.
 * \param order Integer value of \ref WalkOrder.
 * \return The mapped owning value, or an Error.
 */
/*!
 * \brief Structural mutator whose links are runtime ffi.Functions keyed by type index.
 *
 * The dynamic counterpart of \ref StructuralMapMutatorObj. It selects a link by scanning
 * registered type indices where the static version does a compile-time ``as<TSub>()``. The two
 * are kept apart, rather than sharing one template with a mode flag, so each version reads
 * straight through; only the identity remap they share lives in the common base. Keeping this
 * one in the .cc also keeps the link table out of the public header.
 *
 * \tparam order Callback placement relative to child mapping.
 */
template <WalkOrder order>
class StructuralMapDynMutatorObj : public StructuralMapMutatorBaseObj {
 public:
  StructuralMapDynMutatorObj(Array<Tuple<int32_t, Function>> callbacks,
                             Array<Tuple<int32_t, Function>> callbacks_with_def_region_kind)
      : StructuralMapMutatorBaseObj(VTable()),
        callbacks_(std::move(callbacks)),
        callbacks_with_def_region_kind_(std::move(callbacks_with_def_region_kind)) {}

 private:
  static const StructuralMutatorVTable* VTable() {
    static const StructuralMutatorVTable vtable{
        &StructuralMapDynMutatorObj::DispatchMutate,
        &StructuralMapDynMutatorObj::DispatchMaybeInplaceMutate,
        &StructuralMapDynMutatorObj::DispatchVarRemapGet,
        &StructuralMapDynMutatorObj::DispatchVarRemapSet,
    };
    return &vtable;
  }

  static TVMFFIAny DispatchMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                              AnyView value) noexcept {
    return static_cast<StructuralMapDynMutatorObj*>(mutator)->MaybeInplaceMutateImplRaw(value);
  }

  static TVMFFIAny DispatchMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    return static_cast<StructuralMapDynMutatorObj*>(mutator)->MutateImplRaw(value);
  }

  /*!
   * \brief Find the first link registered for \p type_index.
   *
   * \param type_index The input node's runtime type index.
   * \param with_kind Set when the matched link also takes a def-region kind.
   * \param link_type_index Set to the registered type index the link matched on, so a post-order
   *        walk can recheck the descended node against the same target.
   * \return The matched Function, or nullopt when no link applies.
   */
  Optional<Function> FindLink(int32_t type_index, bool* with_kind,
                              int32_t* link_type_index) const noexcept {
    for (const Tuple<int32_t, Function>& entry : callbacks_) {
      if (RuntimeTypeIndexMatch(type_index, entry.get<0>())) {
        *with_kind = false;
        *link_type_index = entry.get<0>();
        return entry.get<1>();
      }
    }
    for (const Tuple<int32_t, Function>& entry : callbacks_with_def_region_kind_) {
      if (RuntimeTypeIndexMatch(type_index, entry.get<0>())) {
        *with_kind = true;
        *link_type_index = entry.get<0>();
        return entry.get<1>();
      }
    }
    return std::nullopt;
  }

  /*!
   * \brief Invoke a matched link, threading the live def-region kind through.
   *
   * \p kind is passed rather than stashed at selection time: the engine only knows the true
   * def-region kind of the node handed back after descent (post-order) or of the matched node
   * itself (pre-order).
   */
  TVM_FFI_INLINE static Expected<Any> InvokeLink(const Function& fn, bool with_kind, AnyView target,
                                                 TVMFFIDefRegionKind kind) noexcept {
    // CallExpected is exception-free: it goes through the safe-call path and returns any raised
    // error as Unexpected, unlike a directly invoked C++ callback.
    return with_kind ? fn.CallExpected<Any>(target, kind) : fn.CallExpected<Any>(target);
  }

  /*!
   * \brief Test the link table against \p value and mutate through the first match.
   *
   * \tparam kMaybeInplace Whether a uniquely owned node may be mutated in place.
   * \param value The borrowed value to test and mutate.
   * \param out Receives the mutated value or Error when a link matched.
   * \return Whether a link matched, in which case \p out was written.
   */
  template <bool kMaybeInplace>
  TVM_FFI_INLINE bool TryLink(AnyView value, Expected<Any>* out) noexcept {
    // Step for step the same walk as StructuralMapMutatorObj::TryLink, and deliberately so:
    // only link detection and invocation differ between the two, and keeping them as separate
    // straight-line copies lets each specialize on its own selection strategy and keeps both
    // readable. Everything below except finding and calling the link is shared semantics, so a
    // change to either copy belongs in both.

    bool with_kind = false;
    int32_t link_type_index = TypeIndex::kTVMFFINone;
    // A local, so descending into a matching child cannot change what this node invokes.
    Optional<Function> matched = FindLink(value.type_index(), &with_kind, &link_type_index);
    if (!matched.has_value()) return false;

    // --- identity remap, entry half -----------------------------------------
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

    // --- callback and descent, in walk order --------------------------------
    const TVMFFIDefRegionKind kind = def_region_kind();
    if constexpr (order == WalkOrder::kPreOrder) {
      // Pre-order: the callback rewrites this node first, then descent runs over what it made.
      Expected<Any> callback_result = InvokeLink(*matched, with_kind, value, kind);
      if (TVM_FFI_PREDICT_FALSE(callback_result.is_err())) {
        UpdateVisitErrorContext(callback_result, value);
        *out = std::move(callback_result);
        return true;
      }
      // Own the callback's result: it is the only reference from here on.
      Any mapped_value = ExpectedUnsafe::GetData(callback_result);
      // Each descent names the node it actually ran on in the error context.
      *out = [&]() -> Expected<Any> {
        if constexpr (kMaybeInplace) {
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
      // Post-order: children are mapped first, so the callback sees the rebuilt node.
      Expected<Any> descended =
          kMaybeInplace ? DefaultMaybeInplaceMutateExpected(value) : DefaultMutateExpected(value);
      if (TVM_FFI_PREDICT_FALSE(descended.is_err())) {
        *out = std::move(descended);
        return true;
      }
      // Held by reference, not moved out: the error path below names this node, so it has to
      // survive the callback.
      const Any& mapped_value = ExpectedUnsafe::GetData(descended);
      // The link was selected on the input node, and the callback must only see the type it
      // registered for. The typed mutator gets this from its `mapped_value.as<TSub>()`; here the
      // registered type index is the same target, so recheck against it.
      if (TVM_FFI_PREDICT_FALSE(
              !RuntimeTypeIndexMatch(mapped_value.type_index(), link_type_index))) {
        *out =
            Unexpected(Error("TypeError", "structural mutate: descent changed the node type", ""));
        UpdateVisitErrorContext(*out, mapped_value);
        return true;
      }
      *out = InvokeLink(*matched, with_kind, mapped_value, kind);
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

  TVM_FFI_INLINE TVMFFIAny MutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if (TryLink<false>(value, &out)) {
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    }
    return DefaultMutateRaw(value);
  }

  TVM_FFI_INLINE TVMFFIAny MaybeInplaceMutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if (TryLink<true>(value, &out)) {
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    }
    return DefaultMaybeInplaceMutateRaw(value);
  }

  Array<Tuple<int32_t, Function>> callbacks_;
  Array<Tuple<int32_t, Function>> callbacks_with_def_region_kind_;
};

Expected<Any> StructuralMapExpected(
    AnyView root, const Array<Tuple<int32_t, Function>>& callbacks,
    const Array<Tuple<int32_t, Function>>& callbacks_with_def_region_kind, int order) noexcept {
  if (order == static_cast<int>(WalkOrder::kPreOrder)) {
    using Mutator = StructuralMapDynMutatorObj<WalkOrder::kPreOrder>;
    StructuralMutator mutator(make_object<Mutator>(callbacks, callbacks_with_def_region_kind));
    return mutator->MaybeInplaceMutateIfUniqueExpected(root);
  } else {
    using Mutator = StructuralMapDynMutatorObj<WalkOrder::kPostOrder>;
    StructuralMutator mutator(make_object<Mutator>(callbacks, callbacks_with_def_region_kind));
    return mutator->MaybeInplaceMutateIfUniqueExpected(root);
  }
}

// ---------------------------------------------------------------------------
// Built-in container structural mutation.
// ---------------------------------------------------------------------------

/*!
 * \brief Structurally mutate the elements of a sequence container.
 *
 * \tparam SeqObj The underlying sequence object type.
 * \param mutator The active structural mutator.
 * \param value The borrowed sequence container.
 * \param self The sequence object stored in \p value.
 * \return The mutated sequence, or an Error.
 */
template <typename SeqObj>
TVMFFIAny MutateSeqContainerRaw(StructuralMutatorObj* mutator, AnyView value,
                                const SeqObj* self) noexcept {
  int64_t size = static_cast<int64_t>(self->size());
  const Any* items = self->begin();
  ObjectPtr<SeqObj> output = nullptr;

  for (int64_t i = 0; i < size; ++i) {
    const Any& item = items[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Any, mapped_value, mutator->MutateExpected(item));

    if (output == nullptr) {
      if (item.same_as(mapped_value)) {
        continue;
      }
      output = SeqObj::CreateRepeated(size, Any());
      output->InitRange(0, items, items + i);
    }
    output->SetItemAfterCheck(i, std::move(mapped_value));
  }

  if (output == nullptr) {
    return AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }
  return AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(output)));
}

/*!
 * \brief Structurally mutate the elements of a sequence container in place when safe.
 *
 * \tparam SeqObj The underlying sequence object type.
 * \param mutator The active structural mutator.
 * \param value The borrowed sequence container, which must be safe to mutate in place.
 * \param self The sequence object stored in \p value.
 * \return The mutated sequence, or an Error.
 */
template <typename SeqObj>
TVMFFIAny MaybeInplaceMutateSeqContainerRaw(StructuralMutatorObj* mutator, AnyView value,
                                            SeqObj* self) noexcept {
  for (int64_t i = 0; i < static_cast<int64_t>(self->size()); ++i) {
    const Any& item = self->begin()[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Any, mapped_value,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));

    if (!item.same_as(mapped_value)) {
      self->SetItemAfterCheck(i, std::move(mapped_value));
    }
  }
  return AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
}

/*!
 * \brief Structurally mutate the values of a map container.
 *
 * \tparam MapObjType The underlying map object type.
 * \param mutator The active structural mutator.
 * \param value The borrowed map container.
 * \param self The map object stored in \p value.
 * \return The mutated map, or an Error.
 */
template <typename MapObjType>
TVMFFIAny MutateMapValuesRaw(StructuralMutatorObj* mutator, AnyView value,
                             const MapObjType* self) noexcept {
  ObjectPtr<Object> output = nullptr;
  MapBaseObj::iterator output_it;
  size_t index = 0;

  for (auto source_it = self->begin(); source_it != self->end(); ++source_it, ++index) {
    const Any& old_value = source_it->second;
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Any, new_value, mutator->MutateExpected(old_value));
    bool changed = !old_value.same_as(new_value);
    if (output == nullptr) {
      if (!changed) {
        continue;
      }
      output = MapObjType::ShallowCopy(self);
      output_it = static_cast<MapBaseObj*>(output.get())->begin();
      for (size_t i = 0; i < index; ++i) {
        ++output_it;
      }
    }
    if (changed) {
      output_it->second = std::move(new_value);
    }
    ++output_it;
  }

  if (output == nullptr) {
    return AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }
  return AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(output)));
}

/*!
 * \brief Structurally mutate the values of a map container in place when safe.
 *
 * \tparam MapObjType The underlying map object type.
 * \param mutator The active structural mutator.
 * \param value The borrowed map container, which must be safe to mutate in place.
 * \param self The map object stored in \p value.
 * \return The mutated map, or an Error.
 */
template <typename MapObjType>
TVMFFIAny MaybeInplaceMutateMapValuesRaw(StructuralMutatorObj* mutator, AnyView value,
                                         MapObjType* self) noexcept {
  for (auto it = self->begin(); it != self->end(); ++it) {
    const Any& old_value = it->second;
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Any, new_value,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(old_value));

    if (!old_value.same_as(new_value)) {
      it->second = std::move(new_value);
    }
  }
  return AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
}

/*! \brief Identity structural mutation hook for immutable String and Bytes leaves. */
TVMFFIAny MutateImmutableLeaf(StructuralMutatorObj*, AnyView value) noexcept {
  Expected<Any> result = Any(value);
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Structural mutation hook for ArrayObj. */
TVMFFIAny MutateArray(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MutateSeqContainerRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ArrayObj>(value));
}

/*! \brief Maybe-in-place structural mutation hook for ArrayObj. */
TVMFFIAny MaybeInplaceMutateArray(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MaybeInplaceMutateSeqContainerRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<ArrayObj>(value));
}

/*! \brief Structural mutation hook for ListObj. */
TVMFFIAny MutateList(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MutateSeqContainerRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ListObj>(value));
}

/*! \brief Maybe-in-place structural mutation hook for ListObj. */
TVMFFIAny MaybeInplaceMutateList(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MaybeInplaceMutateSeqContainerRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<ListObj>(value));
}

/*! \brief Structural mutation hook for MapObj. */
TVMFFIAny MutateMap(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MutateMapValuesRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MapObj>(value));
}

/*! \brief Maybe-in-place structural mutation hook for MapObj. */
TVMFFIAny MaybeInplaceMutateMap(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MaybeInplaceMutateMapValuesRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<MapObj>(value));
}

/*! \brief Structural mutation hook for DictObj. */
TVMFFIAny MutateDict(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MutateMapValuesRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DictObj>(value));
}

/*! \brief Maybe-in-place structural mutation hook for DictObj. */
TVMFFIAny MaybeInplaceMutateDict(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MaybeInplaceMutateMapValuesRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<DictObj>(value));
}
}  // namespace details

// ---------------------------------------------------------------------------
// Static registration.
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<StructuralMutatorObj>();  // NOLINT(bugprone-unused-raii)
  refl::GlobalDef()
      .def_method("ffi.StructuralMutatorMaybeInplaceMutate",
                  &StructuralMutatorObj::MaybeInplaceMutate)
      .def_method("ffi.StructuralMutatorMutate", &StructuralMutatorObj::Mutate)
      .def_method("ffi.StructuralMutatorVarRemapGet",
                  [](const StructuralMutator& mutator, AnyView var) {
                    return mutator->VarRemapGetExpected(var).value();
                  })
      .def_method("ffi.StructuralMutatorVarRemapSet",
                  [](const StructuralMutator& mutator, AnyView var, AnyView mapped_value) {
                    mutator->VarRemapSetExpected(var, mapped_value).value();
                  })
      .def_method("ffi.StructuralMutatorDefRegionKind", &StructuralMutatorObj::def_region_kind)
      .def_method(
          "ffi.StructuralMutatorWithDefRegionKind",
          [](const StructuralMutator& mutator, TVMFFIDefRegionKind kind, const Function& callback) {
            return mutator->WithDefRegionKind(kind, callback);
          })
      .def("ffi.StructuralMap",
           [](AnyView root, const Array<Tuple<int32_t, Function>>& callbacks,
              const Array<Tuple<int32_t, Function>>& callbacks_with_def_region_kind,
              int32_t order) -> Any {
             return details::StructuralMapExpected(root, callbacks, callbacks_with_def_region_kind,
                                                   order)
                 .value();
           });
  refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
  refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMaybeInplaceMutate);
  refl::TypeAttrDef<details::StringObj>()
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateImmutableLeaf)))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateImmutableLeaf)));
  refl::TypeAttrDef<details::BytesObj>()
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateImmutableLeaf)))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateImmutableLeaf)));
  refl::TypeAttrDef<ArrayObj>()
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateArray)))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(
                static_cast<FStructuralMutate>(&details::MaybeInplaceMutateArray)));
  refl::TypeAttrDef<ListObj>()
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateList)))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(
                static_cast<FStructuralMutate>(&details::MaybeInplaceMutateList)));
  refl::TypeAttrDef<MapObj>()
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateMap)))
      .attr(
          refl::type_attr::kStructuralMaybeInplaceMutate,
          reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MaybeInplaceMutateMap)));
  refl::TypeAttrDef<DictObj>()
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateDict)))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(
                static_cast<FStructuralMutate>(&details::MaybeInplaceMutateDict)));
}

}  // namespace ffi
}  // namespace tvm
