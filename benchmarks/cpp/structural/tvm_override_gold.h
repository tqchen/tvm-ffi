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
#ifndef TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_OVERRIDE_GOLD_H_
#define TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_OVERRIDE_GOLD_H_

// GOLD's hooks for the split-fuse fixture, against GOLD's engine (structural_mutate_gold.h).
//
// EXTRACTED FROM bench/377-four-state
// 220f363:benchmarks/cpp/structural/tvm_hook_override_unchanged.h (blob fb4cf266781c, sha256
// cfe4da4793bcada7b7a01889a1ed1f4808cd967dad48141622425b26139a721e), the file the GOLD state was
// timed with. This file is that file's text with the sections the split-fuse fixture never
// reaches left out -- Call, SeqStmt, Evaluate -- and nothing else changed:
//   * the include guard is this file's, and the engine include names structural_mutate_gold.h;
//   * `InstallAll` and `CoveredTypes` list only the six types kept: Var, IntImm, Add, Mul,
//     FloorDiv, FloorMod. The coverage assertion in the driver confirms nothing else is reached.
// Every remaining line -- the banner below, `MinimalMutateRoot`, `Unchanged`, the IntImm, Var and
// binary-operator hooks, `SetAttr`, `Install`, `InstallBinary` -- is the original text.
//

// `tvm_hook_override.h` rewritten for the unchanged descent protocol alone.
//
// Same layout, same function names and signatures, same order, same comments, so that it reads
// as a diff against that file. What changed, and only what changed: where the ported hook built
// an owning `ffi::Any(self)` to say "nothing here moved", this one returns `Unchanged()`.
//
// NO CONDITIONAL COMPILATION INSIDE A HOOK BODY, no template parameter over the result type, no
// base shared with the Expected file. The duplication is the point: each state's hooks are what
// someone would actually write against that protocol, which is the thing being measured. One
// body serving both would be neither.
//
// The review question, per hook: does it return unchanged on exactly the paths where the ported
// hook would have returned a value equal to its input? Not more, not fewer. Where the ported
// hook compared, this one uses `unchanged_or_same_as`, so a built-but-equal result still reports
// unchanged.
//
// AND IT MUST SHORT-CIRCUIT BEFORE CONSTRUCTING ANYTHING. A hook that computes the changed
// result and then discovers nothing changed has already paid the cost the protocol exists to
// avoid, reports "unchanged" correctly, and measures the old cost under a new name.
//
// `port_check.sh` verifies `tvm_hook_override.h` against its apache/tvm source. This file has no
// upstream source to check against -- it is written by hand -- which is why it is reviewed
// instead.

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/structural_mutate_gold.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/stmt.h>

#include <cstring>
#include <utility>
#include <vector>

#include "bench_common.h"

namespace tvm_hooks {

using namespace tvm;              // NOLINT(build/namespaces)
using namespace tvm::tirx;        // NOLINT(build/namespaces)
namespace ffi = tvm::ffi;

/*!
 * \brief The `map_floor` arm's root call into the minimal mutator, in this state's API.
 *
 * `MinimalMutatorObj` dispatches the hook directly and never reaches the engine, so its result
 * is whatever this state's ABI carries. This state's `MutateExpected` is the compatibility
 * form that resolves `unchanged` against its own input before returning, so the root sees a
 * value and has nothing left to resolve.
 */
inline ffi::Expected<ffi::Any> MinimalMutateRoot(ffi::bench::MinimalMutatorObj* mutator,
                                                 ffi::AnyView input, bool moved) noexcept {
  return moved ? mutator->MaybeInplaceMutateIfUniqueExpected(input)
               : mutator->MutateExpected(input);
}

/*!
 * \brief The unchanged result, as the raw ABI value a hook returns.
 *
 * The marker is a type index in an otherwise empty TVMFFIAny, so returning it touches no
 * refcount and allocates nothing. Everywhere the ported hook returns `ffi::Any(self)` to mean
 * "the input is the answer", this file returns this.
 */
inline TVMFFIAny Unchanged() noexcept {
  TVMFFIAny raw{};
  raw.type_index = ffi::TypeIndex::kTVMFFIStructuralMutateUnchanged;
  return raw;
}

// ---------------------------------------------------------------------------
// src/ir/expr.cc -- IntImm
// ---------------------------------------------------------------------------

TVMFFIAny IntImmVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // skips: value
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny IntImmMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  // skips: value -- so this node descends into nothing and can never change. The node fetch
  // existed only to build the owning result the old contract required.
  return Unchanged();
}

TVMFFIAny IntImmMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  // skips: value
  return Unchanged();
}

// ---------------------------------------------------------------------------
// src/ir/expr.cc -- Var
// ---------------------------------------------------------------------------

TVMFFIAny VarVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: name
  const VarNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const VarNode>(value);
  // A PrimType carries only a dtype, so it has nothing to visit.  Broad callbacks do not see this
  // skipped field; dynamically typed Vars still descend through the Type value.
  if (!self->ty.as<PrimTypeNode>()) {
    // Only NonRecursive is clamped: Recursive co-introduces type fields such as BufferType shape
    // variables, so that ambient region must continue through the dynamic type.
    if (visitor->def_region_kind() == kTVMFFIDefRegionKindNonRecursive) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
          kTVMFFIDefRegionKindNone, [&]() { return visitor->VisitExpected(self->ty); }));
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
    }
  }
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny VarMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: name
  const VarNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const VarNode>(value);
  ffi::Expected<ffi::Any> remap_result = mutator->VarRemapGetExpected(value);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap_result);
  if (ffi::details::ExpectedUnsafe::GetData(remap_result).type_index() !=
      ffi::TypeIndex::kTVMFFINone) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(remap_result));
  }
  // The ported hook's `return_mapped_var` splits in two, because the protocol is the only thing
  // that differs between its two callers: the unchanged exit binds the input to itself and says
  // nothing was built, the changed exit binds and hands back what it built.
  auto bind_unchanged = [&]() -> TVMFFIAny {
    auto set_result = mutator->VarRemapSetExpected(value, value);
    if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
      // Hooks propagate errors untouched; the engine names this node.
      return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(set_result).error()));
    }
    return Unchanged();
  };
  auto bind_mapped_var = [&](ffi::Any mapped_var) -> TVMFFIAny {
    auto set_result = mutator->VarRemapSetExpected(value, mapped_var);
    if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
      // Hooks propagate errors untouched; the engine names this node.
      return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(set_result).error()));
    }
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(mapped_var));
  };
  // A PrimType carries only a dtype, so it has nothing to substitute.  Broad callbacks do not see
  // this skipped field; dynamically typed Vars still descend through the Type value.
  if (self->ty.as<PrimTypeNode>()) {
    return bind_unchanged();
  }
  // Only NonRecursive is clamped: Recursive co-introduces type fields such as BufferType shape
  // variables, so that ambient region must continue through the dynamic type.
  auto mutate_ty = [&]() { return mutator->Mutate(self->ty); };
  ffi::MaybeUnchanged<ffi::Any> mapped_ty_result =
      mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
          ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_ty)
          : mutate_ty();
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<Type>, mapped_ty,
                                    std::move(mapped_ty_result));
  if (mapped_ty.unchanged_or_same_as(self->ty)) {
    return bind_unchanged();
  }
  ffi::ObjectPtr<VarNode> copy = ffi::make_object<VarNode>(*self);
  std::move(mapped_ty).move_if_changed(&copy->ty);
  return bind_mapped_var(ffi::Any(std::move(copy)));
}

TVMFFIAny VarMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: name
  VarNode* self = const_cast<VarNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const VarNode>(value));
  ffi::Expected<ffi::Any> remap_result = mutator->VarRemapGetExpected(value);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap_result);
  if (ffi::details::ExpectedUnsafe::GetData(remap_result).type_index() !=
      ffi::TypeIndex::kTVMFFINone) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(remap_result));
  }
  // Every exit here is the input object, so there is one exit and `bind_mapped_var` is
  // unreachable; it is not declared.
  auto bind_unchanged = [&]() -> TVMFFIAny {
    auto set_result = mutator->VarRemapSetExpected(value, value);
    if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
      // Hooks propagate errors untouched; the engine names this node.
      return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(set_result).error()));
    }
    return Unchanged();
  };
  // A PrimType carries only a dtype, so it has nothing to substitute.  Broad callbacks do not see
  // this skipped field; dynamically typed Vars still descend through the Type value.
  if (self->ty.as<PrimTypeNode>()) {
    return bind_unchanged();
  }
  // Only NonRecursive is clamped: Recursive co-introduces type fields such as BufferType shape
  // variables, so that ambient region must continue through the dynamic type.
  auto mutate_ty = [&]() { return mutator->MaybeInplaceMutateIfUnique(self->ty); };
  ffi::MaybeUnchanged<ffi::Any> mapped_ty_result =
      mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
          ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_ty)
          : mutate_ty();
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<Type>, mapped_ty,
                                    std::move(mapped_ty_result));
  // `unchanged` is identity, not contents: `self->ty` is overwritten in place and the node is
  // still the object the caller handed in, so this exit is correct and is the same one the
  // ported hook takes.
  std::move(mapped_ty).move_if_changed(&self->ty);
  return bind_unchanged();
}

// ---------------------------------------------------------------------------
// src/ir/prim/expr.cc -- the binary operators
// ---------------------------------------------------------------------------

template <typename TNode>
TVMFFIAny BinaryVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const TNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->b));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

template <typename TNode>
TVMFFIAny BinaryMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const TNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<PrimExpr>, a, mutator->Mutate(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<PrimExpr>, b, mutator->Mutate(self->b));
  // The short circuit: both operands answered before anything was built.
  if (a.unchanged_or_same_as(self->a) && b.unchanged_or_same_as(self->b)) {
    return Unchanged();
  }
  ffi::ObjectPtr<TNode> copy = ffi::make_object<TNode>(*self);
  // StructuralMap preserves node types. A rewrite that changes operand dtypes must keep the
  // operands compatible and set the result type itself; a generic traversal cannot infer the
  // casts that would require.
  // The copy already carries `*self`'s operands, so an operand that answered unchanged keeps
  // the one that is already there.
  std::move(a).move_if_changed(&copy->a);
  std::move(b).move_if_changed(&copy->b);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

template <typename TNode>
TVMFFIAny BinaryMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  TNode* self = const_cast<TNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<PrimExpr>, a,
                                    mutator->MaybeInplaceMutateIfUnique(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<PrimExpr>, b,
                                    mutator->MaybeInplaceMutateIfUnique(self->b));
  // An unchanged operand is known to be the one already in the slot, so the write is skipped
  // without the comparison the ported hook needed to discover the same thing.
  std::move(a).move_if_changed(&self->a);
  std::move(b).move_if_changed(&self->b);
  return Unchanged();
}

// ---------------------------------------------------------------------------
// Installation, over whatever TVM registered.
//
// HARNESS DEVIATION, and the only one: TVM registers these through `refl::TypeAttrDef<T>()` in
// static-init blocks. The harness cannot use that path -- it has to overwrite an already
// registered attribute from main(), which is what the branch-local
// TVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE build of tvm-ffi permits -- so registration goes
// through TVMFFITypeRegisterAttr directly. The hook bodies above are unaffected.
// ---------------------------------------------------------------------------

inline void SetAttr(int32_t type_index, const char* name, void* fn) {
  TVMFFIByteArray name_array{name, std::strlen(name)};
  TVMFFIAny value_any = ffi::AnyView(fn).CopyToTVMFFIAny();
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index, &name_array, &value_any));
}

inline void Install(int32_t type_index, ffi::FStructuralVisit visit, ffi::FStructuralMutate mutate,
                    ffi::FStructuralMutate inplace) {
  namespace refl = tvm::ffi::reflection;
  SetAttr(type_index, refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(visit));
  SetAttr(type_index, refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(mutate));
  SetAttr(type_index, refl::type_attr::kStructuralMaybeInplaceMutate,
          reinterpret_cast<void*>(inplace));
}

template <typename TNode>
void InstallBinary() {
  Install(TNode::RuntimeTypeIndex(), &BinaryVisit<TNode>, &BinaryMutate<TNode>,
          &BinaryMaybeInplaceMutate<TNode>);
}

/*!
 * \brief Install every hook the benchmark dispatches into. Call once from main().
 *
 * The set was derived empirically, from a dispatch pass over the fixtures rather than from
 * TVM's registry. `real_tvm_bench.cc` asserts coverage: every type a fixture reaches must be
 * one of these, so a new fixture that reaches a new type names it instead of silently falling
 * through to TVM's own hook.
 */
inline void InstallAll() {
  Install(VarNode::RuntimeTypeIndex(), &VarVisit, &VarMutate, &VarMaybeInplaceMutate);
  Install(IntImmNode::RuntimeTypeIndex(), &IntImmVisit, &IntImmMutate, &IntImmMaybeInplaceMutate);
  InstallBinary<prim::AddNode>();
  InstallBinary<prim::MulNode>();
  InstallBinary<prim::FloorDivNode>();
  InstallBinary<prim::FloorModNode>();
}

/*! \brief The type indices InstallAll covers, for the coverage assertion. */
inline std::vector<int32_t> CoveredTypes() {
  return {VarNode::RuntimeTypeIndex(),       IntImmNode::RuntimeTypeIndex(),
          prim::AddNode::RuntimeTypeIndex(), prim::MulNode::RuntimeTypeIndex(),
          prim::FloorDivNode::RuntimeTypeIndex(),
          prim::FloorModNode::RuntimeTypeIndex()};
}

}  // namespace tvm_hooks

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_OVERRIDE_GOLD_H_
