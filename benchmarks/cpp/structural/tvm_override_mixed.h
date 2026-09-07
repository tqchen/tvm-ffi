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
#ifndef TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_OVERRIDE_MIXED_H_
#define TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_OVERRIDE_MIXED_H_

// UC's hooks for the split-fuse fixture, against the mixed engine (structural_mutate_mixed.h):
// GOLD's engine carrying the UC protocol alongside its own.
//
// This file is tvm_override_uc.h with the three descent entries renamed to the names the mixed
// engine gives UC's entries, so they do not collide with GOLD's compatibility forms of the
// same names, and nothing else:
//       MutateExpected                     -> MutateUnchangedOr
//       MaybeInplaceMutateExpected         -> MaybeInplaceMutateUnchangedOr
//       MaybeInplaceMutateIfUniqueExpected -> MaybeInplaceMutateIfUniqueUnchangedOr
// (applied as `\b<name>(` -> `<name'>(`), plus this file's include guard and engine include.
// tvm_override_uc.h's own banner follows, and describes what it is extracted from.
//

// `tvm_hook_override.h` rewritten for the `UnchangedOr` descent protocol alone -- the UC state,
// tvm-ffi `refactor/structural-mutate-unchanged-or`.
//
// Same layout, same function names and signatures, same order, same comments, so that it reads
// as a diff against that file. What changed, and only what changed: where the ported hook built
// an owning `ffi::Any(self)` to say "nothing here moved", this one says so with the protocol's
// own answer.
//
// THE API HERE IS NOT `tvm_hook_override_unchanged.h`'s. That file is the GOLD state and speaks
// the superseded `MaybeUnchanged<T>`, which also carried the error. This state separates them:
// the result is `Expected<UnchangedOr<T>>`, so a hook reads
//
//   * `ffi::UnchangedOr<T>` as the declared type of `TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN`,
//   * `mutator->MutateUnchangedOr(...)` / `MaybeInplaceMutateIfUniqueUnchangedOr(...)` -- the
//     exception-free entry points, since `Mutate` now throws and returns the bare `UnchangedOr`,
//   * `TVM_FFI_S_MUTATE_RETURN_UNCHANGED()` where the GOLD file returned its own raw marker,
//   * `UnchangedOrSameAs` for the short-circuit test, and
//   * `if (!x.IsUnchanged()) slot = std::move(x).ValueUnchecked();` at an assignment site.
//
// THAT LAST LINE IS THE ONLY THING THAT DIFFERS FROM `tvm_hook_override_unchanged_or.h`.
// Sixteen assignment sites, nothing else: same engine, same protocol, same descent entries,
// same short circuits, same declared types.
//
// The documented idiom is `slot = std::move(x).ValueOrUnchanged(std::move(slot))` -- see the
// worked example on `TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN` in `structural_mutate.h` -- and on the
// unchanged path it moves the field out of its own slot and then back: a temporary is
// constructed from `slot`, `slot` is left empty, the temporary is move-assigned into it, and the
// temporary is destroyed. The GOLD file's `std::move(x).move_if_changed(&slot)` skipped the
// write entirely on that path. This file asks whether the gap between UC and GOLD is the
// protocol or the idiom the protocol documents, which no single-construction run can answer --
// the same reason the caller-entry and descent-boundary variants of the ABI widening were both
// run in one interleave.
//
// Both files are UC. Neither carries a name from the superseded `MaybeUnchanged` API.
//
// NO CONDITIONAL COMPILATION INSIDE A HOOK BODY, no template parameter over the result type, no
// base shared with the Expected file or with the GOLD file. The duplication is the point: each
// state's hooks are what someone would actually write against that protocol, which is the thing
// being measured. One body serving several would be none of them.
//
// The review question, per hook: does it return unchanged on exactly the paths where the ported
// hook would have returned a value equal to its input? Not more, not fewer. Where the ported
// hook compared, this one uses `UnchangedOrSameAs`, so a built-but-equal result still reports
// unchanged.
//
// AND IT MUST SHORT-CIRCUIT BEFORE CONSTRUCTING ANYTHING. A hook that computes the changed
// result and then discovers nothing changed has already paid the cost the protocol exists to
// avoid, reports "unchanged" correctly, and measures the old cost under a new name. Three places
// keep an explicit `IsUnchanged()` test rather than funnelling through `ValueOrUnchanged`, all
// of them in the SeqStmt hooks, because there the alternative to the replacement is a `Stmt`
// handle materialized out of a borrowed array slot -- and building it to pass as the fallback
// would pay a refcount on the path that does have a replacement.
//
// `port_check.sh` verifies `tvm_hook_override.h` against its apache/tvm source. This file has no
// upstream source to check against -- it is written by hand -- which is why it is reviewed
// instead.

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/structural_mutate_mixed.h>
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
 * is whatever this state's ABI carries. Here that is `UnchangedOr<Any>`, and the root -- the
 * one caller with nowhere further to hand `unchanged` -- resolves it against its own input.
 * That resolution is once per call, not once per node.
 */
inline ffi::Expected<ffi::Any> MinimalMutateRoot(ffi::bench::MinimalMutatorObj* mutator,
                                                 ffi::AnyView input, bool moved) noexcept {
  ffi::Expected<ffi::UnchangedOr<ffi::Any>> result =
      moved ? mutator->MaybeInplaceMutateIfUniqueUnchangedOr(input) : mutator->MutateUnchangedOr(input);
  if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
    return ffi::Unexpected(std::move(result).error());
  }
  return std::move(result).value().ValueOrUnchanged(input);
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
  TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
}

TVMFFIAny IntImmMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  // skips: value
  TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
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
    TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
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
  auto mutate_ty = [&]() { return mutator->MutateUnchangedOr(self->ty); };
  ffi::Expected<ffi::UnchangedOr<ffi::Any>> mapped_ty_result =
      mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
          ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_ty)
          : mutate_ty();
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    std::move(mapped_ty_result));
  if (mapped_ty.UnchangedOrSameAs(self->ty)) {
    return bind_unchanged();
  }
  ffi::ObjectPtr<VarNode> copy = ffi::make_object<VarNode>(*self);
  if (!mapped_ty.IsUnchanged()) {
    copy->ty = std::move(mapped_ty).ValueUnchecked();
  }
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
    TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
  };
  // A PrimType carries only a dtype, so it has nothing to substitute.  Broad callbacks do not see
  // this skipped field; dynamically typed Vars still descend through the Type value.
  if (self->ty.as<PrimTypeNode>()) {
    return bind_unchanged();
  }
  // Only NonRecursive is clamped: Recursive co-introduces type fields such as BufferType shape
  // variables, so that ambient region must continue through the dynamic type.
  auto mutate_ty = [&]() { return mutator->MaybeInplaceMutateIfUniqueUnchangedOr(self->ty); };
  ffi::Expected<ffi::UnchangedOr<ffi::Any>> mapped_ty_result =
      mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
          ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_ty)
          : mutate_ty();
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    std::move(mapped_ty_result));
  // `unchanged` is identity, not contents: `self->ty` is overwritten in place and the node is
  // still the object the caller handed in, so this exit is correct and is the same one the
  // ported hook takes.
  if (!mapped_ty.IsUnchanged()) {
    self->ty = std::move(mapped_ty).ValueUnchecked();
  }
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
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, a, mutator->MutateUnchangedOr(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, b, mutator->MutateUnchangedOr(self->b));
  // The short circuit: both operands answered before anything was built.
  if (a.UnchangedOrSameAs(self->a) && b.UnchangedOrSameAs(self->b)) {
    TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
  }
  ffi::ObjectPtr<TNode> copy = ffi::make_object<TNode>(*self);
  // StructuralMap preserves node types. A rewrite that changes operand dtypes must keep the
  // operands compatible and set the result type itself; a generic traversal cannot infer the
  // casts that would require.
  // The copy already carries `*self`'s operands, so an operand that answered unchanged keeps
  // the one that is already there.
  if (!a.IsUnchanged()) {
    copy->a = std::move(a).ValueUnchecked();
  }
  if (!b.IsUnchanged()) {
    copy->b = std::move(b).ValueUnchecked();
  }
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

template <typename TNode>
TVMFFIAny BinaryMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  TNode* self = const_cast<TNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, a,
                                    mutator->MaybeInplaceMutateIfUniqueUnchangedOr(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, b,
                                    mutator->MaybeInplaceMutateIfUniqueUnchangedOr(self->b));
  // An unchanged operand is known to be the one already in the slot, so the write is skipped
  // without the comparison the ported hook needed to discover the same thing.
  if (!a.IsUnchanged()) {
    self->a = std::move(a).ValueUnchecked();
  }
  if (!b.IsUnchanged()) {
    self->b = std::move(b).ValueUnchecked();
  }
  TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
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

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_OVERRIDE_MIXED_H_
