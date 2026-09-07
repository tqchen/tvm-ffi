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
#ifndef TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_UC_H_
#define TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_UC_H_

// UC's hooks for every fixture the structural harness runs -- Var, IntImm, Call, Add, Mul,
// FloorDiv, FloorMod, SeqStmt, Evaluate -- against UC's engine, structural_mutate.h. The engine
// base is the head of tvm-ffi `refactor/structural-mutate-unchanged-or` (PR 46) as resolved
// when the tree is built, with apache/tvm-ffi#760 applied on top when that branch does not
// already contain it; the build stamps the resolved hashes, and the tree records them at each
// build (7a569c5 on upstream main e74e58f, which already carries #760 and #761, for the tree
// this banner was written in).
//
// DERIVED FROM bench/377-four-state
// 220f363:benchmarks/cpp/structural/tvm_hook_override_unchanged_or.h (blob 5c51d9a98656, sha256
// 0e1d24cc28a8bd171896e7244e926011279fd9fdef61a52b3aec3170b75b44d4), the unguarded UC hook
// file, with exactly two edits, both at the seq element sites of the copy-on-write SeqStmt
// hooks (`MutateSeqStmtChanged`, `MutateSeqStmtRaw`): the element is moved into
// `ValueOrUnchanged` -- `std::move(mapped).ValueOrUnchanged(std::move(element))` -- rather than
// passed as an lvalue, which would bind the `T&` overload and move from the named handle
// implicitly. The H200 lane's tvm_hook_override_uc.h (bench/reproduce-structural-h200) carries
// the same two edits; the GB200 lane's does not. The include guard is this file's. Nothing else
// changed; the banner that follows is the source file's.
//
// Every mutate hook is the worked example on `TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN` in
// structural_mutate.h, field for field: raw `AnyUnsafe` self cast, one
// `TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN` per traversed field, `TVM_FFI_S_MUTATE_RETURN_UNCHANGED()`
// on the all-unchanged path, `make_object<T>(*self)`, then
// `copy->x = std::move(x).ValueOrUnchanged(std::move(copy->x))` per field. The in-place hooks
// take the same all-unchanged-or-same short circuit before touching the node, then write that
// same line against `self->x` and return unchanged, because the node the caller handed in is
// still the answer. The engine's in-place entries are the `AnyView` ones; nothing here relies
// on a typed `const T&` overload.
//
// The GOLD counterpart, tvm_hook_override_gold.h, keeps GOLD's hook bodies as timed
// (`move_if_changed(&slot)` per field) and takes the same in-place short circuit; `git diff
// --no-index` between the two is the protocol substitution plus that one idiom.
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
//   * `mutator->MutateExpected(...)` / `MaybeInplaceMutateIfUniqueExpected(...)` -- the
//     exception-free entry points, since `Mutate` now throws and returns the bare `UnchangedOr`,
//   * `TVM_FFI_S_MUTATE_RETURN_UNCHANGED()` where the GOLD file returned its own raw marker,
//   * `UnchangedOrSameAs` for the short-circuit test, and
//   * `slot = std::move(x).ValueOrUnchanged(std::move(slot))` at an assignment site, the `T&&`
//     overload, where the GOLD file wrote `std::move(x).move_if_changed(&slot)`.
//
// This is the style the protocol's own documentation prescribes; see the worked example on
// `TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN` in `structural_mutate.h`. A hook left on the GOLD API
// would measure the old protocol under a new name, which is the failure mode here most likely
// to go unnoticed -- so no name from the old API survives in this file.
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
#include <tvm/ffi/extra/structural_mutate.h>
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

using namespace tvm;        // NOLINT(build/namespaces)
using namespace tvm::tirx;  // NOLINT(build/namespaces)
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
      moved ? mutator->MaybeInplaceMutateIfUniqueExpected(input) : mutator->MutateExpected(input);
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
  auto mutate_ty = [&]() { return mutator->MutateExpected(self->ty); };
  ffi::Expected<ffi::UnchangedOr<ffi::Any>> mapped_ty_result =
      mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
          ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_ty)
          : mutate_ty();
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty, std::move(mapped_ty_result));
  if (mapped_ty.UnchangedOrSameAs(self->ty)) {
    return bind_unchanged();
  }
  ffi::ObjectPtr<VarNode> copy = ffi::make_object<VarNode>(*self);
  copy->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(copy->ty));
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
  auto mutate_ty = [&]() { return mutator->MaybeInplaceMutateIfUniqueExpected(self->ty); };
  ffi::Expected<ffi::UnchangedOr<ffi::Any>> mapped_ty_result =
      mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
          ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_ty)
          : mutate_ty();
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty, std::move(mapped_ty_result));
  if (mapped_ty.UnchangedOrSameAs(self->ty)) {
    return bind_unchanged();
  }
  // `unchanged` is identity, not contents: `self->ty` is overwritten in place and the node is
  // still the object the caller handed in, so this exit is correct and is the same one the
  // ported hook takes.
  self->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(self->ty));
  return bind_unchanged();
}

// ---------------------------------------------------------------------------
// src/ir/expr.cc -- Call
//
// The only ported hook with a container field (`args`), and the only one with skip guards:
// a `PrimType` result type, an interned `Op` operator, and an empty `ty_args` are each
// skipped rather than descended.  The `call-split-fuse` fixture exists to reach all three.
// ---------------------------------------------------------------------------

TVMFFIAny CallVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: attrs, constant metadata left untouched like the classic Expr functors.
  const CallNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(value);
  // A PrimType carries only a dtype, so it has nothing to visit.  Broad callbacks do not see this
  // skipped field; dynamically typed Call results still descend through the Type value.
  if (!self->ty.as<PrimTypeNode>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  }
  // An Op is an interned registry singleton, so it has nothing to visit.  Broad callbacks do not
  // see this skipped field; function-valued Call operators still descend through the Expr value.
  if (!self->op.as<OpNode>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->op));
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->args));
  // An empty ty_args has no element to traverse.  Broad callbacks do not see the empty container;
  // nonempty type arguments retain normal container descent and callback behavior.
  if (!self->ty_args.empty()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty_args));
  }
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny CallMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs, constant metadata left untouched like the classic Expr functors.
  const CallNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(value);
  // A PrimType carries only a dtype, so it has nothing to substitute.  Broad callbacks do not see
  // this skipped field; dynamically typed Call results still descend through the Type value.
  // A skipped field is unchanged by construction, so it is initialized to `Unchanged()` and
  // the ported hook's deliberate copy of `self->ty` is not needed at all.
  ffi::UnchangedOr<Type> mapped_ty = ffi::Unchanged();
  if (!self->ty.as<PrimTypeNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, descended_ty,
                                      mutator->MutateExpected(self->ty));
    mapped_ty = std::move(descended_ty);
  }
  // An Op is an interned registry singleton, so it has nothing to substitute.  Broad callbacks do
  // not see this skipped field; function-valued Call operators still descend through the Expr.
  ffi::UnchangedOr<Expr> mapped_op = ffi::Unchanged();
  if (!self->op.as<OpNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, descended_op,
                                      mutator->MutateExpected(self->op));
    mapped_op = std::move(descended_op);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Expr>>, mapped_args,
                                    mutator->MutateExpected(self->args));
  // An empty ty_args has no element to substitute.  Broad callbacks do not see the empty
  // container; nonempty type arguments retain normal container descent and callback behavior.
  ffi::UnchangedOr<ffi::Array<Type>> mapped_ty_args = ffi::Unchanged();
  if (!self->ty_args.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Type>>, descended_ty_args,
                                      mutator->MutateExpected(self->ty_args));
    mapped_ty_args = std::move(descended_ty_args);
  }
  // The short circuit, and the reason this file exists: nothing has been built yet.
  if (mapped_ty.UnchangedOrSameAs(self->ty) && mapped_op.UnchangedOrSameAs(self->op) &&
      mapped_args.UnchangedOrSameAs(self->args) &&
      mapped_ty_args.UnchangedOrSameAs(self->ty_args)) {
    TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
  }
  ffi::ObjectPtr<CallNode> copy = ffi::make_object<CallNode>(*self);
  // The copy already carries `*self`'s fields, so a field that answered unchanged is written
  // by not being written.
  copy->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(copy->ty));
  copy->op = std::move(mapped_op).ValueOrUnchanged(std::move(copy->op));
  copy->args = std::move(mapped_args).ValueOrUnchanged(std::move(copy->args));
  copy->ty_args = std::move(mapped_ty_args).ValueOrUnchanged(std::move(copy->ty_args));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny CallMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs, constant metadata left untouched like the classic Expr functors.
  CallNode* self = const_cast<CallNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(value));
  // A PrimType carries only a dtype, so it has nothing to substitute.  Broad callbacks do not see
  // this skipped field; dynamically typed Call results still descend through the Type value.
  ffi::UnchangedOr<Type> mapped_ty = ffi::Unchanged();
  if (!self->ty.as<PrimTypeNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, descended_ty,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->ty));
    mapped_ty = std::move(descended_ty);
  }
  // An Op is an interned registry singleton, so it has nothing to substitute.  Broad callbacks do
  // not see this skipped field; function-valued Call operators still descend through the Expr.
  ffi::UnchangedOr<Expr> mapped_op = ffi::Unchanged();
  if (!self->op.as<OpNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, descended_op,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->op));
    mapped_op = std::move(descended_op);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Expr>>, mapped_args,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->args));
  // An empty ty_args has no element to substitute.  Broad callbacks do not see the empty
  // container; nonempty type arguments retain normal container descent and callback behavior.
  ffi::UnchangedOr<ffi::Array<Type>> mapped_ty_args = ffi::Unchanged();
  if (!self->ty_args.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Type>>, descended_ty_args,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->ty_args));
    mapped_ty_args = std::move(descended_ty_args);
  }
  if (mapped_ty.UnchangedOrSameAs(self->ty) && mapped_op.UnchangedOrSameAs(self->op) &&
      mapped_args.UnchangedOrSameAs(self->args) &&
      mapped_ty_args.UnchangedOrSameAs(self->ty_args)) {
    TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
  }
  // A field that answered unchanged is known to be the one already in the slot, so the write is
  // skipped without a comparison; every exit is the input object, so every exit is unchanged.
  self->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(self->ty));
  self->op = std::move(mapped_op).ValueOrUnchanged(std::move(self->op));
  self->args = std::move(mapped_args).ValueOrUnchanged(std::move(self->args));
  self->ty_args = std::move(mapped_ty_args).ValueOrUnchanged(std::move(self->ty_args));
  TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
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
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, a,
                                    mutator->MutateExpected(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, b,
                                    mutator->MutateExpected(self->b));
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
  copy->a = std::move(a).ValueOrUnchanged(std::move(copy->a));
  copy->b = std::move(b).ValueOrUnchanged(std::move(copy->b));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

template <typename TNode>
TVMFFIAny BinaryMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  TNode* self = const_cast<TNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, a,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, b,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->b));
  if (a.UnchangedOrSameAs(self->a) && b.UnchangedOrSameAs(self->b)) {
    TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
  }
  // An unchanged operand is known to be the one already in the slot, so the write is skipped
  // without the comparison the ported hook needed to discover the same thing.
  self->a = std::move(a).ValueOrUnchanged(std::move(self->a));
  self->b = std::move(b).ValueOrUnchanged(std::move(self->b));
  TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
}

// ---------------------------------------------------------------------------
// src/tirx/ir/stmt.cc -- SeqStmt
// ---------------------------------------------------------------------------

TVMFFIAny SeqStmtVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const SeqStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->seq));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

// API ADAPTATION, and the file's only signature change: `ffi::AnyView` rather than
// `const Stmt&`. An element that answered unchanged is the one already in the array slot, and
// the whole point is not to form an owning handle to it just to ask a question about it. Both
// a `Stmt` and a borrowed `ffi::Any` convert without a refcount, so every call site is
// unaffected.
bool IsSeqStmtNoOp(ffi::AnyView stmt) {
  const auto* evaluate = stmt.as<EvaluateNode>();
  const auto* value = evaluate == nullptr ? nullptr : evaluate->value.as<IntImmNode>();
  return value != nullptr && value->value == 0;
}

void AppendSeqStmtResult(ffi::Array<Stmt>* output, Stmt mapped) {
  if (IsSeqStmtNoOp(mapped)) {
    return;
  }
  if (const auto* nested = mapped.as<SeqStmtNode>()) {
    for (const Stmt& stmt : nested->seq) {
      output->push_back(stmt);
    }
  } else {
    output->push_back(std::move(mapped));
  }
}

TVMFFIAny MutateSeqStmtChanged(ffi::StructuralMutatorObj* mutator, const SeqStmtNode* self,
                               int64_t index, Stmt mapped) noexcept {
  const int64_t size = static_cast<int64_t>(self->seq.size());
  ffi::ObjectPtr<ffi::ArrayObj> output_obj = ffi::ArrayObj::CreateRepeated(size, ffi::Any());
  output_obj->InitRange(0, self->seq.begin(), self->seq.begin() + index);
  output_obj->resize(index);
  ffi::Array<Stmt> output(std::move(output_obj));
  AppendSeqStmtResult(&output, std::move(mapped));
  for (int64_t i = index + 1; i < size; ++i) {
    Stmt element = self->seq[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped,
                                      mutator->MutateExpected(element));
    AppendSeqStmtResult(&output, std::move(mapped).ValueOrUnchanged(std::move(element)));
  }
  if (output.empty()) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(Evaluate(0)));
  }
  if (output.size() == 1) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(output[0]));
  }
  ffi::ObjectPtr<SeqStmtNode> copy = ffi::make_object<SeqStmtNode>(*self);
  copy->seq = std::move(output);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny MutateSeqStmtRaw(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const SeqStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value);
  const int64_t size = static_cast<int64_t>(self->seq.size());
  for (int64_t i = 0; i < size; ++i) {
    Stmt element = self->seq[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped,
                                      mutator->MutateExpected(element));
    // Before the first change the unchanged branch is the entire loop body: an element that
    // answered unchanged is not compared and no handle to a result is formed.
    if (mapped.UnchangedOrSameAs(element)) {
      continue;
    }
    return MutateSeqStmtChanged(mutator, self, i,
                                std::move(mapped).ValueOrUnchanged(std::move(element)));
  }
  TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
}

TVMFFIAny InplaceSplice(ffi::StructuralMutatorObj* mutator, SeqStmtNode* self, ffi::ArrayObj* seq,
                        int64_t total, int64_t index, const SeqStmtNode* nested) noexcept {
  const int64_t size = static_cast<int64_t>(seq->size());
  ffi::Array<Stmt> output;
  output.reserve(size + static_cast<int64_t>(nested->seq.size()) - 1);
  for (int64_t i = 0; i < total; ++i) {
    output.push_back(seq->begin()[i].cast<Stmt>());
  }
  for (const Stmt& stmt : nested->seq) {
    output.push_back(stmt);
  }
  for (int64_t i = index + 1; i < size; ++i) {
    // Borrow the storage slot rather than naming a Stmt: an owning handle here would make the
    // element non-unique and suppress the in-place path being dispatched into.
    const ffi::Any& item = seq->begin()[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_result,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));
    Stmt mapped =
        mapped_result.IsUnchanged() ? item.cast<Stmt>() : std::move(mapped_result).ValueUnchecked();
    AppendSeqStmtResult(&output, std::move(mapped));
  }
  if (output.empty()) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(Evaluate(0)));
  }
  if (output.size() == 1) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(output[0]));
  }
  self->seq = std::move(output);
  TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
}

TVMFFIAny MaybeInplaceMutateSeqStmtChanged(ffi::StructuralMutatorObj* mutator, SeqStmtNode* self,
                                           ffi::ArrayObj* seq, int64_t index,
                                           Stmt mapped) noexcept {
  const int64_t size = static_cast<int64_t>(seq->size());
  int64_t total = index;
  if (IsSeqStmtNoOp(mapped)) {
    // Drop Evaluate(0), matching SeqStmt::Flatten.
  } else if (const auto* nested = mapped.as<SeqStmtNode>()) {
    const int64_t nested_size = static_cast<int64_t>(nested->seq.size());
    if (total + nested_size > index + 1) {
      return InplaceSplice(mutator, self, seq, total, index, nested);
    }
    for (const Stmt& stmt : nested->seq) {
      seq->SetItemAfterCheck(total++, ffi::Any(stmt));
    }
  } else {
    seq->SetItemAfterCheck(total++, ffi::Any(std::move(mapped)));
  }
  for (int64_t i = index + 1; i < size; ++i) {
    const ffi::Any& item = seq->begin()[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_result,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));
    const bool changed = !mapped_result.IsUnchanged();
    Stmt mapped;
    if (changed) {
      mapped = std::move(mapped_result).ValueUnchecked();
    }
    // The normalization still applies to an unchanged element -- a pre-existing `Evaluate(0)`
    // is dropped from here on, and a pre-existing nested `SeqStmt` is still flattened, both
    // regardless of whether this element is what changed. What the unchanged answer buys is
    // that those questions are asked of the element still in its slot, through a borrowed
    // view, instead of an owning handle built only to be asked and discarded; and that an
    // element the cursors have not moved past is left alone without a write or a comparison.
    const ffi::AnyView result = changed ? ffi::AnyView(mapped) : ffi::AnyView(item);
    if (IsSeqStmtNoOp(result)) {
      continue;
    }
    if (const auto* nested = result.as<SeqStmtNode>()) {
      const int64_t nested_size = static_cast<int64_t>(nested->seq.size());
      if (total + nested_size > i + 1) {
        return InplaceSplice(mutator, self, seq, total, i, nested);
      }
      for (const Stmt& stmt : nested->seq) {
        seq->SetItemAfterCheck(total++, ffi::Any(stmt));
      }
      continue;
    }
    if (total != i || (changed && !item.same_as(mapped))) {
      seq->SetItemAfterCheck(total, changed ? ffi::Any(std::move(mapped)) : ffi::Any(item));
    }
    ++total;
  }
  seq->resize(total);
  if (total == 0) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(Evaluate(0)));
  }
  if (total == 1) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(seq->begin()[0].cast<Stmt>()));
  }
  // `unchanged` is identity: the sequence was rewritten in place and the node the caller handed
  // in is still the answer.
  TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
}

TVMFFIAny MaybeInplaceMutateSeqStmtRaw(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  SeqStmtNode* self = const_cast<SeqStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value));
  // The engine establishes ownership of the SeqStmt, but seq is a field and needs its own check.
  if (!self->seq.unique()) {
    return MutateSeqStmtRaw(mutator, value);
  }
  ffi::ArrayObj* seq = self->seq.GetArrayObj();
  const int64_t size = static_cast<int64_t>(seq->size());
  for (int64_t i = 0; i < size; ++i) {
    // Borrow the storage slot so the element stays unique during in-place dispatch.
    const ffi::Any& item = seq->begin()[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_result,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));
    if (mapped_result.IsUnchanged()) {
      continue;
    }
    Stmt mapped = std::move(mapped_result).ValueUnchecked();
    if (!item.same_as(mapped)) {
      return MaybeInplaceMutateSeqStmtChanged(mutator, self, seq, i, std::move(mapped));
    }
  }
  TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
}

TVMFFIAny SeqStmtMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  return MutateSeqStmtRaw(mutator, value);
}

TVMFFIAny SeqStmtMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                    ffi::AnyView value) noexcept {
  return MaybeInplaceMutateSeqStmtRaw(mutator, value);
}

// ---------------------------------------------------------------------------
// src/tirx/ir/stmt.cc -- Evaluate
// ---------------------------------------------------------------------------

TVMFFIAny EvaluateVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const EvaluateNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny EvaluateMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const EvaluateNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MutateExpected(self->value));
  if (mapped_value.UnchangedOrSameAs(self->value)) {
    TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
  }
  ffi::ObjectPtr<EvaluateNode> copy = ffi::make_object<EvaluateNode>(*self);
  copy->value = std::move(mapped_value).ValueOrUnchanged(std::move(copy->value));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny EvaluateMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                     ffi::AnyView value) noexcept {
  EvaluateNode* self = const_cast<EvaluateNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  if (mapped_value.UnchangedOrSameAs(self->value)) {
    TVM_FFI_S_MUTATE_RETURN_UNCHANGED();
  }
  // `unchanged` is identity, not contents: the value field may be overwritten and the node is
  // still the one the caller handed in.
  self->value = std::move(mapped_value).ValueOrUnchanged(std::move(self->value));
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
  Install(CallNode::RuntimeTypeIndex(), &CallVisit, &CallMutate, &CallMaybeInplaceMutate);
  InstallBinary<prim::AddNode>();
  InstallBinary<prim::MulNode>();
  InstallBinary<prim::FloorDivNode>();
  InstallBinary<prim::FloorModNode>();
  Install(SeqStmtNode::RuntimeTypeIndex(), &SeqStmtVisit, &SeqStmtMutate,
          &SeqStmtMaybeInplaceMutate);
  Install(EvaluateNode::RuntimeTypeIndex(), &EvaluateVisit, &EvaluateMutate,
          &EvaluateMaybeInplaceMutate);
}

/*! \brief The type indices InstallAll covers, for the coverage assertion. */
inline std::vector<int32_t> CoveredTypes() {
  return {VarNode::RuntimeTypeIndex(),
          IntImmNode::RuntimeTypeIndex(),
          CallNode::RuntimeTypeIndex(),
          prim::AddNode::RuntimeTypeIndex(),
          prim::MulNode::RuntimeTypeIndex(),
          prim::FloorDivNode::RuntimeTypeIndex(),
          prim::FloorModNode::RuntimeTypeIndex(),
          SeqStmtNode::RuntimeTypeIndex(),
          EvaluateNode::RuntimeTypeIndex()};
}

}  // namespace tvm_hooks

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_UC_H_
