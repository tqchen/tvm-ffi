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
#ifndef TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_GOLD_H_
#define TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_GOLD_H_

// GOLD's hooks for every fixture the structural harness runs -- Var, IntImm, Call, Add, Mul,
// FloorDiv, FloorMod, SeqStmt, Evaluate -- against GOLD's engine (structural_mutate_gold.h,
// bench/377-gold-759 730d6fc byte for byte).
//
// DERIVED FROM bench/377-four-state
// 220f363:benchmarks/cpp/structural/tvm_hook_override_unchanged.h (blob fb4cf266781c, sha256
// cfe4da4793bcada7b7a01889a1ed1f4808cd967dad48141622425b26139a721e), the file the GOLD state
// was timed with. Its hook bodies are kept as timed -- `std::move(x).move_if_changed(&slot)` at
// every per-field site, since GOLD is the fixed reference -- with one addition, made to both
// files alike: every in-place hook short-circuits with `return Unchanged()` when all of its
// fields answered unchanged or same, mirroring its copy-path test, before any assignment. The
// include guard is this file's and the engine include names structural_mutate_gold.h. Nothing
// else changed; the banner that follows is the source file's.
//
// Read against tvm_hook_override_uc.h, the names differ -- `MaybeUnchanged<T>` for
// `UnchangedOr<T>`, `Mutate` / `MaybeInplaceMutateIfUnique` for the `*Expected` entries,
// `unchanged_or_same_as` for `UnchangedOrSameAs`, `return Unchanged()` for
// `TVM_FFI_S_MUTATE_RETURN_UNCHANGED()` -- and so does the per-field idiom: GOLD's
// `move_if_changed(&slot)` skips the write on the unchanged path, UC's documented
// `slot = std::move(x).ValueOrUnchanged(std::move(slot))` moves the slot through itself.
//
// `git diff --no-index tvm_hook_override_gold.h tvm_hook_override_uc.h` is the review of UC's
// protocol against GOLD's with no cross-branch comparison and no name normalisation.
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

using namespace tvm;        // NOLINT(build/namespaces)
using namespace tvm::tirx;  // NOLINT(build/namespaces)
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
  if (mapped_ty.unchanged_or_same_as(self->ty)) {
    return bind_unchanged();
  }
  // `unchanged` is identity, not contents: `self->ty` is overwritten in place and the node is
  // still the object the caller handed in, so this exit is correct and is the same one the
  // ported hook takes.
  std::move(mapped_ty).move_if_changed(&self->ty);
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
  // A skipped field is unchanged by construction, so a default-constructed result says so and
  // the ported hook's deliberate copy of `self->ty` is not needed at all.
  ffi::MaybeUnchanged<Type> mapped_ty;
  if (!self->ty.as<PrimTypeNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<Type>, descended_ty,
                                      mutator->Mutate(self->ty));
    mapped_ty = std::move(descended_ty);
  }
  // An Op is an interned registry singleton, so it has nothing to substitute.  Broad callbacks do
  // not see this skipped field; function-valued Call operators still descend through the Expr.
  ffi::MaybeUnchanged<Expr> mapped_op;
  if (!self->op.as<OpNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<Expr>, descended_op,
                                      mutator->Mutate(self->op));
    mapped_op = std::move(descended_op);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<ffi::Array<Expr>>, mapped_args,
                                    mutator->Mutate(self->args));
  // An empty ty_args has no element to substitute.  Broad callbacks do not see the empty
  // container; nonempty type arguments retain normal container descent and callback behavior.
  ffi::MaybeUnchanged<ffi::Array<Type>> mapped_ty_args;
  if (!self->ty_args.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<ffi::Array<Type>>, descended_ty_args,
                                      mutator->Mutate(self->ty_args));
    mapped_ty_args = std::move(descended_ty_args);
  }
  // The short circuit, and the reason this file exists: nothing has been built yet.
  if (mapped_ty.unchanged_or_same_as(self->ty) && mapped_op.unchanged_or_same_as(self->op) &&
      mapped_args.unchanged_or_same_as(self->args) &&
      mapped_ty_args.unchanged_or_same_as(self->ty_args)) {
    return Unchanged();
  }
  ffi::ObjectPtr<CallNode> copy = ffi::make_object<CallNode>(*self);
  // The copy already carries `*self`'s fields, so a field that answered unchanged is written
  // by not being written.
  std::move(mapped_ty).move_if_changed(&copy->ty);
  std::move(mapped_op).move_if_changed(&copy->op);
  std::move(mapped_args).move_if_changed(&copy->args);
  std::move(mapped_ty_args).move_if_changed(&copy->ty_args);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny CallMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs, constant metadata left untouched like the classic Expr functors.
  CallNode* self = const_cast<CallNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(value));
  // A PrimType carries only a dtype, so it has nothing to substitute.  Broad callbacks do not see
  // this skipped field; dynamically typed Call results still descend through the Type value.
  ffi::MaybeUnchanged<Type> mapped_ty;
  if (!self->ty.as<PrimTypeNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<Type>, descended_ty,
                                      mutator->MaybeInplaceMutateIfUnique(self->ty));
    mapped_ty = std::move(descended_ty);
  }
  // An Op is an interned registry singleton, so it has nothing to substitute.  Broad callbacks do
  // not see this skipped field; function-valued Call operators still descend through the Expr.
  ffi::MaybeUnchanged<Expr> mapped_op;
  if (!self->op.as<OpNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<Expr>, descended_op,
                                      mutator->MaybeInplaceMutateIfUnique(self->op));
    mapped_op = std::move(descended_op);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<ffi::Array<Expr>>, mapped_args,
                                    mutator->MaybeInplaceMutateIfUnique(self->args));
  // An empty ty_args has no element to substitute.  Broad callbacks do not see the empty
  // container; nonempty type arguments retain normal container descent and callback behavior.
  ffi::MaybeUnchanged<ffi::Array<Type>> mapped_ty_args;
  if (!self->ty_args.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<ffi::Array<Type>>, descended_ty_args,
                                      mutator->MaybeInplaceMutateIfUnique(self->ty_args));
    mapped_ty_args = std::move(descended_ty_args);
  }
  if (mapped_ty.unchanged_or_same_as(self->ty) && mapped_op.unchanged_or_same_as(self->op) &&
      mapped_args.unchanged_or_same_as(self->args) &&
      mapped_ty_args.unchanged_or_same_as(self->ty_args)) {
    return Unchanged();
  }
  // A field that answered unchanged is known to be the one already in the slot, so the write is
  // skipped without a comparison; every exit is the input object, so every exit is unchanged.
  std::move(mapped_ty).move_if_changed(&self->ty);
  std::move(mapped_op).move_if_changed(&self->op);
  std::move(mapped_args).move_if_changed(&self->args);
  std::move(mapped_ty_args).move_if_changed(&self->ty_args);
  return Unchanged();
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
  if (a.unchanged_or_same_as(self->a) && b.unchanged_or_same_as(self->b)) {
    return Unchanged();
  }
  // An unchanged operand is known to be the one already in the slot, so the write is skipped
  // without the comparison the ported hook needed to discover the same thing.
  std::move(a).move_if_changed(&self->a);
  std::move(b).move_if_changed(&self->b);
  return Unchanged();
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
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<Stmt>, mapped, mutator->Mutate(element));
    AppendSeqStmtResult(&output, std::move(mapped).value_or_unchanged(element));
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
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<Stmt>, mapped, mutator->Mutate(element));
    // Before the first change the unchanged branch is the entire loop body: an element that
    // answered unchanged is not compared and no handle to a result is formed.
    if (mapped.unchanged_or_same_as(element)) {
      continue;
    }
    return MutateSeqStmtChanged(mutator, self, i, std::move(mapped).value_or_unchanged(element));
  }
  return Unchanged();
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
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<Stmt>, mapped_result,
                                      mutator->MaybeInplaceMutateIfUnique(item));
    Stmt mapped;
    if (!std::move(mapped_result).move_if_changed(&mapped)) {
      mapped = item.cast<Stmt>();
    }
    AppendSeqStmtResult(&output, std::move(mapped));
  }
  if (output.empty()) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(Evaluate(0)));
  }
  if (output.size() == 1) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(output[0]));
  }
  self->seq = std::move(output);
  return Unchanged();
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
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<Stmt>, mapped_result,
                                      mutator->MaybeInplaceMutateIfUnique(item));
    Stmt mapped;
    const bool changed = std::move(mapped_result).move_if_changed(&mapped);
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
  return Unchanged();
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
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<Stmt>, mapped_result,
                                      mutator->MaybeInplaceMutateIfUnique(item));
    if (mapped_result.unchanged()) {
      continue;
    }
    Stmt mapped;
    std::move(mapped_result).move_if_changed(&mapped);
    if (!item.same_as(mapped)) {
      return MaybeInplaceMutateSeqStmtChanged(mutator, self, seq, i, std::move(mapped));
    }
  }
  return Unchanged();
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
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<Expr>, mapped_value,
                                    mutator->Mutate(self->value));
  if (mapped_value.unchanged_or_same_as(self->value)) {
    return Unchanged();
  }
  ffi::ObjectPtr<EvaluateNode> copy = ffi::make_object<EvaluateNode>(*self);
  std::move(mapped_value).move_if_changed(&copy->value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny EvaluateMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                     ffi::AnyView value) noexcept {
  EvaluateNode* self = const_cast<EvaluateNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::MaybeUnchanged<Expr>, mapped_value,
                                    mutator->MaybeInplaceMutateIfUnique(self->value));
  if (mapped_value.unchanged_or_same_as(self->value)) {
    return Unchanged();
  }
  // `unchanged` is identity, not contents: the value field may be overwritten and the node is
  // still the one the caller handed in.
  std::move(mapped_value).move_if_changed(&self->value);
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

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_GOLD_H_
