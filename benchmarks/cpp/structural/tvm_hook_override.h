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
#ifndef TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_H_
#define TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_H_

// Every structural hook the benchmark dispatches into on real TVM node types, plus the
// installation that puts them over the ones TVM registered from its own static-init blocks.
//
// ===========================================================================================
// PORTED FROM apache/tvm#20275, head a1031a2177 ("[REFACTOR][IR] Add structural hooks for
// Expr and Stmt"), branch redo-expr-stmt-structural-hooks-current, whose `3rdparty/tvm-ffi`
// pin is 9d784c4d.
//
//   hook group   source file             functions
//   ----------   ---------------------   ----------------------------------------------------
//   IntImm       src/ir/expr.cc          IntImmVisit / IntImmMutate / IntImmMaybeInplaceMutate
//   Var          src/ir/expr.cc          VarVisit / VarMutate / VarMaybeInplaceMutate
//   Call         src/ir/expr.cc          CallVisit / CallMutate / CallMaybeInplaceMutate
//   binary ops   src/ir/prim/expr.cc     BinaryVisit / BinaryMutate /
//                                            BinaryMaybeInplaceMutate
//   SeqStmt      src/tirx/ir/stmt.cc     SeqStmtVisit / IsSeqStmtNoOp / AppendSeqStmtResult /
//                                            MutateSeqStmtChanged / MutateSeqStmtRaw /
//                                            InplaceSplice / MaybeInplaceMutateSeqStmtChanged /
//                                            MaybeInplaceMutateSeqStmtRaw
//   Evaluate     src/tirx/ir/stmt.cc     EvaluateVisit / EvaluateMutate /
//                                            EvaluateMaybeInplaceMutate
//
// 20275 is under review and moving, so this port follows it rather than apache/tvm main.
// `./port_check.sh` re-extracts these functions from the recorded sha and diffs them against
// the bodies below; run it before trusting a measurement, and after any rebase of the PR.
//
// The bodies are copied verbatim: same names, same signatures, same order, same internal
// structure, grouped by the TVM file each came from. Nothing is reordered, renamed or tidied,
// because a change prototyped here has to lift back into apache/tvm as a patch. There is one
// intended difference, marked `HARNESS DEVIATION`, and it is in the installation rather than
// in any hook body.
//
// What a1031a2177 changed relative to the previous port at e40167046ed6 -- all of it re-ported
// here, and all of it re-measured. Only `src/tirx/ir/stmt.cc` moved, and only the SeqStmt
// mutate pair within it:
//   * MutateSeqStmtRaw is now a lead loop that returns `self` on the first pass where nothing
//     changed, handing off to a new MutateSeqStmtChanged at the first element that did.
//   * MaybeInplaceMutateSeqStmtRaw is the same shape, handing off to
//     MaybeInplaceMutateSeqStmtChanged, which writes through a `total` cursor and calls a new
//     InplaceSplice when a nested result would overrun the element it replaces.
//   * Two new helpers normalize the result: IsSeqStmtNoOp drops `Evaluate(0)` elements, and a
//     sequence that ends at zero elements returns `Evaluate(0)` while one that ends at a
//     single element returns that element unwrapped.  So these hooks can now return an
//     `Evaluate` or a bare `Stmt` where the input was a `SeqStmt`.
//   * The previous head's `erase`/`insert` cursor and its one-shot overflow rebuild are gone.
//
// A DELIBERATE ASYMMETRY IN THAT NORMALIZATION, which reads as a bug and is not one: a
// pre-existing `Evaluate(0)` in the input SURVIVES a pass in which nothing else changed --
// the lead loop exits on `same_as` and returns `self` untouched -- and is DROPPED as soon as
// any other element changes and the rebuild path runs. Same input, two output shapes,
// depending on an unrelated change elsewhere in the sequence. Dropping no-ops on an unchanged
// pass would rewrite every sequence that contains one and destroy the `same_as` fast path, so
// the asymmetry is the price of that path. `CheckSpliceAgainstReference` pins it.
//
// And what 20275 as a whole still changes relative to apache/tvm main, since it invalidates
// measurements taken against main:
//   * BinaryMutate no longer re-infers the result type at all -- the `BinaryResultType` guard
//     is gone, on the ground that StructuralMap preserves node types.
//   * BinaryMaybeInplaceMutate lost its `same_as` early return.
//   * SeqStmt's hooks were replaced outright by the splice-capable, normalizing pair above.
//   * SeqStmtVisit visits `self->seq` as a value rather than iterating its elements, so the
//     Array itself is now a visited node and the fixtures' node and occurrence counts include
//     it.
//   * Hooks return `ffi::Any(self)` from the raw pointer rather than `ffi::Any(value)`.
// ===========================================================================================
//
// These hooks are always installed; there is no dual mode. `walk_old` and `map_old` run on
// them too, because `PostOrderVisit` and `Substitute` are built on the structural engine, so
// the old-versus-new comparison differs only in the traversal API and not in the hooks
// underneath it. The consequence, which the report states: no number here describes TVM's own
// registered hook implementations.

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

using namespace tvm;              // NOLINT(build/namespaces)
using namespace tvm::tirx;        // NOLINT(build/namespaces)
namespace ffi = tvm::ffi;

/*!
 * \brief The `map_floor` arm's root call into the minimal mutator, in this state's API.
 *
 * `MinimalMutatorObj` dispatches the hook directly and never reaches the engine, so its result
 * is whatever this state's ABI carries. Here that is the mutated value itself, already
 * resolved, so the root has nothing to resolve.
 */
inline ffi::Expected<ffi::Any> MinimalMutateRoot(ffi::bench::MinimalMutatorObj* mutator,
                                                 ffi::AnyView input, bool moved) noexcept {
  return moved ? mutator->MaybeInplaceMutateIfUniqueExpected(input)
               : mutator->MutateExpected(input);
}

// ---------------------------------------------------------------------------
// src/ir/expr.cc -- IntImm
// ---------------------------------------------------------------------------

TVMFFIAny IntImmVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // skips: value
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny IntImmMutate(ffi::StructuralMutatorObj*, ffi::AnyView value) noexcept {
  // skips: value
  const IntImmNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IntImmNode>(value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny IntImmMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView value) noexcept {
  // skips: value
  const IntImmNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IntImmNode>(value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
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
  auto return_mapped_var = [&](ffi::Any mapped_var) -> TVMFFIAny {
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
    return return_mapped_var(ffi::Any(self));
  }
  // Only NonRecursive is clamped: Recursive co-introduces type fields such as BufferType shape
  // variables, so that ambient region must continue through the dynamic type.
  auto mutate_ty = [&]() { return mutator->MutateExpected(self->ty); };
  ffi::Expected<ffi::Any> mapped_ty_result =
      mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
          ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_ty)
          : mutate_ty();
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Type, mapped_ty, std::move(mapped_ty_result));
  ffi::Any mapped_var = ffi::Any(self);
  if (!mapped_ty.same_as(self->ty)) {
    ffi::ObjectPtr<VarNode> copy = ffi::make_object<VarNode>(*self);
    copy->ty = std::move(mapped_ty);
    mapped_var = ffi::Any(std::move(copy));
  }
  return return_mapped_var(std::move(mapped_var));
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
  auto return_mapped_var = [&](ffi::Any mapped_var) -> TVMFFIAny {
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
    return return_mapped_var(ffi::Any(self));
  }
  // Only NonRecursive is clamped: Recursive co-introduces type fields such as BufferType shape
  // variables, so that ambient region must continue through the dynamic type.
  auto mutate_ty = [&]() { return mutator->MaybeInplaceMutateIfUniqueExpected(self->ty); };
  ffi::Expected<ffi::Any> mapped_ty_result =
      mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
          ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_ty)
          : mutate_ty();
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Type, mapped_ty, std::move(mapped_ty_result));
  if (!mapped_ty.same_as(self->ty)) self->ty = std::move(mapped_ty);
  return return_mapped_var(ffi::Any(self));
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
  // Deliberate copy: avoids Any boxing on the dominant primitive skip path.
  Type mapped_ty = self->ty;
  if (!self->ty.as<PrimTypeNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Type, descended_ty, mutator->MutateExpected(self->ty));
    mapped_ty = std::move(descended_ty);
  }
  // An Op is an interned registry singleton, so it has nothing to substitute.  Broad callbacks do
  // not see this skipped field; function-valued Call operators still descend through the Expr.
  Expr mapped_op = self->op;
  if (!self->op.as<OpNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, descended_op, mutator->MutateExpected(self->op));
    mapped_op = std::move(descended_op);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<Expr>, mapped_args,
                                    mutator->MutateExpected(self->args));
  // An empty ty_args has no element to substitute.  Broad callbacks do not see the empty
  // container; nonempty type arguments retain normal container descent and callback behavior.
  ffi::Array<Type> mapped_ty_args = self->ty_args;
  if (!self->ty_args.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<Type>, descended_ty_args,
                                      mutator->MutateExpected(self->ty_args));
    mapped_ty_args = std::move(descended_ty_args);
  }
  if (mapped_ty.same_as(self->ty) && mapped_op.same_as(self->op) &&
      mapped_args.same_as(self->args) && mapped_ty_args.same_as(self->ty_args)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<CallNode> copy = ffi::make_object<CallNode>(*self);
  copy->ty = std::move(mapped_ty);
  copy->op = std::move(mapped_op);
  copy->args = std::move(mapped_args);
  copy->ty_args = std::move(mapped_ty_args);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny CallMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs, constant metadata left untouched like the classic Expr functors.
  CallNode* self = const_cast<CallNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(value));
  // A PrimType carries only a dtype, so it has nothing to substitute.  Broad callbacks do not see
  // this skipped field; dynamically typed Call results still descend through the Type value.
  // Deliberate copy: avoids Any boxing on the dominant primitive skip path.
  Type mapped_ty = self->ty;
  if (!self->ty.as<PrimTypeNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Type, descended_ty,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->ty));
    mapped_ty = std::move(descended_ty);
  }
  // An Op is an interned registry singleton, so it has nothing to substitute.  Broad callbacks do
  // not see this skipped field; function-valued Call operators still descend through the Expr.
  Expr mapped_op = self->op;
  if (!self->op.as<OpNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, descended_op,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->op));
    mapped_op = std::move(descended_op);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<Expr>, mapped_args,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->args));
  // An empty ty_args has no element to substitute.  Broad callbacks do not see the empty
  // container; nonempty type arguments retain normal container descent and callback behavior.
  ffi::Array<Type> mapped_ty_args = self->ty_args;
  if (!self->ty_args.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<Type>, descended_ty_args,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->ty_args));
    mapped_ty_args = std::move(descended_ty_args);
  }
  if (mapped_ty.same_as(self->ty) && mapped_op.same_as(self->op) &&
      mapped_args.same_as(self->args) && mapped_ty_args.same_as(self->ty_args)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->ty = std::move(mapped_ty);
  self->op = std::move(mapped_op);
  self->args = std::move(mapped_args);
  self->ty_args = std::move(mapped_ty_args);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
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
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, a, mutator->MutateExpected(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, b, mutator->MutateExpected(self->b));
  if (a.same_as(self->a) && b.same_as(self->b)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<TNode> copy = ffi::make_object<TNode>(*self);
  // StructuralMap preserves node types. A rewrite that changes operand dtypes must keep the
  // operands compatible and set the result type itself; a generic traversal cannot infer the
  // casts that would require.
  copy->a = std::move(a);
  copy->b = std::move(b);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

template <typename TNode>
TVMFFIAny BinaryMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  TNode* self = const_cast<TNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, a,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, b,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->b));
  if (!a.same_as(self->a)) self->a = std::move(a);
  if (!b.same_as(self->b)) self->b = std::move(b);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
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

bool IsSeqStmtNoOp(const Stmt& stmt) {
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
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped, mutator->MutateExpected(self->seq[i]));
    AppendSeqStmtResult(&output, std::move(mapped));
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
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped, mutator->MutateExpected(self->seq[i]));
    if (!self->seq[i].same_as(mapped)) {
      return MutateSeqStmtChanged(mutator, self, i, std::move(mapped));
    }
  }
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
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
    const ffi::Any& item = seq->begin()[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));
    AppendSeqStmtResult(&output, std::move(mapped));
  }
  if (output.empty()) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(Evaluate(0)));
  }
  if (output.size() == 1) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(output[0]));
  }
  self->seq = std::move(output);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
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
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));
    if (IsSeqStmtNoOp(mapped)) {
      continue;
    }
    if (const auto* nested = mapped.as<SeqStmtNode>()) {
      const int64_t nested_size = static_cast<int64_t>(nested->seq.size());
      if (total + nested_size > i + 1) {
        return InplaceSplice(mutator, self, seq, total, i, nested);
      }
      for (const Stmt& stmt : nested->seq) {
        seq->SetItemAfterCheck(total++, ffi::Any(stmt));
      }
      continue;
    }
    if (total != i || !item.same_as(mapped)) {
      seq->SetItemAfterCheck(total, ffi::Any(std::move(mapped)));
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
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
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
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));
    if (!item.same_as(mapped)) {
      return MaybeInplaceMutateSeqStmtChanged(mutator, self, seq, i, std::move(mapped));
    }
  }
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
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
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, mapped_value, mutator->MutateExpected(self->value));
  if (mapped_value.same_as(self->value)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<EvaluateNode> copy = ffi::make_object<EvaluateNode>(*self);
  copy->value = std::move(mapped_value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny EvaluateMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                     ffi::AnyView value) noexcept {
  EvaluateNode* self = const_cast<EvaluateNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  if (mapped_value.same_as(self->value)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->value = std::move(mapped_value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
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
  return {VarNode::RuntimeTypeIndex(),            IntImmNode::RuntimeTypeIndex(),
          CallNode::RuntimeTypeIndex(),           prim::AddNode::RuntimeTypeIndex(),
          prim::MulNode::RuntimeTypeIndex(),      prim::FloorDivNode::RuntimeTypeIndex(),
          prim::FloorModNode::RuntimeTypeIndex(), SeqStmtNode::RuntimeTypeIndex(),
          EvaluateNode::RuntimeTypeIndex()};
}

}  // namespace tvm_hooks

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_H_
