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
// PORTED FROM apache/tvm#20275, head b51da96381 ("[REFACTOR][IR] Normalize structural hook
// source shape"), branch redo-expr-stmt-structural-hooks-current.
//
//   hook group   source file             functions
//   ----------   ---------------------   ----------------------------------------------------
//   IntImm       src/ir/expr.cc:206-224  IntImmVisit / IntImmMutate / IntImmMaybeInplaceMutate
//   Var          src/ir/expr.cc:281-374  VarVisit / VarMutate / VarMaybeInplaceMutate
//   binary ops   src/ir/prim/expr.cc:51-93   BinaryVisit / BinaryMutate /
//                                            BinaryMaybeInplaceMutate
//   SeqStmt      src/tirx/ir/stmt.cc:504-590  SeqStmtVisit / MutateSeqStmtRaw /
//                                            MaybeInplaceMutateSeqStmtRaw
//   Evaluate     src/tirx/ir/stmt.cc:641-672  EvaluateVisit / EvaluateMutate /
//                                            EvaluateMaybeInplaceMutate
//
// 20275 is under review and moving -- it has a tvm-ffi bump, a visit-macro adoption, a squash
// and two missed hook files pending -- so this port follows it rather than apache/tvm main.
// `./port_check.sh` re-extracts these ranges from the recorded sha and diffs them against the
// bodies below; run it before trusting a measurement, and after any rebase of the PR.
//
// The bodies are copied verbatim: same names, same signatures, same order, same internal
// structure, grouped by the TVM file each came from. Nothing is reordered, renamed or tidied,
// because a change prototyped here has to lift back into apache/tvm as a patch. The only
// intended differences are marked `HARNESS DEVIATION` and there are two, both on SeqStmt.
//
// Note what 20275 changed relative to apache/tvm main, since it invalidates measurements taken
// against main:
//   * BinaryMutate no longer re-infers the result type at all -- the `BinaryResultType` guard
//     is gone, on the ground that StructuralMap preserves node types.
//   * BinaryMaybeInplaceMutate lost its `same_as` early return.
//   * SeqStmt's hooks were replaced outright by the splice-capable MutateSeqStmtRaw /
//     MaybeInplaceMutateSeqStmtRaw, with lazy `output` allocation, a `growing` flag, prefix
//     back-fill, and `self->seq.Set(i, ...)` in place of main's `Array<Stmt> mapped_seq` copy.
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
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/stmt.h>

#include <cstring>
#include <utility>
#include <vector>

#include "bench_common.h"

// Selects between 20275's SeqStmt in-place hook and the repaired variant built on top of it.
#ifndef TVM_SEQSTMT_INPLACE_FIX
#define TVM_SEQSTMT_INPLACE_FIX 1
#endif

namespace tvm_hooks {

using namespace tvm;              // NOLINT(build/namespaces)
using namespace tvm::tirx;        // NOLINT(build/namespaces)
namespace ffi = tvm::ffi;

// ---------------------------------------------------------------------------
// src/ir/expr.cc -- IntImm
// ---------------------------------------------------------------------------

TVMFFIAny IntImmVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // skips: value
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(nullptr));
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
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(nullptr));
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
    return return_mapped_var(ffi::Any(value));
  }
  // Only NonRecursive is clamped: Recursive co-introduces type fields such as BufferType shape
  // variables, so that ambient region must continue through the dynamic type.
  auto mutate_ty = [&]() { return mutator->MutateExpected(self->ty); };
  ffi::Expected<ffi::Any> mapped_ty_result =
      mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive
          ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_ty)
          : mutate_ty();
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Type, mapped_ty, std::move(mapped_ty_result));
  ffi::Any mapped_var = ffi::Any(value);
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
    return return_mapped_var(ffi::Any(value));
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
  return return_mapped_var(ffi::Any(value));
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
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(nullptr));
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
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(nullptr));
}

TVMFFIAny MutateSeqStmtRaw(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const SeqStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value);
  ffi::Array<Stmt> output;
  bool changed = false;
  for (size_t i = 0; i < self->seq.size(); ++i) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped, mutator->MutateExpected(self->seq[i]));
    if (!changed && mapped.same_as(self->seq[i])) {
      continue;
    }
    if (!changed) {
      changed = true;
      output.reserve(self->seq.size());
      for (size_t j = 0; j < i; ++j) {
        output.push_back(self->seq[j]);
      }
    }
    if (const auto* nested = mapped.as<SeqStmtNode>()) {
      for (const Stmt& stmt : nested->seq) {
        output.push_back(stmt);
      }
    } else {
      output.push_back(std::move(mapped));
    }
  }
  if (!changed) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<SeqStmtNode> copy = ffi::make_object<SeqStmtNode>(*self);
  copy->seq = std::move(output);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny MaybeInplaceMutateSeqStmtRaw(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  SeqStmtNode* self = const_cast<SeqStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value));
  ffi::Array<Stmt> output;
  bool growing = false;
  for (size_t i = 0; i < self->seq.size(); ++i) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(self->seq[i]));
    if (!growing) {
      if (mapped.same_as(self->seq[i])) {
        continue;
      }
      if (const auto* nested = mapped.as<SeqStmtNode>()) {
        growing = true;
        output.reserve(self->seq.size());
        for (size_t j = 0; j < i; ++j) {
          output.push_back(self->seq[j]);
        }
        for (const Stmt& stmt : nested->seq) {
          output.push_back(stmt);
        }
      } else {
        self->seq.Set(i, std::move(mapped));
      }
    } else if (const auto* nested = mapped.as<SeqStmtNode>()) {
      for (const Stmt& stmt : nested->seq) {
        output.push_back(stmt);
      }
    } else {
      output.push_back(std::move(mapped));
    }
  }
  if (growing) {
    self->seq = std::move(output);
  }
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

// HARNESS DEVIATION 1 of 2, and the only change to a hook body.
//
// The defect it addresses, stated as a fact about the code and independently of whether
// repairing it is worthwhile: 20275 dropped main's `Array<Stmt> mapped_seq = self->seq` handle
// copy, so the sequence itself is no longer copied. But `Array::operator[]` is
// `const T operator[](int64_t) const` -- it returns *by value* -- so `self->seq[i]` materialises
// a second handle to the element for the duration of the call, and
// `MaybeInplaceMutateIfUniqueExpected` therefore never finds an element unique.
// **20275's hook cannot mutate any element in place.** Only the array-level copy was fixed.
//
// The repair is tvm-ffi's own pattern, `MaybeInplaceMutateSeqContainerRaw`
// (src/ffi/extra/structural_mutate.cc:198): bind `const Any&` into the sequence object's
// storage instead of taking `Array::operator[]`'s by-value return, so no reference is added,
// the element's count stays at one, and the hook sees it as unique. Writing back through
// `SetItemAfterCheck` only when the value actually changed. Nothing is moved out of a slot, so
// there is no restore obligation on the error paths and nothing unsafe to justify.
//
// Two additions over tvm-ffi's version, both because `SeqStmt` differs from a bare container:
//
//   * **The precondition is checked rather than assumed.** tvm-ffi's version is dispatched by
//     the engine only after the engine has established that the container is uniquely owned.
//     Here the sequence is a *field* of the node, so the hook has to establish it itself. When
//     the array is shared this falls through to `MutateSeqStmtRaw` entirely unchanged -- no
//     copy-on-write, no allocation on the shared path. `ObjectRef::unique()` (object.h:499) is
//     `data_ != nullptr && data_->use_count() == 1` and binds no new reference, so reading it
//     does not perturb the count it reads.
//   * **Splice survives.** An element that mutates into a nested `SeqStmt` cannot be written
//     back with `SetItemAfterCheck`, so 20275's `growing` flag and prefix back-fill are kept
//     for that case and the in-place element loop runs only while not splicing.
TVMFFIAny MaybeInplaceMutateSeqStmtRepaired(ffi::StructuralMutatorObj* mutator,
                                            ffi::AnyView value) noexcept {
  SeqStmtNode* self = const_cast<SeqStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value));
  if (!self->seq.unique()) {
    return MutateSeqStmtRaw(mutator, value);
  }
  // Sole owner from here. `get()` hands back the pointer without taking a reference.
  ffi::ArrayObj* arr =
      static_cast<ffi::ArrayObj*>(const_cast<ffi::Object*>(self->seq.get()));
  ffi::Array<Stmt> output;
  bool growing = false;
  for (int64_t i = 0; i < static_cast<int64_t>(arr->size()); ++i) {
    if (!growing) {
      // A const reference into storage: no refcount bump, so the element stays unique and
      // MaybeInplaceMutateIfUniqueExpected can take the in-place path.
      const ffi::Any& item = arr->begin()[i];
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Any, mapped,
                                        mutator->MaybeInplaceMutateIfUniqueExpected(item));
      if (const auto* nested = mapped.as<SeqStmtNode>()) {
        growing = true;
        output.reserve(arr->size());
        for (int64_t j = 0; j < i; ++j) {
          output.push_back(arr->begin()[j].cast<Stmt>());
        }
        for (const Stmt& stmt : nested->seq) {
          output.push_back(stmt);
        }
      } else if (!item.same_as(mapped)) {
        arr->SetItemAfterCheck(i, std::move(mapped));
      }
    } else {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
          Stmt, mapped, mutator->MaybeInplaceMutateIfUniqueExpected(arr->begin()[i]));
      if (const auto* nested = mapped.as<SeqStmtNode>()) {
        for (const Stmt& stmt : nested->seq) {
          output.push_back(stmt);
        }
      } else {
        output.push_back(std::move(mapped));
      }
    }
  }
  if (growing) {
    self->seq = std::move(output);
  }
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny SeqStmtMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  return MutateSeqStmtRaw(mutator, value);
}

TVMFFIAny SeqStmtMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                    ffi::AnyView value) noexcept {
#if TVM_SEQSTMT_INPLACE_FIX
  return MaybeInplaceMutateSeqStmtRepaired(mutator, value);
#else
  return MaybeInplaceMutateSeqStmtRaw(mutator, value);
#endif
}

// ---------------------------------------------------------------------------
// src/tirx/ir/stmt.cc -- Evaluate
// ---------------------------------------------------------------------------

TVMFFIAny EvaluateVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const EvaluateNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(nullptr));
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
// HARNESS DEVIATION 2 of 2: TVM registers these through `refl::TypeAttrDef<T>()` in
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
  Install(SeqStmtNode::RuntimeTypeIndex(), &SeqStmtVisit, &SeqStmtMutate,
          &SeqStmtMaybeInplaceMutate);
  Install(EvaluateNode::RuntimeTypeIndex(), &EvaluateVisit, &EvaluateMutate,
          &EvaluateMaybeInplaceMutate);
}

/*! \brief The type indices InstallAll covers, for the coverage assertion. */
inline std::vector<int32_t> CoveredTypes() {
  return {VarNode::RuntimeTypeIndex(),           IntImmNode::RuntimeTypeIndex(),
          prim::AddNode::RuntimeTypeIndex(),     prim::MulNode::RuntimeTypeIndex(),
          prim::FloorDivNode::RuntimeTypeIndex(), prim::FloorModNode::RuntimeTypeIndex(),
          SeqStmtNode::RuntimeTypeIndex(),       EvaluateNode::RuntimeTypeIndex()};
}

}  // namespace tvm_hooks

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_H_
