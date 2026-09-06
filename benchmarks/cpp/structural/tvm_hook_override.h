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
// PORTED FROM apache/tvm#20275, head e40167046ed6 ("[REFACTOR][IR] Add structural hooks for
// Expr and Stmt"), branch redo-expr-stmt-structural-hooks-current.
//
//   hook group   source file             functions
//   ----------   ---------------------   ----------------------------------------------------
//   IntImm       src/ir/expr.cc          IntImmVisit / IntImmMutate / IntImmMaybeInplaceMutate
//   Var          src/ir/expr.cc          VarVisit / VarMutate / VarMaybeInplaceMutate
//   binary ops   src/ir/prim/expr.cc     BinaryVisit / BinaryMutate /
//                                            BinaryMaybeInplaceMutate
//   SeqStmt      src/tirx/ir/stmt.cc     SeqStmtVisit / MutateSeqStmtRaw /
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
// because a change prototyped here has to lift back into apache/tvm as a patch. The only
// intended differences are marked `HARNESS DEVIATION` and there are two, both on SeqStmt.
//
// What e40167046ed6 changed relative to the previous port at b51da96381 -- all of it re-ported
// here, and all of it re-measured:
//   * `3rdparty/tvm-ffi` bumped 4e754f9f -> 9d784c4d, which is where
//     TVM_FFI_S_VISIT_RETURN_NONE() is defined.
//   * every visit hook now ends in TVM_FFI_S_VISIT_RETURN_NONE() rather than spelling out
//     `MoveAnyToTVMFFIAny(ffi::Any(nullptr))`.  The same value, named.
//   * VarMutate and VarMaybeInplaceMutate return `ffi::Any(self)` where they returned
//     `ffi::Any(value)` -- four sites, on the split/fuse hot path.
//   * MaybeInplaceMutateSeqStmtRaw is rewritten: a `.unique()` precondition falling through to
//     MutateSeqStmtRaw, a borrowed `const Any&` into storage instead of `self->seq[i]`'s
//     by-value return, a cursor with `erase`/`insert` splicing, and a one-shot overflow path
//     that finishes the suffix and allocates the exact flattened size.  See the note above
//     MaybeInplaceMutateSeqStmtRepaired for what this does to the defect the repair addressed.
//   * two hook files the earlier port did not cover were normalized (src/tirx/ir/tirx_stmt.cc,
//     src/tirx/ir/layout/tile_core.cc); neither holds a hook this harness dispatches into.
//
// And what 20275 as a whole still changes relative to apache/tvm main, since it invalidates
// measurements taken against main:
//   * BinaryMutate no longer re-infers the result type at all -- the `BinaryResultType` guard
//     is gone, on the ground that StructuralMap preserves node types.
//   * BinaryMaybeInplaceMutate lost its `same_as` early return.
//   * SeqStmt's hooks were replaced outright by the splice-capable MutateSeqStmtRaw /
//     MaybeInplaceMutateSeqStmtRaw.
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
  // The engine establishes ownership of the SeqStmt, but seq is a field and needs its own check.
  if (!self->seq.unique()) {
    return MutateSeqStmtRaw(mutator, value);
  }
  ffi::ArrayObj* seq = self->seq.GetArrayObj();
  // A null output means the uniquely owned field is still the mutation destination.
  ffi::ObjectPtr<ffi::ArrayObj> output = nullptr;
  int64_t cursor = 0;
  while (cursor < static_cast<int64_t>(seq->size())) {
    // Borrow the storage slot so the element stays unique during in-place dispatch.
    const ffi::Any& item = seq->begin()[cursor];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));

    const auto* nested = mapped.as<SeqStmtNode>();
    int64_t contribution = nested == nullptr ? 1 : static_cast<int64_t>(nested->seq.size());
    int64_t new_size = static_cast<int64_t>(seq->size()) - 1 + contribution;
    if (new_size > static_cast<int64_t>(seq->capacity())) {
      // Finish the suffix exactly once, then allocate the exact flattened size without replaying
      // callbacks over the already-mutated prefix.
      seq->SetItemAfterCheck(cursor, ffi::Any(std::move(mapped)));
      int64_t source_size = static_cast<int64_t>(seq->size());
      int64_t output_size = cursor + contribution;
      for (int64_t read_cursor = cursor + 1; read_cursor < source_size; ++read_cursor) {
        const ffi::Any& source_item = seq->begin()[read_cursor];
        TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, source_mapped,
                                          mutator->MaybeInplaceMutateIfUniqueExpected(source_item));
        if (!source_item.same_as(source_mapped)) {
          seq->SetItemAfterCheck(read_cursor, ffi::Any(std::move(source_mapped)));
        }
        const auto* source_nested = seq->begin()[read_cursor].as<SeqStmtNode>();
        output_size +=
            source_nested == nullptr ? 1 : static_cast<int64_t>(source_nested->seq.size());
      }

      output = ffi::ArrayObj::CreateRepeated(output_size, ffi::Any());
      int64_t write_cursor = 0;
      for (int64_t read_cursor = 0; read_cursor < source_size; ++read_cursor) {
        const ffi::Any& source_item = seq->begin()[read_cursor];
        if (const auto* source_nested = source_item.as<SeqStmtNode>()) {
          for (const Stmt& stmt : source_nested->seq) {
            output->SetItemAfterCheck(write_cursor++, ffi::Any(stmt));
          }
        } else {
          output->SetItemAfterCheck(write_cursor++, ffi::Any(source_item.cast<Stmt>()));
        }
      }
      TVM_FFI_ICHECK_EQ(write_cursor, output_size);
      break;
    }

    if (nested == nullptr) {
      if (!item.same_as(mapped)) {
        seq->SetItemAfterCheck(cursor, ffi::Any(std::move(mapped)));
      }
      ++cursor;
    } else {
      const ffi::ArrayObj* nested_seq = nested->seq.GetArrayObj();
      seq->erase(cursor);
      seq->insert(cursor, nested_seq->begin(), nested_seq->end());
      cursor += contribution;
    }
  }
  if (output != nullptr) {
    self->seq = ffi::Array<Stmt>(std::move(output));
  }
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

// HARNESS DEVIATION 1 of 2, and the only change to a hook body.
//
// **The defect this was written against is fixed upstream.** At b51da96381 the hook took
// `self->seq[i]`, and `Array::operator[]` is `const T operator[](int64_t) const` -- it returns
// *by value* -- so the element gained a second handle for the duration of the call and
// `MaybeInplaceMutateIfUniqueExpected` never found one unique.  e40167046ed6 adopts the
// pattern this deviation was prototyping: a `.unique()` precondition on the field, a borrowed
// `const ffi::Any&` into storage, and `SetItemAfterCheck` write-back.  So the shipped hook
// now mutates elements in place, and the gap this variant existed to close is closed.
//
// It is kept, behind TVM_SEQSTMT_INPLACE_FIX, for two things it still does:
//
//   * It is the **differential reference for the splice matrix**.  The two implementations
//     splice differently -- upstream now uses `erase`/`insert` with a cursor plus a one-shot
//     overflow rebuild; this one uses a single read/write cursor pair with spill-on-overrun --
//     and `real_tvm_bench.cc` runs both against `MutateSeqStmtRaw` across the same twelve
//     cases.  Two independent in-place implementations agreeing with the reference is a
//     stronger check on the reference than either alone.
//   * It prices **spill-on-grow against shift-on-grow**, which is a real trade the upstream
//     choice settles one way and this one the other.  Upstream reuses the array on any splice
//     that fits its capacity, shifting the tail with `insert`, at O(n) per splice.  This
//     variant never shifts and spills to a fresh array whenever a write would pass the read
//     cursor -- which a growing splice does even with capacity to spare.
//
// The mechanics of this variant, unchanged: bind `const Any&` into the sequence object's
// storage rather than taking `Array::operator[]`'s by-value return, so no reference is added
// and the element stays unique; write back through `SetItemAfterCheck` only when the value
// actually changed.  Nothing is moved out of a slot, so there is no restore obligation on the
// error paths.  The `.unique()` precondition is checked rather than assumed because the
// sequence is a *field* of the node rather than a container the engine has already established
// ownership of; a shared array falls through to `MutateSeqStmtRaw` unchanged.
// `ObjectRef::unique()` binds no new reference, so reading it does not perturb what it reads.
//
// Splice is handled in the same pass without direction cases: an element mapping to a nested
// `SeqStmt` of n statements replaces itself with all n, and the write cursor advances by
// however many statements the result contributes.  `output` is null while writing in place and
// is allocated once, carrying the already-mutated prefix, at the first write that has nowhere
// to go.  The spill condition is `write >= capacity || write > read`: capacity alone is not
// sufficient, because a write also has nowhere to go when it would pass the read cursor and
// clobber input not yet consumed.  Slack opened by an earlier shrink is reused by a later
// grow, so a pass that nets out even or shorter never allocates.
TVMFFIAny MaybeInplaceMutateSeqStmtRepaired(ffi::StructuralMutatorObj* mutator,
                                            ffi::AnyView value) noexcept {
  SeqStmtNode* self = const_cast<SeqStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value));
  if (!self->seq.unique()) {
    return MutateSeqStmtRaw(mutator, value);
  }
  // Sole owner. One pass, a read cursor and a write cursor over the array the node already
  // owns. Growing and shrinking are not cases: the write cursor simply advances by however
  // many statements each result contributes -- zero for an empty nested SeqStmt, one for an
  // ordinary result, several for a nested one.
  ffi::ArrayObj* arr = static_cast<ffi::ArrayObj*>(const_cast<ffi::Object*>(self->seq.get()));
  const int64_t size = static_cast<int64_t>(arr->size());
  const int64_t capacity = static_cast<int64_t>(self->seq.capacity());
  ffi::Array<Stmt> output;  // null until a write has nowhere to go
  bool spilled = false;
  int64_t write = 0;

  // A write has nowhere to go when it would run past the buffer, or when it would run past the
  // read cursor and clobber input not yet consumed. Slack opened by an earlier shrink is
  // therefore reused by a later grow, and a sequence that nets out even or shorter never
  // allocates at all.
  auto emit = [&](Stmt stmt, int64_t read) {
    if (!spilled && (write >= capacity || write > read)) {
      spilled = true;
      output.reserve(size);
      for (int64_t j = 0; j < write; ++j) {
        output.push_back(arr->begin()[j].cast<Stmt>());  // the already-mutated prefix
      }
    }
    if (spilled) {
      output.push_back(std::move(stmt));
    } else {
      arr->SetItemAfterCheck(write, std::move(stmt));
    }
    ++write;
  };

  for (int64_t read = 0; read < size; ++read) {
    // A const reference into storage: no refcount bump, so the element stays unique and
    // MaybeInplaceMutateIfUniqueExpected can take the in-place path. This is the whole fix.
    // Reading precedes any write for this element, and while not spilled `write <= read`, so
    // no write can clobber an element still to be read.
    const ffi::Any& item = arr->begin()[read];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Any, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));
    if (const auto* nested = mapped.as<SeqStmtNode>()) {
      for (const Stmt& stmt : nested->seq) emit(stmt, read);
    } else {
      emit(ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<Stmt>(std::move(mapped)), read);
    }
  }
  if (spilled) {
    self->seq = std::move(output);
  } else if (write != size) {
    self->seq.resize(write);  // the sequence shrank; drop the tail in place
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
