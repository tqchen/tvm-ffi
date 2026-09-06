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
// This is the point of the real-TVM harness rather than a refinement of it.  With the hooks
// here, a hook experiment is an edit to this file and a rebuild of one translation unit;
// without them it is a TVM branch and a full TVM build.
//
// They are always installed -- there is no dual mode.  `walk_old` and `map_old` run on them
// too, because `PostOrderVisit` and `Substitute` are built on the structural engine, so the
// old-versus-new comparison then differs only in the traversal API and not in the hooks
// underneath it.  The consequence, which the report states plainly: no number here describes
// TVM's own registered hook implementations.
//
// Fidelity.  The bodies are a port of what apache/tvm registers today, grouped in the same
// order as TVM's own registration blocks so they are easy to diff:
//
//   src/ir/prim/expr.cc     Var, IntImm, FloatImm, the binary operators
//   src/tirx/ir/stmt.cc     Evaluate, SeqStmt
//
// Deliberate divergences, both named in the report:
//
//   * result-type re-inference copies the left operand's type instead of running TVM's
//     `BinaryResultType`, which is internal to TVM.  Equivalent for equal-typed scalar
//     operands, which is every case these fixtures build.
//   * `MaybeInplaceMutateSeqStmt` has two shapes, selected by TVM_SEQSTMT_INPLACE_FIX.  TVM's
//     current body takes a handle copy of the sequence, so no element is ever uniquely owned
//     and every one takes the copy-on-write path; the repaired body moves each element out of
//     a uniquely owned array instead.  Both are compiled so before and after are measured on
//     the same footing.
//
// Which types appear here was derived empirically, from a dispatch-count pass over the
// fixtures rather than from reading TVM's registry: `Var`, `IntImm`, `prim::Add`, `prim::Mul`,
// `prim::FloorDiv`, `prim::FloorMod`, `Evaluate`, `SeqStmt`.  `real_tvm_bench.cc` asserts
// coverage -- every type a fixture dispatches on must be one of these -- so a new fixture that
// reaches a new type names it instead of silently falling through to TVM's hook.

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/stmt.h>

#include <cstring>
#include <utility>
#include <vector>

#include "bench_common.h"

// Whether SeqStmt's in-place hook is the repaired shape.  The same switch exists in
// `mini_tir.h`, so the before/after is measured identically on both node sets.
#ifndef TVM_SEQSTMT_INPLACE_FIX
#define TVM_SEQSTMT_INPLACE_FIX 1
#endif

namespace tvm_hooks {

using namespace tvm;              // NOLINT(build/namespaces)
using namespace tvm::tirx;        // NOLINT(build/namespaces)
namespace ffi = tvm::ffi;

// ---------------------------------------------------------------------------
// Result carriers.
// ---------------------------------------------------------------------------

TVM_FFI_INLINE TVMFFIAny VisitDone() {
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
      ffi::Expected<ffi::Optional<ffi::VisitInterrupt>>(std::nullopt));
}
TVM_FFI_INLINE TVMFFIAny KeepValue(ffi::AnyView value) {
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(value));
}

// ---------------------------------------------------------------------------
// src/ir/prim/expr.cc -- Var
// ---------------------------------------------------------------------------

TVMFFIAny VisitVar(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return VisitDone();
}

/*!
 * \brief Port of tvm::MutateVar: consult the remap first, record the answer on the way out.
 *
 * This is what makes a substitution consistent across occurrences.  The first occurrence of
 * an identity runs the callback and records the result; every later occurrence is served from
 * the remap without re-entering a hook, which is why a matching map arm's dispatch count is
 * `occurrences - remap hits` rather than `occurrences`.
 */
TVMFFIAny MutateVar(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  ffi::Expected<ffi::Any> cached = mutator->VarRemapGetExpected(value);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(cached);
  if (ffi::details::ExpectedUnsafe::GetData(cached).type_index() != ffi::TypeIndex::kTVMFFINone) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(cached));
  }
  ffi::Expected<void> set_result = mutator->VarRemapSetExpected(value, ffi::AnyView(value));
  if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(set_result).error()));
  }
  return KeepValue(value);
}

// ---------------------------------------------------------------------------
// src/ir/prim/expr.cc -- IntImm
// ---------------------------------------------------------------------------

TVMFFIAny VisitIntImm(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return VisitDone();
}

TVMFFIAny MutateIntImm(ffi::StructuralMutatorObj*, ffi::AnyView value) noexcept {
  return KeepValue(value);
}

// ---------------------------------------------------------------------------
// src/ir/prim/expr.cc -- the binary operators
//
// Written once per operator would be eight near-identical function pairs, so the operator
// type is the one thing parameterized here; the bodies are otherwise literal.
// ---------------------------------------------------------------------------

template <typename TNode>
TVMFFIAny VisitBinary(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const TNode* self = value.cast<const TNode*>();
  auto a_result = visitor->VisitExpected(self->a);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(a_result);
  auto b_result = visitor->VisitExpected(self->b);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(b_result);
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(b_result));
}

template <typename TNode>
TVMFFIAny MutateBinary(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const TNode* self = value.cast<const TNode*>();
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, a, mutator->MutateExpected(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, b, mutator->MutateExpected(self->b));
  if (a.same_as(self->a) && b.same_as(self->b)) return KeepValue(value);
  ffi::ObjectPtr<TNode> copy = ffi::make_object<TNode>(*self);
  // The result type is a function of the operand types alone, so when neither operand's type
  // moved the copy already carries the right one, and substituting a variable of the same
  // type is the common case. This is TVM's own guard.
  if (!a->ty.same_as(self->a->ty) || !b->ty.same_as(self->b->ty)) {
    copy->ExprNode::ty = a->ty;
  }
  copy->a = std::move(a);
  copy->b = std::move(b);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(PrimExpr(std::move(copy))));
}

template <typename TNode>
TVMFFIAny MaybeInplaceMutateBinary(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  TNode* self = const_cast<TNode*>(value.cast<const TNode*>());
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, a,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, b,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->b));
  if (a.same_as(self->a) && b.same_as(self->b)) return KeepValue(value);
  if (!a->ty.same_as(self->a->ty) || !b->ty.same_as(self->b->ty)) {
    self->ExprNode::ty = a->ty;
  }
  self->a = std::move(a);
  self->b = std::move(b);
  return KeepValue(value);
}

// ---------------------------------------------------------------------------
// src/tirx/ir/stmt.cc -- Evaluate
// ---------------------------------------------------------------------------

TVMFFIAny VisitEvaluate(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const EvaluateNode* self = value.cast<const EvaluateNode*>();
  auto result = visitor->VisitExpected(self->value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(result);
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

TVMFFIAny MutateEvaluate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const EvaluateNode* self = value.cast<const EvaluateNode*>();
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, mapped, mutator->MutateExpected(self->value));
  if (mapped.same_as(self->value)) return KeepValue(value);
  ffi::ObjectPtr<EvaluateNode> copy = ffi::make_object<EvaluateNode>(*self);
  copy->value = std::move(mapped);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(Stmt(std::move(copy))));
}

TVMFFIAny MaybeInplaceMutateEvaluate(ffi::StructuralMutatorObj* mutator,
                                     ffi::AnyView value) noexcept {
  EvaluateNode* self = const_cast<EvaluateNode*>(value.cast<const EvaluateNode*>());
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, mapped,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  if (mapped.same_as(self->value)) return KeepValue(value);
  self->value = std::move(mapped);
  return KeepValue(value);
}

// ---------------------------------------------------------------------------
// src/tirx/ir/stmt.cc -- SeqStmt
// ---------------------------------------------------------------------------

TVMFFIAny VisitSeqStmt(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const SeqStmtNode* self = value.cast<const SeqStmtNode*>();
  ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> result = std::nullopt;
  for (const Stmt& statement : self->seq) {
    result = visitor->VisitExpected(statement);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(result);
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(result);
  return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

TVMFFIAny MutateSeqStmt(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const SeqStmtNode* self = value.cast<const SeqStmtNode*>();
  ffi::Array<Stmt> mapped = self->seq;
  for (size_t i = 0; i < mapped.size(); ++i) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, statement, mutator->MutateExpected(mapped[i]));
    if (!statement.same_as(mapped[i])) mapped.Set(i, std::move(statement));
  }
  if (mapped.same_as(self->seq)) return KeepValue(value);
  ffi::ObjectPtr<SeqStmtNode> copy = ffi::make_object<SeqStmtNode>(*self);
  copy->seq = std::move(mapped);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(Stmt(std::move(copy))));
}

/*!
 * \brief SeqStmt's in-place hook, in both shapes.
 *
 * TVM's current body is the `#else` branch.  Binding `Array<Stmt> mapped = self->seq` gives
 * the sequence -- and through `mapped[i]` every element -- a second reference, so no element
 * is ever uniquely owned and all of them take the copy-on-write path.  Only the SeqStmt node
 * itself is mutated in place; nothing below it is.
 *
 * The repaired body extends the path invariant rather than breaking it.  In-place mutation is
 * sound only while every node from the root down to the value is uniquely owned, and this
 * hook is dispatched under exactly that condition; `MutateByApply` on a uniquely owned array
 * moves each element out and clears its slot before calling back, so the handle handed down
 * is the sole reference.  A shared array still takes the copy-on-write path, which is what
 * correctness requires.
 */
TVMFFIAny MaybeInplaceMutateSeqStmt(ffi::StructuralMutatorObj* mutator,
                                    ffi::AnyView value) noexcept {
  SeqStmtNode* self = const_cast<SeqStmtNode*>(value.cast<const SeqStmtNode*>());
#if TVM_SEQSTMT_INPLACE_FIX
  if (self->seq.unique()) {
    ffi::Any failure;
    bool failed = false;
    self->seq.MutateByApply([&](Stmt statement) -> Stmt {
      if (TVM_FFI_PREDICT_FALSE(failed)) return statement;
      ffi::Expected<ffi::Any> mapped = mutator->MaybeInplaceMutateIfUniqueExpected(statement);
      if (TVM_FFI_PREDICT_FALSE(mapped.is_err())) {
        failed = true;
        failure = ffi::Any(std::move(mapped).error());
        return statement;
      }
      return ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<Stmt>(std::move(mapped).value());
    });
    if (TVM_FFI_PREDICT_FALSE(failed)) {
      return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(failure));
    }
    return KeepValue(value);
  }
#endif
  ffi::Array<Stmt> mapped = self->seq;
  for (size_t i = 0; i < mapped.size(); ++i) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, statement,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(mapped[i]));
    if (!statement.same_as(mapped[i])) mapped.Set(i, std::move(statement));
  }
  if (mapped.same_as(self->seq)) return KeepValue(value);
  self->seq = std::move(mapped);
  return KeepValue(value);
}

// ---------------------------------------------------------------------------
// Installation, over whatever TVM registered.
//
// Re-registering a type attribute is what the branch-local
// TVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE build of tvm-ffi exists for: TVM installs its hooks
// from static-init blocks, which run before main(), so the harness has to overwrite them.
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

/*! \brief Type indices the harness registered a hook for; the coverage assertion reads this. */
inline std::vector<int32_t>* CoveredTypes() {
  static std::vector<int32_t> covered;
  return &covered;
}

template <typename TNode>
void InstallBinary() {
  Install(TNode::RuntimeTypeIndex(), &VisitBinary<TNode>, &MutateBinary<TNode>,
          &MaybeInplaceMutateBinary<TNode>);
  CoveredTypes()->push_back(TNode::RuntimeTypeIndex());
}

/*!
 * \brief Install every hook the benchmark dispatches into. Call once from main().
 *
 * The set was derived from a dispatch-count pass over the fixtures, not from TVM's registry,
 * and `real_tvm_bench.cc` fails the run if a fixture reaches a type that is not here.
 */
inline void InstallAll() {
  Install(VarNode::RuntimeTypeIndex(), &VisitVar, &MutateVar, &MutateVar);
  CoveredTypes()->push_back(VarNode::RuntimeTypeIndex());
  Install(IntImmNode::RuntimeTypeIndex(), &VisitIntImm, &MutateIntImm, &MutateIntImm);
  CoveredTypes()->push_back(IntImmNode::RuntimeTypeIndex());
  InstallBinary<prim::AddNode>();
  InstallBinary<prim::MulNode>();
  InstallBinary<prim::FloorDivNode>();
  InstallBinary<prim::FloorModNode>();
  Install(EvaluateNode::RuntimeTypeIndex(), &VisitEvaluate, &MutateEvaluate,
          &MaybeInplaceMutateEvaluate);
  CoveredTypes()->push_back(EvaluateNode::RuntimeTypeIndex());
  Install(SeqStmtNode::RuntimeTypeIndex(), &VisitSeqStmt, &MutateSeqStmt,
          &MaybeInplaceMutateSeqStmt);
  CoveredTypes()->push_back(SeqStmtNode::RuntimeTypeIndex());
}

}  // namespace tvm_hooks

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_H_
