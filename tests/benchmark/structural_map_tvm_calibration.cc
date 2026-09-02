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
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace tvm;
using namespace tvm::tirx;

static volatile size_t sink = 0;

template <bool WithIdentityCheck>
class LadderMutatorObj : public ffi::StructuralMutatorObj {
 public:
  LadderMutatorObj() : StructuralMutatorObj(VTable()) {}

 private:
  static const ffi::StructuralMutatorVTable* VTable() {
    static const ffi::StructuralMutatorVTable vtable{&DispatchMutate, &DispatchMutate,
                                                     &DispatchVarRemapGet, &DispatchVarRemapSet};
    return &vtable;
  }
  static TVMFFIAny DispatchMutate(ffi::StructuralMutatorObj* base, ffi::AnyView value) noexcept {
    auto* self = static_cast<LadderMutatorObj*>(base);
    ffi::Expected<ffi::Any> result = [&]() -> ffi::Expected<ffi::Any> {
      if constexpr (WithIdentityCheck) {
        return self->MutateWithIdentityRemapExpected(
            value, [&] { return self->DefaultMutateExpected(value); });
      } else {
        return self->DefaultMutateExpected(value);
      }
    }();
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }
  static TVMFFIAny DispatchVarRemapGet(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
        ffi::Expected<ffi::Any>(ffi::Any(nullptr)));
  }
  static TVMFFIAny DispatchVarRemapSet(ffi::StructuralMutatorObj*, ffi::AnyView,
                                       ffi::AnyView) noexcept {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(ffi::Expected<void>());
  }
};

using Ladder2a = LadderMutatorObj<false>;
using Ladder2b = LadderMutatorObj<true>;

class OldPostOrder : public StmtExprVisitor {
 public:
  explicit OldPostOrder(std::function<void(const ffi::ObjectRef&)> callback)
      : callback_(std::move(callback)) {}
  void VisitExpr(const Expr& expr) final {
    if (!visited_.insert(expr.get()).second) return;
    ExprVisitor::VisitExpr(expr);
    callback_(expr);
  }
  void VisitStmt(const Stmt& stmt) final {
    if (!visited_.insert(stmt.get()).second) return;
    StmtVisitor::VisitStmt(stmt);
    callback_(stmt);
  }
  void VisitBufferDef(const BufferVar&, bool) final {}
  void VisitBufferUse(const BufferVar&) final {}

 private:
  std::function<void(const ffi::ObjectRef&)> callback_;
  std::unordered_set<const ffi::Object*> visited_;
};

class OldSubstitute : public StmtExprMutator {
 public:
  OldSubstitute(Var target, Expr replacement)
      : target_(std::move(target)), replacement_(std::move(replacement)) {}
  size_t visits() const { return visits_; }
  Expr VisitExpr(const Expr& expr) final {
    ++visits_;
    return ExprMutator::VisitExpr(expr);
  }
  Expr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);
    return var.same_as(target_) ? replacement_ : Expr(var);
  }

 private:
  Var target_;
  Expr replacement_;
  size_t visits_{0};
};

template <typename F>
double MedianNs(int repeats, F&& workload) {
  std::vector<double> samples;
  for (int sample = 0; sample < 11; ++sample) {
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) workload();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now() - begin)
                       .count();
    samples.push_back(static_cast<double>(elapsed) / repeats);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void Run(const char* name, const PrimExpr& root, const Var& target, const PrimExpr& replacement,
         int repeats) {
  PrimExpr stable_owner = root;
  size_t walk_nodes = 0;
  OldPostOrder count([&](const ffi::ObjectRef&) { ++walk_nodes; });
  count(root);
  OldSubstitute old_count(target, replacement);
  old_count(root);
  size_t old_mutate_nodes = old_count.visits();
  size_t map_nodes = name == std::string("base_distinct") ? 15 : 155;

  auto old_walk = [&] {
    OldPostOrder visitor([](const ffi::ObjectRef& value) {
      if (value.as<VarNode>()) ++sink;
    });
    visitor(root);
  };
  auto structural_walk = [&] {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder, true>(root, [](const Var&) -> ffi::WalkResult {
      ++sink;
      return ffi::WalkResult::Advance();
    });
  };
  auto old_changed = [&] {
    OldSubstitute mutator(target, replacement);
    sink += mutator(root).defined();
  };
  auto structural_changed = [&] {
    sink +=
        ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(root, [&](const VarNode* node) -> ffi::Any {
          Var value = ffi::GetRef<Var>(node);
          if (value.same_as(target)) return ffi::Any(replacement);
          return ffi::Any(Expr(value));
        }).type_index();
  };
  auto old_noop = [&] {
    OldSubstitute mutator(target, target);
    sink += mutator(root).defined();
  };
  auto structural_noop = [&] {
    sink +=
        ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(root, [](const VarNode* node) -> ffi::Any {
          return ffi::Any(ffi::GetRef<Var>(node));
        }).type_index();
  };
  auto rung2a = [&] {
    auto mutator = ffi::make_object<Ladder2a>();
    sink += mutator->Mutate(root).type_index();
  };
  auto rung2b = [&] {
    auto mutator = ffi::make_object<Ladder2b>();
    sink += mutator->Mutate(root).type_index();
  };
  auto rung3 = [&] {
    sink += ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
                root, [](const ffi::String& value) -> ffi::Any { return ffi::Any(value); })
                .type_index();
  };

  old_walk();
  structural_walk();
  old_changed();
  structural_changed();
  old_noop();
  structural_noop();
  rung2a();
  rung2b();
  rung3();
  std::cout << std::fixed << std::setprecision(3) << "TVM fixture=" << name
            << " walk_nodes=" << walk_nodes << " old_mutate_nodes=" << old_mutate_nodes
            << " map_nodes=" << map_nodes
            << " old_walk=" << MedianNs(repeats, old_walk) / walk_nodes
            << " structural_walk=" << MedianNs(repeats, structural_walk) / walk_nodes
            << " old_changed=" << MedianNs(repeats / 10, old_changed) / old_mutate_nodes
            << " structural_changed=" << MedianNs(repeats / 10, structural_changed) / map_nodes
            << " old_noop=" << MedianNs(repeats / 10, old_noop) / old_mutate_nodes
            << " structural_noop=" << MedianNs(repeats / 10, structural_noop) / map_nodes
            << " rung1=" << MedianNs(repeats, structural_walk) / walk_nodes
            << " rung2a=" << MedianNs(repeats / 10, rung2a) / old_mutate_nodes
            << " rung2b=" << MedianNs(repeats / 10, rung2b) / old_mutate_nodes
            << " rung3=" << MedianNs(repeats / 10, rung3) / map_nodes
            << " rung4=" << MedianNs(repeats / 10, structural_changed) / map_nodes << '\n';
}

int main() {
  PrimVar fo("fo"), fi("fi"), replacement("replacement");
  PrimExpr fused_a = fo * 16 + fi;
  PrimExpr fused_b = fo * 16 + fi;
  PrimExpr base = indexdiv(fused_a, 32) * 32 + indexmod(fused_b, 32);
  std::function<PrimExpr(int)> nest = [&](int depth) -> PrimExpr {
    if (depth == 0) return fo * 16 + fi;
    static constexpr int factors[] = {0, 32, 16, 8, 4};
    PrimExpr left = nest(depth - 1);
    PrimExpr right = nest(depth - 1);
    return indexdiv(left, factors[depth]) * factors[depth] + indexmod(right, factors[depth]);
  };
  int repeats = std::getenv("TVM_FFI_BENCH_REPEATS")
                    ? std::atoi(std::getenv("TVM_FFI_BENCH_REPEATS"))
                    : 200000;
  Run("base_distinct", base, fo, replacement, repeats);
  Run("nested4_distinct", nest(4), fo, replacement, repeats / 20);
  std::cout << "SINK " << sink << '\n';
}
