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
 * \file tests/benchmark/structural_map_tvm_ast_cost.cc
 * \brief The five-rung structural cost ladder measured on the TVM primitive
 *        expression node set copied into src/ffi/testing/structural_map_tvm_ast.h.
 *
 * The driver is the one from structural_map_cost.cc with the synthetic node set
 * replaced by the copied one, so the two targets differ only in the AST.
 */
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

#include "../../src/ffi/testing/structural_map_tvm_ast.h"

namespace ffi = tvm::ffi;
namespace ast = tvm::ffi::testing::tvmast;

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

  // Deliberately empty: rung 2b isolates the metadata test from hash-table work.
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

// --------------------------------------------------------------------------
// Registration gate.  A node type without all three compiled hooks silently
// falls back to reflection-driven traversal, which measures something else.
// --------------------------------------------------------------------------
void RequireCompiledHooks() {
  namespace refl = ffi::reflection;
  const refl::TypeAttrColumn columns[] = {
      refl::TypeAttrColumn(refl::type_attr::kStructuralVisit),
      refl::TypeAttrColumn(refl::type_attr::kStructuralMutate),
      refl::TypeAttrColumn(refl::type_attr::kStructuralMaybeInplaceMutate),
  };
  const char* column_names[] = {"kStructuralVisit", "kStructuralMutate",
                                "kStructuralMaybeInplaceMutate"};
  const std::pair<const char*, int32_t> node_types[] = {
      {"Var", ast::VarNode::RuntimeTypeIndex()},
      {"IntImm", ast::IntImmNode::RuntimeTypeIndex()},
      {"Add", ast::AddNode::RuntimeTypeIndex()},
      {"Mul", ast::MulNode::RuntimeTypeIndex()},
      {"FloorDiv", ast::FloorDivNode::RuntimeTypeIndex()},
      {"FloorMod", ast::FloorModNode::RuntimeTypeIndex()},
  };
  bool ok = true;
  for (size_t c = 0; c < 3; ++c) {
    for (const auto& [name, type_index] : node_types) {
      ffi::AnyView entry = columns[c][type_index];
      if (entry.type_index() == ffi::TypeIndex::kTVMFFINone) {
        std::cerr << "FATAL missing type attribute " << column_names[c] << " for " << name << '\n';
        ok = false;
      }
    }
  }
  if (!ok) {
    std::cerr << "FATAL the copied AST would fall back to reflection-driven traversal; "
                 "the measurement would not describe the compiled hook path.\n";
    std::exit(1);
  }
  std::cout << "HOOKS ok node_types=6 columns=3\n";
}

// --------------------------------------------------------------------------
// Fixtures: the #345 split/fuse index expressions rebuilt on the copied AST.
//   ((fo * 16 + fi) // 32) * 32 + ((fo * 16 + fi) % 32)
// --------------------------------------------------------------------------
ast::PrimExpr Int(int64_t value) { return ast::IntImm::Int32(value); }
ast::PrimExpr Add(ast::PrimExpr a, ast::PrimExpr b) {
  return ast::Add(std::move(a), std::move(b));
}
ast::PrimExpr Mul(ast::PrimExpr a, ast::PrimExpr b) {
  return ast::Mul(std::move(a), std::move(b));
}
ast::PrimExpr FloorDiv(ast::PrimExpr a, int64_t b) {
  return ast::FloorDiv(std::move(a), Int(b));
}
ast::PrimExpr FloorMod(ast::PrimExpr a, int64_t b) {
  return ast::FloorMod(std::move(a), Int(b));
}

ast::PrimExpr Fused(const ast::PrimVar& fo, const ast::PrimVar& fi) {
  return Add(Mul(fo, Int(16)), fi);
}

ast::PrimExpr SplitBase(const ast::PrimVar& fo, const ast::PrimVar& fi, bool shared) {
  ast::PrimExpr left = Fused(fo, fi);
  ast::PrimExpr right = shared ? left : Fused(fo, fi);
  return Add(Mul(FloorDiv(left, 32), Int(32)), FloorMod(right, 32));
}

ast::PrimExpr SplitNested(const ast::PrimVar& fo, const ast::PrimVar& fi, int depth, bool shared) {
  if (depth == 0) return Fused(fo, fi);
  static constexpr int factors[] = {0, 32, 16, 8, 4};
  ast::PrimExpr left = SplitNested(fo, fi, depth - 1, shared);
  ast::PrimExpr right = shared ? left : SplitNested(fo, fi, depth - 1, shared);
  return Add(Mul(FloorDiv(left, factors[depth]), Int(factors[depth])),
             FloorMod(right, factors[depth]));
}

size_t CountWalkNodes(const ast::PrimExpr& root) {
  size_t count = 0;
#ifdef TVM_FFI_BENCHMARK_WALK_DEDUP_TEMPLATE
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder, true>(root,
                                                        [&](const ast::Expr&) -> ffi::WalkResult {
                                                          ++count;
                                                          return ffi::WalkResult::Advance();
                                                        });
#else
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(root, [&](const ast::Expr&) -> ffi::WalkResult {
    ++count;
    return ffi::WalkResult::Advance();
  });
#endif
  return count;
}

size_t CountMapNodes(const ast::PrimExpr& root) {
  size_t count = 0;
  ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(root, [&](const ast::Expr& value) -> ffi::Any {
    ++count;
    return ffi::Any(value);
  });
  return count;
}

// Occurrence count: the denominator the old recursive mutator and the ladder
// mutators actually process, since neither deduplicates shared subtrees.
size_t CountOccurrences(const ast::PrimExpr& value) {
  if (const auto* op = value.as<ast::AddNode>()) {
    return 1 + CountOccurrences(op->a) + CountOccurrences(op->b);
  }
  if (const auto* op = value.as<ast::MulNode>()) {
    return 1 + CountOccurrences(op->a) + CountOccurrences(op->b);
  }
  if (const auto* op = value.as<ast::FloorDivNode>()) {
    return 1 + CountOccurrences(op->a) + CountOccurrences(op->b);
  }
  if (const auto* op = value.as<ast::FloorModNode>()) {
    return 1 + CountOccurrences(op->a) + CountOccurrences(op->b);
  }
  return 1;
}

class OldVisitor {
 public:
  void Visit(const ast::PrimExpr& value) {
    if (!seen_.insert(value.get()).second) return;
    if (const auto* op = value.as<ast::AddNode>())
      VisitBinary(op);
    else if (const auto* op = value.as<ast::MulNode>())
      VisitBinary(op);
    else if (const auto* op = value.as<ast::FloorDivNode>())
      VisitBinary(op);
    else if (const auto* op = value.as<ast::FloorModNode>())
      VisitBinary(op);
    if (value.as<ast::VarNode>()) ++sink;
  }

 private:
  template <typename TNode>
  void VisitBinary(const TNode* op) {
    Visit(op->a);
    Visit(op->b);
  }
  std::unordered_set<const ffi::Object*> seen_;
};

class OldMutator {
 public:
  OldMutator(ast::PrimVar target, ast::PrimExpr replacement)
      : target_(std::move(target)), replacement_(std::move(replacement)) {}
  ast::PrimExpr Mutate(const ast::PrimExpr& value) {
    if (value.same_as(target_)) return replacement_;
    if (const auto* op = value.as<ast::AddNode>()) return MutateBinary<ast::Add>(op, value);
    if (const auto* op = value.as<ast::MulNode>()) return MutateBinary<ast::Mul>(op, value);
    if (const auto* op = value.as<ast::FloorDivNode>()) {
      return MutateBinary<ast::FloorDiv>(op, value);
    }
    if (const auto* op = value.as<ast::FloorModNode>()) {
      return MutateBinary<ast::FloorMod>(op, value);
    }
    return value;
  }

 private:
  template <typename TRef, typename TNode>
  ast::PrimExpr MutateBinary(const TNode* op, const ast::PrimExpr& original) {
    ast::PrimExpr a = Mutate(op->a);
    ast::PrimExpr b = Mutate(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) return original;
    return TRef(std::move(a), std::move(b));
  }
  ast::PrimVar target_;
  ast::PrimExpr replacement_;
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

void Run(const char* name, const ast::PrimExpr& root, const ast::PrimVar& target,
         const ast::PrimExpr& replacement, int repeats) {
  ast::PrimExpr stable_owner = root;
  size_t walk_nodes = CountWalkNodes(root);
  size_t mutation_nodes = CountOccurrences(root);
  size_t map_nodes = CountMapNodes(root);
  auto old_walk = [&] {
    OldVisitor visitor;
    visitor.Visit(root);
  };
  auto structural_walk = [&] {
#ifdef TVM_FFI_BENCHMARK_WALK_DEDUP_TEMPLATE
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder, true>(root,
                                                          [](const ast::Var&) -> ffi::WalkResult {
                                                            ++sink;
                                                            return ffi::WalkResult::Advance();
                                                          });
#else
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(root, [](const ast::Var&) -> ffi::WalkResult {
      ++sink;
      return ffi::WalkResult::Advance();
    });
#endif
  };
  auto old_changed = [&] {
    OldMutator mutator(target, replacement);
    sink += mutator.Mutate(root).defined();
  };
  auto old_noop = [&] {
    OldMutator mutator(target, target);
    sink += mutator.Mutate(root).defined();
  };
  auto rung2a = [&] {
    auto mutator = ffi::make_object<Ladder2a>();
    sink += mutator->Mutate(root).type_index();
  };
  auto rung2b = [&] {
    auto mutator = ffi::make_object<Ladder2b>();
    sink += mutator->Mutate(root).type_index();
  };
  auto map_noop = [&] {
    sink += ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
                root, [](const ffi::String& value) -> ffi::Any { return ffi::Any(value); })
                .type_index();
  };
  auto map_matching_noop = [&] {
    sink += ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
                root,
                [](const ast::VarNode* node) -> ffi::Any {
                  return ffi::Any(ffi::GetRef<ast::Var>(node));
                })
                .type_index();
  };
  auto map_changed = [&] {
    sink += ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
                root,
                [&](const ast::VarNode* node) -> ffi::Any {
                  ast::Var value = ffi::GetRef<ast::Var>(node);
                  if (value.same_as(target)) return ffi::Any(replacement);
                  return ffi::Any(value);
                })
                .type_index();
  };

  old_walk();
  structural_walk();
  old_changed();
  old_noop();
  rung2a();
  rung2b();
  map_noop();
  map_matching_noop();
  map_changed();
  std::cout << std::fixed << std::setprecision(3) << "RESULT fixture=" << name
            << " walk_nodes=" << walk_nodes << " mutation_nodes=" << mutation_nodes
            << " map_nodes=" << map_nodes
            << " old_walk=" << MedianNs(repeats, old_walk) / walk_nodes
            << " structural_walk=" << MedianNs(repeats, structural_walk) / walk_nodes
            << " old_changed=" << MedianNs(repeats / 10, old_changed) / mutation_nodes
            << " structural_changed=" << MedianNs(repeats / 10, map_changed) / map_nodes
            << " old_noop=" << MedianNs(repeats / 10, old_noop) / mutation_nodes
            << " structural_noop=" << MedianNs(repeats / 10, map_matching_noop) / map_nodes
            << " rung1=" << MedianNs(repeats, structural_walk) / walk_nodes
            << " rung2a=" << MedianNs(repeats / 10, rung2a) / mutation_nodes
            << " rung2b=" << MedianNs(repeats / 10, rung2b) / mutation_nodes
            << " rung3=" << MedianNs(repeats / 10, map_noop) / map_nodes
            << " rung4=" << MedianNs(repeats / 10, map_changed) / map_nodes << '\n';
}

int main() {
  ast::RegisterTvmAstTypes();
  RequireCompiledHooks();
  ast::PrimVar fo("fo"), fi("fi"), replacement("replacement");
  int repeats = std::getenv("TVM_FFI_BENCH_REPEATS")
                    ? std::atoi(std::getenv("TVM_FFI_BENCH_REPEATS"))
                    : 200000;
  Run("base_shared", SplitBase(fo, fi, true), fo, replacement, repeats);
  Run("base_distinct", SplitBase(fo, fi, false), fo, replacement, repeats);
  Run("nested4_shared", SplitNested(fo, fi, 4, true), fo, replacement, repeats / 5);
  Run("nested4_distinct", SplitNested(fo, fi, 4, false), fo, replacement, repeats / 20);
  std::cout << "SINK " << sink << '\n';
}
