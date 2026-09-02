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
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../src/ffi/testing/structural_map_benchmark.h"

namespace ffi = tvm::ffi;
namespace test = tvm::ffi::testing;

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

test::TestExpr Int(int64_t value) { return test::TestIntImm(value); }
test::TestExpr Add(test::TestExpr a, test::TestExpr b) {
  return test::TestAdd(std::move(a), std::move(b));
}
test::TestExpr Mul(test::TestExpr a, test::TestExpr b) {
  return test::TestMul(std::move(a), std::move(b));
}
test::TestExpr FloorDiv(test::TestExpr a, int64_t b) {
  return test::TestFloorDiv(std::move(a), Int(b));
}
test::TestExpr FloorMod(test::TestExpr a, int64_t b) {
  return test::TestFloorMod(std::move(a), Int(b));
}

test::TestExpr Fused(const test::TestVar& fo, const test::TestVar& fi) {
  return Add(Mul(fo, Int(16)), fi);
}

test::TestExpr SplitBase(const test::TestVar& fo, const test::TestVar& fi, bool shared) {
  test::TestExpr left = Fused(fo, fi);
  test::TestExpr right = shared ? left : Fused(fo, fi);
  return Add(Mul(FloorDiv(left, 32), Int(32)), FloorMod(right, 32));
}

test::TestExpr SplitNested(const test::TestVar& fo, const test::TestVar& fi, int depth,
                           bool shared) {
  if (depth == 0) return Fused(fo, fi);
  static constexpr int factors[] = {0, 32, 16, 8, 4};
  test::TestExpr left = SplitNested(fo, fi, depth - 1, shared);
  test::TestExpr right = shared ? left : SplitNested(fo, fi, depth - 1, shared);
  return Add(Mul(FloorDiv(left, factors[depth]), Int(factors[depth])),
             FloorMod(right, factors[depth]));
}

size_t CountWalkNodes(const test::TestExpr& root) {
  size_t count = 0;
#ifdef TVM_FFI_BENCHMARK_WALK_DEDUP_TEMPLATE
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder, true>(
      root, [&](const test::TestExpr&) -> ffi::WalkResult {
        ++count;
        return ffi::WalkResult::Advance();
      });
#else
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(root,
                                                  [&](const test::TestExpr&) -> ffi::WalkResult {
                                                    ++count;
                                                    return ffi::WalkResult::Advance();
                                                  });
#endif
  return count;
}

size_t CountMapNodes(const test::TestExpr& root) {
  size_t count = 0;
  ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(root,
                                                 [&](const test::TestExpr& value) -> ffi::Any {
                                                   ++count;
                                                   return ffi::Any(value);
                                                 });
  return count;
}

size_t CountOccurrences(const test::TestExpr& value) {
  if (const auto* op = value.as<test::TestAddObj>()) {
    return 1 + CountOccurrences(op->a) + CountOccurrences(op->b);
  }
  if (const auto* op = value.as<test::TestMulObj>()) {
    return 1 + CountOccurrences(op->a) + CountOccurrences(op->b);
  }
  if (const auto* op = value.as<test::TestFloorDivObj>()) {
    return 1 + CountOccurrences(op->a) + CountOccurrences(op->b);
  }
  if (const auto* op = value.as<test::TestFloorModObj>()) {
    return 1 + CountOccurrences(op->a) + CountOccurrences(op->b);
  }
  return 1;
}

class OldVisitor {
 public:
  void Visit(const test::TestExpr& value) {
    if (!seen_.insert(value.get()).second) return;
    if (const auto* op = value.as<test::TestAddObj>())
      VisitBinary(op);
    else if (const auto* op = value.as<test::TestMulObj>())
      VisitBinary(op);
    else if (const auto* op = value.as<test::TestFloorDivObj>())
      VisitBinary(op);
    else if (const auto* op = value.as<test::TestFloorModObj>())
      VisitBinary(op);
    if (value.as<test::TestVarObj>()) ++sink;
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
  OldMutator(test::TestVar target, test::TestExpr replacement)
      : target_(std::move(target)), replacement_(std::move(replacement)) {}
  test::TestExpr Mutate(const test::TestExpr& value) {
    if (value.same_as(target_)) return replacement_;
    if (const auto* op = value.as<test::TestAddObj>())
      return MutateBinary<test::TestAdd>(op, value);
    if (const auto* op = value.as<test::TestMulObj>())
      return MutateBinary<test::TestMul>(op, value);
    if (const auto* op = value.as<test::TestFloorDivObj>()) {
      return MutateBinary<test::TestFloorDiv>(op, value);
    }
    if (const auto* op = value.as<test::TestFloorModObj>()) {
      return MutateBinary<test::TestFloorMod>(op, value);
    }
    return value;
  }

 private:
  template <typename TRef, typename TNode>
  test::TestExpr MutateBinary(const TNode* op, const test::TestExpr& original) {
    test::TestExpr a = Mutate(op->a);
    test::TestExpr b = Mutate(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) return original;
    return TRef(std::move(a), std::move(b));
  }
  test::TestVar target_;
  test::TestExpr replacement_;
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

void Run(const char* name, const test::TestExpr& root, const test::TestVar& target,
         const test::TestExpr& replacement, int repeats) {
  test::TestExpr stable_owner = root;
  size_t walk_nodes = CountWalkNodes(root);
  size_t mutation_nodes = CountOccurrences(root);
  size_t map_nodes = CountMapNodes(root);
  auto old_walk = [&] {
    OldVisitor visitor;
    visitor.Visit(root);
  };
  auto structural_walk = [&] {
#ifdef TVM_FFI_BENCHMARK_WALK_DEDUP_TEMPLATE
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder, true>(
        root, [](const test::TestVar&) -> ffi::WalkResult {
          ++sink;
          return ffi::WalkResult::Advance();
        });
#else
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(root,
                                                    [](const test::TestVar&) -> ffi::WalkResult {
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
                [](const test::TestVarObj* node) -> ffi::Any {
                  return ffi::Any(ffi::GetRef<test::TestVar>(node));
                })
                .type_index();
  };
  auto map_changed = [&] {
    sink += ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
                root,
                [&](const test::TestVar& value) -> ffi::Any {
                  return ffi::Any(value.same_as(target) ? replacement : test::TestExpr(value));
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
  test::RegisterStructuralMapBenchmarkTypes();
  test::TestVar fo("fo"), fi("fi"), replacement("replacement");
  int repeats = std::getenv("TVM_FFI_BENCH_REPEATS")
                    ? std::atoi(std::getenv("TVM_FFI_BENCH_REPEATS"))
                    : 200000;
  Run("base_shared", SplitBase(fo, fi, true), fo, replacement, repeats);
  Run("base_distinct", SplitBase(fo, fi, false), fo, replacement, repeats);
  Run("nested4_shared", SplitNested(fo, fi, 4, true), fo, replacement, repeats / 5);
  Run("nested4_distinct", SplitNested(fo, fi, 4, false), fo, replacement, repeats / 20);
  std::cout << "SINK " << sink << '\n';
}
