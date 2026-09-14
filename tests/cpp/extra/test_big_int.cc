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
#include <gtest/gtest.h>
#include <tvm/ffi/big_int.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/extra/structural_mutate.h>

#include <limits>
#include <sstream>
#include <utility>

namespace {
using namespace tvm::ffi;
TEST(BigIntExtra, JSONGraphRoundTrip) {
  BigInt wide = BigInt(1) << 100;
  for (const auto& [value, decimal] :
       {std::pair<BigInt, String>{wide, "1267650600228229401496703205376"},
        {-wide, "-1267650600228229401496703205376"},
        {BigInt(std::numeric_limits<int64_t>::max()) + 1, "9223372036854775808"},
        {BigInt(std::numeric_limits<int64_t>::min()) - 1, "-9223372036854775809"}}) {
    SCOPED_TRACE(decimal);
    json::Object expected{
        {"root_index", 0},
        {"nodes", json::Array{json::Object{{"type", "ffi.BigInt"}, {"data", decimal}}}}};
    auto graph = ToJSONGraph(value);
    EXPECT_TRUE(StructuralEqual()(graph, expected));
    Any restored = FromJSONGraph(json::Parse(json::Stringify(graph)));
    EXPECT_EQ(restored.type_index(), TypeIndex::kTVMFFIBigInt);
    EXPECT_EQ(restored.cast<BigInt>(), value);
  }
  Array<Any> nested{7, Map<String, Any>{{"values", Array<Any>{wide, -wide}}}};
  EXPECT_TRUE(
      StructuralEqual()(FromJSONGraph(json::Parse(json::Stringify(ToJSONGraph(nested)))), nested));
}

TEST(BigIntExtra, JSONGraphCanonicalInt64) {
  for (const auto& [decimal, value] :
       {std::pair<String, int64_t>{"0", 0},
        {"-0", 0},
        {"+0007", 7},
        {"9223372036854775807", std::numeric_limits<int64_t>::max()},
        {"-9223372036854775808", std::numeric_limits<int64_t>::min()}}) {
    SCOPED_TRACE(decimal);
    json::Object graph{
        {"root_index", 0},
        {"nodes", json::Array{json::Object{{"type", "ffi.BigInt"}, {"data", decimal}}}}};
    Any restored = FromJSONGraph(graph);
    EXPECT_EQ(restored.type_index(), TypeIndex::kTVMFFIInt);
    EXPECT_EQ(restored.cast<int64_t>(), value);
    json::Object expected{{"root_index", 0},
                          {"nodes", json::Array{json::Object{{"type", "int"}, {"data", value}}}}};
    EXPECT_TRUE(StructuralEqual()(ToJSONGraph(BigInt(value)), expected));
    EXPECT_EQ(FromJSONGraph(expected).cast<BigInt>(), value);
  }
}

TEST(BigIntExtra, JSONGraphInvalidData) {
  for (const Any& data : Array<Any>{"", "+", "-", "x", "12x", " 1", "1 ", "1.0", String("12\0x", 4),
                                    nullptr, true, 1, 1.0, json::Array{}, json::Object{}}) {
    SCOPED_TRACE(ReprPrint(data));
    json::Object graph{
        {"root_index", 0},
        {"nodes", json::Array{json::Object{{"type", "ffi.BigInt"}, {"data", data}}}}};
    EXPECT_THROW(FromJSONGraph(graph), Error);
  }
}

TEST(BigIntExtra, NumericValueAndImmutableCopy) {
  BigInt a = BigInt(1) << 255;
  BigInt b = BigInt(2) << 254;
  BigInt c = BigInt(1) << 256;
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));
  EXPECT_TRUE(RecursiveEq(a, b));
  EXPECT_FALSE(RecursiveEq(a, c));
  EXPECT_TRUE(RecursiveLt(a, c));
  BigInt next = a + 1;
  EXPECT_FALSE(StructuralEqual()(a, next));
  EXPECT_FALSE(RecursiveEq(a, next));
  EXPECT_TRUE(RecursiveLt(a, next));
  EXPECT_TRUE(RecursiveLt(-c, -a));
  EXPECT_TRUE(RecursiveLt(-a, b));
  EXPECT_TRUE(RecursiveGt(a, -b));
  EXPECT_EQ(RecursiveHash(a), RecursiveHash(b));
  Array<Any> nested{a, -c};
  Array<Any> equal{b, -c};
  EXPECT_TRUE(RecursiveEq(nested, equal));
  EXPECT_EQ(RecursiveHash(nested), RecursiveHash(equal));
  EXPECT_TRUE(DeepCopy(a).same_as(Any(a)));
  EXPECT_TRUE(RecursiveEq(DeepCopy(nested), nested));
  std::ostringstream stream;
  stream << a;
  EXPECT_EQ(ReprPrint(a), stream.str());
}
TEST(BigIntExtra, RecursiveOrderingAcrossRepresentations) {
  BigInt max = std::numeric_limits<int64_t>::max();
  BigInt min = std::numeric_limits<int64_t>::min();
  BigInt above = max + 1;
  BigInt below = min - 1;
  EXPECT_TRUE(RecursiveLt(max, above));
  EXPECT_TRUE(RecursiveGt(above, max));
  EXPECT_FALSE(RecursiveLt(above, max));
  EXPECT_TRUE(RecursiveLt(below, min));
  EXPECT_TRUE(RecursiveGt(min, below));
  EXPECT_FALSE(RecursiveLt(min, below));
  EXPECT_TRUE(RecursiveLt(below, 1));
  EXPECT_TRUE(RecursiveGt(1, below));
  EXPECT_TRUE(RecursiveLt(Array<Any>{7, max}, Array<Any>{7, above}));
  EXPECT_TRUE(RecursiveLt(Array<Any>{7, below}, Array<Any>{7, min}));
  EXPECT_FALSE(RecursiveEq(max, above));
  EXPECT_FALSE(RecursiveEq(below, min));
  EXPECT_THROW(RecursiveLt(1.0, above), Error);
  EXPECT_THROW(RecursiveLt(above, String("1")), Error);
  EXPECT_THROW(RecursiveLt(above, Array<Any>{1}), Error);
}
TEST(BigIntExtra, StructuralMutation) {
  BigInt value = BigInt(1) << 255;
  BigInt alias = value;
  auto never_matches = [](int64_t, StructuralMutatorObj*) -> Expected<Any> { return Any(); };
  using Mutator = StructuralMutateEngine<StructuralMapEngineBase, decltype(never_matches)>;
  StructuralMutator mutator(make_object<Mutator>(never_matches));
  EXPECT_TRUE(mutator->Mutate(AnyView(value)).IsUnchanged());
  EXPECT_TRUE(mutator->MaybeInplaceMutate(AnyView(value)).IsUnchanged());
  EXPECT_TRUE(StructuralMutateExpected(Any(value), never_matches).value().same_as(Any(value)));
  auto map_int = [](int64_t x) -> Expected<Any> { return Any(x + 1); };
  EXPECT_TRUE(StructuralMapExpected<WalkOrder::kPostOrder>(Any(value), map_int)
                  .value()
                  .same_as(Any(value)));
  Array<Any> nested{value, 1};
  auto result =
      StructuralMapExpected<WalkOrder::kPostOrder>(Any(nested), map_int).value().cast<Array<Any>>();
  EXPECT_TRUE(result[0].same_as(Any(alias)));
  EXPECT_EQ(result[1].cast<int64_t>(), 2);
  EXPECT_EQ(value, alias);
}
TEST(BigIntExtra, StructuralMapKeyHashDoesNotDependOnAliases) {
  BigInt a = BigInt(1) << 255;
  BigInt b = BigInt(2) << 254;
  BigInt c = BigInt(4) << 253;
  Array<Any> lhs{a, Map<Any, Any>{{a, 7}}};
  Array<Any> rhs{b, Map<Any, Any>{{c, 7}}};
  EXPECT_TRUE(StructuralEqual()(lhs, rhs));
  EXPECT_EQ(StructuralHash()(lhs), StructuralHash()(rhs));
  Map<Any, Any> first{{a, 7}, {-a, 8}};
  EXPECT_EQ(first[b].cast<int64_t>(), 7);
  Map<Any, Any> second{{-b, 8}, {c, 7}};
  EXPECT_TRUE(StructuralEqual()(first, second));
  EXPECT_EQ(StructuralHash()(first), StructuralHash()(second));
  EXPECT_NE(StructuralHash()(first), StructuralHash()(Map<Any, Any>{{a, 9}, {-a, 8}}));
}
}  // namespace
