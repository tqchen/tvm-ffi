
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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include "../testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;
namespace refl = tvm::ffi::reflection;

TEST(StructuralEqualHash, Array) {
  Array<int> a = {1, 2, 3};
  Array<int> b = {1, 2, 3};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  Array<int> c = {1, 3};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));
  auto diff_a_c = StructuralEqual::GetFirstMismatch(a, c);

  // first directly interepret diff,
  EXPECT_TRUE(diff_a_c.has_value());
  auto lhs_steps = (*diff_a_c).get<0>()->ToSteps();
  auto rhs_steps = (*diff_a_c).get<1>()->ToSteps();
  EXPECT_EQ(lhs_steps[0]->kind, refl::AccessKind::kArrayItem);
  EXPECT_EQ(rhs_steps[0]->kind, refl::AccessKind::kArrayItem);
  EXPECT_EQ(lhs_steps[0]->key.cast<int64_t>(), 1);
  EXPECT_EQ(rhs_steps[0]->key.cast<int64_t>(), 1);
  EXPECT_EQ(lhs_steps.size(), 1);
  EXPECT_EQ(rhs_steps.size(), 1);

  // use structural equal for checking in future parts
  // given we have done some basic checks above by directly interepret diff,
  Array<int> d = {1, 2};
  auto diff_a_d = StructuralEqual::GetFirstMismatch(a, d);
  auto expected_diff_a_d = refl::AccessPathPair(refl::AccessPath::FromSteps({
                                                    refl::AccessStep::ArrayItem(2),
                                                }),
                                                refl::AccessPath::FromSteps({
                                                    refl::AccessStep::ArrayItemMissing(2),
                                                }));
  // then use structural equal to check it
  EXPECT_TRUE(StructuralEqual()(diff_a_d, expected_diff_a_d));
}

TEST(StructuralEqualHash, Map) {
  // same map but different insertion order
  Map<String, int> a = {{"a", 1}, {"b", 2}, {"c", 3}};
  Map<String, int> b = {{"b", 2}, {"c", 3}, {"a", 1}};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  Map<String, int> c = {{"a", 1}, {"b", 2}, {"c", 4}};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));

  auto diff_a_c = StructuralEqual::GetFirstMismatch(a, c);
  auto expected_diff_a_c = refl::AccessPathPair(refl::AccessPath::Root()->MapItem("c"),
                                                refl::AccessPath::Root()->MapItem("c"));
  EXPECT_TRUE(diff_a_c.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_c, expected_diff_a_c));
}

TEST(StructuralEqualHash, NestedMapArray) {
  Map<String, Array<Any>> a = {{"a", {1, 2, 3}}, {"b", {4, "hello", 6}}};
  Map<String, Array<Any>> b = {{"a", {1, 2, 3}}, {"b", {4, "hello", 6}}};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  Map<String, Array<Any>> c = {{"a", {1, 2, 3}}, {"b", {4, "world", 6}}};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));

  auto diff_a_c = StructuralEqual::GetFirstMismatch(a, c);
  auto expected_diff_a_c =
      refl::AccessPathPair(refl::AccessPath::Root()->MapItem("b")->ArrayItem(1),
                           refl::AccessPath::Root()->MapItem("b")->ArrayItem(1));
  EXPECT_TRUE(diff_a_c.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_c, expected_diff_a_c));

  Map<String, Array<Any>> d = {{"a", {1, 2, 3}}};
  auto diff_a_d = StructuralEqual::GetFirstMismatch(a, d);
  auto expected_diff_a_d = refl::AccessPathPair(refl::AccessPath::Root()->MapItem("b"),
                                                refl::AccessPath::Root()->MapItemMissing("b"));
  EXPECT_TRUE(diff_a_d.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_d, expected_diff_a_d));

  auto diff_d_a = StructuralEqual::GetFirstMismatch(d, a);
  auto expected_diff_d_a = refl::AccessPathPair(refl::AccessPath::Root()->MapItemMissing("b"),
                                                refl::AccessPath::Root()->MapItem("b"));
}

TEST(StructuralEqualHash, Dict) {
  // same dict but different insertion order
  Dict<String, int> a = {{"a", 1}, {"b", 2}, {"c", 3}};
  Dict<String, int> b = {{"b", 2}, {"c", 3}, {"a", 1}};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  Dict<String, int> c = {{"a", 1}, {"b", 2}, {"c", 4}};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));

  auto diff_a_c = StructuralEqual::GetFirstMismatch(a, c);
  auto expected_diff_a_c = refl::AccessPathPair(refl::AccessPath::Root()->MapItem("c"),
                                                refl::AccessPath::Root()->MapItem("c"));
  EXPECT_TRUE(diff_a_c.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_c, expected_diff_a_c));
}

TEST(StructuralEqualHash, NestedDictArray) {
  Dict<String, Array<Any>> a = {{"a", {1, 2, 3}}, {"b", {4, "hello", 6}}};
  Dict<String, Array<Any>> b = {{"a", {1, 2, 3}}, {"b", {4, "hello", 6}}};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  Dict<String, Array<Any>> c = {{"a", {1, 2, 3}}, {"b", {4, "world", 6}}};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));

  auto diff_a_c = StructuralEqual::GetFirstMismatch(a, c);
  auto expected_diff_a_c =
      refl::AccessPathPair(refl::AccessPath::Root()->MapItem("b")->ArrayItem(1),
                           refl::AccessPath::Root()->MapItem("b")->ArrayItem(1));
  EXPECT_TRUE(diff_a_c.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_c, expected_diff_a_c));

  Dict<String, Array<Any>> d = {{"a", {1, 2, 3}}};
  auto diff_a_d = StructuralEqual::GetFirstMismatch(a, d);
  auto expected_diff_a_d = refl::AccessPathPair(refl::AccessPath::Root()->MapItem("b"),
                                                refl::AccessPath::Root()->MapItemMissing("b"));
  EXPECT_TRUE(diff_a_d.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_d, expected_diff_a_d));
}

TEST(StructuralEqualHash, DictVsMapDifferentType) {
  Map<String, int> m = {{"a", 1}, {"b", 2}};
  Dict<String, int> d = {{"a", 1}, {"b", 2}};
  // Different type_index => not equal
  EXPECT_FALSE(StructuralEqual()(m, d));
  // Different type_key_hash => different hash (very likely)
  EXPECT_NE(StructuralHash()(m), StructuralHash()(d));
}

TEST(StructuralEqualHash, FreeVar) {
  TVar a = TVar("a");
  TVar b = TVar("b");
  EXPECT_TRUE(StructuralEqual::Equal(a, b, /*map_free_vars=*/true));
  EXPECT_FALSE(StructuralEqual::Equal(a, b));

  EXPECT_NE(StructuralHash()(a), StructuralHash()(b));
  EXPECT_EQ(StructuralHash::Hash(a, /*map_free_vars=*/true),
            StructuralHash::Hash(b, /*map_free_vars=*/true));
}

TEST(StructuralEqualHash, FuncDefAndIgnoreField) {
  TVar x = TVar("x");
  TVar y = TVar("y");
  // comment fields are ignored
  TFunc fa = TFunc({x}, {TInt(1), x}, String("comment a"));
  TFunc fb = TFunc({y}, {TInt(1), y}, String("comment b"));

  TFunc fc = TFunc({x}, {TInt(1), TInt(2)}, String("comment c"));

  EXPECT_TRUE(StructuralEqual()(fa, fb));
  EXPECT_EQ(StructuralHash()(fa), StructuralHash()(fb));

  EXPECT_FALSE(StructuralEqual()(fa, fc));
  auto diff_fa_fc = StructuralEqual::GetFirstMismatch(fa, fc);
  auto expected_diff_fa_fc = refl::AccessPathPair(refl::AccessPath::FromSteps({
                                                      refl::AccessStep::Attr("body"),
                                                      refl::AccessStep::ArrayItem(1),
                                                  }),
                                                  refl::AccessPath::FromSteps({
                                                      refl::AccessStep::Attr("body"),
                                                      refl::AccessStep::ArrayItem(1),
                                                  }));
  EXPECT_TRUE(diff_fa_fc.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_fa_fc, expected_diff_fa_fc));
}

TEST(StructuralEqualHash, CustomTreeNode) {
  TVar x = TVar("x");
  TVar y = TVar("y");
  // comment fields are ignored
  TCustomFunc fa = TCustomFunc({x}, {TInt(1), x}, "comment a");
  TCustomFunc fb = TCustomFunc({y}, {TInt(1), y}, "comment b");

  TCustomFunc fc = TCustomFunc({x}, {TInt(1), TInt(2)}, "comment c");

  EXPECT_TRUE(StructuralEqual()(fa, fb));
  EXPECT_EQ(StructuralHash()(fa), StructuralHash()(fb));

  EXPECT_FALSE(StructuralEqual()(fa, fc));
  auto diff_fa_fc = StructuralEqual::GetFirstMismatch(fa, fc);
  auto expected_diff_fa_fc =
      refl::AccessPathPair(refl::AccessPath::Root()->Attr("body")->ArrayItem(1),
                           refl::AccessPath::Root()->Attr("body")->ArrayItem(1));
  EXPECT_TRUE(diff_fa_fc.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_fa_fc, expected_diff_fa_fc));
}

TEST(StructuralEqualHash, List) {
  List<int> a = {1, 2, 3};
  List<int> b = {1, 2, 3};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  List<int> c = {1, 3};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));
}

TEST(StructuralEqualHash, ListVsArrayDifferentType) {
  Array<int> arr = {1, 2, 3};
  List<int> lst = {1, 2, 3};
  // Different type_index => not equal
  EXPECT_FALSE(StructuralEqual()(arr, lst));
  // Different type_key_hash => different hash (very likely)
  EXPECT_NE(StructuralHash()(arr), StructuralHash()(lst));
}

TEST(StructuralEqualHash, DISABLED_ListCycleDetection) {
  List<Any> lst;
  lst.push_back(42);
  lst.push_back(lst);  // creates a cycle
  EXPECT_ANY_THROW(StructuralHash()(lst));
  EXPECT_ANY_THROW(StructuralEqual()(lst, lst));
}

TEST(StructuralEqualHash, ArraySelfInsertProducesSnapshot) {
  Array<Any> arr;
  arr.push_back(arr);

  Array<Any> snapshot = arr[0].cast<Array<Any>>();
  EXPECT_TRUE(snapshot.empty());
  EXPECT_FALSE(snapshot.same_as(arr));

  EXPECT_TRUE(StructuralEqual()(arr, arr));
  EXPECT_EQ(StructuralHash()(arr), StructuralHash()(arr));
}

// Test the FFI global "ffi.StructuralEqual" — the cheap boolean path that
// avoids constructing AccessPath mismatch pairs (unlike GetFirstMismatch).
TEST(StructuralEqualHash, FFIGlobalStructuralEqual) {
  Function ffi_equal = Function::GetGlobalRequired("ffi.StructuralEqual");
  Function ffi_mismatch = Function::GetGlobalRequired("ffi.GetFirstStructuralMismatch");

  // Equal arrays: both the FFI global and GetFirstMismatch must agree.
  Array<int> a = {1, 2, 3};
  Array<int> b = {1, 2, 3};
  EXPECT_TRUE(ffi_equal(a, b, false, false).cast<bool>());
  EXPECT_FALSE(
      ffi_mismatch(a, b, false, false).cast<Optional<reflection::AccessPathPair>>().has_value());

  // Unequal arrays: FFI global returns false; GetFirstMismatch returns a value.
  Array<int> c = {1, 2, 4};
  EXPECT_FALSE(ffi_equal(a, c, false, false).cast<bool>());
  EXPECT_TRUE(
      ffi_mismatch(a, c, false, false).cast<Optional<reflection::AccessPathPair>>().has_value());

  // Consistency: bool result from FFI global matches (mismatch is None) for equal objects.
  Array<int> d = {10, 20};
  Array<int> e = {10, 20};
  bool equal_result = ffi_equal(d, e, false, false).cast<bool>();
  bool no_mismatch =
      !ffi_mismatch(d, e, false, false).cast<Optional<reflection::AccessPathPair>>().has_value();
  EXPECT_EQ(equal_result, no_mismatch);
}

}  // namespace
