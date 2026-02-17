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
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>

#include "./testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(Dict, Basic) {
  Dict<TInt, int> dict0;
  TInt k0(0);
  dict0.Set(k0, 1);

  EXPECT_EQ(dict0.size(), 1);

  dict0.Set(k0, 2);
  EXPECT_EQ(dict0.size(), 1);

  auto it = dict0.find(k0);
  EXPECT_TRUE(it != dict0.end());
  EXPECT_EQ((*it).second, 2);
}

TEST(Dict, PODKey) {
  Dict<Any, Any> dict0;

  // int as key
  dict0.Set(1, 2);
  // float key is different
  dict0.Set(1.1, 3);
  EXPECT_EQ(dict0.size(), 2);

  auto it = dict0.find(1.1);
  EXPECT_TRUE(it != dict0.end());
  EXPECT_EQ((*it).second.cast<int>(), 3);
}

TEST(Dict, Object) {
  TInt x(1);
  TInt z(100);
  TInt zz(1000);
  Dict<TInt, TInt> dict{{x, z}, {z, zz}};
  EXPECT_EQ(dict.size(), 2);
  EXPECT_TRUE(dict[x].same_as(z));
  EXPECT_TRUE(dict.count(z));
  EXPECT_TRUE(!dict.count(zz));
}

TEST(Dict, Str) {
  TInt z(100);
  Dict<String, TInt> dict{{"x", z}, {"z", z}};
  EXPECT_EQ(dict.size(), 2);
  EXPECT_TRUE(dict["x"].same_as(z));
}

TEST(Dict, SharedMutation) {
  // Dict is mutable and does NOT use copy-on-write.
  // When two Dict variables share the same underlying object,
  // in-place mutations (update existing keys) through one
  // are visible through the other.
  TInt x(1);
  TInt z(100);
  TInt zz(1000);
  Dict<TInt, TInt> dict{{x, z}, {z, zz}};

  auto dict2 = dict;
  EXPECT_TRUE(dict2.same_as(dict));  // same underlying object

  // Update an existing key - mutation is visible through dict2
  dict.Set(x, zz);
  EXPECT_TRUE(dict2[x].same_as(zz));
  EXPECT_EQ(dict2.size(), 2);
}

TEST(Dict, Clear) {
  TInt x(1);
  TInt z(100);
  Dict<TInt, TInt> dict{{x, z}, {z, z}};
  EXPECT_EQ(dict.size(), 2);
  dict.clear();
  EXPECT_EQ(dict.size(), 0);
}

TEST(Dict, Insert) {
  auto check = [](const Dict<String, int64_t>& result,
                  std::unordered_map<std::string, int64_t> expected) {
    EXPECT_EQ(result.size(), expected.size());
    for (const auto& kv : result) {
      EXPECT_TRUE(expected.count(kv.first));
      EXPECT_EQ(expected[kv.first], kv.second);
      expected.erase(kv.first);
    }
  };
  Dict<String, int64_t> result;
  std::unordered_map<std::string, int64_t> expected;
  char key = 'a';
  int64_t val = 1;
  for (int i = 0; i < 26; ++i, ++key, ++val) {
    std::string s(1, key);
    result.Set(s, val);
    expected[s] = val;
    check(result, expected);
  }
}

TEST(Dict, Erase) {
  auto check = [](const Dict<String, int64_t>& result,
                  std::unordered_map<std::string, int64_t> expected) {
    EXPECT_EQ(result.size(), expected.size());
    for (const auto& kv : result) {
      EXPECT_TRUE(expected.count(kv.first));
      EXPECT_EQ(expected[kv.first], kv.second);
      expected.erase(kv.first);
    }
  };
  std::unordered_map<std::string, int64_t> stl{{"a", 1}, {"b", 2}, {"c", 3}, {"d", 4}, {"e", 5}};
  for (char c = 'a'; c <= 'e'; ++c) {
    // Recreate dict each iteration since Dict is mutable (no COW).
    Dict<String, int64_t> dict{{"a", 1}, {"b", 2}, {"c", 3}, {"d", 4}, {"e", 5}};
    std::unordered_map<std::string, int64_t> expected(stl);
    std::string key(1, c);
    dict.erase(key);
    expected.erase(key);
    check(dict, expected);
  }
}

TEST(Dict, TypeIndex) {
  // Verify that Dict objects get the correct type index
  Dict<String, int64_t> dict{{"a", 1}};
  Any any_dict = dict;
  EXPECT_EQ(any_dict.type_index(), TypeIndex::kTVMFFIDict);

  // Map objects should still get kTVMFFIMap
  Map<String, int64_t> map{{"a", 1}};
  Any any_map = map;
  EXPECT_EQ(any_map.type_index(), TypeIndex::kTVMFFIMap);
}

TEST(Dict, CrossConversion) {
  // Map can be cross-converted to Dict and vice versa
  Map<String, int64_t> map{{"a", 1}, {"b", 2}};
  Any any_map = map;

  // Dict accepts Map via cross-conversion (creates a copy)
  auto dict = any_map.cast<Dict<String, int64_t>>();
  EXPECT_FALSE(dict.same_as(map));
  EXPECT_EQ(dict["a"], 1);
  EXPECT_EQ(dict["b"], 2);

  // Map accepts Dict via cross-conversion (creates a copy)
  Dict<String, int64_t> dict2{{"x", 10}, {"y", 20}};
  Any any_dict = dict2;
  auto map2 = any_dict.cast<Map<String, int64_t>>();
  EXPECT_FALSE(map2.same_as(dict2));
  EXPECT_EQ(map2["x"], 10);
  EXPECT_EQ(map2["y"], 20);
}

TEST(Dict, AnyImplicitConversion) {
  Dict<Any, Any> dict0;
  dict0.Set(1, 2);
  dict0.Set(2, 3.1);
  EXPECT_EQ(dict0.size(), 2);

  AnyView view0 = dict0;
  auto dict1 = view0.cast<Dict<int, double>>();
  EXPECT_TRUE(!dict1.same_as(dict0));
  EXPECT_EQ(dict1[1], 2);
  EXPECT_EQ(dict1[2], 3.1);

  auto dict2 = view0.cast<Dict<int, Any>>();
  EXPECT_TRUE(dict2.same_as(dict0));

  auto dict3 = view0.cast<Dict<Any, double>>();
  EXPECT_TRUE(!dict3.same_as(dict0));
}

TEST(Dict, FunctionGetItem) {
  Function f = Function::FromTyped(
      [](const MapBaseObj* n, const Any& k) -> Any { return n->at(k); }, "dict_get_item");
  Dict<String, int64_t> dict{{"x", 1}, {"y", 2}};
  Any k("x");
  Any v = f(dict, k);
  EXPECT_EQ(v.cast<int>(), 1);
}

TEST(Dict, Upcast) {
  Dict<int, int> d0 = {{1, 2}, {3, 4}};
  Dict<Any, Any> d1 = d0;
  EXPECT_EQ(d1[1].cast<int>(), 2);
  EXPECT_EQ(d1[3].cast<int>(), 4);
  static_assert(details::type_contains_v<Dict<Any, Any>, Dict<String, int>>);
}

TEST(Dict, InsertOrder) {
  // test that dict preserves the insertion order
  auto get_reverse_order = [](size_t size) {
    std::vector<int> reverse_order;
    for (int i = static_cast<int>(size); i != 0; --i) {
      reverse_order.push_back(i - 1);
    }
    return reverse_order;
  };

  auto check_dict = [&](Dict<String, int> d0, size_t size, const std::vector<int>& order) {
    auto lhs = d0.begin();
    auto rhs = order.begin();
    while (lhs != d0.end()) {
      TVM_FFI_ICHECK_EQ((*lhs).first, "hello" + std::to_string(*rhs));
      TVM_FFI_ICHECK_EQ((*lhs).second, *rhs);
      ++lhs;
      ++rhs;
    }
  };

  auto check_order = [&](std::vector<int> order) {
    Dict<String, int> d0;
    for (size_t i = 0; i < order.size(); ++i) {
      d0.Set("hello" + std::to_string(order[i]), order[i]);
      check_dict(d0, i + 1, order);
    }
    check_dict(d0, order.size(), order);
  };
  // test with 17 items: DenseMapObj
  check_order(get_reverse_order(17));
  // test with 4 items: SmallMapObj
  check_order(get_reverse_order(4));
}

TEST(Dict, EmptyIter) {
  Dict<String, int> d0;
  EXPECT_EQ(d0.begin(), d0.end());
  // create a big dict and then erase to keep a dense map empty
  for (int i = 0; i < 10; ++i) {
    d0.Set("hello" + std::to_string(i), i);
  }
  for (int i = 0; i < 10; ++i) {
    d0.erase("hello" + std::to_string(i));
  }
  EXPECT_EQ(d0.size(), 0);
  EXPECT_EQ(d0.begin(), d0.end());
}

TEST(Dict, DuplicatedKeysInit) {
  std::vector<std::pair<String, int>> data = {{"a", 1}, {"a", 2}, {"a", 3}};
  Dict<String, int> dict(data.begin(), data.end());
  EXPECT_EQ(dict.size(), 1);
  EXPECT_EQ(dict["a"], 3);
}

TEST(Dict, TypeIndexPreservedThroughRehash) {
  // Ensure the type_index stays kTVMFFIDict through Small→Dense transition
  Dict<String, int64_t> dict;
  for (int i = 0; i < 20; ++i) {
    dict.Set("key" + std::to_string(i), i);
  }
  Any any_dict = dict;
  EXPECT_EQ(any_dict.type_index(), TypeIndex::kTVMFFIDict);
  EXPECT_EQ(dict.size(), 20);
  EXPECT_EQ(dict["key0"], 0);
  EXPECT_EQ(dict["key19"], 19);
}

}  // namespace
