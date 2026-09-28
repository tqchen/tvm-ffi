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
#include <tvm/ffi/reflection/native_function.h>

#include <optional>
#include <string>
#include <type_traits>

namespace {

using namespace tvm::ffi;
using tvm::ffi::reflection::NativeFunction;
using tvm::ffi::reflection::NativeFunctionView;

static_assert(sizeof(NativeFunctionView<Expected<int>(int)>) == sizeof(TVMFFIAny));
static_assert(sizeof(NativeFunction<Expected<int>(int)>) == sizeof(Any));
static_assert(!std::is_default_constructible_v<NativeFunctionView<Expected<int>(int)>>);
static_assert(!std::is_constructible_v<NativeFunctionView<Expected<int>(int)>, std::nullptr_t>);
static_assert(std::is_convertible_v<TypedFunction<Expected<int>(int)>&,
                                    NativeFunctionView<Expected<int>(int)>>);
static_assert(!std::is_constructible_v<NativeFunctionView<Expected<int>(int)>,
                                       TypedFunction<Expected<int>(int)>&&>);
static_assert(!std::is_constructible_v<NativeFunctionView<Expected<int>(int)>, AnyView>);
static_assert(std::is_same_v<decltype(std::declval<NativeFunctionView<Expected<int>(int)>>()(0)),
                             Expected<int>>);

Expected<int> IncrementHook(int value) noexcept { return value + 1; }
Expected<int> FailHook(int) noexcept { return Unexpected(Error("ValueError", "hook failed", "")); }

TEST(NativeFunctionView, NativeAndBorrowedPacked) {
  auto native = NativeFunctionView<Expected<int>(int)>::FromNative<&IncrementHook>();
  EXPECT_EQ(native(3).value(), 4);
  EXPECT_EQ(AnyView(native).type_index(), TypeIndex::kTVMFFIOpaquePtr);
  EXPECT_EQ(Any(native).cast<NativeFunctionView<Expected<int>(int)>>()(3).value(), 4);
  EXPECT_EQ(TypeTraits<NativeFunctionView<Expected<int>(int)>>::TypeStr().find("Variant<"), 0);
  EXPECT_NE(
      TypeTraits<NativeFunctionView<Expected<int>(int)>>::TypeSchema().find("\"type\":\"Variant\""),
      std::string::npos);

  TypedFunction<Expected<int>(int)> packed(
      Function::FromTyped([](int x) -> Expected<int> { return x + 2; }));
  auto* obj = packed.packed().get();
  auto before = obj->use_count();
  NativeFunctionView<Expected<int>(int)> borrowed = packed;
  EXPECT_EQ(obj->use_count(), before);
  EXPECT_EQ(borrowed(3).value(), 5);
}

TEST(NativeFunctionView, ErrorAndNullRejection) {
  auto native = NativeFunctionView<Expected<int>(int)>::FromNative<&FailHook>();
  auto failed = native(0);
  ASSERT_TRUE(failed.is_err());
  EXPECT_EQ(failed.error().kind(), "ValueError");

  Error packed_error("ValueError", "packed failure", "");
  TypedFunction<Expected<int>(int)> packed(Function::FromTyped(
      [packed_error](int) -> Expected<int> { return Unexpected(packed_error); }));
  auto packed_failed = NativeFunctionView<Expected<int>(int)>(packed)(0);
  ASSERT_TRUE(packed_failed.is_err());
  EXPECT_TRUE(packed_failed.error().same_as(packed_error));

  TypedFunction<Expected<void>()> packed_void(Function::FromTyped(
      []() -> Expected<void> { return Unexpected(Error("ValueError", "void failure", "")); }));
  auto void_failed = NativeFunctionView<Expected<void>()>(packed_void)();
  ASSERT_TRUE(void_failed.is_err());
  EXPECT_EQ(void_failed.error().message(), "void failure");

  TypedFunction<Expected<int>(int)> empty(nullptr);
  EXPECT_ANY_THROW(NativeFunctionView<Expected<int>(int)> invalid(empty));
  EXPECT_FALSE(Any(nullptr).try_cast<NativeFunctionView<Expected<int>(int)>>().has_value());
  EXPECT_FALSE(Any(nullptr).try_cast<NativeFunction<Expected<int>(int)>>().has_value());
  EXPECT_FALSE(Any(static_cast<void*>(nullptr))
                   .try_cast<NativeFunctionView<Expected<int>(int)>>()
                   .has_value());
  EXPECT_FALSE(
      Any(static_cast<void*>(nullptr)).try_cast<NativeFunction<Expected<int>(int)>>().has_value());
}

TEST(NativeFunctionView, OwningLifetimeAndAnyRoundtrip) {
  std::optional<NativeFunction<Expected<int>(int)>> owned;
  {
    TypedFunction<Expected<int>(int)> packed(
        Function::FromTyped([](int value) -> Expected<int> { return value + 2; }));
    auto* obj = packed.packed().get();
    auto before = obj->use_count();
    owned.emplace(NativeFunctionView<Expected<int>(int)>(packed));
    EXPECT_EQ(obj->use_count(), before + 1);
    NativeFunction<Expected<int>(int)> copy(*owned);
    EXPECT_EQ(obj->use_count(), before + 2);
    NativeFunction<Expected<int>(int)> moved(std::move(copy));
    EXPECT_EQ(moved(3).value(), 5);
  }
  EXPECT_EQ((*owned)(3).value(), 5);
  Any attr = *owned;
  EXPECT_EQ(attr.cast<NativeFunction<Expected<int>(int)>>()(3).value(), 5);
  Any moved_attr = std::move(*owned);
  EXPECT_EQ(moved_attr.cast<NativeFunction<Expected<int>(int)>>()(3).value(), 5);
  EXPECT_EQ(NativeFunction<Expected<int>(int)>::FromNative<&IncrementHook>()(3).value(), 4);
}

}  // namespace
