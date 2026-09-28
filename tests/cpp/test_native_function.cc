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
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

using namespace tvm::ffi;
using tvm::ffi::reflection::NativeFunction;
using tvm::ffi::reflection::NativeFunctionView;

static_assert(sizeof(NativeFunctionView<int(int)>) == sizeof(TVMFFIAny));
static_assert(sizeof(NativeFunction<int(int)>) == sizeof(Any));
static_assert(!std::is_default_constructible_v<NativeFunctionView<int(int)>>);
static_assert(!std::is_constructible_v<NativeFunctionView<int(int)>, std::nullptr_t>);
static_assert(std::is_convertible_v<TypedFunction<int(int)>&, NativeFunctionView<int(int)>>);
static_assert(
    std::is_convertible_v<TypedFunction<Expected<int>(int)>&, NativeFunctionView<int(int)>>);
static_assert(!std::is_constructible_v<NativeFunctionView<int(int)>, TypedFunction<int(int)>&&>);
static_assert(
    !std::is_constructible_v<NativeFunctionView<int(int)>, TypedFunction<Expected<int>(int)>&&>);
static_assert(!std::is_constructible_v<NativeFunctionView<int(int)>, AnyView>);
static_assert(std::is_same_v<decltype(std::declval<NativeFunctionView<int(int)>>()(0)), int>);
static_assert(std::is_same_v<decltype(std::declval<NativeFunctionView<int(int)>>().CallExpected(0)),
                             Expected<int>>);
static_assert(std::is_same_v<decltype(std::declval<NativeFunction<void(int)>>()(0)), void>);
static_assert(std::is_same_v<decltype(std::declval<NativeFunction<void(int)>>().CallExpected(0)),
                             Expected<void>>);

Expected<int> IncrementHook(int value) noexcept { return value + 1; }
Expected<int> FailHook(int) noexcept { return Unexpected(Error("ValueError", "hook failed", "")); }
int PlainHook(int value) noexcept { return value + 2; }
void VoidHook(int value) {
  if (value < 0) throw std::runtime_error("void failure");
}
Expected<void> ExpectedVoidHook(int value) noexcept {
  if (value < 0) return Unexpected(Error("ValueError", "expected void failure", ""));
  return Expected<void>();
}
const Error& OriginalError() {
  static Error error("ValueError", "original error", "original trace");
  return error;
}
int ThrowFFIError(int) { throw OriginalError(); }
int ThrowStdError(int) { throw std::runtime_error("standard failure"); }

TEST(NativeFunctionView, NativeAndBorrowedPacked) {
  auto native = NativeFunctionView<int(int)>::FromNative<&IncrementHook>();
  EXPECT_EQ(native(3), 4);
  EXPECT_EQ(native.CallExpected(3).value(), 4);
  EXPECT_EQ(AnyView(native).type_index(), TypeIndex::kTVMFFIOpaquePtr);
  EXPECT_EQ(Any(native).cast<NativeFunctionView<int(int)>>()(3), 4);
  EXPECT_EQ(TypeTraits<NativeFunctionView<int(int)>>::TypeStr().find("Variant<"), 0);
  EXPECT_NE(TypeTraits<NativeFunctionView<int(int)>>::TypeSchema().find("\"type\":\"Variant\""),
            std::string::npos);

  EXPECT_EQ(NativeFunctionView<int(int)>::FromNative<&PlainHook>()(3), 5);

  TypedFunction<int(int)> packed(Function::FromTyped([](int x) { return x + 2; }));
  auto* obj = packed.packed().get();
  auto before = obj->use_count();
  NativeFunctionView<int(int)> borrowed = packed;
  EXPECT_EQ(obj->use_count(), before);
  EXPECT_EQ(borrowed(3), 5);
  EXPECT_EQ(borrowed.CallExpected(3).value(), 5);
  EXPECT_EQ(NativeFunction<int(int)>(packed)(3), 5);
}

TEST(NativeFunctionView, ErrorAndNullRejection) {
  auto native = NativeFunctionView<int(int)>::FromNative<&FailHook>();
  auto failed = native.CallExpected(0);
  ASSERT_TRUE(failed.is_err());
  EXPECT_EQ(failed.error().kind(), "ValueError");
  EXPECT_THROW(native(0), Error);

  Error packed_error("ValueError", "packed failure", "");
  TypedFunction<Expected<int>(int)> packed(Function::FromTyped(
      [packed_error](int) -> Expected<int> { return Unexpected(packed_error); }));
  auto packed_failed = NativeFunctionView<int(int)>(packed).CallExpected(0);
  ASSERT_TRUE(packed_failed.is_err());
  EXPECT_TRUE(packed_failed.error().same_as(packed_error));

  TypedFunction<int(int)> packed_throw(
      Function::FromTyped([](int) -> int { throw Error("ValueError", "packed throw", ""); }));
  auto packed_raised = NativeFunctionView<int(int)>(packed_throw).CallExpected(0);
  ASSERT_TRUE(packed_raised.is_err());
  EXPECT_EQ(packed_raised.error().message(), "packed throw");

  TypedFunction<Expected<void>()> packed_void(Function::FromTyped(
      []() -> Expected<void> { return Unexpected(Error("ValueError", "void failure", "")); }));
  auto void_failed = NativeFunctionView<void()>(packed_void).CallExpected();
  ASSERT_TRUE(void_failed.is_err());
  EXPECT_EQ(void_failed.error().message(), "void failure");
  TypedFunction<void()> packed_void_ok(Function::FromTyped([]() {}));
  EXPECT_TRUE(NativeFunctionView<void()>(packed_void_ok).CallExpected().has_value());

  auto ffi_error = NativeFunctionView<int(int)>::FromNative<&ThrowFFIError>();
  EXPECT_TRUE(ffi_error.CallExpected(0).error().same_as(OriginalError()));
  try {
    ffi_error(0);
    FAIL() << "Expected ffi::Error";
  } catch (const Error& error) {
    EXPECT_TRUE(error.same_as(OriginalError()));
  }

  auto std_error = NativeFunctionView<int(int)>::FromNative<&ThrowStdError>();
  auto converted = std_error.CallExpected(0);
  ASSERT_TRUE(converted.is_err());
  EXPECT_EQ(converted.error().kind(), "InternalError");
  EXPECT_EQ(converted.error().message(), "standard failure");
  EXPECT_TRUE(converted.error().backtrace().empty());
  EXPECT_THROW(std_error(0), Error);

  auto void_native = NativeFunctionView<void(int)>::FromNative<&VoidHook>();
  EXPECT_TRUE(void_native.CallExpected(1).has_value());
  EXPECT_EQ(void_native.CallExpected(-1).error().kind(), "InternalError");
  auto void_expected = NativeFunctionView<void(int)>::FromNative<&ExpectedVoidHook>();
  EXPECT_TRUE(void_expected.CallExpected(1).has_value());
  EXPECT_EQ(void_expected.CallExpected(-1).error().kind(), "ValueError");

  TypedFunction<Expected<int>(int)> empty(nullptr);
  EXPECT_ANY_THROW(NativeFunctionView<int(int)> invalid(empty));
  EXPECT_FALSE(Any(nullptr).try_cast<NativeFunctionView<int(int)>>().has_value());
  EXPECT_FALSE(Any(nullptr).try_cast<NativeFunction<int(int)>>().has_value());
  EXPECT_FALSE(
      Any(static_cast<void*>(nullptr)).try_cast<NativeFunctionView<int(int)>>().has_value());
  EXPECT_FALSE(Any(static_cast<void*>(nullptr)).try_cast<NativeFunction<int(int)>>().has_value());
}

TEST(NativeFunctionView, OwningLifetimeAndAnyRoundtrip) {
  std::optional<NativeFunction<int(int)>> owned;
  {
    TypedFunction<Expected<int>(int)> packed(
        Function::FromTyped([](int value) -> Expected<int> { return value + 2; }));
    auto* obj = packed.packed().get();
    auto before = obj->use_count();
    owned.emplace(NativeFunctionView<int(int)>(packed));
    EXPECT_EQ(obj->use_count(), before + 1);
    NativeFunction<int(int)> copy(*owned);
    EXPECT_EQ(obj->use_count(), before + 2);
    NativeFunction<int(int)> moved(std::move(copy));
    EXPECT_EQ(moved(3), 5);
  }
  EXPECT_EQ((*owned)(3), 5);
  EXPECT_EQ(owned->CallExpected(3).value(), 5);
  Any attr = *owned;
  EXPECT_EQ(attr.cast<NativeFunction<int(int)>>()(3), 5);
  Any moved_attr = std::move(*owned);
  EXPECT_EQ(moved_attr.cast<NativeFunction<int(int)>>()(3), 5);
  auto native_owned = NativeFunction<int(int)>::FromNative<&IncrementHook>();
  EXPECT_EQ(native_owned(3), 4);
  EXPECT_EQ(native_owned.CallExpected(3).value(), 4);
  EXPECT_TRUE(NativeFunction<void(int)>::FromNative<&VoidHook>().CallExpected(1).has_value());
  EXPECT_EQ(
      NativeFunction<int(int)>::FromNative<&ThrowStdError>().CallExpected(0).error().message(),
      "standard failure");
}

}  // namespace
