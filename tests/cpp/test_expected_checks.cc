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
#include <tvm/ffi/expected.h>
#include <tvm/ffi/extra/structural_mutate.h>

#include <algorithm>
#include <iomanip>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using namespace tvm::ffi;

template <typename T>
class ExpectedChecks : public ::testing::Test {};
using ExpectedCheckTypes = ::testing::Types<int, UnchangedOr<Any>>;
TYPED_TEST_SUITE(ExpectedChecks, ExpectedCheckTypes, );

template <typename T>
Expected<T> RunCheck(int check, int x, int y, int* streamed) noexcept {
  switch (check) {
    case 0:
      TVM_FFI_RET_CHECK(x < y, ValueError) << " detail " << ++*streamed;
      break;
    case 1:
      TVM_FFI_RET_ICHECK(x < y) << " detail " << ++*streamed;
      break;
    case 2:
      TVM_FFI_RET_CHECK_EQ(x, y, ValueError) << " detail " << ++*streamed;
      break;
    case 3:
      TVM_FFI_RET_ICHECK_EQ(x, y) << " detail " << ++*streamed;
      break;
    case 4:
      TVM_FFI_RET_CHECK_NE(x, y, ValueError) << " detail " << ++*streamed;
      break;
    case 5:
      TVM_FFI_RET_ICHECK_NE(x, y) << " detail " << ++*streamed;
      break;
    case 6:
      TVM_FFI_RET_CHECK_LT(x, y, ValueError) << " detail " << ++*streamed;
      break;
    case 7:
      TVM_FFI_RET_ICHECK_LT(x, y) << " detail " << ++*streamed;
      break;
    case 8:
      TVM_FFI_RET_CHECK_LE(x, y, ValueError) << " detail " << ++*streamed;
      break;
    case 9:
      TVM_FFI_RET_ICHECK_LE(x, y) << " detail " << ++*streamed;
      break;
    case 10:
      TVM_FFI_RET_CHECK_GT(x, y, ValueError) << " detail " << ++*streamed;
      break;
    case 11:
      TVM_FFI_RET_ICHECK_GT(x, y) << " detail " << ++*streamed;
      break;
    case 12:
      TVM_FFI_RET_CHECK_GE(x, y, ValueError) << " detail " << ++*streamed;
      break;
    case 13:
      TVM_FFI_RET_ICHECK_GE(x, y) << " detail " << ++*streamed;
      break;
    default:
      return TVM_FFI_UNEXPECTED(TypeError) << "unexpected detail " << ++*streamed;
  }
  return T(42);
}

TYPED_TEST(ExpectedChecks, AllFormsReturnErrorsAndFallThroughOnSuccess) {
  struct CheckCase {
    bool passes[3];  // x < y, x == y, x > y
    const char* expression;
  };
  const CheckCase cases[] = {{{true, false, false}, "(x < y) is false"},
                             {{false, true, false}, "x == y"},
                             {{true, false, true}, "x != y"},
                             {{true, false, false}, "x < y"},
                             {{true, true, false}, "x <= y"},
                             {{false, false, true}, "x > y"},
                             {{false, true, true}, "x >= y"}};
  for (int check = 0; check < 14; ++check) {
    SCOPED_TRACE(check);
    const auto& test = cases[check / 2];
    for (int order = 0; order < 3; ++order) {
      SCOPED_TRACE(order);
      int streamed = 0;
      auto result = RunCheck<TypeParam>(check, 3, 4 - order, &streamed);
      ASSERT_EQ(result.is_ok(), test.passes[order]);
      EXPECT_EQ(streamed, test.passes[order] ? 0 : 1);
      if (result.is_ok()) {
        if constexpr (std::is_same_v<TypeParam, int>) {
          EXPECT_EQ(result.value(), 42);
        } else {
          EXPECT_EQ(std::move(result).value().ValueOrUnchanged(Any(0)).template cast<int>(), 42);
        }
      } else {
        EXPECT_EQ(result.error().kind(), check % 2 == 0 ? "ValueError" : "InternalError");
        EXPECT_NE(result.error().message().find(test.expression), std::string::npos);
        EXPECT_NE(result.error().message().find(" detail 1"), std::string::npos);
        if (check >= 2) {
          EXPECT_NE(result.error().message().find("(3 vs. " + std::to_string(4 - order) + ")"),
                    std::string::npos);
        }
      }
    }
  }
  int streamed = 0;
  auto unexpected = RunCheck<TypeParam>(14, 0, 0, &streamed);
  ASSERT_TRUE(unexpected.is_err());
  EXPECT_EQ(unexpected.error().kind(), "TypeError");
  EXPECT_EQ(unexpected.error().message(), "unexpected detail 1");
  EXPECT_EQ(streamed, 1);
}

TEST(ExpectedChecks, BuilderConversionsStreamingAndSourceLocation) {
  using Builder = details::UnexpectedBuilder;
  static_assert(std::is_convertible_v<Builder&&, Unexpected<Error>>);
  static_assert(std::is_convertible_v<Builder&&, Expected<int>>);
  static_assert(std::is_convertible_v<Builder&&, Expected<void>>);
  static_assert(std::is_convertible_v<Builder&&, Expected<UnchangedOr<Any>>>);
  static_assert(!std::is_convertible_v<Builder&, Unexpected<Error>>);
  static_assert(!std::is_convertible_v<Builder&, Expected<int>>);
  static_assert(!std::is_convertible_v<const Builder&, Expected<int>>);
  static_assert(std::is_same_v<decltype(std::declval<Builder>() << 1), Builder&&>);

  Unexpected<Error> raw = TVM_FFI_UNEXPECTED(ValueError) << std::hex << 31 << std::endl
                                                         << std::setw(3) << 1 << std::flush;
  EXPECT_EQ(raw.error().kind(), "ValueError");
  EXPECT_EQ(raw.error().message(), "1f\n  1");
  auto any = []() noexcept -> Expected<Any> { return TVM_FFI_UNEXPECTED(TypeError) << "any"; }();
  EXPECT_TRUE(any.is_err());
  auto empty = []() noexcept -> Expected<void> {
    TVM_FFI_RET_CHECK(false, ValueError);
    return {};
  }();
  EXPECT_TRUE(empty.is_err());
  int source_line = __LINE__ + 1;
  auto no_message = []() noexcept -> Expected<int> { return TVM_FFI_UNEXPECTED(ValueError); }();
  EXPECT_TRUE(no_message.is_err());
  EXPECT_EQ(no_message.error().message(), "");
  std::string backtrace = no_message.error().backtrace();
  EXPECT_NE(backtrace.find(__FILE__), std::string::npos);
  EXPECT_NE(backtrace.find("line " + std::to_string(source_line) + ", in "), std::string::npos);
  EXPECT_NE(backtrace.find("BuilderConversionsStreamingAndSourceLocation"), std::string::npos);
  EXPECT_EQ(std::count(backtrace.begin(), backtrace.end(), '\n'), 1);
}

TEST(ExpectedChecks, ControlFlowAndSingleEvaluation) {
  // These intentionally unbraced bodies ensure the macro cannot capture the user's else.
  // NOLINTBEGIN(google-readability-braces-around-statements)
  int condition_calls = 0;
  auto conditional = [&](bool outer, bool inner) noexcept -> Expected<int> {
    if (outer)
      TVM_FFI_RET_ICHECK((++condition_calls, inner));
    else
      return 2;
    return 1;
  };
  EXPECT_EQ(conditional(false, false).value(), 2);
  EXPECT_EQ(conditional(false, true).value(), 2);
  EXPECT_EQ(condition_calls, 0);
  EXPECT_EQ(conditional(true, true).value(), 1);
  EXPECT_EQ(condition_calls, 1);
  EXPECT_TRUE(conditional(true, false).is_err());
  EXPECT_EQ(condition_calls, 2);
  int left_calls = 0;
  int right_calls = 0;
  auto binary = [&](bool outer, int x) noexcept -> Expected<int> {
    if (outer)
      TVM_FFI_RET_ICHECK_EQ((++left_calls, x), (++right_calls, 3));
    else
      return 2;
    return 1;
  };
  EXPECT_EQ(binary(false, 4).value(), 2);
  EXPECT_EQ(binary(false, 3).value(), 2);
  EXPECT_EQ(left_calls, 0);
  EXPECT_EQ(right_calls, 0);
  EXPECT_EQ(binary(true, 3).value(), 1);
  EXPECT_EQ(left_calls, 1);
  EXPECT_EQ(right_calls, 1);
  EXPECT_TRUE(binary(true, 4).is_err());
  EXPECT_EQ(left_calls, 2);
  EXPECT_EQ(right_calls, 2);
  auto loop = [](int limit) noexcept -> Expected<int> {
    for (int i = 0; i < limit; ++i) TVM_FFI_RET_CHECK_LT(i, 3, ValueError);
    return 1;
  };
  EXPECT_EQ(loop(3).value(), 1);
  EXPECT_TRUE(loop(4).is_err());
  // NOLINTEND(google-readability-braces-around-statements)
}

template <typename Return>
Return UnsafeMutateAssign(Expected<Any> result, int* evaluations) noexcept {
  TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN(int, value, (++*evaluations, std::move(result)));
  if constexpr (std::is_same_v<Return, TVMFFIAny>) {
    return AnyView(value + 1).CopyToTVMFFIAny();
  } else {
    return value + 1;
  }
}

TEST(ExpectedChecks, UnsafeStructuralAssignUnwrapsOrReturnsError) {
  int evaluations = 0;
  EXPECT_EQ(UnsafeMutateAssign<Expected<int>>(Any(41), &evaluations).value(), 42);
  auto raw_ok = details::ExpectedUnsafe::MoveFromTVMFFIAny<int>(
      UnsafeMutateAssign<TVMFFIAny>(Any(41), &evaluations));
  EXPECT_EQ(raw_ok.value(), 42);
  Error error("ValueError", "unsafe assign", "");
  auto typed_error = UnsafeMutateAssign<Expected<int>>(error, &evaluations);
  ASSERT_TRUE(typed_error.is_err());
  EXPECT_EQ(typed_error.error().message(), "unsafe assign");
  auto raw_error = details::ExpectedUnsafe::MoveFromTVMFFIAny<int>(
      UnsafeMutateAssign<TVMFFIAny>(error, &evaluations));
  ASSERT_TRUE(raw_error.is_err());
  EXPECT_EQ(raw_error.error().message(), "unsafe assign");
  EXPECT_EQ(evaluations, 4);
}

}  // namespace
