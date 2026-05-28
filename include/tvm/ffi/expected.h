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
 * \file tvm/ffi/expected.h
 * \brief Runtime Expected container type for exception-free error handling.
 */
#ifndef TVM_FFI_EXPECTED_H_
#define TVM_FFI_EXPECTED_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/error.h>

#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief Wrapper to explicitly construct an Expected in the error state.
 * \tparam E The error type, must derive from Error.
 */
template <typename E = Error>
class Unexpected {
  static_assert(std::is_base_of_v<Error, std::remove_cv_t<E>>,
                "Unexpected<E> requires E to be Error or a subclass of Error.");

 public:
  /*! \brief Construct from an error value. */
  explicit Unexpected(E error) : error_(std::move(error)) {}

  /*! \brief Access the stored error. */
  const E& error() const& noexcept { return error_; }
  /*! \brief Access the stored error. */
  E& error() & noexcept { return error_; }
  /*! \brief Access the stored error (rvalue). */
  const E&& error() const&& noexcept { return std::move(error_); }
  /*! \brief Access the stored error (rvalue). */
  E&& error() && noexcept { return std::move(error_); }

 private:
  E error_;
};

#ifndef TVM_FFI_DOXYGEN_MODE
template <typename E>
Unexpected(E) -> Unexpected<E>;
#endif

/*!
 * \brief Expected<T> provides exception-free error handling for FFI functions.
 *
 * Expected<T> is similar to Rust's Result<T, Error> or C++23's std::expected.
 * It can hold either a success value of type T or an error of type Error.
 *
 * \tparam T The success type. Must be Any-compatible and cannot be Error.
 *
 * Usage:
 * \code
 * Expected<int> divide(int a, int b) {
 *   if (b == 0) {
 *     return Error("ValueError", "Division by zero");
 *   }
 *   return a / b;
 * }
 *
 * Expected<int> result = divide(10, 2);
 * if (result.is_ok()) {
 *   int value = result.value();
 * } else {
 *   Error err = result.error();
 * }
 * \endcode
 */
template <typename T>
class Expected {
 public:
  static_assert(!std::is_same_v<T, Error>, "Expected<Error> is not allowed. Use Error directly.");

  /*!
   * \brief Implicit constructor from a success value.
   * \param value The success value.
   */
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  Expected(T value) : data_(Any(std::move(value))) {}

  /*!
   * \brief Implicit constructor from an error.
   * \param error The error value.
   */
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  Expected(Error error) : data_(Any(std::move(error))) {}

  /*! \brief Implicit constructor from an Unexpected wrapper. */
  template <typename E, typename = std::enable_if_t<std::is_base_of_v<Error, std::remove_cv_t<E>>>>
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  Expected(Unexpected<E> unexpected) : data_(Any(std::move(unexpected).error())) {}

  /*! \brief Returns true if the Expected contains a success value. */
  TVM_FFI_INLINE bool is_ok() const noexcept {
    return data_.type_index() != TypeIndex::kTVMFFIError;
  }

  /*! \brief Returns true if the Expected contains an error. */
  TVM_FFI_INLINE bool is_err() const noexcept {
    return data_.type_index() == TypeIndex::kTVMFFIError;
  }

  /*! \brief Alias for is_ok(). */
  TVM_FFI_INLINE bool has_value() const noexcept { return is_ok(); }

  /*! \brief Returns the success value, or throws the contained error. */
  TVM_FFI_INLINE T value() const& {
    if (TVM_FFI_PREDICT_TRUE(is_ok())) {
      return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(data_);
    }
    throw details::AnyUnsafe::CopyFromAnyViewAfterCheck<Error>(data_);
  }

  /*! \brief Returns the success value (moved out), or throws the contained error. */
  TVM_FFI_INLINE T value() && {
    if (TVM_FFI_PREDICT_TRUE(is_ok())) {
      return details::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
    }
    throw details::AnyUnsafe::MoveFromAnyAfterCheck<Error>(std::move(data_));
  }

  /*! \brief Returns the contained error, or throws RuntimeError if is_ok(). */
  TVM_FFI_INLINE Error error() const& {
    // No branch hint: error() is itself a cold path — callers only invoke it
    // after observing !is_ok(), so the branch direction here doesn't matter.
    if (is_ok()) {
      TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
    }
    return details::AnyUnsafe::CopyFromAnyViewAfterCheck<Error>(data_);
  }

  /*! \brief Returns the contained error (moved out), or throws RuntimeError if is_ok(). */
  TVM_FFI_INLINE Error error() && {
    // No branch hint: error() is itself a cold path — callers only invoke it
    // after observing !is_ok(), so the branch direction here doesn't matter.
    if (is_ok()) {
      TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
    }
    return details::AnyUnsafe::MoveFromAnyAfterCheck<Error>(std::move(data_));
  }

  /*!
   * \brief Returns the success value, or \p default_value if the Expected holds an error.
   */
  template <typename U = std::remove_cv_t<T>>
  TVM_FFI_INLINE T value_or(U&& default_value) const& {
    if (TVM_FFI_PREDICT_TRUE(is_ok())) {
      return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(data_);
    }
    return T(std::forward<U>(default_value));
  }

  /*!
   * \brief Returns the success value (moved out), or \p default_value if the Expected holds an
   * error.
   */
  template <typename U = std::remove_cv_t<T>>
  TVM_FFI_INLINE T value_or(U&& default_value) && {
    if (TVM_FFI_PREDICT_TRUE(is_ok())) {
      return details::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
    }
    return T(std::forward<U>(default_value));
  }

 private:
  Any data_;  // Invariant: holds a T (type_index != kTVMFFIError) or an Error.
};

// TypeTraits specialization for Expected<T>
template <typename T>
inline constexpr bool use_default_type_traits_v<Expected<T>> = false;

template <typename T>
struct TypeTraits<Expected<T>> : public TypeTraitsBase {
  TVM_FFI_INLINE static void CopyToAnyView(const Expected<T>& src, TVMFFIAny* result) {
    if (src.is_err()) {
      TypeTraits<Error>::CopyToAnyView(src.error(), result);
    } else {
      TypeTraits<T>::CopyToAnyView(src.value(), result);
    }
  }

  TVM_FFI_INLINE static void MoveToAny(Expected<T> src, TVMFFIAny* result) {
    if (src.is_err()) {
      TypeTraits<Error>::MoveToAny(std::move(src).error(), result);
    } else {
      TypeTraits<T>::MoveToAny(std::move(src).value(), result);
    }
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return TypeTraits<T>::CheckAnyStrict(src) || TypeTraits<Error>::CheckAnyStrict(src);
  }

  TVM_FFI_INLINE static Expected<T> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if (TypeTraits<T>::CheckAnyStrict(src)) {
      return TypeTraits<T>::CopyFromAnyViewAfterCheck(src);
    }
    return TypeTraits<Error>::CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static Expected<T> MoveFromAnyAfterCheck(TVMFFIAny* src) {
    if (TypeTraits<T>::CheckAnyStrict(src)) {
      return TypeTraits<T>::MoveFromAnyAfterCheck(src);
    }
    return TypeTraits<Error>::MoveFromAnyAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<Expected<T>> TryCastFromAnyView(const TVMFFIAny* src) {
    if (auto opt = TypeTraits<T>::TryCastFromAnyView(src)) {
      return Expected<T>(*std::move(opt));
    }
    if (auto opt_err = TypeTraits<Error>::TryCastFromAnyView(src)) {
      return Expected<T>(*std::move(opt_err));
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "Expected<" + TypeTraits<T>::TypeStr() + ">";
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"Expected","args":[)" + details::TypeSchema<T>::v() +
           R"(,{"type":"ffi.Error"}]})";
  }
};

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXPECTED_H_
