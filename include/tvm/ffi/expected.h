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

template <typename T>
class Expected;

namespace details {

struct ExpectedUnsafe;

template <typename T>
inline constexpr bool is_expected_v = false;

template <typename T>
inline constexpr bool is_expected_v<Expected<T>> = true;

template <typename T>
inline constexpr bool is_unexpected_v = false;

template <typename E>
inline constexpr bool is_unexpected_v<Unexpected<E>> = true;

}  // namespace details

/*!
 * \brief Expected<T> provides exception-free error handling for FFI functions.
 *
 * Expected<T> is similar to Rust's Result<T, Error> or C++23's std::expected.
 * It can hold either a success value of type T or an error of type Error.
 *
 * \tparam T The success type. Must be Any-compatible and cannot be Error. The ``void`` type is
 *         supported as a success state without a value. \sa Expected<void>
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
  static_assert(
      !std::is_void_v<T>,
      "Expected with a cv-qualified void success type is not allowed. Use Expected<void>.");
  static_assert(!std::is_same_v<T, Error>, "Expected<Error> is not allowed. Use Error directly.");

  /*!
   * \brief Implicit constructor from a success value.
   * \param value The success value.
   */
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  Expected(T value) : data_(Any(std::move(value))) {}

  /*!
   * \brief Implicit constructor from a different success value type.
   * \tparam U Source type implicitly convertible to ``T``.
   * \param value The success value to convert.
   */
  // Excludes Error, Unexpected, and Expected deliberately: Any subsumes all three, so without
  // these an Expected<Any> built from an error would store it as a success value. The Expected
  // exclusion also keeps this overload disjoint from Expected(Expected<U>) instead of relying on
  // partial ordering to choose between two paths that must agree.
  //
  // std::expected admits constructible sources and uses C++20 explicit(bool) to separate its
  // implicit subset. Under C++17, convertibility keeps exactly that implicit subset and drops only
  // explicit-only conversions; is_constructible plus explicit(bool) can extend it after an upgrade.
  template <typename U, typename = std::enable_if_t<!details::is_expected_v<std::decay_t<U>> &&
                                                    !details::is_unexpected_v<std::decay_t<U>> &&
                                                    !std::is_base_of_v<Error, std::decay_t<U>> &&
                                                    std::is_convertible_v<U, T>>>
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  Expected(U&& value) : data_(Any(T(std::forward<U>(value)))) {}

  /*!
   * \brief Implicit converting constructor from another Expected success type.
   * \tparam U Source success type whose storage is subsumed by or implicitly convertible to ``T``.
   * \param other The Expected value to convert.
   */
  // Subsumption belongs only here: this source already contains a materialized U or Error whose
  // representation may be reused. Applying type_subsumes_v<Any, U> to the bare-value constructor
  // would accept every U, including types that cannot be materialized as Any, and fail in its body.
  // Taking by value gives a local to move from, copying an lvalue source and moving an rvalue. The
  // implicit copy constructor still wins for Expected<T> itself by the non-template tiebreaker.
  template <typename U,
            typename = std::enable_if_t<!std::is_void_v<U> &&
                                        (type_subsumes_v<T, U> || std::is_convertible_v<U, T>)>>
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  Expected(Expected<U> other) {
    if constexpr (type_subsumes_v<T, U>) {
      // data_ holds a T or an Error. Subsumption proves the source representation already
      // satisfies that invariant, so the Any moves without inspecting its state. Do not make
      // this unconditional: value() reads back through MoveFromAnyAfterCheck<T>, whose check is
      // the success/error state, not the type.
      data_ = std::move(other.data_);
    } else {
      data_ = other.is_err() ? Any(std::move(other).error()) : Any(T(std::move(other).value()));
    }
  }

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

  /*! \brief Return the raw stored type index. */
  TVM_FFI_INLINE int32_t type_index() const noexcept { return data_.type_index(); }

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
  template <typename>
  friend class Expected;
  Expected() = default;

  friend struct details::ExpectedUnsafe;

  Any data_;  // Invariant: holds a T (type_index != kTVMFFIError) or an Error.
};

/*!
 * \brief Specialization of Expected for a successful operation without a value.
 *
 * Expected<void> stores FFI None when successful and an Error when unsuccessful. A successful
 * value is constructed with the default constructor, and \ref value validates success without
 * returning a value.
 */
template <>
class Expected<void> {
 public:
  /*! \brief Construct a successful Expected<void>. */
  Expected() = default;

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

  /*! \brief Return the raw stored type index. */
  TVM_FFI_INLINE int32_t type_index() const noexcept { return data_.type_index(); }

  /*! \brief Returns true if this Expected represents successful completion. */
  TVM_FFI_INLINE bool is_ok() const noexcept {
    return data_.type_index() != TypeIndex::kTVMFFIError;
  }

  /*! \brief Returns true if this Expected contains an error. */
  TVM_FFI_INLINE bool is_err() const noexcept {
    return data_.type_index() == TypeIndex::kTVMFFIError;
  }

  /*! \brief Alias for is_ok(). */
  TVM_FFI_INLINE bool has_value() const noexcept { return is_ok(); }

  /*! \brief Validate successful completion, or throw the contained error. */
  TVM_FFI_INLINE void value() const& {
    if (TVM_FFI_PREDICT_FALSE(is_err())) {
      throw details::AnyUnsafe::CopyFromAnyViewAfterCheck<Error>(data_);
    }
  }

  /*! \brief Validate successful completion, or throw the contained error by move. */
  TVM_FFI_INLINE void value() && {
    if (TVM_FFI_PREDICT_FALSE(is_err())) {
      throw details::AnyUnsafe::MoveFromAnyAfterCheck<Error>(std::move(data_));
    }
  }

  /*! \brief Returns the contained error, or throws RuntimeError if is_ok(). */
  TVM_FFI_INLINE Error error() const& {
    if (is_ok()) {
      TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
    }
    return details::AnyUnsafe::CopyFromAnyViewAfterCheck<Error>(data_);
  }

  /*! \brief Returns the contained error (moved out), or throws RuntimeError if is_ok(). */
  TVM_FFI_INLINE Error error() && {
    if (is_ok()) {
      TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
    }
    return details::AnyUnsafe::MoveFromAnyAfterCheck<Error>(std::move(data_));
  }

 private:
  friend struct details::ExpectedUnsafe;

  Any data_;  // Invariant: holds FFI None on success or an Error.
};

namespace details {

/*!
 * \brief Unsafe raw-storage helpers for Expected.
 *
 * These helpers bypass normal value checking and are intended for ABI boundaries
 * that already know the underlying Any storage holds either a valid T or Error.
 */
struct ExpectedUnsafe {
  /*!
   * \brief Move a raw TVMFFIAny into Expected storage.
   * \tparam T The Expected success type.
   * \param raw The raw FFI value to move.
   * \return Expected backed by moved Any storage.
   */
  template <typename T>
  TVM_FFI_INLINE static Expected<T> MoveFromTVMFFIAny(TVMFFIAny raw) {
    Expected<T> result;
    result.data_ = AnyUnsafe::MoveTVMFFIAnyToAny(&raw);
    return result;
  }

  /*!
   * \brief Move Expected storage to a raw TVMFFIAny.
   * \tparam T The Expected success type.
   * \param result The Expected value to move from.
   * \return Raw FFI value containing moved underlying Any storage.
   */
  template <typename T>
  TVM_FFI_INLINE static TVMFFIAny MoveToTVMFFIAny(Expected<T>&& result) {
    return AnyUnsafe::MoveAnyToTVMFFIAny(std::move(result.data_));
  }

  /*!
   * \brief Return the underlying Any storage as an rvalue reference.
   * \tparam T The Expected success type.
   * \param result The Expected value to move from.
   * \return Rvalue reference to the raw Any storage.
   *
   * \note This overload is what makes ``std::move(GetData(x))`` actually move. Without it the
   *       const overload below is the only candidate, so the expression yields ``const Any&&``
   *       and binds the copy constructor, taking a reference count the call site looks like it
   *       is avoiding. Nothing is moved until the returned reference is consumed, so \p result
   *       is left intact when the caller only inspects it.
   */
  template <typename T>
  TVM_FFI_INLINE static Any&& GetData(Expected<T>& result) noexcept {
    return std::move(result.data_);
  }

  /*!
   * \brief Return the underlying Any storage.
   * \tparam T The Expected success type.
   * \param result The Expected value to inspect.
   * \return Const reference to the raw Any storage.
   */
  template <typename T>
  TVM_FFI_INLINE static const Any& GetData(const Expected<T>& result) noexcept {
    return result.data_;
  }

  /*!
   * \brief Proxy that converts a moved-out Expected payload to the destination type.
   *
   * \note Produced only by MoveDataAutoCast. The conversion is resolved against the declared
   *       type of the initialization target, so no explicit type argument appears at the call
   *       site. The conversion is unchecked in release builds; the debug assert is the only
   *       guard, so the caller must already know the payload type.
   */
  struct CastProxy {
    /*! \brief The moved-out payload awaiting conversion. */
    Any value;

    /*!
     * \brief Convert the payload to the destination type.
     * \tparam T The destination type, deduced from the initialization target.
     * \return The payload decoded as T.
     */
    template <typename T>
    TVM_FFI_INLINE operator T() && {  // NOLINT(runtime/explicit)
      if constexpr (!std::is_same_v<T, Any>) {
        TVM_FFI_DCHECK(value.as<T>().has_value());
      }
      return AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(value));
    }
  };

  /*!
   * \brief Move the payload out of an Expected without re-testing its success state.
   * \tparam U The Expected success type.
   * \param ref The Expected to move the payload out of. Left holding a moved-from Any.
   * \return Proxy that converts to the destination type.
   *
   * \note For use directly after an early-return check has established that \p ref holds a
   *       value. Neither the skipped success test nor the unchecked conversion appears at the
   *       call site.
   *
   * \note ``auto`` on the left does not work: it deduces CastProxy instead of triggering the
   *       conversion. The destination must be a concrete type, which is the normal case for a
   *       structural hook writing a field of declared type. Write ``Any x = ...`` when the
   *       payload should stay type-erased; ``MoveFromAnyAfterCheck<Any>`` short-circuits to a
   *       plain move, so that costs nothing.
   */
  template <typename U>
  TVM_FFI_INLINE static CastProxy MoveDataAutoCast(Expected<U>& ref) {
    return CastProxy{std::move(GetData(ref))};
  }

  /*!
   * \brief Read an Expected success value as a compatible raw storage type.
   * \tparam T The type to read from the underlying Any storage, or ``void`` to validate an
   *           ``Expected<void>`` success state.
   * \tparam U The Expected success type.
   * \param result The Expected value to read from.
   * \return The stored value decoded as T, or no value when T is ``void``.
   *
   * \note When T is ``void``, U must also be ``void``. Otherwise, this assumes \p result stores
   *       T-compatible Any storage, or Error.
   */
  template <typename T, typename U>
  TVM_FFI_INLINE static T ValueAs(const Expected<U>& result) {
    if constexpr (std::is_void_v<T>) {
      static_assert(std::is_void_v<U>, "ExpectedUnsafe::ValueAs<void> requires an Expected<void>");
      result.value();
    } else {
      const Any& data = result.data_;
      if (TVM_FFI_PREDICT_TRUE(data.type_index() != TypeIndex::kTVMFFIError)) {
        return AnyUnsafe::CopyFromAnyViewAfterCheck<T>(data);
      }
      throw AnyUnsafe::CopyFromAnyViewAfterCheck<Error>(data);
    }
  }
};

}  // namespace details

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

/*! \brief TypeTraits specialization for Expected<void>. */
template <>
struct TypeTraits<Expected<void>> : public TypeTraitsBase {
  TVM_FFI_INLINE static void CopyToAnyView(const Expected<void>& src, TVMFFIAny* result) {
    if (src.is_err()) {
      TypeTraits<Error>::CopyToAnyView(src.error(), result);
    } else {
      TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
    }
  }

  TVM_FFI_INLINE static void MoveToAny(Expected<void> src, TVMFFIAny* result) {
    if (src.is_err()) {
      TypeTraits<Error>::MoveToAny(std::move(src).error(), result);
    } else {
      TypeTraits<std::nullptr_t>::MoveToAny(nullptr, result);
    }
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return TypeTraits<std::nullptr_t>::CheckAnyStrict(src) ||
           TypeTraits<Error>::CheckAnyStrict(src);
  }

  TVM_FFI_INLINE static Expected<void> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if (TypeTraits<std::nullptr_t>::CheckAnyStrict(src)) {
      return Expected<void>();
    }
    return TypeTraits<Error>::CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static Expected<void> MoveFromAnyAfterCheck(TVMFFIAny* src) {
    if (TypeTraits<std::nullptr_t>::CheckAnyStrict(src)) {
      return Expected<void>();
    }
    return TypeTraits<Error>::MoveFromAnyAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<Expected<void>> TryCastFromAnyView(const TVMFFIAny* src) {
    if (TypeTraits<std::nullptr_t>::CheckAnyStrict(src)) {
      return Expected<void>();
    }
    if (auto opt_err = TypeTraits<Error>::TryCastFromAnyView(src)) {
      return Expected<void>(*std::move(opt_err));
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return "Expected<void>"; }

  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"Expected","args":[)" + TypeTraits<std::nullptr_t>::TypeSchema() +
           R"(,{"type":"ffi.Error"}]})";
  }
};

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXPECTED_H_
