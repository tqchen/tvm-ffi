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

//! Typed unchanged-or-replacement results for structural mutation.

use std::marker::PhantomData;

use crate::any::{Any, AnyView, TryFromTemp};
use crate::error::{Error, Result, TYPE_ERROR};
use crate::tvm_ffi_sys::{TVMFFIAny, TVMFFIMutationMarkerKind, TVMFFITypeIndex};
use crate::type_traits::{AnyCompatible, ContainerElement};

/// A structural mutation that keeps its input without acquiring another owner.
///
/// This marker may be returned directly from a map or mutation callback. A
/// pre-order map still descends into the original value's children.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unchanged;

// SAFETY: the marker payload contains no owned resources.
unsafe impl AnyCompatible for Unchanged {
    unsafe fn copy_to_any_view(_src: &Self, data: &mut TVMFFIAny) {
        *data = TVMFFIAny::new();
        data.type_index = TVMFFITypeIndex::kTVMFFIMutationMarker as i32;
        data.data_union.v_int64 = TVMFFIMutationMarkerKind::kTVMFFIMutationMarkerUnchanged as i64;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        Self::copy_to_any_view(&src, data);
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        data.type_index == TVMFFITypeIndex::kTVMFFIMutationMarker as i32
            && data.small_str_len == 0
            && data.data_union.v_int64
                == TVMFFIMutationMarkerKind::kTVMFFIMutationMarkerUnchanged as i64
    }

    unsafe fn copy_from_any_view_after_check(_data: &TVMFFIAny) -> Self {
        Self
    }

    unsafe fn move_from_any_after_check(_data: &mut TVMFFIAny) -> Self {
        Self
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> std::result::Result<Self, ()> {
        Self::check_any_strict(data).then_some(Self).ok_or(())
    }

    fn type_str() -> std::string::String {
        "Unchanged".into()
    }
}

crate::impl_try_from_any!(Unchanged);

/// Retain the original identity while reporting a change to its value or subtree.
///
/// Like [`Unchanged`], this marker does not acquire ownership of the original.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UpdatedInPlace;

// SAFETY: the marker payload contains no owned resources.
unsafe impl AnyCompatible for UpdatedInPlace {
    unsafe fn copy_to_any_view(_src: &Self, data: &mut TVMFFIAny) {
        *data = TVMFFIAny::new();
        data.type_index = TVMFFITypeIndex::kTVMFFIMutationMarker as i32;
        data.data_union.v_int64 =
            TVMFFIMutationMarkerKind::kTVMFFIMutationMarkerUpdatedInPlace as i64;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        Self::copy_to_any_view(&src, data);
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        data.type_index == TVMFFITypeIndex::kTVMFFIMutationMarker as i32
            && data.small_str_len == 0
            && data.data_union.v_int64
                == TVMFFIMutationMarkerKind::kTVMFFIMutationMarkerUpdatedInPlace as i64
    }

    unsafe fn copy_from_any_view_after_check(_data: &TVMFFIAny) -> Self {
        Self
    }

    unsafe fn move_from_any_after_check(_data: &mut TVMFFIAny) -> Self {
        Self
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> std::result::Result<Self, ()> {
        Self::check_any_strict(data).then_some(Self).ok_or(())
    }

    fn type_str() -> std::string::String {
        "UpdatedInPlace".into()
    }
}

crate::impl_try_from_any!(UpdatedInPlace);

/// A replacement, [`Unchanged`], or [`UpdatedInPlace`], stored in one FFI Any cell.
///
/// `T` may be an FFI value type or [`Any`]. Use `Result<MutationResult<T>>` to
/// propagate errors. Callbacks can return this wrapper, either marker, a
/// replacement value, or a `Result` containing any of these.
///
/// Converting this wrapper to `Any` preserves the marker. Use
/// [`Self::value_or_original`] or [`Self::value_or_original_else`] when an actual value is needed.
#[repr(transparent)]
pub struct MutationResult<T: ContainerElement = Any> {
    data: Any,
    _marker: PhantomData<T>,
}

impl<T: ContainerElement> MutationResult<T> {
    /// Keep the original value without borrowing or cloning it.
    #[inline]
    pub fn unchanged() -> Self {
        Self {
            data: Unchanged.into(),
            _marker: PhantomData,
        }
    }

    /// Keep the original identity while reporting an in-place subtree change.
    #[inline]
    pub fn updated_in_place() -> Self {
        Self {
            data: UpdatedInPlace.into(),
            _marker: PhantomData,
        }
    }

    /// Supply an owning replacement value.
    /// Returning the original identity asserts that its subtree is unchanged.
    #[inline]
    pub fn changed(value: T) -> Self {
        let mut raw = TVMFFIAny::new();
        // SAFETY: ContainerElement transfers exactly one owning value.
        unsafe {
            T::container_move_to_any(value, &mut raw);
            Self {
                data: Any::from_raw_ffi_any(raw),
                _marker: PhantomData,
            }
        }
    }

    /// Whether the original value and its subtree are unchanged.
    #[inline]
    pub fn is_unchanged(&self) -> bool {
        is_unchanged(&self.data)
    }

    /// Whether the original identity is retained with an in-place subtree change.
    #[inline]
    pub fn is_updated_in_place(&self) -> bool {
        is_updated_in_place(&self.data)
    }

    /// Whether a replacement is stored, including a null replacement.
    #[inline]
    pub fn has_value(&self) -> bool {
        !is_marker(&self.data)
    }

    /// Whether the result is unchanged or has the original shallow identity.
    /// Always false for [`UpdatedInPlace`].
    #[inline]
    pub fn unchanged_or_same_as(&self, original: &T) -> bool {
        if self.is_unchanged() {
            return true;
        }
        if self.is_updated_in_place() {
            return false;
        }
        let mut raw = TVMFFIAny::new();
        // SAFETY: the borrowed cell is used only while original is alive.
        unsafe {
            T::container_copy_to_any_view(original, &mut raw);
        }
        super::structural_common::same_shallow(raw, *self.data.as_raw_ffi_any())
    }

    /// Move out the replacement, returning None for either marker.
    #[inline]
    pub fn into_option(self) -> Option<T> {
        if !self.has_value() {
            return None;
        }
        // SAFETY: constructors and conversions maintain the declared T.
        unsafe {
            let mut raw = Any::into_raw_ffi_any(self.data);
            Some(T::container_move_from_any_after_check(&mut raw))
        }
    }

    /// Move the replacement or the supplied original value out of this result.
    #[inline]
    pub fn value_or_original(self, original: T) -> T {
        self.value_or_original_else(|| original)
    }

    /// Materialize the original only if this result is either marker.
    #[inline]
    pub fn value_or_original_else(self, original: impl FnOnce() -> T) -> T {
        self.into_option().unwrap_or_else(original)
    }

    /// Convert only the replacement, preserving either marker payload.
    #[inline]
    pub fn map<U: ContainerElement>(self, convert: impl FnOnce(T) -> U) -> MutationResult<U> {
        if !self.has_value() {
            return MutationResult {
                data: self.data,
                _marker: PhantomData,
            };
        }
        MutationResult::changed(convert(self.into_option().unwrap()))
    }

    /// Fallibly convert only the replacement, propagating its error unchanged.
    #[inline]
    pub fn try_map<U: ContainerElement, E>(
        self,
        convert: impl FnOnce(T) -> std::result::Result<U, E>,
    ) -> std::result::Result<MutationResult<U>, E> {
        if !self.has_value() {
            return Ok(MutationResult {
                data: self.data,
                _marker: PhantomData,
            });
        }
        convert(self.into_option().unwrap()).map(MutationResult::changed)
    }

    /// Check a replacement's FFI type without cloning it.
    ///
    /// An unchanged result is compatible with every replacement type. Erased
    /// and narrowing conversions retain the strict type check used by
    /// [`Any::try_as`]. For a known typed conversion, use `map(Into::into)`.
    #[inline]
    pub fn try_cast<U: ContainerElement>(self) -> Result<MutationResult<U>> {
        if unsafe { MutationResult::<U>::check_any_strict(self.data.as_raw_ffi_any()) } {
            Ok(MutationResult {
                data: self.data,
                _marker: PhantomData,
            })
        } else {
            Err(Error::new(
                TYPE_ERROR,
                &format!(
                    "structural mutation result does not match {}",
                    U::container_type_str()
                ),
                "",
            ))
        }
    }
}

impl MutationResult<Any> {
    #[inline]
    pub(crate) fn from_carrier(data: Any) -> Result<Self> {
        if data.type_index() == TVMFFITypeIndex::kTVMFFIError as i32 {
            return match Error::try_from(data) {
                Ok(error) | Err(error) => Err(error),
            };
        }
        Self::try_from(data)
    }
}

impl<T: ContainerElement> From<Unchanged> for MutationResult<T> {
    fn from(_: Unchanged) -> Self {
        Self::unchanged()
    }
}

impl<T: ContainerElement> From<UpdatedInPlace> for MutationResult<T> {
    fn from(_: UpdatedInPlace) -> Self {
        Self::updated_in_place()
    }
}

impl<T: ContainerElement> Clone for MutationResult<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            _marker: PhantomData,
        }
    }
}

// SAFETY: the carrier contains either the resource-free marker or exactly
// T's owning representation. Borrowing never acquires ownership; moving
// transfers the carrier once. Non-strict conversions materialize a valid T.
unsafe impl<T: ContainerElement> AnyCompatible for MutationResult<T> {
    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        *data = *src.data.as_raw_ffi_any();
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        *data = Any::into_raw_ffi_any(src.data);
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        if data.type_index == TVMFFITypeIndex::kTVMFFIMutationMarker as i32 {
            return Unchanged::check_any_strict(data) || UpdatedInPlace::check_any_strict(data);
        }
        if T::CONTAINER_IS_ANY {
            // An erased successful result excludes the ABI's error channel.
            data.type_index != TVMFFITypeIndex::kTVMFFIError as i32
        } else {
            T::container_check_any_strict(data)
        }
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        if data.type_index == TVMFFITypeIndex::kTVMFFIMutationMarker as i32 {
            return Self {
                data: Any::from_raw_ffi_any(*data),
                _marker: PhantomData,
            };
        }
        // Materialize T, including numeric narrowing, before storing its value.
        Self::changed(T::container_copy_from_any_view_after_check(data))
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        Self {
            data: Any::from_raw_ffi_any(std::mem::replace(data, TVMFFIAny::new())),
            _marker: PhantomData,
        }
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> std::result::Result<Self, ()> {
        if T::CONTAINER_IS_ANY && data.type_index == TVMFFITypeIndex::kTVMFFIError as i32 {
            return Err(());
        }
        if data.type_index == TVMFFITypeIndex::kTVMFFIMutationMarker as i32 {
            return if Self::check_any_strict(data) {
                Ok(Self::copy_from_any_view_after_check(data))
            } else {
                Err(())
            };
        }
        T::container_try_cast_from_any_view(data).map(Self::changed)
    }

    fn type_str() -> std::string::String {
        format!("MutationResult<{}>", T::container_type_str())
    }
}

impl<T: ContainerElement> TryFrom<Any> for MutationResult<T> {
    type Error = Error;
    #[inline]
    fn try_from(value: Any) -> Result<Self> {
        TryFromTemp::<Self>::try_from(value).map(TryFromTemp::into_value)
    }
}

impl<'a, T: ContainerElement> TryFrom<AnyView<'a>> for MutationResult<T> {
    type Error = Error;
    #[inline]
    fn try_from(value: AnyView<'a>) -> Result<Self> {
        TryFromTemp::<Self>::try_from(value).map(TryFromTemp::into_value)
    }
}

#[inline]
pub(crate) fn is_marker(value: &Any) -> bool {
    value.type_index() == TVMFFITypeIndex::kTVMFFIMutationMarker as i32
}

#[inline]
pub(crate) fn is_unchanged(value: &Any) -> bool {
    unsafe { Unchanged::check_any_strict(value.as_raw_ffi_any()) }
}

#[inline]
pub(crate) fn is_updated_in_place(value: &Any) -> bool {
    unsafe { UpdatedInPlace::check_any_strict(value.as_raw_ffi_any()) }
}
