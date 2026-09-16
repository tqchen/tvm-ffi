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

use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFIMutationMarkerKind};
use tvm_ffi::{
    Any, AnyCompatible, AnyView, Array, Function, MutationResult, ObjectArc, ObjectRefCore, Result,
    TypeIndex, Unchanged, UpdatedInPlace,
};

#[test]
fn mutation_result_states_and_resolution() {
    let original = Array::new(vec![7i64]);
    let owners = ObjectArc::strong_count(ObjectRefCore::data(&original));
    for updated in [false, true] {
        let result: MutationResult<Array<i64>> = if updated {
            UpdatedInPlace.into()
        } else {
            Unchanged.into()
        };
        assert_eq!(result.is_unchanged(), !updated);
        assert_eq!(result.is_updated_in_place(), updated);
        assert!(!result.has_value());
        assert_eq!(result.unchanged_or_same_as(&original), !updated);
        assert_eq!(
            ObjectArc::strong_count(ObjectRefCore::data(&original)),
            owners
        );
        assert!(result.clone().into_option().is_none());
        let borrowed = result.clone().value_or_original_else(|| original.clone());
        assert_eq!(
            unsafe { ObjectArc::as_raw(ObjectRefCore::data(&borrowed)) },
            unsafe { ObjectArc::as_raw(ObjectRefCore::data(&original)) }
        );
        drop(borrowed);
        let erased = result.clone().try_cast::<Any>().unwrap();
        assert_eq!(erased.is_updated_in_place(), updated);
        assert_eq!(
            ObjectArc::strong_count(ObjectRefCore::data(&original)),
            owners
        );
        let mut cell = TVMFFIAny::new();
        unsafe { MutationResult::copy_to_any_view(&result, &mut cell) };
        let raw: Any = result.into();
        assert_eq!(raw.type_index(), TypeIndex::kTVMFFIMutationMarker as i32);
        assert_eq!(cell.small_str_len, 0);
        assert_eq!(
            unsafe { cell.data_union.v_int64 },
            if updated {
                TVMFFIMutationMarkerKind::kTVMFFIMutationMarkerUpdatedInPlace as i64
            } else {
                TVMFFIMutationMarkerKind::kTVMFFIMutationMarkerUnchanged as i64
            }
        );
        let copied = MutationResult::<Array<i64>>::try_from(AnyView::from(&raw)).unwrap();
        assert_eq!(copied.is_updated_in_place(), updated);
        let moved = MutationResult::<Array<i64>>::try_from(raw).unwrap();
        assert_eq!(moved.is_updated_in_place(), updated);
    }
    let same = MutationResult::changed(original.clone());
    assert!(same.has_value());
    assert!(!same.is_unchanged());
    assert!(!same.is_updated_in_place());
    assert!(same.unchanged_or_same_as(&original));
    let null = MutationResult::<Any>::changed(Any::new());
    assert!(null.has_value());
    assert_eq!(
        null.into_option().unwrap().type_index(),
        TypeIndex::kTVMFFINone as i32
    );
    assert_eq!(
        MutationResult::changed(12i64).value_or_original_else(|| panic!("unused")),
        12
    );
    let pointer = unsafe { ObjectArc::as_raw(ObjectRefCore::data(&original)) };
    let resolved = MutationResult::updated_in_place().value_or_original(original);
    assert_eq!(
        unsafe { ObjectArc::as_raw(ObjectRefCore::data(&resolved)) },
        pointer
    );
}

#[test]
fn mutation_result_conversions_preserve_payload_without_running_value_mapping() {
    for result in [
        MutationResult::<i64>::unchanged(),
        MutationResult::updated_in_place(),
    ] {
        let updated = result.is_updated_in_place();
        let mapped = result.clone().map::<f64>(|_| panic!("marker has no value"));
        assert_eq!(mapped.is_updated_in_place(), updated);
        let mapped = result
            .try_map::<f64, ()>(|_| panic!("marker has no value"))
            .unwrap();
        assert_eq!(mapped.is_updated_in_place(), updated);
        assert!(mapped.try_cast::<Array<i64>>().is_ok());
    }
    let integer = Any::from(3i64);
    let float = MutationResult::<f64>::try_from(AnyView::from(&integer)).unwrap();
    assert_eq!(float.into_option(), Some(3.0));
    assert_eq!(
        MutationResult::changed(4i64)
            .map(|x| x as f64)
            .into_option(),
        Some(4.0)
    );
    assert!(MutationResult::changed(4i64)
        .try_map::<i64, _>(|_| Err("failure"))
        .is_err());
}

#[test]
fn marker_tags_and_checked_boundaries_reject_reserved_payloads() {
    assert!(Unchanged::try_from(Any::from(UpdatedInPlace)).is_err());
    assert!(UpdatedInPlace::try_from(Any::from(Unchanged)).is_err());
    assert_eq!(
        UpdatedInPlace::try_from(Any::from(UpdatedInPlace)).unwrap(),
        UpdatedInPlace
    );
    for (payload, padding) in [(2, 0), (-1, 0), (0, 1), (1, 1)] {
        let mut raw = TVMFFIAny::new();
        raw.type_index = TypeIndex::kTVMFFIMutationMarker as i32;
        raw.data_union.v_int64 = payload;
        raw.small_str_len = padding;
        unsafe {
            assert!(!MutationResult::<i64>::check_any_strict(&raw));
            assert!(!MutationResult::<Any>::check_any_strict(&raw));
            assert!(MutationResult::<i64>::try_cast_from_any_view(&raw).is_err());
            assert!(MutationResult::<Any>::try_cast_from_any_view(&raw).is_err());
            assert!(!Unchanged::check_any_strict(&raw));
            assert!(!UpdatedInPlace::check_any_strict(&raw));
        }
    }
}

#[test]
fn packed_result_callbacks_forward_both_markers() {
    let function =
        Function::from_typed(|| -> Result<MutationResult<i64>> { Ok(UpdatedInPlace.into()) });
    let result = function.call_tuple(()).unwrap();
    assert!(MutationResult::<i64>::try_from(result)
        .unwrap()
        .is_updated_in_place());
}
