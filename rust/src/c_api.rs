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
#![allow(non_camel_case_types)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64};

// DLPack related declarations
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DLDeviceType {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
    kDLMAIA = 17,
    kDLTrn = 18,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}

/// DLPack data type code enum
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DLDataTypeCode {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLBfloat = 4,
    kDLComplex = 5,
    kDLOpaqueHandle = 3,
    kDLBool = 6,
    kDLFloat8_e3m4 = 7,
    kDLFloat8_e4m3 = 8,
    kDLFloat8_e4m3b11fnuz = 9,
    kDLFloat8_e4m3fn = 10,
    kDLFloat8_e4m3fnuz = 11,
    kDLFloat8_e5m2 = 12,
    kDLFloat8_e5m2fnuz = 13,
    kDLFloat8_e8m0fnu = 14,
    kDLFloat6_e2m3fn = 15,
    kDLFloat6_e3m2fn = 16,
    kDLFloat4_e2m1fn = 17,
}

/// DLPack data type struct
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

///  The index type of the FFI objects
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TVMFFITypeIndex {
    /// None/nullptr value
    kTVMFFINone = 0,
    /// POD int value
    kTVMFFIInt = 1,
    /// POD bool value
    kTVMFFIBool = 2,
    /// POD float value
    kTVMFFIFloat = 3,
    /// Opaque pointer object
    kTVMFFIOpaquePtr = 4,
    /// DLDataType
    kTVMFFIDataType = 5,
    /// DLDevice
    kTVMFFIDevice = 6,
    /// DLTensor*
    kTVMFFIDLTensorPtr = 7,
    /// const char*
    kTVMFFIRawStr = 8,
    /// TVMFFIByteArray*
    kTVMFFIByteArrayPtr = 9,
    /// R-value reference to ObjectRef
    kTVMFFIObjectRValueRef = 10,
    /// Small string on stack
    kTVMFFISmallStr = 11,
    /// Small bytes on stack
    kTVMFFISmallBytes = 12,
    /// Start of statically defined objects.
    kTVMFFIStaticObjectBegin = 64,
    /// String object, layout = { TVMFFIObject, TVMFFIByteArray, ... }
    kTVMFFIStr = 65,
    /// Bytes object, layout = { TVMFFIObject, TVMFFIByteArray, ... }
    kTVMFFIBytes = 66,
    /// Error object.
    kTVMFFIError = 67,
    /// Function object.
    kTVMFFIFunction = 68,
    /// Shape object, layout = { TVMFFIObject, { const int64_t*, size_t }, ... }
    kTVMFFIShape = 69,
    /// Tensor object, layout = { TVMFFIObject, DLTensor, ... }
    kTVMFFITensor = 70,
    /// Array object.
    kTVMFFIArray = 71,
    //----------------------------------------------------------------
    // more complex objects
    //----------------------------------------------------------------
    /// Map object.
    kTVMFFIMap = 72,
    /// Runtime dynamic loaded module object.
    kTVMFFIModule = 73,
    /// Opaque python object.
    kTVMFFIOpaquePyObject = 74,
}

/// Handle to Object from C API's pov
pub type TVMFFIObjectHandle = *mut c_void;
pub type TVMFFIObjectDeleter = unsafe extern "C" fn(self_ptr: *mut c_void, flags: i32);

#[repr(C)]
pub struct TVMFFIObject {
    pub type_index: i32,
    pub weak_ref_count: AtomicU32,
    pub strong_ref_count: AtomicU64,
    pub deleter: Option<TVMFFIObjectDeleter>,
    // private padding to ensure 8 bytes alignment
    #[cfg(target_pointer_width = "32")]
    __padding: u32,
}

/// Second union in TVMFFIAny - 8 bytes
#[repr(C)]
pub union TVMFFIAnyData {
    /// Integers
    pub v_int64: i64,
    /// Floating-point numbers
    pub v_float64: f64,
    /// Typeless pointers
    pub v_ptr: *mut c_void,
    /// Raw C-string
    pub v_c_str: *const i8,
    /// Ref counted objects
    pub v_obj: *mut TVMFFIObject,
    /// Data type
    pub v_dtype: DLDataType,
    /// Device
    pub v_device: DLDevice,
    /// Small string
    pub v_bytes: [u8; 8],
    /// uint64 repr mainly used for hashing
    pub v_uint64: u64,
}

/// TVM FFI Any value - a union type that can hold various data types
#[repr(C)]
pub struct TVMFFIAny {
    /// Type index of the object.
    /// The type index of Object and Any are shared in FFI.
    pub type_index: i32,
    /// small string length or zero padding
    pub small_str_len: u32,
    /// Second union - 8 bytes
    pub data: TVMFFIAnyData,
}

/// Byte array data structure used by String and Bytes.
#[repr(C)]
pub struct TVMFFIByteArray {
    pub data: *const u8,
    pub size: usize,
}

pub type TVMFFISafeCallType = unsafe extern "C" fn(
    handle: *mut c_void,
    args: *const TVMFFIAny,
    num_args: i32,
    result: *mut TVMFFIAny,
) -> i32;

#[link(name = "tvm_ffi")]
unsafe extern "C" {
    pub fn TVMFFIFunctionGetGlobal(
        name: *const TVMFFIByteArray,
        out: *mut TVMFFIObjectHandle,
    ) -> i32;
    pub fn TVMFFIFunctionSetGlobal(
        name: *const TVMFFIByteArray,
        f: TVMFFIObjectHandle,
        can_override: i32,
    ) -> i32;
    pub fn TVMFFIFunctionCreate(
        self_ptr: *mut c_void,
        safe_call: TVMFFISafeCallType,
        deleter: *mut c_void,
        out: *mut TVMFFIObjectHandle,
    ) -> i32;
    pub fn TVMFFIAnyViewToOwnedAny(any_view: *const TVMFFIAny, out: *mut TVMFFIAny) -> i32;
    pub fn TVMFFIFunctionCall(
        func: TVMFFIObjectHandle,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> i32;
    pub fn TVMFFIErrorMoveFromRaised(result: *mut TVMFFIObjectHandle);
    pub fn TVMFFIErrorSetRaised(error: TVMFFIObjectHandle);
    pub fn TVMFFIErrorSetRaisedFromCStr(kind: *const i8, message: *const i8);
    pub fn TVMFFIErrorCreate(
        kind: *const i8,
        message: *const i8,
        traceback: *const i8,
    ) -> TVMFFIObjectHandle;
    pub fn TVMFFITensorFromDLPack(
        from: *mut c_void,
        require_alignment: i32,
        require_contiguous: i32,
        out: *mut TVMFFIObjectHandle,
    ) -> i32;
    pub fn TVMFFITensorToDLPack(from: TVMFFIObjectHandle, out: *mut *mut c_void) -> i32;
    pub fn TVMFFITensorFromDLPackVersioned(
        from: *mut c_void,
        require_alignment: i32,
        require_contiguous: i32,
        out: *mut TVMFFIObjectHandle,
    ) -> i32;
    pub fn TVMFFITensorToDLPackVersioned(from: TVMFFIObjectHandle, out: *mut *mut c_void) -> i32;
    pub fn TVMFFIStringFromByteArray(input: *const TVMFFIByteArray, out: *mut TVMFFIAny) -> i32;
    pub fn TVMFFIBytesFromByteArray(input: *const TVMFFIByteArray, out: *mut TVMFFIAny) -> i32;
    pub fn TVMFFIDataTypeFromString(str: *const i8, out: *mut DLDataType) -> i32;
    pub fn TVMFFIDataTypeToString(dtype: *const DLDataType, out: *mut TVMFFIAny) -> i32;
}
