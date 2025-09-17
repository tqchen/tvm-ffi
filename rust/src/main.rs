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
use std::ptr;
use tvm_ffi::c_api;


fn main() {
    unsafe {
        let mut func_ptr: c_api::TVMFFIObjectHandle = ptr::null_mut();
        let fun_name: c_api::TVMFFIByteArray = c_api::TVMFFIByteArray {
            data: b"add".as_ptr(),
            size: b"add".len(),
        };
        let ret: i32 = c_api::TVMFFIFunctionGetGlobal(
            &fun_name as *const c_api::TVMFFIByteArray,
            &mut func_ptr as *mut c_api::TVMFFIObjectHandle);
        println!("ret: {}", ret);
        println!("func_ptr: {:p}", func_ptr);
    }
}
