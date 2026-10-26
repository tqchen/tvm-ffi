// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
.text
.globl __tvm_ffi_probe
.p2align 2
__tvm_ffi_probe:
  // TVMFFISafeCall: the result pointer is x3; all clobbers are volatile.
  adrp x9, data + 4096
  add x9, x9, :lo12:data + 4096
  ldr x10, [x9]
  adrp x9, target - 16
  add x9, x9, :lo12:target - 16
  ldr x11, [x9]
  add x10, x10, x11
  adrp x9, target
  ldr q0, [x9, :lo12:target]
  umov x11, v0.d[0]
  add x10, x10, x11
  mov x9, #1
  str x9, [x3]
  str x10, [x3, #8]
  mov w0, #0
  ret

.data
.p2align 12
data:
  .zero 4080
  .quad 17, 0
target:
  .quad 42, 0
