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
#ifndef TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_OLD_H_
#define TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_OLD_H_

// The OLD state's hook selection for build_mix.sh: the OLD engine (structural_mutate_old.h,
// upstream main e74e58f's engine verbatim) under the existing, untouched OLD hook file
// tvm_hook_override.h. The engine header is included first and carries the original include
// guard, so the hook file's own `#include <tvm/ffi/extra/structural_mutate.h>` resolves to
// nothing and every hook compiles against the OLD engine. The stamped hook hash is
// tvm_hook_override.h's, not this file's.

#include <tvm/ffi/extra/structural_mutate_old.h>

#include "tvm_hook_override.h"

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TVM_HOOK_OVERRIDE_OLD_H_
