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
 * \file backtrace_win.cc
 * \brief Backtrace implementation on windows platform
 * \note We use the term "backtrace" to be consistent with python naming convention.
 */
#ifdef _MSC_VER

// clang-format off
//#include <windows.h>
//#include <dbghelp.h>  // NOLINT(*)
// clang-format on

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>

#include <iostream>
#include <vector>

#include "./backtrace_utils.h"

const TVMFFIByteArray* TVMFFIBacktrace(const char* filename, int lineno, const char* func,
                                       int cross_ffi_boundary) {
  return nullptr;
}
#endif  // _MSC_VER
