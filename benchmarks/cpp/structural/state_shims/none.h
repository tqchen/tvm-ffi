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
#ifndef TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_STATE_SHIMS_NONE_H_
#define TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_STATE_SHIMS_NONE_H_

// The empty build shim: a state that needs nothing still gets force-included, so both states
// of a two-state run are compiled with the identical `-include <shim>` flag string and the
// only difference between them is what the header contains.  A flag present on one side and
// absent on the other is itself a build difference, and this is a two-state run's whole
// premise.

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_STATE_SHIMS_NONE_H_
