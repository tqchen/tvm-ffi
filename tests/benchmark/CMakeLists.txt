# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

add_executable(tvm_ffi_benchmark_structural_map_cost structural_map_cost.cc)
set_target_properties(
  tvm_ffi_benchmark_structural_map_cost
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
target_link_libraries(tvm_ffi_benchmark_structural_map_cost PRIVATE tvm_ffi_shared)
tvm_ffi_add_cxx_warning(tvm_ffi_benchmark_structural_map_cost)

add_executable(tvm_ffi_benchmark_structural_map_tvm_ast_cost structural_map_tvm_ast_cost.cc)
set_target_properties(
  tvm_ffi_benchmark_structural_map_tvm_ast_cost
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
target_link_libraries(tvm_ffi_benchmark_structural_map_tvm_ast_cost PRIVATE tvm_ffi_shared)
tvm_ffi_add_cxx_warning(tvm_ffi_benchmark_structural_map_tvm_ast_cost)

# The calibrated 2b50d28 API exposed walk deduplication as a template flag;
# current main has removed that flag. Keep the benchmark source buildable at
# both the calibration pin and the branch base.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${CMAKE_SOURCE_DIR}/include;${CMAKE_SOURCE_DIR}/3rdparty/dlpack/include")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
check_cxx_source_compiles(
  "#include <tvm/ffi/extra/structural_visit.h>
   int main() {
     tvm::ffi::StructuralWalk<tvm::ffi::WalkOrder::kPostOrder, true>(
       tvm::ffi::Any(nullptr), [](tvm::ffi::AnyView) {
         return tvm::ffi::WalkResult::Advance();
       });
   }"
  TVM_FFI_BENCHMARK_HAS_WALK_DEDUP_TEMPLATE
)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
if (TVM_FFI_BENCHMARK_HAS_WALK_DEDUP_TEMPLATE)
  target_compile_definitions(
    tvm_ffi_benchmark_structural_map_cost PRIVATE TVM_FFI_BENCHMARK_WALK_DEDUP_TEMPLATE=1
  )
  target_compile_definitions(
    tvm_ffi_benchmark_structural_map_tvm_ast_cost
    PRIVATE TVM_FFI_BENCHMARK_WALK_DEDUP_TEMPLATE=1
  )
endif ()
