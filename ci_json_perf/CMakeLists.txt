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

cmake_minimum_required(VERSION 3.26)
project(tvm_ffi LANGUAGES CXX C)
add_subdirectory("${FFI_SOURCE}" ffi)
add_executable(json_probe probe.cc)
target_link_libraries(json_probe PRIVATE tvm_ffi_shared)
file(
  GENERATE
  OUTPUT "${CMAKE_BINARY_DIR}/paths-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:json_probe>\n$<TARGET_FILE:tvm_ffi_shared>\n"
)
if (WIN32)
  add_custom_command(
    TARGET json_probe
    POST_BUILD
    COMMENT "Copy runtime library next to benchmark executable"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:tvm_ffi_shared>
            $<TARGET_FILE_DIR:json_probe>
  )
endif ()
