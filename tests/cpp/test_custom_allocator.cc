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
#include <gtest/gtest.h>
#include <tvm/ffi/c_api.h>

#include <cstddef>
#include <cstdint>

namespace {

class AllocatorReset {
 public:
  ~AllocatorReset() { TVMFFISetCustomAllocator(nullptr); }
};

void* TestAllocate(size_t, size_t, int32_t, void*) { return nullptr; }

TEST(CustomAllocator, BuiltinDefaultContract) {
  AllocatorReset reset;
  ASSERT_EQ(TVMFFISetCustomAllocator(nullptr), 0);
  TVMFFICustomAllocator* allocator = TVMFFIGetCustomAllocator();
  ASSERT_NE(allocator, nullptr);
  ASSERT_NE(allocator->allocate, nullptr);

  constexpr size_t kSize = 64;
  constexpr size_t kAlignment = alignof(std::max_align_t);
  void* body = allocator->allocate(kSize, kAlignment, /*type_index=*/0, allocator->context);
  ASSERT_NE(body, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(body) % kAlignment, 0U);

  auto* header = reinterpret_cast<TVMFFIObjectAllocHeader*>(
      static_cast<char*>(body) - sizeof(TVMFFIObjectAllocHeader));
  ASSERT_NE(header->delete_space, nullptr);
  header->delete_space(body);
}

TEST(CustomAllocator, SetAndRestore) {
  AllocatorReset reset;
  TVMFFICustomAllocator custom{&TestAllocate, /*context=*/nullptr};
  ASSERT_EQ(TVMFFISetCustomAllocator(&custom), 0);
  EXPECT_EQ(TVMFFIGetCustomAllocator(), &custom);

  ASSERT_EQ(TVMFFISetCustomAllocator(nullptr), 0);
  EXPECT_NE(TVMFFIGetCustomAllocator(), nullptr);
  EXPECT_NE(TVMFFIGetCustomAllocator(), &custom);
}

}  // namespace
