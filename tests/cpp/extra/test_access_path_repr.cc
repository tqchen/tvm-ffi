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
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/reflection/access_path.h>

namespace {

TEST(AccessPathRepr, StreamlinedFormat) {
  using namespace tvm::ffi;
  using namespace tvm::ffi::reflection;
  AccessPath p = AccessPath::Root()
                     ->Attr("config")
                     ->Attr("layers")
                     ->ArrayItem(0)
                     ->MapItem(String("name"))
                     ->AttrMissing("weights")
                     ->ArrayItemMissing(3)
                     ->MapItemMissing(String("bias"));
  EXPECT_EQ(std::string(ReprPrint(Any(p)).c_str()),
            "<root>.config.layers[0][\"name\"]"
            "[<missing:\"weights\">][<missing:3>][<missing:\"bias\">]");
}

}  // namespace
