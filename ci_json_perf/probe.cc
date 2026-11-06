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

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/json.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <locale>
#include <random>
#include <string>
#include <vector>
using namespace tvm::ffi;
struct CommaPunctuation : std::numpunct<char> {
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '.'; }
  std::string do_grouping() const override { return std::string(1, 3); }
};
int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "validate";
  if (mode == "validate") {
    std::locale previous =
        std::locale::global(std::locale(std::locale::classic(), new CommaPunctuation));
    if (json::Stringify(1234.5) != "1234.5" || json::Stringify(-0.0) != "-0.0" ||
        json::Stringify(1234.0) != "1234.0")
      return 3;
    std::locale::global(previous);
    std::vector<std::string> inputs = {"0",
                                       "-0",
                                       "-0.0",
                                       "00",
                                       "01",
                                       "1.",
                                       "1e+",
                                       "-",
                                       "--1",
                                       "1+2",
                                       "9223372036854775807",
                                       "-9223372036854775808",
                                       "9223372036854775808",
                                       "18446744073709551615",
                                       "1e309",
                                       "1e-9999",
                                       "0e-9999",
                                       "2.2250738585072012e-308",
                                       "2.2250738585072013e-308",
                                       "2.2250738585072014e-308",
                                       "2.2250738585072015e-308",
                                       "4.9406564584124654e-324",
                                       "NaN",
                                       "Infinity",
                                       "-Infinity"};
    std::mt19937_64 random(955);
    for (int i = 0; i < 20000; ++i) {
      uint64_t bits = random();
      double value;
      std::memcpy(&value, &bits, 8);
      if (!std::isfinite(value)) continue;
      char buffer[64];
      std::snprintf(buffer, sizeof(buffer), "%.17g", value);
      inputs.emplace_back(buffer);
      std::cout << "w " << bits << ' ' << std::string(json::Stringify(value)) << '\n';
    }
    for (auto& input : inputs) {
      String error;
      auto value = json::Parse(String(input), &error);
      std::cout << "p " << input << ' ' << value.type_index();
      if (value.type_index() == TypeIndex::kTVMFFIInt) std::cout << ' ' << value.cast<int64_t>();
      if (value.type_index() == TypeIndex::kTVMFFIFloat) {
        double number = value.cast<double>();
        uint64_t bits;
        std::memcpy(&bits, &number, 8);
        std::cout << ' ' << bits;
      }
      std::cout << ' ' << std::string(error) << '\n';
    }
    return 0;
  }
  std::string kind = argv[2];
  int iterations = std::stoi(argv[3]);
  std::string input;
  if (kind == "int") input = "1234567890123";
  if (kind == "float") input = "12345.6789012345";
  if (kind == "integral_float") input = "1234567.0";
  if (kind == "scientific") input = "1.2345678901234567e-123";
  if (kind == "ints" || kind == "floats") {
    input = "[";
    for (int i = 0; i < 512; ++i) {
      if (i) input += ",";
      input += std::to_string(i * 1789 - 123456);
      if (kind == "floats") input += ".123456789012345";
    }
    input += "]";
  }
  if (kind == "mixed") {
    input = "{";
    for (int i = 0; i < 32; ++i) {
      if (i) input += ",";
      input += '"' + std::to_string(i) + '"';
      input += ":[123.456789012345,1234567890,\"payload-string\",true,null]";
    }
    input += "}";
  }
  String text(input);
  auto value = json::Parse(text);
  uint64_t checksum = 0;
  for (int i = 0; i < 100; ++i) {
    auto parsed = json::Parse(text);
    checksum += json::Stringify(parsed).size();
  }
  auto start = std::chrono::steady_clock::now();
  if (mode == "parse")
    for (int i = 0; i < iterations; ++i) checksum += json::Parse(text).type_index();
  else
    for (int i = 0; i < iterations; ++i) checksum += json::Stringify(value).size();
  auto stop = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double, std::nano>(stop - start).count();
  std::cout << elapsed / iterations << ' ' << checksum << '\n';
}
