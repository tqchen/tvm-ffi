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
#ifndef TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TIMER_H_
#define TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TIMER_H_

// The two timing loops, and nothing else.
//
// There are only two kinds of arm.  A stationary arm leaves the fixture unchanged, so a fixed
// loop over one fixture measures it.  A consuming arm does not: an in-place map on a fixture
// with a pointer-shared subtree un-shares it, so the second iteration would traverse a
// different graph.  That one runs over a pool of independent copies, rebuilt untimed between
// timed passes.
//
// Both are templates, so the arm is inlined into the loop and nothing type-erased sits in the
// timed path -- `walk_floor` is a ~4 ns/node quantity and an indirect call would show.
//
// Both also take a `drain`, run after the clock is read. A rebuilding arm parks its output
// rather than destroying it, because releasing a rebuilt subgraph walks and frees every node in
// it and that teardown would otherwise be charged to the arm that built it -- inflating exactly
// the arms that allocate most.
//
// Repeat counts are fixed per fixture, declared where the fixture is, and chosen so a sample
// lands in the milliseconds: far above the two clock reads bracketing it.

#include <algorithm>
#include <chrono>
#include <vector>

namespace tvm {
namespace ffi {
namespace bench {

/*! \brief One untimed warm-up pass, then this many timed samples; the median is reported. */
constexpr int kSamples = 9;
/*! \brief Independent pinned processes whose medians the reporter medians again. */
constexpr int kProcessRuns = 5;
/*! \brief Independent copies a consuming arm traverses per timed pass. */
constexpr int kPoolSize = 64;

constexpr const char* kMethodDescription =
    "one untimed warm-up pass, 9 timed samples per arm per process with a fixed per-fixture "
    "repeat count, process value = median of its 9 samples, reported value = median of 5 "
    "pinned process medians";

inline double NowNs() {
  return std::chrono::duration<double, std::nano>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline double Median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

/*!
 * \brief Time a stationary arm: \p repeats traversals of one fixture, nine times.
 * \return Nanoseconds per traversal, the median of the nine samples.
 */
template <typename Run, typename Drain>
double MeasureStationary(int repeats, Run run, Drain drain) {
  for (int i = 0; i < repeats; ++i) run();  // warm-up, untimed
  drain();
  std::vector<double> samples;
  samples.reserve(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    double begin = NowNs();
    for (int i = 0; i < repeats; ++i) run();
    double elapsed = NowNs() - begin;
    drain();  // teardown of everything the batch built, outside the clock
    samples.push_back(elapsed / repeats);
  }
  return Median(std::move(samples));
}

/*!
 * \brief Time a consuming arm over a pool of independent copies.
 *
 * `refill` rebuilds the pool untimed; `run(i)` traverses copy `i`.  One timed pass covers the
 * whole pool, and \p passes of them make a sample, so the pool is rebuilt between passes and
 * no traversal ever sees a graph an earlier one already changed.
 */
template <typename Refill, typename Run, typename Drain>
double MeasurePooled(int passes, Refill refill, Run run, Drain drain) {
  refill();
  for (int i = 0; i < kPoolSize; ++i) run(i);  // warm-up, untimed
  drain();
  std::vector<double> samples;
  samples.reserve(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    double total = 0;
    for (int pass = 0; pass < passes; ++pass) {
      refill();
      double begin = NowNs();
      for (int i = 0; i < kPoolSize; ++i) run(i);
      total += NowNs() - begin;
      drain();  // outside the clock
    }
    samples.push_back(total / (static_cast<double>(passes) * kPoolSize));
  }
  return Median(std::move(samples));
}

}  // namespace bench
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_TIMER_H_
