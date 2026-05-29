// Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "direct/common/_gaussian_impl.h"
#include "direct/common/_poisson_impl.h"
#include "direct/ssl/_gaussian_fill_impl.h"

namespace direct::common {
namespace {

TEST(GaussianMask1D, FillsExactlyNonzeroCountPlusOne) {
  constexpr int kN = 256;
  constexpr int kNonzero = 32;
  std::vector<std::int64_t> mask(kN, 0);
  GaussianMask1D(kNonzero, kN, kN / 2, 6.0, mask, 1234);
  EXPECT_EQ(std::accumulate(mask.begin(), mask.end(), std::int64_t{0}),
            kNonzero + 1);
  for (int i = 0; i < kN; ++i) {
    EXPECT_TRUE(mask[i] == 0 || mask[i] == 1);
  }
}

TEST(GaussianMask1D, IsDeterministicForSameSeed) {
  constexpr int kN = 128;
  std::vector<std::int64_t> a(kN, 0);
  std::vector<std::int64_t> b(kN, 0);
  GaussianMask1D(20, kN, kN / 2, 5.0, a, 7);
  GaussianMask1D(20, kN, kN / 2, 5.0, b, 7);
  EXPECT_EQ(a, b);
}

TEST(GaussianMask2D, FillsExactlyNonzeroCountPlusOne) {
  constexpr int kRows = 64;
  constexpr int kCols = 64;
  constexpr int kNonzero = 80;
  std::vector<std::int64_t> mask(static_cast<std::size_t>(kRows * kCols), 0);
  GaussianMask2D(kNonzero, kRows, kCols, kRows / 2, kCols / 2, 4.0, 4.0, mask,
                 42);
  EXPECT_EQ(std::accumulate(mask.begin(), mask.end(), std::int64_t{0}),
            kNonzero + 1);
}

TEST(Poisson, AllSamplesRespectApproximateMinimumDistance) {
  // The algorithm rejects candidates in float space and then stores them at
  // their integer truncation, so two stored cells can sit at most sqrt(2)
  // closer than the radius.
  constexpr int kNx = 40;
  constexpr int kNy = 40;
  constexpr double kRadius = 3.5;
  const double kTolerance = std::sqrt(2.0);
  std::vector<std::int64_t> mask(static_cast<std::size_t>(kNx * kNy), 0);
  std::vector<double> rx(static_cast<std::size_t>(kNx * kNy), kRadius);
  std::vector<double> ry(static_cast<std::size_t>(kNx * kNy), kRadius);
  Poisson(kNx, kNy, /*max_attempts=*/30, mask, rx, ry, 7);
  EXPECT_GE(std::accumulate(mask.begin(), mask.end(), std::int64_t{0}), 1);
  for (int x1 = 0; x1 < kNx; ++x1) {
    for (int y1 = 0; y1 < kNy; ++y1) {
      if (mask[static_cast<std::size_t>(x1 * kNy + y1)] != 1) {
        continue;
      }
      for (int x2 = 0; x2 < kNx; ++x2) {
        for (int y2 = 0; y2 < kNy; ++y2) {
          if ((x1 == x2 && y1 == y2) ||
              mask[static_cast<std::size_t>(x2 * kNy + y2)] != 1) {
            continue;
          }
          const double dx = static_cast<double>(x1 - x2);
          const double dy = static_cast<double>(y1 - y2);
          const double dist = std::sqrt(dx * dx + dy * dy);
          EXPECT_GE(dist, kRadius - kTolerance);
        }
      }
    }
  }
}

}  // namespace
}  // namespace direct::common

namespace direct::ssl {
namespace {

TEST(GaussianFill, OnlyFillsPositionsWhereInputMaskIsSet) {
  constexpr int kRows = 64;
  constexpr int kCols = 64;
  constexpr int kNonzero = 50;
  std::vector<std::int64_t> input(
      static_cast<std::size_t>(kRows * kCols), 0);
  for (int r = kRows / 4; r < 3 * kRows / 4; ++r) {
    for (int c = kCols / 4; c < 3 * kCols / 4; ++c) {
      input[static_cast<std::size_t>(r * kCols + c)] = 1;
    }
  }
  std::vector<std::int64_t> output(
      static_cast<std::size_t>(kRows * kCols), 0);
  GaussianFill(kNonzero, kRows, kCols, kRows / 2, kCols / 2, 4.0, input, output,
               13);
  EXPECT_EQ(std::accumulate(output.begin(), output.end(), std::int64_t{0}),
            kNonzero + 1);
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (output[i] == 1) {
      EXPECT_EQ(input[i], 1);
    }
  }
}

}  // namespace
}  // namespace direct::ssl
