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

#ifndef DIRECT_COMMON_GAUSSIAN_IMPL_H_
#define DIRECT_COMMON_GAUSSIAN_IMPL_H_

#include <cstdint>
#include <span>

#include "direct/common/rng.h"

namespace direct::common {

/// Rejection-sample positions in {0, ..., n-1} from N(center, std) and
/// flip the corresponding entries of `mask` to 1. Loops until exactly
/// `nonzero_count + 1` distinct hits are accumulated, mirroring the
/// "<=" termination in the original Cython.
inline void GaussianMask1D(int nonzero_count, int n, int center, double std,
                           std::span<std::int64_t> mask, int seed) {
  Rng rng(seed);
  int count = 0;
  while (count <= nonzero_count) {
    const double sample = rng.normal_1d(static_cast<double>(center), std);
    const int ind = static_cast<int>(sample);
    if (ind >= 0 && ind < n && mask[static_cast<std::size_t>(ind)] != 1) {
      mask[static_cast<std::size_t>(ind)] = 1;
      ++count;
    }
  }
}

/// Same idea as GaussianMask1D but for a 2D row-major mask of shape
/// (nrow, ncol). `std_x` and `std_y` are the per-axis Gaussian widths.
inline void GaussianMask2D(int nonzero_count, int nrow, int ncol, int center_x,
                           int center_y, double std_x, double std_y,
                           std::span<std::int64_t> mask, int seed) {
  Rng rng(seed);
  int count = 0;
  while (count <= nonzero_count) {
    auto [sx, sy] = rng.normal_2d(static_cast<double>(center_x),
                                  static_cast<double>(center_y), std_x, std_y);
    const int ix = static_cast<int>(sx);
    const int iy = static_cast<int>(sy);
    if (ix >= 0 && ix < nrow && iy >= 0 && iy < ncol) {
      const std::size_t flat =
          static_cast<std::size_t>(ix) * static_cast<std::size_t>(ncol) +
          static_cast<std::size_t>(iy);
      if (mask[flat] != 1) {
        mask[flat] = 1;
        ++count;
      }
    }
  }
}

}  // namespace direct::common

#endif  // DIRECT_COMMON_GAUSSIAN_IMPL_H_
