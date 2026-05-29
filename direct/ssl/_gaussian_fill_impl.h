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

#ifndef DIRECT_SSL_GAUSSIAN_FILL_IMPL_H_
#define DIRECT_SSL_GAUSSIAN_FILL_IMPL_H_

#include <cstddef>
#include <cstdint>
#include <span>

#include "direct/common/rng.h"

namespace direct::ssl {

/// Fill `output_mask` with `nonzero_mask_count + 1` distinct hits drawn
/// from a 2D Gaussian centered at (center_x, center_y) with widths
/// (nrow - 1) / std_scale and (ncol - 1) / std_scale, accepting only
/// positions where the input `mask` is set.
inline void GaussianFill(int nonzero_mask_count, int nrow, int ncol,
                         int center_x, int center_y, double std_scale,
                         std::span<const std::int64_t> mask,
                         std::span<std::int64_t> output_mask, int seed) {
  direct::common::Rng rng(seed);

  const double std_x =
      static_cast<double>(nrow - 1) / std_scale;
  const double std_y =
      static_cast<double>(ncol - 1) / std_scale;

  int count = 0;
  while (count <= nonzero_mask_count) {
    auto [sx, sy] = rng.normal_2d(static_cast<double>(center_x),
                                  static_cast<double>(center_y), std_x, std_y);
    const int ix = static_cast<int>(sx);
    const int iy = static_cast<int>(sy);
    if (ix >= 0 && ix < nrow && iy >= 0 && iy < ncol) {
      const std::size_t flat =
          static_cast<std::size_t>(ix) * static_cast<std::size_t>(ncol) +
          static_cast<std::size_t>(iy);
      if (mask[flat] == 1 && output_mask[flat] != 1) {
        output_mask[flat] = 1;
        ++count;
      }
    }
  }
}

}  // namespace direct::ssl

#endif  // DIRECT_SSL_GAUSSIAN_FILL_IMPL_H_
