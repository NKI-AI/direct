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

#ifndef DIRECT_COMMON_POISSON_IMPL_H_
#define DIRECT_COMMON_POISSON_IMPL_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <vector>

#include "direct/common/rng.h"

namespace direct::common {

/// Variable-density Poisson-disc sampler. The grid is row-major
/// (nx, ny); `mask`, `radius_x`, `radius_y` are the same shape.
///
/// Algorithm adapted from
/// https://github.com/mikgroup/sigpy/blob/1817ff849d34d7cbbbcb503a1b310e7d8f95c242/sigpy/mri/samp.py#L158
/// (BSD-3, Copyright (c) 2016, Frank Ong / The Regents of the University
/// of California). The original Cython port lived in _poisson.pyx.
inline void Poisson(int nx, int ny, int max_attempts,
                    std::span<std::int64_t> mask,
                    std::span<const double> radius_x,
                    std::span<const double> radius_y, int seed) {
  Rng rng(seed);

  const auto idx = [ny](int x, int y) {
    return static_cast<std::size_t>(x) * static_cast<std::size_t>(ny) +
           static_cast<std::size_t>(y);
  };

  std::vector<std::int64_t> pxs(static_cast<std::size_t>(nx) *
                                static_cast<std::size_t>(ny));
  std::vector<std::int64_t> pys(static_cast<std::size_t>(nx) *
                                static_cast<std::size_t>(ny));

  pxs[0] = rng.randint(nx);
  pys[0] = rng.randint(ny);

  std::int64_t num_actives = 1;

  while (num_actives > 0) {
    const int i = rng.randint(static_cast<int>(num_actives));
    const std::int64_t px = pxs[static_cast<std::size_t>(i)];
    const std::int64_t py = pys[static_cast<std::size_t>(i)];
    const double rx = radius_x[idx(static_cast<int>(px), static_cast<int>(py))];
    const double ry = radius_y[idx(static_cast<int>(px), static_cast<int>(py))];

    bool done = false;
    int k = 0;
    double qx = 0.0;
    double qy = 0.0;

    while (!done && k < max_attempts) {
      const double v = rng.uniform() + 1.0;
      const double t = 2.0 * std::numbers::pi * rng.uniform();
      qx = static_cast<double>(px) + v * rx * std::cos(t);
      qy = static_cast<double>(py) + v * ry * std::sin(t);

      if (qx >= 0.0 && qx < static_cast<double>(nx) && qy >= 0.0 &&
          qy < static_cast<double>(ny)) {
        const int startx = std::max(static_cast<int>(qx - rx), 0);
        const int endx = std::min(static_cast<int>(qx + rx + 1.0), nx);
        const int starty = std::max(static_cast<int>(qy - ry), 0);
        const int endy = std::min(static_cast<int>(qy + ry + 1.0), ny);

        done = true;
        for (int x = startx; x < endx && done; ++x) {
          for (int y = starty; y < endy; ++y) {
            const std::size_t flat = idx(x, y);
            const double dx = (qx - static_cast<double>(x)) / radius_x[flat];
            const double dy = (qy - static_cast<double>(y)) / radius_y[flat];
            const double distance = dx * dx + dy * dy;
            if (mask[flat] == 1 && distance < 1.0) {
              done = false;
              break;
            }
          }
        }
      }

      ++k;
    }

    if (done) {
      pxs[static_cast<std::size_t>(num_actives)] = static_cast<std::int64_t>(qx);
      pys[static_cast<std::size_t>(num_actives)] = static_cast<std::int64_t>(qy);
      mask[idx(static_cast<int>(pxs[static_cast<std::size_t>(num_actives)]),
               static_cast<int>(pys[static_cast<std::size_t>(num_actives)]))] =
          1;
      ++num_actives;
    } else {
      --num_actives;
      pxs[static_cast<std::size_t>(i)] =
          pxs[static_cast<std::size_t>(num_actives)];
      pys[static_cast<std::size_t>(i)] =
          pys[static_cast<std::size_t>(num_actives)];
    }
  }
}

}  // namespace direct::common

#endif  // DIRECT_COMMON_POISSON_IMPL_H_
