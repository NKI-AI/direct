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

#ifndef DIRECT_COMMON_RNG_H_
#define DIRECT_COMMON_RNG_H_

#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <utility>

namespace direct::common {

/// Cross-platform deterministic RNG used by the mask-sampling kernels.
class Rng {
 public:
  explicit Rng(int seed)
      : gen_(static_cast<std::uint64_t>(seed)), uniform_(0.0, 1.0) {}

  /// Uniform sample in (0, 1). The Box-Muller path below requires the
  /// open interval to avoid log(0); std::generate_canonical can return 0,
  /// so we resample on the (extremely rare) zero draw.
  double uniform() {
    double u = uniform_(gen_);
    while (u <= 0.0) {
      u = uniform_(gen_);
    }
    return u;
  }

  /// Random integer in {0, 1, ..., upper - 1}.
  int randint(int upper) {
    return static_cast<int>(uniform_(gen_) * static_cast<double>(upper));
  }

  /// Single Box-Muller draw from N(mu, std).
  double normal_1d(double mu, double std) {
    const double r = std::sqrt(-2.0 * std::log(uniform()));
    const double theta = 2.0 * std::numbers::pi * uniform();
    return mu + r * std::cos(theta) * std;
  }

  /// Pair of (correlated via shared radius/angle) Box-Muller draws,
  /// matching the original Cython random_normal_2d behavior.
  std::pair<double, double> normal_2d(double mu_x, double mu_y, double std_x,
                                      double std_y) {
    const double r = std::sqrt(-2.0 * std::log(uniform()));
    const double theta = 2.0 * std::numbers::pi * uniform();
    const double x = mu_x + r * std::cos(theta) * std_x;
    const double y = mu_y + r * std::sin(theta) * std_y;
    return {x, y};
  }

 private:
  std::mt19937_64 gen_;
  std::uniform_real_distribution<double> uniform_;
};

}  // namespace direct::common

#endif  // DIRECT_COMMON_RNG_H_
