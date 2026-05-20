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

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "direct/common/_poisson_impl.h"

namespace nb = nanobind;

namespace direct::common {

namespace {

using Mask2D =
    nb::ndarray<std::int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Radius2D =
    nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

void BindPoisson(int nx, int ny, int max_attempts, Mask2D mask,
                 Radius2D radius_x, Radius2D radius_y, int seed) {
  const std::size_t expected_rows = static_cast<std::size_t>(nx);
  const std::size_t expected_cols = static_cast<std::size_t>(ny);
  if (mask.shape(0) != expected_rows || mask.shape(1) != expected_cols ||
      radius_x.shape(0) != expected_rows ||
      radius_x.shape(1) != expected_cols ||
      radius_y.shape(0) != expected_rows ||
      radius_y.shape(1) != expected_cols) {
    throw std::invalid_argument(
        "mask, radius_x and radius_y must all have shape (nx, ny)");
  }

  std::span<std::int64_t> mask_view(mask.data(), expected_rows * expected_cols);
  std::span<const double> rx_view(radius_x.data(),
                                  expected_rows * expected_cols);
  std::span<const double> ry_view(radius_y.data(),
                                  expected_rows * expected_cols);

  nb::gil_scoped_release release;
  Poisson(nx, ny, max_attempts, mask_view, rx_view, ry_view, seed);
}

}  // namespace

}  // namespace direct::common

NB_MODULE(_poisson, m) {
  m.doc() = "Variable-density Poisson-disc sub-sampling.";
  m.def("poisson", &direct::common::BindPoisson, nb::arg("nx"), nb::arg("ny"),
        nb::arg("max_attempts"), nb::arg("mask").noconvert(),
        nb::arg("radius_x").noconvert(), nb::arg("radius_y").noconvert(),
        nb::arg("seed"));
}
