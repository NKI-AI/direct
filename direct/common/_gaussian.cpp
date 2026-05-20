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

#include "direct/common/_gaussian_impl.h"

namespace nb = nanobind;

namespace direct::common {

namespace {

using Mask1D =
    nb::ndarray<std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Mask2D =
    nb::ndarray<std::int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using StdArr =
    nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

void BindGaussianMask1D(int nonzero_count, int n, int center, double std,
                        Mask1D mask, int seed) {
  if (static_cast<std::size_t>(n) > mask.shape(0)) {
    throw std::invalid_argument("mask must have at least n entries");
  }
  std::span<std::int64_t> view(mask.data(), mask.shape(0));
  nb::gil_scoped_release release;
  GaussianMask1D(nonzero_count, n, center, std, view, seed);
}

void BindGaussianMask2D(int nonzero_count, int nrow, int ncol, int center_x,
                        int center_y, StdArr std, Mask2D mask, int seed) {
  if (std.shape(0) < 2) {
    throw std::invalid_argument("std must have at least 2 entries");
  }
  if (mask.shape(0) != static_cast<std::size_t>(nrow) ||
      mask.shape(1) != static_cast<std::size_t>(ncol)) {
    throw std::invalid_argument("mask shape does not match (nrow, ncol)");
  }
  const double std_x = std.data()[0];
  const double std_y = std.data()[1];
  std::span<std::int64_t> view(mask.data(), mask.shape(0) * mask.shape(1));
  nb::gil_scoped_release release;
  GaussianMask2D(nonzero_count, nrow, ncol, center_x, center_y, std_x, std_y,
                 view, seed);
}

}  // namespace

}  // namespace direct::common

NB_MODULE(_gaussian, m) {
  m.doc() = "Gaussian rejection sampling for sub-sampling masks.";
  m.def("gaussian_mask_1d", &direct::common::BindGaussianMask1D,
        nb::arg("nonzero_count"), nb::arg("n"), nb::arg("center"),
        nb::arg("std"), nb::arg("mask").noconvert(), nb::arg("seed"));
  m.def("gaussian_mask_2d", &direct::common::BindGaussianMask2D,
        nb::arg("nonzero_count"), nb::arg("nrow"), nb::arg("ncol"),
        nb::arg("center_x"), nb::arg("center_y"),
        nb::arg("std").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("seed"));
}
