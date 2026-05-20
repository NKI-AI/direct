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

#include "direct/ssl/_gaussian_fill_impl.h"

namespace nb = nanobind;

namespace direct::ssl {

namespace {

using Mask2D =
    nb::ndarray<std::int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using ConstMask2D =
    nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

Mask2D BindGaussianFill(int nonzero_mask_count, int nrow, int ncol,
                        int center_x, int center_y, double std_scale,
                        ConstMask2D mask, Mask2D output_mask, int seed) {
  const std::size_t expected_rows = static_cast<std::size_t>(nrow);
  const std::size_t expected_cols = static_cast<std::size_t>(ncol);
  if (mask.shape(0) != expected_rows || mask.shape(1) != expected_cols ||
      output_mask.shape(0) != expected_rows ||
      output_mask.shape(1) != expected_cols) {
    throw std::invalid_argument(
        "mask and output_mask must have shape (nrow, ncol)");
  }

  std::span<const std::int64_t> mask_view(mask.data(),
                                          expected_rows * expected_cols);
  std::span<std::int64_t> out_view(output_mask.data(),
                                   expected_rows * expected_cols);
  {
    nb::gil_scoped_release release;
    GaussianFill(nonzero_mask_count, nrow, ncol, center_x, center_y, std_scale,
                 mask_view, out_view, seed);
  }
  return output_mask;
}

}  // namespace

}  // namespace direct::ssl

NB_MODULE(_gaussian_fill, m) {
  m.doc() = "Gaussian rejection sampling restricted to an input mask.";
  m.def("gaussian_fill", &direct::ssl::BindGaussianFill,
        nb::arg("nonzero_mask_count"), nb::arg("nrow"), nb::arg("ncol"),
        nb::arg("center_x"), nb::arg("center_y"), nb::arg("std_scale"),
        nb::arg("mask").noconvert(), nb::arg("output_mask").noconvert(),
        nb::arg("seed"));
}
