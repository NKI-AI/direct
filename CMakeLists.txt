# Copyright 2026 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18...3.30)
project(direct LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# scikit-build-core sets Python_EXECUTABLE; locate the matching dev module.
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

# nanobind is found through its installed CMake package. When building
# via scikit-build-core, the helper command below resolves the path
# for the build-time nanobind wheel.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

set(_direct_includes "${CMAKE_CURRENT_SOURCE_DIR}")

nanobind_add_module(_gaussian
  STABLE_ABI
  NB_STATIC
  direct/common/_gaussian.cpp
)
target_include_directories(_gaussian PRIVATE "${_direct_includes}")

nanobind_add_module(_poisson
  STABLE_ABI
  NB_STATIC
  direct/common/_poisson.cpp
)
target_include_directories(_poisson PRIVATE "${_direct_includes}")

nanobind_add_module(_gaussian_fill
  STABLE_ABI
  NB_STATIC
  direct/ssl/_gaussian_fill.cpp
)
target_include_directories(_gaussian_fill PRIVATE "${_direct_includes}")

install(TARGETS _gaussian      LIBRARY DESTINATION direct/common)
install(TARGETS _poisson       LIBRARY DESTINATION direct/common)
install(TARGETS _gaussian_fill LIBRARY DESTINATION direct/ssl)
