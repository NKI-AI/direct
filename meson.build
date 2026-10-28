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

project('direct', 'cpp',
  version : '2.2.1',
  license : 'Apache-2.0',
  meson_version : '>=1.3',
  default_options : [
    'warning_level=1',
    'cpp_std=c++20',
    'default_library=static',
    # Optimized by default (mirrors the Bazel `--config=release` -O3 -DNDEBUG).
    # Override on the CLI with e.g. `meson setup builddir --buildtype=debug`.
    'buildtype=release',
    'b_ndebug=if-release',
    # Stable-ABI (abi3) builds are opt-in; see the Python extension section
    # below. Off by default so a plain source build targets the running
    # interpreter and works on every supported version -- including 3.11, which
    # nanobind cannot build against the stable ABI. The release workflow turns
    # it on (`-Dpython.allow_limited_api=true`) for the cp312 wheels.
    'python.allow_limited_api=false',
  ])

py = import('python').find_installation(pure : false)

# nanobind resolves from subprojects/nanobind.wrap (with subprojects/robin-map.wrap
# as its transitive dependency). Under --wrap-mode=forcefallback (set in
# pyproject.toml) it always builds from source, so the extensions are
# self-contained. Pinned to the same version as `nanobind_bazel` in MODULE.bazel.
nanobind_dep = dependency('nanobind')

# Repo root on the include path, matching CMake's `_direct_includes` and Bazel's
# `strip_include_prefix = "/"`, so the sources can `#include "direct/common/..."`.
inc = include_directories('.')

# Target CPython's stable ABI (abi3) only when it is both opted in AND available:
# `python.allow_limited_api` is off by default (see default_options above), and
# nanobind only supports the stable ABI from 3.12 onwards, so 3.11 always gets a
# version-specific extension whatever the option says. When abi3 is on, a single
# cp312-abi3 wheel serves every CPython >= 3.12. ('' is Meson's "no limited API"
# default; meson-python keys its abi3 wheel tag off the same option.)
if get_option('python.allow_limited_api') and py.language_version().version_compare('>=3.12')
  limited_api = '3.12'
else
  limited_api = ''
endif

# Each entry: [module name, source file, package subdirectory].
extensions = [
  ['_gaussian', 'direct/common/_gaussian.cpp', 'direct/common'],
  ['_poisson', 'direct/common/_poisson.cpp', 'direct/common'],
  ['_gaussian_fill', 'direct/ssl/_gaussian_fill.cpp', 'direct/ssl'],
]

foreach ext : extensions
  py.extension_module(ext[0], ext[1],
    include_directories : inc,
    dependencies : nanobind_dep,
    limited_api : limited_api,
    install : true,
    subdir : ext[2])
endforeach

# Pure-Python package. The native extensions above install alongside these into
# the same site-packages `direct/` directory. The C++ sources/headers, Bazel
# build files, and bytecode caches have no place in the wheel, so drop them;
# the `.pyi` type stubs are kept.
install_subdir('direct',
  install_dir : py.get_install_dir(),
  exclude_files : [
    'BUILD.bazel',
    'common/BUILD.bazel',
    'common/_gaussian.cpp',
    'common/_gaussian_impl.h',
    'common/_gaussian_test.cc',
    'common/_poisson.cpp',
    'common/_poisson_impl.h',
    'common/rng.h',
    'ssl/BUILD.bazel',
    'ssl/_gaussian_fill.cpp',
    'ssl/_gaussian_fill_impl.h',
  ],
  exclude_directories : ['__pycache__'])
