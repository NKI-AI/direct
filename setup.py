#!/usr/bin/env python
# coding=utf-8
"""The setup script."""

import ast
import pathlib

from setuptools import Extension, find_packages, setup  # type: ignore
from setuptools.command.build_ext import build_ext


class _build_ext(build_ext):
    def run(self):
        import numpy as np

        self.include_dirs.append(np.get_include())
        super().run()

    def finalize_options(self):
        from Cython.Build import cythonize

        self.distribution.ext_modules = cythonize(self.distribution.ext_modules)
        super().finalize_options()


with open("direct/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break

with open("README.rst") as readme_file:
    readme = readme_file.read()


setup(
    author="Jonas Teuwen, George Yiasemis",
    author_email="j.teuwen@nki.nl, g.yiasemis@nki.nl",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="DIRECT - Deep Image REConsTruction - is a deep learning" " framework for MRI reconstruction.",
    entry_points={
        "console_scripts": [
            "direct=direct.cli:main",
        ],
    },
    setup_requires=["numpy", "cython"],
    install_requires=[
        "numpy>=1.21.2",
        "h5py==3.3.0",
        "omegaconf==2.1.1",
        "torch==1.11.0",
        "torchvision",
        "scikit-image>=0.19.0",
        "scikit-learn>=1.0.1",
        "tensorboard>=2.7.0",
        "tqdm",
        "protobuf==3.20.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "sphinx_copybutton",
            "numpydoc",
            "myst_parser",
            "sphinx-book-theme",
            "pylint",
            "packaging",
            "boto3",
            "ismrmrd>=1.9.5",
            "pyxb",
        ],
    },
    license="Apache Software License 2.0",
    long_description=readme,
    include_package_data=True,
    keywords="direct",
    name="direct",
    packages=find_packages(include=["direct", "direct.*"]),
    test_suite="tests",
    url="https://github.com/NKI-AI/direct",
    version=version,
    zip_safe=False,
    cmdclass={"build_ext": _build_ext},
    ext_modules=[
        Extension("direct.common._poisson", sources=[str(pathlib.Path(".") / "direct" / "common" / "_poisson.pyx")])
    ],
)
