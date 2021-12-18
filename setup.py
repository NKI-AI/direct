#!/usr/bin/env python
# coding=utf-8
"""The setup script."""
import ast

from setuptools import find_packages, setup  # type: ignore

with open("direct/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break

with open("README.md") as readme_file:
    readme = readme_file.read()


setup(
    author="Jonas Teuwen",
    author_email="j.teuwen@nki.nl",
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
    description="DIRECT - Deep Image REConsTruction - is a deep learning"
    " framework for MRI reconstruction.",
    entry_points={
        "console_scripts": [
            "direct=direct.cli:main",
        ],
    },
    install_requires=[
        "numpy>=1.20.0",
        "h5py>=2.10.0",
        "omegaconf>=2.0.0",
        "torch==1.10.0",
        "torchvision",
        "scikit-image>=0.18.1",
        "scikit-learn>=0.24.2",
        "pyxb==1.2.6",
        "ismrmrd==1.9.1",
        "tensorboard>=2.5.0",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "sphinx_copybutton",
            "numpydoc",
            "myst_parser",
            "sphinx-book-theme",
            "pylint",
            "sewar",
            "packaging",
        ],
    },
    license="Apache Software License 2.0",
    long_description=readme,
    include_package_data=True,
    keywords="direct",
    name="direct",
    packages=find_packages(include=["direct", "direct.*"]),
    test_suite="tests",
    url="https://github.com/directgroup/direct",
    version=version,
    zip_safe=False,
)
