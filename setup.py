#!/usr/bin/env python

"""The setup script."""
import ast

with open("direct/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("history.md") as history_file:
    history = history_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().split("\n")

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]


setup(
    author="Jonas Teuwen",
    author_email="j.teuwen@nki.nl",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    description="DIRECT - Deep Image REConsTruction - is a deep learning"
    " framework for medical  data reconstruction.",
    install_requires=requirements,
    license="Apache 2.0 License",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="direct",
    name="direct",
    packages=find_packages(include=["direct", "direct.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/directgroup/direct",
    version=version,
    zip_safe=False,
)
