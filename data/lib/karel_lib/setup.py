#!/usr/bin/env python


"""
Setup script for karel
"""
import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme_content = "README: https://github.com/carpedm20/karel"

requirements = [
    "ply",
    "numpy",
    "prompt_toolkit",
]

version = None
author = None
email = None
source = None
description = None

with open(os.path.join("karel", "__init__.py")) as f:
    for line in f:
        if line.strip().startswith("__version__"):
            version = line.split("=")[1].strip().replace('"', "").replace("'", "")
        elif line.strip().startswith("__author__"):
            author = line.split("=")[1].strip().replace('"', "").replace("'", "")
        elif line.strip().startswith("__email__"):
            email = line.split("=")[1].strip().replace('"', "").replace("'", "")
        elif line.strip().startswith("__source__"):
            source = line.split("=")[1].strip().replace('"', "").replace("'", "")
        elif line.strip().startswith("__description__"):
            description = line.split("=")[1].strip().replace('"', "").replace("'", "")
        elif None not in (version, author, email, source, description):
            break

setup(
    name="karel",
    author=author,
    author_email=email,
    license="BSD License",
    keywords=["Interpreter and code generator of Karel"],
    description=description,
    long_description=readme_content,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: BSD License",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Communications :: Chat",
    ],
    include_package_data=True,
    packages=["karel"],
    install_requires=requirements,
    url=source,
    version=version,
    zip_safe=True,
)
