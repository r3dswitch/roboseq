"""Installation script for the 'isaacgymenvs' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "gym",
    "matplotlib==3.5.1",
    "tb-nightly",
    "tqdm",
    "ipdb",
]

# Installation operation
setup(
    name="roboseq",
    author="r3dswitch",
    version="0.1",
    description="Sequential Dextrous Hand Manipulation",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
    ],
    zip_safe=False,
)

# EOF
